import asyncio
import base64
import cv2
import numpy as np
import re  # Thêm re
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from openai import AsyncOpenAI
from paddleocr import PaddleOCR

from config import settings, logger

# Khởi tạo PaddleOCR
try:
    ocr = PaddleOCR(
        use_angle_cls=settings.paddle_use_angle_cls,
        lang=settings.paddle_language,
        use_gpu=settings.paddle_use_gpu,
    )
    logger.info("PaddleOCR engine đã được khởi tạo.")
except Exception as e:
    logger.error(f"Lỗi nghiêm trọng khi khởi tạo PaddleOCR: {e}", exc_info=True)
    ocr = None  # Đặt là None nếu lỗi

# Khởi tạo OpenAI client (nếu có key và được bật)
openai_client = None
if ocr and settings.enable_openai_fallback:  # Chỉ bật fallback nếu PaddleOCR chạy được
    if settings.openai_api_key:
        try:
            openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("Đã khởi tạo OpenAI client cho fallback.")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo OpenAI client: {e}", exc_info=True)
            openai_client = None  # Đặt là None nếu lỗi
    else:
        logger.warning(
            "OpenAI fallback được bật nhưng OPENAI_API_KEY chưa được cung cấp trong .env. Fallback sẽ không hoạt động."
        )
else:
    logger.info("OpenAI fallback không được bật hoặc PaddleOCR khởi tạo lỗi.")


# --- Hàm nội bộ ---


async def _run_paddle_ocr(plate_crop: np.ndarray, executor: ThreadPoolExecutor) -> str:
    """Thực hiện PaddleOCR trong executor."""
    if ocr is None:
        logger.error("PaddleOCR chưa được khởi tạo, không thể chạy OCR.")
        return ""
    if plate_crop is None or plate_crop.size == 0:
        return ""

    loop = asyncio.get_running_loop()
    try:

        def perform_ocr(image_array):
            # Logic OCR giống như trong main.py cũ
            results = ocr.ocr(image_array, cls=settings.paddle_use_angle_cls)
            text_list = []
            if results and results[0]:
                for line in results[0]:
                    if (
                        line
                        and len(line) >= 2
                        and isinstance(line[1], (tuple, list))
                        and len(line[1]) >= 1
                    ):
                        text_list.append(line[1][0])
            final_text = " ".join(text_list).strip()
            logger.debug(f"Kết quả PaddleOCR raw: {results}")
            return final_text

        return await loop.run_in_executor(executor, perform_ocr, plate_crop)
    except Exception as e:
        logger.error(f"Lỗi khi chạy PaddleOCR trong thread: {e}", exc_info=True)
        return ""


async def _run_openai_ocr(
    plate_crop: np.ndarray,
    executor: ThreadPoolExecutor,
    full_image: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> str:
    """Thực hiện OpenAI OCR trong executor, ưu tiên full_image nếu được cung cấp."""
    if openai_client is None:
        return ""

    image_to_encode = full_image if full_image is not None else plate_crop
    if image_to_encode is None or image_to_encode.size == 0:
        logger.warning("Image to encode for OpenAI is empty or None.")
        return ""

    loop = asyncio.get_running_loop()
    try:

        def encode_image(image_array):
            _, buffer = cv2.imencode(".jpg", image_array)
            return base64.b64encode(buffer).decode("utf-8")

        base64_image = await loop.run_in_executor(
            executor, encode_image, image_to_encode
        )

        prompt = (
            "You are an expert OCR specialized in Vietnamese vehicle license plates. "
            "Extract ONLY the license plate text from the image. "
            "Focus on Vietnamese formats like XX-YZ.ZZZZ, XX-YZZ.ZZ, XXYZ.ZZZZZ, XX-Y ZZZ.ZZ. "
        )
        if full_image is not None and bbox:
            prompt += f"The approximate bounding box of the plate within the full image is [x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}]. Focus your analysis there. "
        else:
            prompt += (
                "The provided image is likely a close-up crop of the license plate. "
            )

        prompt += (
            "Respond only with the extracted text, no extra formatting or explanations. "
            "If text cannot be extracted, respond with 'None'."
        )

        logger.info(
            f"Sending image (shape: {image_to_encode.shape}, source: {'full' if full_image is not None else 'crop'}) to OpenAI model: {settings.openai_model}"
        )
        response = await openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=50,
            timeout=15.0,
        )

        extracted_text = response.choices[0].message.content.strip()
        logger.info(f"Kết quả từ OpenAI: '{extracted_text}'")

        if not extracted_text or extracted_text.lower() == "none":
            return ""
        return extracted_text

    except Exception as e:
        if "timed out" in str(e).lower():
            logger.warning(f"Lỗi timeout khi gọi OpenAI API: {e}")
        else:
            logger.error(f"Lỗi khác khi gọi OpenAI API: {e}", exc_info=True)
        return ""


def _is_potentially_valid(ocr_text: str) -> bool:
    """Kiểm tra nhanh xem text có khả năng hợp lệ không (để quyết định fallback)."""
    if not ocr_text:
        return False
    # Logic kiểm tra đơn giản: độ dài tối thiểu và có cả chữ và số
    normalized = re.sub(r"[^A-Z0-9]", "", ocr_text.upper())
    if len(normalized) < 6:  # Độ dài tối thiểu
        return False
    if not re.search(r"\d", normalized) or not re.search(
        r"[A-Z]", normalized
    ):  # Phải có cả số và chữ
        return False
    return True


# --- Hàm chính để gọi từ main.py ---


async def get_ocr_result(
    plate_crop: np.ndarray,
    executor: ThreadPoolExecutor,
    full_image: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[str, str]:
    """
    Lấy kết quả OCR tốt nhất, xử lý fallback nếu cần.

    Args:
        plate_crop: The cropped image of the license plate.
        executor: ThreadPoolExecutor for running tasks.
        full_image: The original full image (optional, used for OpenAI fallback).
        bbox: The bounding box coordinates on the full_image (optional).

    Returns:
        A tuple containing the final OCR text and the engine used ('paddleocr' or 'openai').
    """
    engine_used = "paddleocr"  # Mặc định
    paddle_ocr_text = await _run_paddle_ocr(plate_crop, executor)
    logger.info(f"Kết quả PaddleOCR: '{paddle_ocr_text}'")

    # Kiểm tra nhanh validity của kết quả Paddle
    is_paddle_valid = _is_potentially_valid(paddle_ocr_text)
    logger.info(
        f"Kết quả kiểm tra nhanh PaddleOCR: {'Hợp lệ' if is_paddle_valid else 'KHÔNG hợp lệ'}"
    )

    # Kiểm tra điều kiện fallback
    should_fallback = (
        settings.enable_openai_fallback
        and openai_client is not None
        and not is_paddle_valid
    )
    logger.debug(
        f"Kiểm tra fallback: enable={settings.enable_openai_fallback}, "
        f"client_ready={openai_client is not None}, is_paddle_invalid={not is_paddle_valid} "
        f"==> should_fallback={should_fallback}"
    )

    if should_fallback:
        logger.info(
            f"Kết quả PaddleOCR ('{paddle_ocr_text}') không hợp lệ/đáng ngờ. Thử fallback với OpenAI..."
        )
        # Pass full_image and bbox to OpenAI OCR function
        openai_ocr_text = await _run_openai_ocr(
            plate_crop, executor, full_image=full_image, bbox=bbox
        )
        # Even if OpenAI fails, we return its (empty) result and 'openai' engine
        # This prevents returning the known bad Paddle result.
        return openai_ocr_text, "openai"
    else:
        return paddle_ocr_text, "paddleocr"

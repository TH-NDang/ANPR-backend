import asyncio
import base64
import cv2
import numpy as np
import re
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from openai import AsyncOpenAI
from paddleocr import PaddleOCR
import google.generativeai as genai
from PIL import Image
import io

from config import settings, logger

try:
    ocr = PaddleOCR(
        use_angle_cls=settings.paddle_use_angle_cls,
        lang=settings.paddle_language,
        use_gpu=settings.paddle_use_gpu,
    )
    logger.info("PaddleOCR engine đã được khởi tạo.")
except Exception as e:
    logger.error(f"Lỗi nghiêm trọng khi khởi tạo PaddleOCR: {e}", exc_info=True)
    ocr = None

openai_client = None
if settings.enable_openai_fallback and settings.openai_api_key:
    try:
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info("Đã khởi tạo OpenAI client cho fallback.")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo OpenAI client: {e}", exc_info=True)
        openai_client = None
else:
    logger.warning(
        "OpenAI fallback không được bật hoặc OPENAI_API_KEY chưa được cung cấp."
    )

gemini_client = None
if settings.enable_gemini_fallback and settings.gemini_api_key:
    try:
        genai.configure(api_key=settings.gemini_api_key)
        gemini_client = genai.GenerativeModel(settings.gemini_model)
        logger.info(
            f"Đã khởi tạo Gemini client model '{settings.gemini_model}' cho fallback."
        )
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo Gemini client: {e}", exc_info=True)
        gemini_client = None
else:
    logger.info(
        "Gemini fallback không được bật hoặc GEMINI_API_KEY chưa được cung cấp."
    )


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
            return final_text

        return await loop.run_in_executor(executor, perform_ocr, plate_crop)
    except RuntimeError as re:
        logger.error(
            f"Lỗi RuntimeError khi chạy PaddleOCR (có thể do lỗi primitive): {re}",
            exc_info=True,
        )
        return ""
    except Exception as e:
        logger.error(
            f"Lỗi không xác định khi chạy PaddleOCR trong thread: {e}", exc_info=True
        )
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

        if not extracted_text or extracted_text.lower() == "none":
            return ""
        return extracted_text

    except Exception as e:
        if "timed out" in str(e).lower():
            logger.warning(f"Lỗi timeout khi gọi OpenAI API: {e}")
        else:
            logger.error(f"Lỗi khác khi gọi OpenAI API: {e}", exc_info=True)
        return ""


async def _run_gemini_ocr(
    plate_crop: np.ndarray,
    executor: ThreadPoolExecutor,
    full_image: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> str:
    """Thực hiện Gemini OCR trong executor, ưu tiên full_image nếu được cung cấp."""
    if gemini_client is None:
        logger.warning(
            "Gemini client chưa khởi tạo, không thể chạy Gemini OCR fallback."
        )
        return ""
    image_to_process = full_image if full_image is not None else plate_crop
    if image_to_process is None or image_to_process.size == 0:
        logger.warning("Ảnh đầu vào cho Gemini OCR là None hoặc trống.")
        return ""
    loop = asyncio.get_running_loop()
    try:

        def encode_image_for_gemini(image_array_np):
            try:
                image_pil = Image.fromarray(
                    cv2.cvtColor(image_array_np, cv2.COLOR_BGR2RGB)
                )
                byte_arr = io.BytesIO()
                image_pil.save(byte_arr, format="JPEG")
                return byte_arr.getvalue()
            except Exception as encode_err:
                logger.error(f"Lỗi khi encode ảnh cho Gemini: {encode_err}")
                return None

        image_bytes = await loop.run_in_executor(
            executor, encode_image_for_gemini, image_to_process
        )
        if not image_bytes:
            return ""
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        prompt_parts = [
            "You are an expert OCR specialized in Vietnamese vehicle license plates. ",
            "Extract ONLY the license plate text from the image. ",
            "Focus on Vietnamese formats like XX-YZ.ZZZZ, XX-YZZ.ZZ, XXYZ.ZZZZZ, XX-Y ZZZ.ZZ etc. ",
        ]
        if full_image is not None and bbox:
            prompt_parts.append(
                f"The approximate bounding box of the plate within the full image is [x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}]. Focus your analysis there. "
            )
        else:
            prompt_parts.append(
                "The provided image is likely a close-up crop of the license plate. "
            )
        prompt_parts.extend(
            [
                "Respond only with the extracted text, no extra formatting or explanations. ",
                "If the text cannot be reliably extracted, respond with 'None'.",
                image_part,
            ]
        )
        response = await gemini_client.generate_content_async(
            prompt_parts,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=50, temperature=0.1
            ),
        )
        extracted_text = response.text.strip()
        if not extracted_text or extracted_text.lower() == "none":
            logger.info("Gemini không thể đọc hoặc trả về 'None'.")
            return ""
        return extracted_text
    except Exception as e:
        logger.error(f"Lỗi khi gọi Gemini API: {e}", exc_info=True)
        return ""


def _is_potentially_valid(ocr_text: str) -> bool:
    """Kiểm tra nhanh xem text có khả năng hợp lệ không (để quyết định fallback)."""
    if not ocr_text:
        return False
    normalized = re.sub(r"[^A-Z0-9]", "", ocr_text.upper())
    if len(normalized) < 6:
        return False
    if not re.search(r"\d", normalized) or not re.search(r"[A-Z]", normalized):
        return False
    return True


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
        A tuple containing the final OCR text and the engine used ('paddleocr' or 'gemini').
    """
    paddle_ocr_text = await _run_paddle_ocr(plate_crop, executor)
    is_paddle_valid = _is_potentially_valid(paddle_ocr_text)

    should_fallback = (
        settings.enable_gemini_fallback
        and gemini_client is not None
        and not is_paddle_valid
    )
    if should_fallback:
        try:
            gemini_ocr_text = await _run_gemini_ocr(
                plate_crop, executor, full_image=full_image, bbox=bbox
            )
            return gemini_ocr_text, "gemini"
        except Exception as e:
            logger.error(f"Lỗi khi gọi Gemini API: {e}", exc_info=True)
            return paddle_ocr_text, "paddleocr"
    else:
        return paddle_ocr_text, "paddleocr"

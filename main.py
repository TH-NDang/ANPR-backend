from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import logging

# Import config và schemas
from config import settings
from schemas import (
    ProcessImageResponse,
    DetectionResult,
    PlateAnalysisResult,
    ErrorResponse,
    ImageUrlRequest,
)

# Import các processor và utils
from detection_processor import run_detection
from ocr_processor import get_ocr_result
from image_utils import decode_image, encode_image_to_base64, download_image_from_url
from analysis import analyze_license_plate


# Lấy logger cho module main
logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level.upper())
logger.info(
    f"Logger '{__name__}' được cấu hình với level: {settings.log_level.upper()}"
)

# *** KHỞI TẠO ỨNG DỤNG FastAPI TẠI ĐÂY ***
app = FastAPI(
    title="License Plate Recognition API",
    description="API nhận dạng biển số xe sử dụng YOLO và các engine OCR",
    version="1.1.0",  # Phiên bản sau khi refactor
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool executor cho xử lý đa luồng
executor = ThreadPoolExecutor()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Lỗi không mong muốn xảy ra: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Lỗi máy chủ nội bộ."},
    )


async def process_image_for_detection(
    image_np: np.ndarray, source: str = "unknown"
) -> ProcessImageResponse:
    """
    Hàm chung để xử lý ảnh dùng cho cả file upload và URL image.
    Args:
        image_np: Ảnh đầu vào dạng NumPy array.
        source: Nguồn gốc của ảnh ('file' hoặc 'url').
    """
    if image_np is None:
        raise HTTPException(
            status_code=422, detail="Không thể đọc hoặc giải mã file ảnh."
        )

    logger.info(
        f"[{source}] Bắt đầu process_image_for_detection. Kích thước ảnh: {image_np.shape}, dtype: {image_np.dtype}"
    )

    with ThreadPoolExecutor() as executor:
        # 1. Phát hiện
        detections_yolo = await run_detection(image_np, executor)
        logger.info(
            f"[{source}] YOLO phát hiện được tổng cộng {len(detections_yolo)} vùng."
        )
        all_results = []
        processed_image_np = image_np.copy()

        if not detections_yolo:
            logger.info(f"[{source}] Không phát hiện được biển số nào.")
            encoded_image = encode_image_to_base64(processed_image_np)
            return ProcessImageResponse(
                detections=[],
                processed_image_url=encoded_image,
                error="Không phát hiện được biển số.",
            )

        # 2. Xử lý từng detection

        # Xử lý từng biển số phát hiện được
        for i, (bbox, conf) in enumerate(detections_yolo):
            # *** Thêm log ERROR để kiểm tra luồng ***
            logger.error(
                f"[{source}] >>> ĐANG Ở ĐẦU VÒNG LẶP XỬ LÝ DETECTION {i+1} <<<"
            )
            # *** Thêm log bắt đầu vòng lặp ***
            logger.debug(f"[{source}] --- Bắt đầu xử lý detection {i+1} ---")

            logger.info(f"[{source}] Xử lý detection {i+1}/{len(detections_yolo)}")
            x1, y1, x2, y2 = bbox

            logger.warning(
                f"[{source}] Bounding box: ({x1}, {y1}, {x2}, {y2})"
            )  # Giữ lại log bbox này

            # Đảm bảo tọa độ nằm trong giới hạn của ảnh
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)

            if x1 >= x2 or y1 >= y2:
                continue
            plate_crop = image_np[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # 3. Lấy kết quả OCR (đã bao gồm fallback)
            final_ocr_text, engine_used = await get_ocr_result(plate_crop, executor)
            logger.info(
                f"[{source}] Kết quả OCR cuối cùng cho detection {i+1}: '{final_ocr_text}' (Engine: {engine_used})"
            )

            # 4. Phân tích kết quả OCR cuối cùng (Gọi trực tiếp analyze_license_plate)
            try:
                logger.debug(
                    f"[{source}] Chuẩn bị gọi analyze_license_plate với text: '{final_ocr_text}'"
                )
                final_analysis = analyze_license_plate(final_ocr_text, plate_crop)
                logger.info(
                    f"[{source}] Kết quả phân tích cuối cùng cho detection {i+1}: {final_analysis.__dict__}"
                )
            except Exception as e:
                logger.error(
                    f"[{source}] Lỗi khi phân tích biển số cuối cùng cho detection {i+1}: {e}",
                    exc_info=True,
                )
                # Tạo kết quả analysis rỗng/mặc định nếu lỗi
                final_analysis = PlateAnalysisResult(original=final_ocr_text or "N/A")

            # Vẽ kết quả lên ảnh
            draw_text = (
                final_ocr_text if final_ocr_text else "N/A"
            )  # Text để vẽ lên ảnh
            cv2.rectangle(
                processed_image_np,
                (x1, y1),
                (x2, y2),
                (0, 255, 0) if final_analysis.is_valid_format else (0, 0, 255),
                2,
            )
            cv2.putText(
                processed_image_np,
                draw_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if final_analysis.is_valid_format else (0, 0, 255),
                2,
            )

            # Tạo kết quả detection cuối cùng
            detection_entry = DetectionResult(
                plate_number=final_ocr_text or "Không đọc được",
                confidence_detection=conf,
                bounding_box=(x1, y1, x2, y2),
                plate_analysis=final_analysis,
                ocr_engine_used=engine_used,
            )
            all_results.append(detection_entry)
            logger.debug(f"[{source}] --- Kết thúc xử lý detection {i+1} ---")

    # Encode ảnh kết quả sang base64
    encoded_image_url = encode_image_to_base64(processed_image_np)

    logger.info(f"[{source}] Hoàn thành xử lý. {len(all_results)} biển số được xử lý.")
    return ProcessImageResponse(
        detections=all_results, processed_image_url=encoded_image_url
    )


@app.post(
    "/process-image",
    response_model=ProcessImageResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Yêu cầu không hợp lệ"},
        422: {
            "model": ErrorResponse,
            "description": "File không hợp lệ hoặc không phải ảnh",
        },
        500: {"model": ErrorResponse, "description": "Lỗi máy chủ nội bộ"},
    },
)
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint nhận ảnh và trả về kết quả nhận dạng biển số.
    """
    # Kiểm tra loại file
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Loại file không hợp lệ: {file.content_type}")
        raise HTTPException(
            status_code=415,
            detail=f"Loại file không được hỗ trợ. Chỉ chấp nhận file ảnh.",
        )

    # Đọc và decode ảnh
    contents = await file.read()
    # Giới hạn kích thước file (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Kích thước file quá lớn (tối đa {MAX_FILE_SIZE // 1024 // 1024}MB).",
        )

    image_np = decode_image(contents)
    logger.info(
        f"[file] Đã nhận và decode ảnh thành công, kích thước: {image_np.shape if image_np is not None else 'None'}"
    )

    # Truyền source='file' vào hàm xử lý
    return await process_image_for_detection(image_np, source="file")


@app.post(
    "/process-image-url",
    response_model=ProcessImageResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Yêu cầu không hợp lệ"},
        422: {
            "model": ErrorResponse,
            "description": "URL không hợp lệ hoặc không phải ảnh",
        },
        500: {"model": ErrorResponse, "description": "Lỗi máy chủ nội bộ"},
    },
)
async def process_image_url_endpoint(request: ImageUrlRequest):
    """
    Endpoint nhận URL hình ảnh và trả về kết quả nhận dạng biển số.
    """
    logger.info(f"[url] Đã nhận yêu cầu xử lý ảnh từ URL: {request.url}")

    # Tải ảnh từ URL
    image_np = download_image_from_url(str(request.url))
    if image_np is None:
        raise HTTPException(
            status_code=422, detail="Không thể tải hoặc xử lý ảnh từ URL cung cấp."
        )

    logger.info(f"[url] Đã tải ảnh từ URL thành công, kích thước: {image_np.shape}")

    # Truyền source='url' vào hàm xử lý
    return await process_image_for_detection(image_np, source="url")


# Thêm endpoint kiểm tra trạng thái
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "openai_enabled": settings.enable_openai_fallback
        and bool(settings.openai_api_key),
    }


if __name__ == "__main__":
    import uvicorn

    # Chạy uvicorn một cách đơn giản nhất, dựa vào command line cho log level
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,  # Quan trọng: Tắt reload để tránh hành vi không mong muốn
    )

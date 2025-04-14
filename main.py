from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import logging
import time  # Import time module

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
from services.detection_processor import run_detection
from services.ocr_processor import get_ocr_result
from utils.image_utils import (
    decode_image,
    encode_image_to_base64,
    download_image_from_url,
)
from utils.analysis import analyze_license_plate

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
    start_time = time.perf_counter()  # Start timer

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
            processing_time_ms = (
                time.perf_counter() - start_time
            ) * 1000  # Calculate time
            return ProcessImageResponse(
                detections=[],
                processed_image_url=encoded_image,
                error="No license plates detected.",
                processing_time_ms=processing_time_ms,
            )

        # 2. Process each detected plate
        for i, (bbox, conf) in enumerate(detections_yolo):
            logger.debug(f"[{source}] --- Starting processing detection {i+1} ---")
            logger.info(f"[{source}] Processing detection {i+1}/{len(detections_yolo)}")
            x1, y1, x2, y2 = bbox
            logger.warning(f"[{source}] Bounding box: ({x1}, {y1}, {x2}, {y2})")

            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)

            if x1 >= x2 or y1 >= y2:
                logger.warning(
                    f"[{source}] Skipping detection {i+1} due to invalid bbox after clamping."
                )
                continue
            plate_crop = image_np[y1:y2, x1:x2]
            if plate_crop.size == 0:
                logger.warning(
                    f"[{source}] Skipping detection {i+1} due to empty crop."
                )
                continue

            # 3. Get OCR result (includes fallback logic)
            # Pass the full image and bbox for potential OpenAI use
            final_ocr_text, engine_used = await get_ocr_result(
                plate_crop, executor, full_image=image_np, bbox=bbox
            )
            logger.info(
                f"[{source}] Final OCR result for detection {i+1}: '{final_ocr_text}' (Engine: {engine_used})"
            )

            # 4. Analyze final OCR result
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
                final_analysis = PlateAnalysisResult(original=final_ocr_text or "N/A")

            # Draw result on the image
            draw_text = final_ocr_text if final_ocr_text else "N/A"
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
                plate_number=final_ocr_text or "Cannot read",
                confidence_detection=conf,
                bounding_box=(x1, y1, x2, y2),
                plate_analysis=final_analysis,
                ocr_engine_used=engine_used,
            )
            all_results.append(detection_entry)
            logger.debug(f"[{source}] --- Finished processing detection {i+1} ---")

    # Encode result image to base64
    encoded_image_url = encode_image_to_base64(processed_image_np)

    processing_time_ms = (time.perf_counter() - start_time) * 1000  # Calculate time
    logger.info(
        f"[{source}] Processing complete. {len(all_results)} plates processed in {processing_time_ms:.2f} ms."
    )
    return ProcessImageResponse(
        detections=all_results,
        processed_image_url=encoded_image_url,
        processing_time_ms=processing_time_ms,
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
    Endpoint receives an image and returns license plate recognition results.
    """
    start_time = time.perf_counter()  # Start timer for the endpoint
    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Only image files are accepted.",
        )

    # Read and decode image
    contents = await file.read()
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

    # Call the processing function
    response = await process_image_for_detection(image_np, source="file")

    # Add endpoint processing time if not already set by the core function (e.g., if error occurred early)
    if response.processing_time_ms is None:
        response.processing_time_ms = (time.perf_counter() - start_time) * 1000

    return response


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
    Endpoint receives an image URL and returns license plate recognition results.
    """
    start_time = time.perf_counter()  # Start timer for the endpoint
    try:
        image_np = await download_image_from_url(str(request.url))
        logger.info(
            f"[url] Downloaded and decoded image from URL successfully, shape: {image_np.shape if image_np is not None else 'None'}"
        )
    except Exception as e:
        logger.error(
            f"[url] Error downloading or decoding image from URL: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=422, detail=f"Failed to process image from URL: {e}"
        )

    # Call the processing function
    response = await process_image_for_detection(image_np, source="url")

    # Add endpoint processing time if not already set
    if response.processing_time_ms is None:
        response.processing_time_ms = (time.perf_counter() - start_time) * 1000

    return response


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    # Basic check: Can we access settings?
    if settings.app_name:
        return {"status": "ok", "app_name": settings.app_name}
    else:
        raise HTTPException(status_code=503, detail="Service configuration missing")


if __name__ == "__main__":
    import uvicorn

    # Chạy uvicorn một cách đơn giản nhất, dựa vào command line cho log level
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,  # Quan trọng: Tắt reload để tránh hành vi không mong muốn
    )

# main.py - API nhận dạng biển số xe
import asyncio
import re
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import os
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, HttpUrl

# Import các module cần thiết
from ultralytics import YOLO
from paddleocr import PaddleOCR
from config import settings, logger
from schemas import ProcessImageResponse, DetectionResult, PlateAnalysisResult, ErrorResponse
from image_utils import decode_image, encode_image_to_base64, download_image_from_url
from analysis import analyze_license_plate

# Schema cho xử lý URL
class ImageUrlRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL của ảnh cần xử lý")

# Khởi tạo model
model_path = "best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy model YOLO tại {model_path}")

model = YOLO(model_path)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

app = FastAPI(
    title="License Plate Recognition API",
    description="API nhận dạng biển số xe sử dụng YOLO và PaddleOCR",
    version="1.0.0"
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

async def run_detection(image: np.ndarray) -> List[Tuple[List[int], float]]:
    """Chạy YOLO detection trong executor."""
    loop = asyncio.get_running_loop()
    try:
        def detect_plates():
            results = model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    xyxy = box.xyxy[0]
                    conf = box.conf[0]
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append(([x1, y1, x2, y2], float(conf)))
            
            return detections
            
        return await loop.run_in_executor(executor, detect_plates)
    except Exception as e:
        logger.error(f"Lỗi khi chạy detection: {e}", exc_info=True)
        return []

async def run_ocr(plate_image: np.ndarray) -> str:
    """Thực hiện OCR trong executor."""
    loop = asyncio.get_running_loop()
    try:
        def perform_ocr(image_array):
            if image_array is None or not isinstance(image_array, np.ndarray):
                return ""
                
            results = ocr.ocr(image_array, rec=True)
            return ' '.join([result[1][0] for result in results[0]] if results[0] else "")
            
        return await loop.run_in_executor(executor, perform_ocr, plate_image)
    except Exception as e:
        logger.error(f"Lỗi khi chạy OCR: {e}", exc_info=True)
        return ""

def process_plate_text(ocr_text: str, plate_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Phân tích biển số xe - kết hợp cả phân tích đơn giản và nâng cao.
    """
    if not ocr_text:
        # Trả về kết quả rỗng nếu không có text
        normalized = ""
        return {
            "original": ocr_text or "N/A",
            "normalized": normalized,
            "province_code": None,
            "province_name": None,
            "serial": None,
            "number": None,
            "plate_type": "unknown",
            "plate_type_info": {
                "name": "Không xác định",
                "description": "Chưa phân tích loại biển số"
            },
            "detected_color": None,
            "is_valid_format": False,
            "format_description": None
        }
        
    try:
        # Gọi hàm phân tích từ module analysis
        analysis_result = analyze_license_plate(ocr_text, plate_image)
        return analysis_result.__dict__
    except Exception as e:
        logger.error(f"Lỗi khi phân tích biển số: {e}", exc_info=True)
        # Fallback về phân tích đơn giản nếu có lỗi
        normalized = re.sub(r'[^A-Z0-9]', '', ocr_text.upper()) if ocr_text else ""
        return {
            "original": ocr_text,
            "normalized": normalized,
            "province_code": None,
            "province_name": None,
            "serial": None,
            "number": None,
            "plate_type": "unknown",
            "plate_type_info": {
                "name": "Không xác định",
                "description": "Chưa phân tích loại biển số"
            },
            "detected_color": None,
            "is_valid_format": False,
            "format_description": None
        }

async def process_image_for_detection(image_np: np.ndarray) -> ProcessImageResponse:
    """
    Hàm chung để xử lý ảnh dùng cho cả file upload và URL image.
    """
    if image_np is None:
        raise HTTPException(status_code=422, detail="Không thể đọc hoặc giải mã file ảnh.")
        
    # Chạy phát hiện biển số (YOLO)
    detections_yolo = await run_detection(image_np)

    all_results = []
    processed_image_np = image_np.copy() # Ảnh để vẽ kết quả lên

    if not detections_yolo:
        logger.info("Không phát hiện được biển số nào.")
        encoded_image = encode_image_to_base64(processed_image_np)
        return ProcessImageResponse(detections=[], processed_image_url=encoded_image, error="Không phát hiện được biển số.")

    # Xử lý từng biển số phát hiện được
    for bbox, conf in detections_yolo:
        x1, y1, x2, y2 = bbox
        
        # Đảm bảo tọa độ nằm trong giới hạn của ảnh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
        
        # Kiểm tra xem bbox có hợp lệ không
        if x1 >= x2 or y1 >= y2:
            continue
            
        # Cắt và xử lý ảnh biển số
        plate_crop = image_np[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue
            
        # Thực hiện OCR trên ảnh crop
        ocr_text = await run_ocr(plate_crop)
        
        # Vẽ kết quả lên ảnh
        cv2.rectangle(processed_image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(processed_image_np, ocr_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Phân tích biển số
        final_text = ocr_text if ocr_text else "Không đọc được"
        analysis_result = PlateAnalysisResult(**process_plate_text(ocr_text, plate_crop))
        
        # Tạo kết quả detection
        detection_entry = DetectionResult(
            plate_number=final_text,
            confidence_detection=conf,
            bounding_box=(x1, y1, x2, y2),
            plate_analysis=analysis_result
        )
        all_results.append(detection_entry)

    # Encode ảnh kết quả sang base64
    encoded_image_url = encode_image_to_base64(processed_image_np)

    logger.info(f"Hoàn thành xử lý. {len(all_results)} biển số được xử lý.")
    return ProcessImageResponse(detections=all_results, processed_image_url=encoded_image_url)

@app.post("/process-image",
          response_model=ProcessImageResponse,
          responses={
              400: {"model": ErrorResponse, "description": "Yêu cầu không hợp lệ"},
              422: {"model": ErrorResponse, "description": "File không hợp lệ hoặc không phải ảnh"},
              500: {"model": ErrorResponse, "description": "Lỗi máy chủ nội bộ"}
          })
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint nhận ảnh và trả về kết quả nhận dạng biển số.
    """
    # Kiểm tra loại file
    if not file.content_type or not file.content_type.startswith('image/'):
        logger.warning(f"Loại file không hợp lệ: {file.content_type}")
        raise HTTPException(status_code=415, detail=f"Loại file không được hỗ trợ. Chỉ chấp nhận file ảnh.")

    # Đọc và decode ảnh
    contents = await file.read()
    # Giới hạn kích thước file (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Kích thước file quá lớn (tối đa {MAX_FILE_SIZE // 1024 // 1024}MB).")

    image_np = decode_image(contents)
    logger.info(f"Đã nhận và decode ảnh thành công, kích thước: {image_np.shape}")
    
    return await process_image_for_detection(image_np)

@app.post("/process-image-url",
          response_model=ProcessImageResponse,
          responses={
              400: {"model": ErrorResponse, "description": "Yêu cầu không hợp lệ"},
              422: {"model": ErrorResponse, "description": "URL không hợp lệ hoặc không phải ảnh"},
              500: {"model": ErrorResponse, "description": "Lỗi máy chủ nội bộ"}
          })
async def process_image_url_endpoint(request: ImageUrlRequest):
    """
    Endpoint nhận URL hình ảnh và trả về kết quả nhận dạng biển số.
    """
    logger.info(f"Đã nhận yêu cầu xử lý ảnh từ URL: {request.url}")
    
    # Tải ảnh từ URL
    image_np = download_image_from_url(str(request.url))
    if image_np is None:
        raise HTTPException(status_code=422, detail="Không thể tải hoặc xử lý ảnh từ URL cung cấp.")
    
    logger.info(f"Đã tải ảnh từ URL thành công, kích thước: {image_np.shape}")
    
    return await process_image_for_detection(image_np)

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Khởi chạy server tại http://{settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=True
    ) 
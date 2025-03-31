# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from pydantic import HttpUrl


class PlateAnalysisResult(BaseModel):
    original: str = Field(..., description="Biển số gốc đọc được từ OCR")
    normalized: str = Field(
        "", description="Biển số đã chuẩn hóa (loại bỏ ký tự đặc biệt)"
    )
    province_code: Optional[str] = Field(
        None, description="Mã tỉnh/thành phố (2 số đầu)"
    )
    province_name: Optional[str] = Field(None, description="Tên tỉnh/thành phố")
    serial: Optional[str] = Field(None, description="Phần seri của biển số")
    number: Optional[str] = Field(None, description="Phần số đăng ký của biển số")
    plate_type: str = Field(
        "unknown",
        description="Loại biển số (personal, commercial, government, etc.) dựa trên màu và ký tự",
    )
    plate_type_info: Optional[dict] = Field(
        None, description="Thông tin chi tiết về loại biển số"
    )
    detected_color: Optional[str] = Field(
        None,
        description="Màu sắc chủ đạo phát hiện được của nền biển số (white, yellow, blue, red, unknown)",
    )
    is_valid_format: bool = Field(
        False, description="Biển số có khớp với định dạng phổ biến không?"
    )
    format_description: Optional[str] = Field(
        None, description="Mô tả định dạng khớp được"
    )


class DetectionResult(BaseModel):
    plate_number: str = Field(
        ..., description="Biển số nhận dạng được (có thể là 'Không đọc được')"
    )
    confidence_detection: float = Field(
        ..., description="Độ tin cậy của việc phát hiện biển số (từ YOLO)"
    )
    # confidence_ocr: Optional[float] = Field(None, description="Độ tin cậy của OCR (nếu có)") # Khó lấy từ PaddleOCR
    bounding_box: Tuple[int, int, int, int] = Field(
        ..., description="Tọa độ hộp giới hạn [x1, y1, x2, y2]"
    )
    plate_analysis: Optional[PlateAnalysisResult] = Field(
        None, description="Kết quả phân tích chi tiết biển số"
    )
    ocr_engine_used: Optional[str] = Field(
        None, description="Engine OCR đã sử dụng ('paddleocr' hoặc 'openai')"
    )


class ProcessImageResponse(BaseModel):
    detections: List[DetectionResult] = Field(
        [], description="Danh sách các biển số phát hiện và nhận dạng được"
    )
    processed_image_url: Optional[str] = Field(
        None,
        description="URL Data của ảnh đã xử lý với các hộp giới hạn (Base64 encoded)",
    )
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có")


class ErrorResponse(BaseModel):
    detail: str


class ImageUrlRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL của ảnh cần xử lý")

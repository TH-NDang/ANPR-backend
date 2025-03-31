# config.py
import os
import torch
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Optional

load_dotenv() # Tải biến môi trường từ file .env (nếu có)

class Settings(BaseSettings):
    # Đường dẫn tới model YOLO (ưu tiên biến môi trường)
    yolo_model_path: str = os.getenv("YOLO_MODEL_PATH", "models/best.pt")
    yolo_fallback_model: str = "models/yolov11n.pt"  # Model dùng nếu best.pt không tồn tại

    # Ngưỡng tin cậy cho việc phát hiện biển số của YOLO
    yolo_conf_threshold: float = float(os.getenv("YOLO_CONF_THRESHOLD", 0.4))

    # Ngưỡng tin cậy cho việc nhận dạng ký tự của OCR (nếu thư viện hỗ trợ)
    ocr_conf_threshold: float = float(os.getenv("OCR_CONF_THRESHOLD", 0.5))

    # Tham số tiền xử lý ảnh OCR
    ocr_preprocess_clahe_clip_limit: float = float(os.getenv("OCR_CLAHE_CLIP_LIMIT", 2.0))
    ocr_preprocess_clahe_tile_size: int = int(os.getenv("OCR_CLAHE_TILE_SIZE", 8))
    ocr_preprocess_thresh_block_size: int = int(os.getenv("OCR_THRESH_BLOCK_SIZE", 11))
    ocr_preprocess_thresh_c: int = int(os.getenv("OCR_THRESH_C", 5))

    # Bật/tắt các bước tiền xử lý và cài đặt nâng cao
    enable_ocr_preprocessing: bool = bool(os.getenv("ENABLE_OCR_PREPROCESSING", True))
    enable_multiple_ocr_methods: bool = bool(os.getenv("ENABLE_MULTIPLE_OCR_METHODS", True))
    max_ocr_versions: int = int(os.getenv("MAX_OCR_VERSIONS", 5))
    enable_deskew: bool = bool(os.getenv("ENABLE_DESKEW", True))

    # Cấu hình PaddleOCR
    paddle_use_gpu: bool = bool(os.getenv("PADDLE_USE_GPU", torch.cuda.is_available()))
    paddle_language: str = os.getenv("PADDLE_LANGUAGE", "en")
    paddle_use_angle_cls: bool = bool(os.getenv("PADDLE_USE_ANGLE_CLS", True))

    # API Settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 5000))
    log_level: str = os.getenv("LOG_LEVEL", "debug")

    # OpenAI Settings
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Bật/tắt fallback OpenAI
    enable_openai_fallback: bool = bool(os.getenv("ENABLE_OPENAI_FALLBACK", True))

    # Torch settings
    torch_weights_only: bool = False # Giữ False để load model custom

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Bỏ qua các biến môi trường không được định nghĩa

settings = Settings()

# Monkey patch torch.load nếu cần (giữ nguyên từ code gốc)
original_torch_load = torch.load
def patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = settings.torch_weights_only
    return original_torch_load(f, *args, **kwargs)
torch.load = patched_torch_load

# --- Logging Setup ---
import logging
import sys

# Thiết lập logging cơ bản
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout) # Log ra console
        # Thêm FileHandler nếu muốn log ra file
        # logging.FileHandler("app.log"),
    ]
)
logger = logging.getLogger(__name__)

logger.info("Configuration loaded successfully.")
logger.info(f"YOLO Model Path: {settings.yolo_model_path}")
logger.info(f"YOLO Fallback Model: {settings.yolo_fallback_model}")
logger.info(f"YOLO Confidence Threshold: {settings.yolo_conf_threshold}")
logger.info(f"Enable OCR Preprocessing: {settings.enable_ocr_preprocessing}")
logger.info(f"Enable Multiple OCR Methods: {settings.enable_multiple_ocr_methods}")
logger.info(f"PaddleOCR Language: {settings.paddle_language}")
logger.info(f"PaddleOCR GPU Enabled: {settings.paddle_use_gpu}")
logger.info(f"OpenAI API Key Provided: {'Yes' if settings.openai_api_key else 'No'}")
logger.info(f"OpenAI Model: {settings.openai_model}")
logger.info(f"OpenAI Fallback Enabled: {settings.enable_openai_fallback}") 
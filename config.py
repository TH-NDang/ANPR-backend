# config.py
import os
import torch
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Settings(BaseSettings):
    # --- YOLO Settings ---
    yolo_model_path: str = os.getenv("YOLO_MODEL_PATH", "models/best.pt")
    yolo_fallback_model: str = "models/yolov11n.pt"
    yolo_conf_threshold: float = 0.4

    # --- OCR Settings ---
    ocr_conf_threshold: float = 0.5
    ocr_preprocess_clahe_clip_limit: float = 2.0
    ocr_preprocess_clahe_tile_size: int = 8
    ocr_preprocess_thresh_block_size: int = 11
    ocr_preprocess_thresh_c: int = 5
    enable_ocr_preprocessing: bool = True
    enable_multiple_ocr_methods: bool = True
    max_ocr_versions: int = 5
    enable_deskew: bool = True

    # --- PaddleOCR Settings ---
    paddle_use_gpu: bool = torch.cuda.is_available()
    paddle_language: str = "en"
    paddle_use_angle_cls: bool = True

    # --- API Settings ---
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 5000))
    log_level: str = os.getenv("LOG_LEVEL", "info")

    # --- OpenAI/Gemini Settings (Keep .env priority for keys and models) ---
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    enable_openai_fallback: bool = True

    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # --- Torch settings ---
    torch_weights_only: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = settings.torch_weights_only
    return original_torch_load(f, *args, **kwargs)


torch.load = patched_torch_load

import logging
import sys

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
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
logger.info(f"Gemini API Key Provided: {'Yes' if settings.gemini_api_key else 'No'}")
logger.info(f"Gemini Model: {settings.gemini_model}")

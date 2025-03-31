import asyncio
import os
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

# Import các thành phần cần thiết từ config
from config import settings, logger

# Khởi tạo model YOLO một lần khi module được load
model_path = settings.yolo_model_path
if not os.path.exists(model_path):
    logger.warning(
        f"Không tìm thấy model YOLO chính tại {model_path}. Sử dụng model fallback {settings.yolo_fallback_model}"
    )
    model_path = settings.yolo_fallback_model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy model YOLO chính và fallback tại {settings.yolo_model_path} và {settings.yolo_fallback_model}"
        )

try:
    model = YOLO(model_path)
    logger.info(f"Model YOLO đã được load thành công từ: {model_path}")
except Exception as e:
    logger.error(f"Lỗi nghiêm trọng khi load model YOLO: {e}", exc_info=True)
    raise RuntimeError(f"Không thể load model YOLO từ {model_path}") from e


async def run_detection(
    image: np.ndarray, executor: ThreadPoolExecutor
) -> List[Tuple[List[int], float]]:
    """
    Chạy YOLO detection trong executor.
    Args:
        image: Ảnh đầu vào dạng NumPy array.
        executor: ThreadPoolExecutor để chạy tác vụ nặng.
    Returns:
        Danh sách các tuple chứa (bounding_box, confidence).
    """
    loop = asyncio.get_running_loop()
    try:

        def detect_plates():
            results = model(
                image, conf=settings.yolo_conf_threshold, verbose=False
            )  # Thêm verbose=False
            detections = []

            # Duyệt qua kết quả một cách an toàn hơn
            if results:
                for result in results:
                    if result.boxes:  # Kiểm tra xem có boxes không
                        for box in result.boxes:
                            # Lấy tọa độ xyxy (luôn là list 1 phần tử)
                            xyxy_list = box.xyxy.tolist()
                            # Lấy confidence (luôn là list 1 phần tử)
                            conf_list = box.conf.tolist()

                            if xyxy_list and conf_list:
                                x1, y1, x2, y2 = map(int, xyxy_list[0])
                                conf = float(conf_list[0])

                                # Kiểm tra bbox hợp lệ
                                if x1 < x2 and y1 < y2:
                                    detections.append(([x1, y1, x2, y2], conf))
                                else:
                                    logger.warning(
                                        f"Bounding box từ YOLO không hợp lệ bị bỏ qua: ({x1}, {y1}, {x2}, {y2})"
                                    )
            logger.debug(f"YOLO phát hiện {len(detections)} detections hợp lệ.")
            return detections

        return await loop.run_in_executor(executor, detect_plates)

    except Exception as e:
        logger.error(f"Lỗi khi chạy detection trong thread: {e}", exc_info=True)
        return []

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os
import matplotlib.pyplot as plt
import argparse # Import thư viện argparse

def perform_ocr(image_array):
    """
    Thực hiện OCR trên ảnh được cung cấp và trả về chuỗi văn bản trích xuất được.
    """
    if image_array is None:
        raise ValueError("Ảnh đầu vào không được là None.")
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Ảnh đầu vào phải là một mảng numpy.")
    results = ocr.ocr(image_array, rec=True)
    return ' '.join([result[1][0] for result in results[0]] if results[0] else "")

def process_image(image_path, model, ocr, display=True, save_path=None):
    """
    Xử lý ảnh để phát hiện và nhận diện biển số xe.

    Args:
        image_path (str): Đường dẫn đến ảnh cần xử lý.
        model (YOLO): Mô hình YOLO đã được load.
        ocr (PaddleOCR): Mô hình OCR đã được load.
        display (bool, optional): Có hiển thị ảnh kết quả hay không. Mặc định là True.
        save_path (str, optional): Đường dẫn để lưu ảnh kết quả. Nếu là None, ảnh sẽ không được lưu.
    """
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại đường dẫn: {image_path}")
        return

    # Đọc ảnh đầu vào
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
        return

    # Phát hiện biển số xe bằng YOLO
    results = model(frame)

    # Lặp qua các kết quả phát hiện
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            xyxy = box.xyxy[0]
            conf = box.conf[0]

            # Chuyển đổi tọa độ về kiểu int
            x1, y1, x2, y2 = map(int, xyxy)

            # Crop biển số xe từ frame
            license_plate_img = frame[y1:y2, x1:x2]

            # Thực hiện OCR trên ảnh biển số xe đã crop
            ocr_text = perform_ocr(license_plate_img)

            # In ra kết quả OCR và độ tin cậy
            print(f"Biển số xe: {ocr_text}, Độ tin cậy: {conf:.2f}")

            # Vẽ bounding box và hiển thị kết quả lên frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, ocr_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị và lưu ảnh (tùy chọn)
    if display:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 5))
        plt.imshow(frame_rgb)
        plt.title("Nhận diện biển số xe")
        plt.axis('off')
        plt.show()

    if save_path:
        cv2.imwrite(save_path, frame)
        print(f"Đã lưu ảnh kết quả tại: {save_path}")

if __name__ == "__main__":
    # Khởi tạo argparse để nhận các tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Nhận diện biển số xe từ ảnh")
    parser.add_argument("image_path", help="Đường dẫn đến ảnh cần xử lý")
    parser.add_argument("--model_path", default="best.pt", help="Đường dẫn đến file weights YOLO")
    parser.add_argument("--display", action="store_true", help="Hiển thị ảnh kết quả")
    parser.add_argument("--save_path", default=None, help="Đường dẫn để lưu ảnh kết quả")
    args = parser.parse_args()

    # Load model YOLO
    if not os.path.exists(args.model_path):
        print(f"Lỗi: Không tìm thấy mô hình YOLO tại đường dẫn: {args.model_path}")
        print("Vui lòng đảm bảo bạn đã huấn luyện mô hình YOLO và cung cấp đường dẫn đúng.")
        exit()
    model = YOLO(args.model_path)

    # Khởi tạo OCR system
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # Xử lý ảnh
    process_image(args.image_path, model, ocr, args.display, args.save_path)

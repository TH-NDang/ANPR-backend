# image_utils.py
import cv2
import numpy as np
import base64
from typing import Optional, List, Tuple
from config import logger, settings
from constants import COLOR_RANGES_HSV
import requests

def download_image_from_url(url: str) -> Optional[np.ndarray]:
    """
    Tải ảnh từ URL internet và chuyển thành mảng numpy.

    Args:
        url: URL của ảnh cần tải.

    Returns:
        Mảng numpy chứa ảnh hoặc None nếu có lỗi.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        image = decode_image(response.content)
        if image is None:
            logger.error(f"Không thể decode dữ liệu ảnh từ URL: {url}")
            return None

        return image
    except requests.RequestException as e:
        logger.error(f"Lỗi khi tải ảnh từ URL {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Lỗi không xác định khi xử lý ảnh từ URL {url}: {e}")
        return None


def decode_image(contents: bytes) -> Optional[np.ndarray]:
    """Đọc dữ liệu byte của ảnh và chuyển thành mảng numpy."""
    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(
                "Không thể decode ảnh. Định dạng không hợp lệ hoặc file bị lỗi."
            )
        return img
    except Exception as e:
        logger.error(f"Lỗi khi decode ảnh: {e}")
        return None


def encode_image_to_base64(image: np.ndarray) -> Optional[str]:
    """Mã hóa ảnh numpy array thành chuỗi base64 Data URL."""
    try:
        if image is None or image.size == 0:
            logger.warning("Ảnh đầu vào để encode là None hoặc trống.")
            return None
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Không thể encode ảnh sang JPG.")
        encoded_string = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Lỗi khi encode ảnh sang base64: {e}")
        return None

def apply_unsharp_mask(image: np.ndarray, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Áp dụng unsharp mask filter để tăng độ sắc nét của ảnh."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def auto_canny(image: np.ndarray, sigma=0.33):
    """Tự động chọn ngưỡng tối ưu cho Canny edge detection."""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def deskew(image: np.ndarray) -> np.ndarray:
    """Chỉnh sửa góc nghiêng của ảnh biển số."""
    try:
        # Tìm các cạnh trong ảnh
        edges = auto_canny(image)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return image

        # Tính góc nghiêng trung bình
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return image

        # Tính góc nghiêng trung bình
        angle = np.median(angles)

        # Lấy kích thước ảnh
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Tạo ma trận xoay
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Áp dụng xoay cho ảnh
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        return rotated
    except Exception as e:
        logger.warning(f"Lỗi khi deskew ảnh: {e}")
        return image  # Trả về ảnh gốc nếu có lỗi

def try_multiple_thresholds(gray_image: np.ndarray) -> List[np.ndarray]:
    """Thử nghiệm nhiều phương pháp threshold khác nhau và trả về danh sách kết quả."""
    results = []

    # 1. Global binary threshold with Otsu's method
    _, otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results.append(otsu)

    # 2. Global binary threshold
    _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    results.append(binary)

    # 3. Adaptive Gaussian Threshold - normal
    adaptive1 = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    results.append(adaptive1)

    # 4. Adaptive Gaussian Threshold - inverted
    adaptive2 = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    results.append(adaptive2)

    # 5. Tạo ảnh đảo ngược (nếu cần thêm kỹ thuật)
    inverted = cv2.bitwise_not(gray_image)
    results.append(inverted)

    return results

def preprocess_plate_for_ocr(plate_image: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý ảnh biển số để cải thiện OCR.
    Có nhiều cách xử lý khác nhau, trả về version tốt nhất.
    """
    if not settings.enable_ocr_preprocessing:
        return plate_image

    if plate_image is None or plate_image.size == 0:
        logger.warning("Ảnh biển số đầu vào cho tiền xử lý OCR là None hoặc trống.")
        return plate_image

    # Lưu bản sao của ảnh gốc để so sánh
    original = plate_image.copy()
    processed_versions = []

    # --- Version 1: Xử lý cơ bản ---
    try:
        # Resize ảnh nếu quá nhỏ
        min_height = 40
        h, w = original.shape[:2]
        if h < min_height:
            scale = min_height / h
            resized = cv2.resize(original, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            resized = original.copy()

        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Tăng độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # Làm mờ nhẹ để giảm nhiễu
        blurred = cv2.GaussianBlur(contrast_enhanced, (3, 3), 0)

        # Áp dụng nhiều phương pháp threshold khác nhau
        threshold_versions = try_multiple_thresholds(blurred)

        # Thêm các phiên bản xử lý vào danh sách
        processed_versions.extend(threshold_versions)

        # Áp dụng kỹ thuật deskew cho từng phiên bản
        deskewed_versions = [deskew(img) for img in threshold_versions]
        processed_versions.extend(deskewed_versions)
        
        # Áp dụng morphology để làm rõ ký tự
        kernel = np.ones((3, 3), np.uint8)
        for img in threshold_versions:
            # Erosion để làm mỏng các ký tự
            erosion = cv2.erode(img, kernel, iterations=1)
            processed_versions.append(erosion)
            
            # Dilation để làm dày các ký tự
            dilation = cv2.dilate(img, kernel, iterations=1)
            processed_versions.append(dilation)
        
        # Chọn phương pháp ban đầu của chúng ta với binary threshold
        binary_threshold = cv2.adaptiveThreshold(
            contrast_enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            settings.ocr_preprocess_thresh_block_size,
            settings.ocr_preprocess_thresh_c
        )
        processed_versions.append(binary_threshold)
        
        # Thêm phiên bản tăng độ sắc nét
        sharpened = apply_unsharp_mask(contrast_enhanced)
        processed_versions.append(sharpened)
        
        # Nếu số lượng phương pháp quá nhiều, hãy giảm bớt để tránh quá tải
        max_versions = 5
        if len(processed_versions) > max_versions:
            processed_versions = processed_versions[:max_versions]
        
        logger.debug(f"Đã tạo {len(processed_versions)} phiên bản tiền xử lý khác nhau")
        
        # Mặc định trả về phiên bản binary threshold (phiên bản 0)
        # PaddleOCR/OCR engine sẽ thử từng phiên bản để tìm ra kết quả tốt nhất
        # Trả về các phiên bản ảnh trong 1 tuple (để có thể lặp qua trong OCR)
        return processed_versions[0]
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình tiền xử lý OCR: {e}")
        # Trả về ảnh gốc nếu có lỗi
        return cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

def draw_detections(image: np.ndarray, detections: List) -> np.ndarray:
    """Vẽ các hộp giới hạn và thông tin lên ảnh."""
    output_image = image.copy()
    if len(output_image.shape) == 2:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    elif output_image.shape[2] == 4:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGBA2BGR)

    for det in detections:
        x1, y1, x2, y2 = det.bounding_box
        text = det.plate_number
        color = (0, 0, 255)  # Màu đỏ cho lỗi/không đọc được
        if det.plate_analysis and det.plate_analysis.is_valid_format:
            color = (0, 255, 0)  # Màu xanh lá cho biển hợp lệ

        # Vẽ hộp
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        # Chuẩn bị text để hiển thị
        label = f"{text}"
        if det.plate_analysis:
            label += f" ({det.plate_analysis.plate_type})"

        # Tính toán vị trí đặt text
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10

        # Vẽ nền cho text
        cv2.rectangle(
            output_image,
            (x1, text_y - text_height - baseline),
            (x1 + text_width, text_y + baseline),
            color,
            -1,
        )
        # Vẽ text
        cv2.putText(
            output_image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output_image


def get_plate_color(plate_image: np.ndarray) -> str:
    """Xác định màu nền chủ đạo của ảnh biển số."""
    if plate_image is None or plate_image.size < 100:
        return "unknown"

    try:
        # Chuyển sang HSV
        hsv_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)

        hsv_image = cv2.medianBlur(hsv_image, 5)
        max_pixels = 0
        detected_color = "unknown"

        for color_name, (lower, upper) in COLOR_RANGES_HSV.items():
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)

            # Tạo mask
            mask = cv2.inRange(hsv_image, lower_np, upper_np)
            num_pixels = cv2.countNonZero(mask)

            if color_name.startswith("red"):
                current_red_pixels = num_pixels
                if detected_color == "red":
                    num_pixels += max_pixels
                color_name = "red"

            if num_pixels > max_pixels:
                max_pixels = num_pixels
                if color_name == "red":
                    max_pixels = current_red_pixels
                detected_color = color_name

        total_pixels = plate_image.shape[0] * plate_image.shape[1]
        if max_pixels < total_pixels * 0.1:
            return "unknown"

        return detected_color

    except Exception as e:
        logger.error(f"Lỗi khi xác định màu biển số: {e}")
        return "unknown" 
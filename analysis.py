# analysis.py
import re
import numpy as np
from typing import Optional
from schemas import PlateAnalysisResult
from constants import PROVINCE_CODES, PLATE_TYPES
from config import logger
from image_utils import get_plate_color

def analyze_license_plate(plate_text: str, plate_image: Optional[np.ndarray] = None) -> PlateAnalysisResult:
    """
    Phân tích chuỗi biển số và ảnh (nếu có) để trích xuất thông tin.

    Args:
        plate_text: Chuỗi ký tự biển số đã được OCR và hậu xử lý.
        plate_image: Ảnh crop của biển số (dùng để xác định màu).

    Returns:
        Đối tượng PlateAnalysisResult chứa thông tin phân tích.
    """
    analysis = PlateAnalysisResult(original=plate_text if plate_text else "N/A")

    if not plate_text:
        logger.debug("Chuỗi biển số rỗng, không thể phân tích.")
        return analysis # Trả về kết quả rỗng nếu không có text

    # 1. Chuẩn hóa (đã làm ở OCR postprocess, nhưng làm lại để chắc chắn)
    # Lưu lại biển số gốc trước khi chuẩn hóa để dùng trong regex
    original_text = plate_text
    normalized = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    analysis.normalized = normalized

    # 2. Xác định màu sắc (nếu có ảnh)
    detected_color = "unknown"
    if plate_image is not None and plate_image.size > 0:
        detected_color = get_plate_color(plate_image)
    analysis.detected_color = detected_color

    # 3. Xác định loại biển số dựa trên màu sắc và ký tự đặc biệt
    plate_type_key = "unknown"
    if "NG" in normalized:
        plate_type_key = "diplomatic_ng"
    elif "QT" in normalized:
        plate_type_key = "diplomatic_qt"
    elif "NN" in normalized:
        plate_type_key = "foreign_nn"
    elif "TM" in normalized or detected_color == "red": # Biển quân đội thường có TM hoặc màu đỏ
         plate_type_key = "military"
    elif detected_color == "blue":
        # Phân biệt xanh trung ương (mã 80) và địa phương
        if normalized.startswith("80"):
             plate_type_key = "government_central"
        else:
             plate_type_key = "government_local"
             # Có thể thêm logic kiểm tra ký hiệu đặc biệt của công an nếu có
    elif detected_color == "yellow":
        plate_type_key = "commercial"
    elif detected_color == "white":
        plate_type_key = "personal" # Mặc định cho biển trắng nếu không phải loại đặc biệt
    # Có thể thêm logic cho biển tạm (chữ T)

    # Nếu vẫn là unknown nhưng màu rõ ràng, gán theo màu
    if plate_type_key == "unknown":
        if detected_color == "blue": plate_type_key = "government_local"
        elif detected_color == "yellow": plate_type_key = "commercial"
        elif detected_color == "white": plate_type_key = "personal"
        elif detected_color == "red": plate_type_key = "military"

    analysis.plate_type = plate_type_key
    analysis.plate_type_info = PLATE_TYPES.get(plate_type_key)


    # 4. Áp dụng Regex để trích xuất cấu trúc và kiểm tra định dạng
    # Ưu tiên các mẫu phổ biến trước
    # Lưu ý: Các regex này có thể cần mở rộng cho nhiều trường hợp hơn (xe máy, rơ mooc, biển vuông...)
    patterns = [
        # === Biển số xe máy 2 dòng (thêm mới) ===
        # Mẫu 68-G1 668.86 hoặc 68-G1 668,86
        (r'^(\d{2})[-\s]*([A-Z])(\d)[\s\.,-]*(\d{3})[\s\.,-]*(\d{2})$', "Biển số xe máy 2 dòng"),
        # Mẫu không có dấu gạch: 68G1 668.86
        (r'^(\d{2})([A-Z])(\d)[\s\.,-]*(\d{3})[\s\.,-]*(\d{2})$', "Biển số xe máy 2 dòng không dấu gạch"),

        # === Biển dài thông thường (ô tô) ===
        # 2 số - 1 chữ - 5 số (mới nhất): 51K12345 hoặc 51-K-12345
        (r'^(\d{2})[-\s]*([A-Z])[-\s]*(\d{5})$', "2 số - 1 chữ - 5 số"),
        # 2 số - 1 chữ - 4 số (cũ): 51A1234 hoặc 51-A-1234
        (r'^(\d{2})[-\s]*([A-Z])[-\s]*(\d{4})$', "2 số - 1 chữ - 4 số"),
        # 2 số - 2 chữ - 5 số (mới): 51AA12345 hoặc 51-AA-12345
        (r'^(\d{2})[-\s]*([A-Z]{2})[-\s]*(\d{5})$', "2 số - 2 chữ - 5 số"),
        # 2 số - 2 chữ - 4 số (cũ): 51AB1234 hoặc 51-AB-1234
        (r'^(\d{2})[-\s]*([A-Z]{2})[-\s]*(\d{4})$', "2 số - 2 chữ - 4 số"),

        # === Biển dành cho cơ quan/tổ chức đặc biệt ===
        # Biển xanh 80: 80A12345, 80B1234, 80NG12345 hoặc có dấu phân cách
        (r'^(80)[-\s]*([A-Z]{1,2}|NG)[-\s]*(\d{3,5})$', "Biển xanh TW (80)"),
        # Biển ngoại giao NG: xxNGxxxx(x) hoặc có dấu phân cách
        (r'^(\d{2})[-\s]*(NG)[-\s]*(\d{3,5})$', "Biển Ngoại giao (NG)"),
        # Biển QT: xxQTxxxx(x) hoặc có dấu phân cách
        (r'^(\d{2})[-\s]*(QT)[-\s]*(\d{3,5})$', "Biển Quốc tế (QT)"),
        # Biển NN: xxNNxxxx(x) hoặc có dấu phân cách
        (r'^(\d{2})[-\s]*(NN)[-\s]*(\d{3,5})$', "Biển Nước Ngoài (NN)"),

        # === Biển Quân đội (chữ cái đầu) ===
        # Ví dụ: TM1234, QP12345 hoặc có dấu phân cách: TM-1234
        (r'^([A-Z]{2})[-\s]*(\d{4,5})$', "Biển Quân đội (2 chữ cái đầu)"),

        # === Biển xe máy (có thể có 4 hoặc 5 số cuối) ===
        # 2 số - 1 chữ + 1 số/chữ - 4/5 số: 29H112345, 29P11234 hoặc 29-H1-12345
        (r'^(\d{2})[-\s]*([A-Z][A-Z0-9])[-\s]*(\d{4,5})$', "Biển xe máy (2 số - seri 2 ký tự - 4/5 số)"),
        # Biển xe máy 5 số cũ (Hà Nội/HCM): 29X112345 hoặc 29-X1-12345
        (r'^(\d{2})[-\s]*([A-Z]\d)[-\s]*(\d{5})$', "Biển xe máy 5 số (cũ)"),
        # Biển xe máy 4 số cũ: 29F11234 hoặc 29-F1-1234
        (r'^(\d{2})[-\s]*([A-Z]\d)[-\s]*(\d{4})$', "Biển xe máy 4 số (cũ)"),

        # === Các loại khác (Rơ moóc, Sơ mi rơ moóc - chữ R, M) ===
        # Cần thêm regex cụ thể nếu muốn nhận dạng chính xác
    ]

    found_match = False
    for pattern, desc in patterns:
        match = re.match(pattern, normalized) or re.search(pattern, original_text)
        if match:
            analysis.is_valid_format = True
            analysis.format_description = desc
            groups = match.groups()

            # Logic trích xuất dựa trên số lượng group và loại pattern
            if len(groups) == 3:
                analysis.province_code = groups[0]
                analysis.serial = groups[1]
                analysis.number = groups[2]
            elif len(groups) == 2 and desc.startswith("Biển Quân đội"):
                 # Biển quân đội không có mã tỉnh theo số
                 analysis.province_code = None
                 analysis.serial = groups[0] # Ký hiệu đơn vị
                 analysis.number = groups[1]
            elif len(groups) == 5 and desc.startswith("Biển số xe máy 2 dòng"):
                # Xử lý biển số xe máy 2 dòng kiểu 68-G1 668.86
                analysis.province_code = groups[0]
                analysis.serial = groups[1] + groups[2]
                analysis.number = groups[3] + groups[4]
                analysis.plate_type = "motorcycle"
                analysis.plate_type_info = PLATE_TYPES.get("motorcycle", PLATE_TYPES.get("personal"))
            # Thêm các else if cho các cấu trúc group khác nếu cần

            # Lấy tên tỉnh (nếu có mã tỉnh)
            if analysis.province_code:
                analysis.province_name = PROVINCE_CODES.get(analysis.province_code, "Không xác định")
                # Điều chỉnh lại loại biển xanh nếu cần
                if analysis.province_code == "80" and analysis.plate_type != "government_central":
                     analysis.plate_type = "government_central"
                     analysis.plate_type_info = PLATE_TYPES.get("government_central")

            found_match = True
            logger.debug(f"Biển số '{normalized}' khớp với mẫu: {desc}")
            break # Dừng lại khi tìm thấy mẫu đầu tiên khớp

    if not found_match:
        # Thử nhận dạng biển số xe máy 2 dòng bằng cách tách chuỗi thủ công
        motorcycle_2line_pattern = re.search(r'(\d{2})[-\s]*([A-Z]\d)[-\s\.]*(\d{3})[-\s\.]*(\d{2})', original_text)
        if motorcycle_2line_pattern:
            groups = motorcycle_2line_pattern.groups()
            analysis.province_code = groups[0]
            analysis.serial = groups[1]
            analysis.number = groups[2] + groups[3]
            analysis.plate_type = "motorcycle"
            analysis.plate_type_info = PLATE_TYPES.get("motorcycle", PLATE_TYPES.get("personal"))
            analysis.is_valid_format = True
            analysis.format_description = "Biển số xe máy 2 dòng (phân tích thủ công)"
            analysis.province_name = PROVINCE_CODES.get(analysis.province_code, "Không xác định")
            found_match = True
            logger.debug(f"Biển số '{original_text}' được phân tích thủ công là biển xe máy 2 dòng")
        else:
            # Thử tìm kiếm mẫu biển số cơ bản với dấu phân cách
            basic_pattern = re.search(r'(\d{2})[-\s.]*([A-Z0-9]{1,2})[-\s.]*(\d{4,5})', original_text)
            if basic_pattern:
                groups = basic_pattern.groups()
                analysis.province_code = groups[0]
                analysis.serial = groups[1]
                analysis.number = groups[2]
                analysis.province_name = PROVINCE_CODES.get(analysis.province_code, "Không xác định")
                analysis.is_valid_format = True
                analysis.format_description = "Biển số cơ bản (phân tích thủ công)"
                
                # Đoán loại biển số dựa trên độ dài và ký tự
                if len(groups[1]) == 2 and groups[1][0].isalpha() and groups[1][1].isdigit():
                    analysis.plate_type = "motorcycle"
                    analysis.plate_type_info = PLATE_TYPES.get("motorcycle", PLATE_TYPES.get("personal"))
                    analysis.format_description = "Biển số xe máy (phân tích thủ công)"
                
                found_match = True
                logger.debug(f"Biển số '{original_text}' được phân tích thủ công là biển số cơ bản")
            else:
                logger.warning(f"Biển số '{normalized}' (gốc: '{analysis.original}') không khớp với bất kỳ định dạng phổ biến nào.")
                # Có thể thử phân tích cơ bản nếu không khớp regex
                if len(normalized) >= 6: # heuristic đơn giản
                    potential_code = normalized[:2]
                    if potential_code.isdigit():
                         analysis.province_code = potential_code
                         analysis.province_name = PROVINCE_CODES.get(potential_code, "Không xác định")
                         # Phần còn lại có thể là serial + number gộp lại
                         analysis.serial = normalized[2:4] # Đoán đại
                         analysis.number = normalized[4:]

    # Nếu phát hiện biển xe máy dựa trên regex nhưng chưa đặt plate_type
    if analysis.is_valid_format and "xe máy" in (analysis.format_description or ""):
        analysis.plate_type = "motorcycle"
        analysis.plate_type_info = PLATE_TYPES.get("motorcycle", PLATE_TYPES.get("personal"))

    return analysis 
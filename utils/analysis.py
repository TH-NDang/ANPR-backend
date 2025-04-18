import re
import numpy as np
from typing import Optional
from schemas import PlateAnalysisResult
from constants import PROVINCE_CODES, PLATE_TYPES
from config import logger
from .image_utils import get_plate_color


def analyze_license_plate(
    plate_text: str, plate_image: Optional[np.ndarray] = None
) -> PlateAnalysisResult:
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
        return analysis

    # 1. Chuẩn hóa
    original_text = plate_text
    logger.warning(f"Đang phân tích biển số: '{plate_text}'")
    normalized = re.sub(r"[^A-Z0-9]", "", plate_text.upper())
    analysis.normalized = normalized

    MIN_PLATE_LENGTH = 6
    if len(normalized) < MIN_PLATE_LENGTH:
        logger.warning(
            f"Biển số chuẩn hóa '{normalized}' (gốc: '{original_text}') quá ngắn (dưới {MIN_PLATE_LENGTH} ký tự). Coi là không hợp lệ."
        )
        analysis.is_valid_format = False
    else:
        patterns = [
            # === Biển số xe máy 2 dòng (thêm mới) ===
            # Mẫu 68-G1 668.86 hoặc 68-G1 668,86
            (
                r"^(\d{2})[-\s]*([A-Z])(\d)[\s\.,-]*(\d{3})[\s\.,-]*(\d{2})$",
                "Biển số xe máy 2 dòng",
            ),
            # Mẫu không có dấu gạch: 68G1 668.86
            (
                r"^(\d{2})([A-Z])(\d)[\s\.,-]*(\d{3})[\s\.,-]*(\d{2})$",
                "Biển số xe máy 2 dòng không dấu gạch",
            ),
            # === Biển dài thông thường (ô tô) ===
            # 2 số - 1 chữ - 5 số (mới nhất): 51K12345 hoặc 51-K-12345
            (r"^(\d{2})[-\s]*([A-Z])[-\s]*(\d{5})$", "2 số - 1 chữ - 5 số"),
            # 2 số - 1 chữ - 4 số (cũ): 51A1234 hoặc 51-A-1234
            (r"^(\d{2})[-\s]*([A-Z])[-\s]*(\d{4})$", "2 số - 1 chữ - 4 số"),
            # 2 số - 2 chữ - 5 số (mới): 51AA12345 hoặc 51-AA-12345
            (r"^(\d{2})[-\s]*([A-Z]{2})[-\s]*(\d{5})$", "2 số - 2 chữ - 5 số"),
            # 2 số - 2 chữ - 4 số (cũ): 51AB1234 hoặc 51-AB-1234
            (r"^(\d{2})[-\s]*([A-Z]{2})[-\s]*(\d{4})$", "2 số - 2 chữ - 4 số"),
            # === Biển dành cho cơ quan/tổ chức đặc biệt ===
            # Biển xanh 80: 80A12345, 80B1234, 80NG12345 hoặc có dấu phân cách
            (r"^(80)[-\s]*([A-Z]{1,2}|NG)[-\s]*(\d{3,5})$", "Biển xanh TW (80)"),
            # Biển ngoại giao NG: xxNGxxxx(x) hoặc có dấu phân cách
            (r"^(\d{2})[-\s]*(NG)[-\s]*(\d{3,5})$", "Biển Ngoại giao (NG)"),
            # Biển QT: xxQTxxxx(x) hoặc có dấu phân cách
            (r"^(\d{2})[-\s]*(QT)[-\s]*(\d{3,5})$", "Biển Quốc tế (QT)"),
            # Biển NN: xxNNxxxx(x) hoặc có dấu phân cách
            (r"^(\d{2})[-\s]*(NN)[-\s]*(\d{3,5})$", "Biển Nước Ngoài (NN)"),
            # === Biển Quân đội (chữ cái đầu) ===
            # Ví dụ: TM1234, QP12345 hoặc có dấu phân cách: TM-1234
            (r"^([A-Z]{2})[-\s]*(\d{4,5})$", "Biển Quân đội (2 chữ cái đầu)"),
            # === Biển xe máy (có thể có 4 hoặc 5 số cuối) ===
            # 2 số - 1 chữ + 1 số/chữ - 4/5 số: 29H112345, 29P11234 hoặc 29-H1-12345
            (
                r"^(\d{2})[-\s]*([A-Z][A-Z0-9])[-\s]*(\d{4,5})$",
                "Biển xe máy (2 số - seri 2 ký tự - 4/5 số)",
            ),
            # Biển xe máy 5 số cũ (Hà Nội/HCM): 29X112345 hoặc 29-X1-12345
            (r"^(\d{2})[-\s]*([A-Z]\d)[-\s]*(\d{5})$", "Biển xe máy 5 số (cũ)"),
            # Biển xe máy 4 số cũ: 29F11234 hoặc 29-F1-1234
            (r"^(\d{2})[-\s]*([A-Z]\d)[-\s]*(\d{4})$", "Biển xe máy 4 số (cũ)"),
        ]

        found_match = False
        for pattern, desc in patterns:
            match = re.match(pattern, normalized)
            if not match and original_text:
                match = re.search(pattern, original_text)
            if match:
                groups = match.groups()
                is_components_reasonable = True
                extracted_province = None
                extracted_serial = None
                extracted_number = None

                if len(groups) == 3:
                    extracted_province = groups[0]
                    extracted_serial = groups[1]
                    extracted_number = groups[2]
                    if not (
                        extracted_province
                        and extracted_serial
                        and extracted_number
                        and len(extracted_number) >= 4
                    ):
                        is_components_reasonable = False
                elif len(groups) == 2 and desc.startswith("Biển Quân đội"):
                    extracted_serial = groups[0]
                    extracted_number = groups[1]
                    if not (
                        extracted_serial
                        and extracted_number
                        and len(extracted_number) >= 4
                    ):
                        is_components_reasonable = False
                elif len(groups) == 5 and desc.startswith("Biển số xe máy 2 dòng"):
                    extracted_province = groups[0]
                    extracted_serial = groups[1] + groups[2]
                    extracted_number = groups[3] + groups[4]
                    if not (
                        extracted_province
                        and extracted_serial
                        and extracted_number
                        and len(extracted_number) >= 5
                    ):
                        is_components_reasonable = False

                if is_components_reasonable:
                    analysis.is_valid_format = True
                    analysis.format_description = desc
                    analysis.province_code = extracted_province
                    analysis.serial = extracted_serial
                    analysis.number = extracted_number

                    # Lấy tên tỉnh
                    if analysis.province_code:
                        analysis.province_name = PROVINCE_CODES.get(
                            analysis.province_code, "Không xác định"
                        )
                        # Điều chỉnh lại loại biển xanh nếu cần
                        if (
                            analysis.province_code == "80"
                            and analysis.plate_type != "government_central"
                        ):
                            analysis.plate_type = "government_central"
                            analysis.plate_type_info = PLATE_TYPES.get(
                                "government_central"
                            )

                    found_match = True
                    logger.debug(f"Biển số '{normalized}' khớp với mẫu hợp lệ: {desc}")
                    break
                else:
                    logger.debug(
                        f"Biển số '{normalized}' khớp với mẫu {desc} nhưng các thành phần không hợp lý, tiếp tục tìm kiếm."
                    )

        if not found_match:
            logger.warning(
                f"Biển số '{normalized}' (gốc: '{analysis.original}') không khớp với bất kỳ định dạng phổ biến hợp lệ nào sau khi kiểm tra độ dài và thành phần."
            )
            analysis.is_valid_format = False

    # 2. Xác định màu sắc
    detected_color = "unknown"
    if plate_image is not None and plate_image.size > 0:
        detected_color = get_plate_color(plate_image)
    analysis.detected_color = detected_color

    # 3. Xác định loại biển số
    plate_type_key = "unknown"
    if "NG" in normalized:
        plate_type_key = "diplomatic_ng"
    elif "QT" in normalized:
        plate_type_key = "diplomatic_qt"
    elif "NN" in normalized:
        plate_type_key = "foreign_nn"
    elif (
        "TM" in normalized or detected_color == "red"
    ):
        plate_type_key = "military"
    elif detected_color == "blue":
        if normalized.startswith("80"):
            plate_type_key = "government_central"
        else:
            plate_type_key = "government_local"
    elif detected_color == "yellow":
        plate_type_key = "commercial"
    elif detected_color == "white":
        plate_type_key = "personal"

    if plate_type_key == "unknown":
        if detected_color == "blue":
            plate_type_key = "government_local"
        elif detected_color == "yellow":
            plate_type_key = "commercial"
        elif detected_color == "white":
            plate_type_key = "personal"
        elif detected_color == "red":
            plate_type_key = "military"

    analysis.plate_type = plate_type_key
    analysis.plate_type_info = PLATE_TYPES.get(plate_type_key)

    if (
        analysis.is_valid_format
        and analysis.plate_type == "unknown"
        and "xe máy" in (analysis.format_description or "")
    ):
        analysis.plate_type = "motorcycle"
        analysis.plate_type_info = PLATE_TYPES.get(
            "motorcycle", PLATE_TYPES.get("personal")
        )

    return analysis

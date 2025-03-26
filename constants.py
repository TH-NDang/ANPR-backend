# constants.py
import cv2
import numpy as np

# Dữ liệu biển số tỉnh/thành phố (Giữ nguyên từ code gốc, kiểm tra lại tính chính xác)
PROVINCE_CODES = {
    "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh", "15": "Hải Phòng",
    "16": "Hải Phòng", "17": "Thái Bình", "18": "Nam Định", "19": "Phú Thọ",
    "20": "Thái Nguyên", "21": "Yên Bái", "22": "Tuyên Quang", "23": "Hà Giang",
    "24": "Lào Cai", "25": "Lai Châu", "26": "Sơn La", "27": "Điện Biên",
    "28": "Hòa Bình", "29": "Hà Nội", "30": "Hà Nội", "31": "Hà Nội",
    "32": "Hà Nội", "33": "Hà Nội", "34": "Hải Dương", "35": "Ninh Bình",
    "36": "Thanh Hóa", "37": "Nghệ An", "38": "Hà Tĩnh", "39": "Đồng Nai", # Mã 39 Đồng Nai là cũ, hiện chủ yếu 60
    "40": "Hà Nội", "41": "TP. Hồ Chí Minh", # Mã 41 TP.HCM là cũ, hiện chủ yếu 5x
    "43": "TP. Đà Nẵng", "47": "Đắk Lắk", "48": "Đắk Nông", "49": "Lâm Đồng",
    "50": "TP. Hồ Chí Minh", "51": "TP. Hồ Chí Minh", "52": "TP. Hồ Chí Minh",
    "53": "TP. Hồ Chí Minh", "54": "TP. Hồ Chí Minh", "55": "TP. Hồ Chí Minh",
    "56": "TP. Hồ Chí Minh", "57": "TP. Hồ Chí Minh", "58": "TP. Hồ Chí Minh",
    "59": "TP. Hồ Chí Minh", "60": "Đồng Nai", "61": "Bình Dương", "62": "Long An",
    "63": "Tiền Giang", "64": "Vĩnh Long", "65": "Cần Thơ", "66": "Đồng Tháp",
    "67": "An Giang", "68": "Kiên Giang", "69": "Cà Mau", "70": "Tây Ninh",
    "71": "Bến Tre", "72": "Bà Rịa - Vũng Tàu", "73": "Quảng Bình", "74": "Quảng Trị",
    "75": "Thừa Thiên Huế", "76": "Quảng Ngãi", "77": "Bình Định", "78": "Phú Yên",
    "79": "Khánh Hòa", "80": "Cục CSGT (Biển xanh TW)", # Thêm biển xanh 80
    "81": "Gia Lai", "82": "Kon Tum", "83": "Sóc Trăng", "84": "Trà Vinh",
    "85": "Ninh Thuận", "86": "Bình Thuận", "88": "Vĩnh Phúc", "89": "Hưng Yên",
    "90": "Hà Nam", "92": "Quảng Nam", "93": "Bình Phước", "94": "Bạc Liêu",
    "95": "Hậu Giang", "97": "Bắc Kạn", "98": "Bắc Giang", "99": "Bắc Ninh"
    # Cần bổ sung/cập nhật thêm các mã mới hoặc mã đặc biệt nếu có
}

# Phân loại loại biển số dựa trên màu sắc và ký tự đặc biệt
PLATE_TYPES = {
    "personal": {"name": "Xe cá nhân", "description": "Biển trắng, chữ đen"},
    "commercial": {"name": "Xe kinh doanh vận tải", "description": "Biển vàng, chữ đen"},
    "government_local": {"name": "Xe cơ quan nhà nước (Địa phương)", "description": "Biển xanh dương, chữ trắng"},
    "government_central": {"name": "Xe cơ quan nhà nước (Trung ương)", "description": "Biển xanh dương (mã 80), chữ trắng"},
    "military": {"name": "Xe quân đội", "description": "Biển đỏ, chữ trắng"},
    # Police có thể trùng màu với government nhưng có thể có ký hiệu riêng
    "police": {"name": "Xe công an", "description": "Biển xanh dương, chữ trắng"},
    "diplomatic_ng": {"name": "Xe ngoại giao (NG)", "description": "Biển trắng, chữ đen, ký hiệu NG"},
    "diplomatic_qt": {"name": "Xe ngoại giao (QT)", "description": "Biển trắng, chữ đen, ký hiệu QT"},
    "foreign_nn": {"name": "Xe nước ngoài (NN)", "description": "Biển trắng, chữ đen, ký hiệu NN"}, # Thêm NN
    "temporary": {"name": "Biển tạm", "description": "Thường có chữ T"}, # Cần thêm logic nhận dạng
    "motorcycle": {"name": "Xe máy", "description": "Biển số xe máy (thường là biển trắng, 2 dòng hoặc 1 dòng)"},
    "unknown": {"name": "Không xác định", "description": "Không thể phân loại"},
}

# Dải màu HSV gần đúng cho việc phân loại màu nền biển số
# Lưu ý: Các giá trị này cần được tinh chỉnh dựa trên dữ liệu thực tế
COLOR_RANGES_HSV = {
    "white": ([0, 0, 150], [180, 50, 255]),  # Trắng/Xám sáng
    "yellow": ([20, 100, 100], [30, 255, 255]), # Vàng
    "blue": ([90, 80, 50], [130, 255, 255]),   # Xanh dương
    "red1": ([0, 70, 50], [10, 255, 255]),    # Đỏ (phần 1)
    "red2": ([170, 70, 50], [180, 255, 255]), # Đỏ (phần 2)
}

# Ký tự có thể bị OCR nhầm lẫn và cách sửa (Sử dụng trong hậu xử lý)
OCR_CORRECTION_MAP = {
    'O': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'B': '8',
    # Thêm các cặp khác nếu cần
} 
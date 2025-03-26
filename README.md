# Hệ Thống Nhận Dạng Biển Số Xe (ANPR)

Hệ thống API nhận dạng biển số xe sử dụng YOLO và PaddleOCR, được phát triển bằng FastAPI.

## Tính năng

- Phát hiện biển số xe trong ảnh bằng YOLO
- OCR biển số xe sử dụng PaddleOCR
- Phân tích và phân loại biển số (màu sắc, loại biển, mã tỉnh)
- API RESTful đơn giản dễ tích hợp
- Xử lý bất đồng bộ giúp tăng hiệu suất

## Yêu cầu hệ thống

- Python 3.8 hoặc cao hơn
- Khoảng 4GB RAM trở lên
- (Tùy chọn) GPU với CUDA để tăng tốc độ

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/yourusername/ANPR-backend.git
cd ANPR-backend
```

2. Tạo môi trường ảo Python:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

3. Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

4. Tải model YOLO:
Đặt file model YOLO (`best.pt`) đã huấn luyện vào thư mục gốc, hoặc
chỉnh sửa file `.env` để chỉ định đường dẫn model.

## Cấu hình

Chỉnh sửa file `.env` để thay đổi các cài đặt:

```
# Đường dẫn đến model YOLO
YOLO_MODEL_PATH=best.pt
# Ngưỡng tin cậy cho model YOLO
YOLO_CONF_THRESHOLD=0.4
# Cài đặt tiền xử lý OCR
ENABLE_OCR_PREPROCESSING=true
# Cài đặt API server
API_HOST=0.0.0.0
API_PORT=5000
LOG_LEVEL=info
```

## Sử dụng

1. Khởi động API server:
```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

2. Truy cập tài liệu API (Swagger UI):
```
http://localhost:5000/docs
```

3. Gửi yêu cầu nhận dạng biển số:
```bash
curl -X POST "http://localhost:5000/process-image" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/image.jpg"
```

## Cấu trúc dự án

```
ANPR-backend/
├── .env                # Cấu hình môi trường
├── README.md           # Tài liệu hướng dẫn
├── requirements.txt    # Danh sách thư viện cần thiết
├── main.py             # FastAPI application
├── constants.py        # Các hằng số và dữ liệu tĩnh
├── config.py           # Quản lý cấu hình
├── schemas.py          # Định nghĩa cấu trúc dữ liệu
├── image_utils.py      # Các tiện ích xử lý ảnh
├── detection.py        # Xử lý model YOLO
├── ocr.py              # Xử lý OCR (PaddleOCR)
└── analysis.py         # Phân tích biển số
```

## API Endpoints

### POST /process-image

Upload ảnh và nhận dạng biển số xe.

**Request:**
- Form-data với key `file` chứa ảnh cần phân tích

**Response:**
```json
{
  "detections": [
    {
      "plate_number": "51F12345",
      "confidence_detection": 0.95,
      "bounding_box": [100, 200, 300, 250],
      "plate_analysis": {
        "original": "51F12345",
        "normalized": "51F12345",
        "province_code": "51",
        "province_name": "TP. Hồ Chí Minh",
        "serial": "F",
        "number": "12345",
        "plate_type": "personal",
        "plate_type_info": {
          "name": "Xe cá nhân", 
          "description": "Biển trắng, chữ đen"
        },
        "detected_color": "white",
        "is_valid_format": true,
        "format_description": "2 số - 1 chữ - 5 số"
      }
    }
  ],
  "processed_image_url": "data:image/jpeg;base64,..."
}
```

## Lưu ý

- Cần cài đặt model YOLO đã được huấn luyện cho biển số xe
- Có thể cần điều chỉnh các tham số cho phù hợp với biển số của từng quốc gia/khu vực
- Nếu không có GPU, tốc độ xử lý sẽ bị ảnh hưởng đáng kể

## License

[MIT License](LICENSE) 
# Hệ Thống Nhận Dạng Biển Số Xe (ANPR)

Hệ thống API nhận dạng biển số xe sử dụng YOLO, PaddleOCR, và tùy chọn OpenAI, được phát triển bằng FastAPI.

## Tính năng

- Phát hiện biển số xe trong ảnh bằng YOLO.
- OCR biển số xe sử dụng PaddleOCR.
- (Tùy chọn) Fallback sang OpenAI (gpt-4o-mini hoặc model khác) để OCR nếu PaddleOCR thất bại hoặc kết quả không hợp lệ.
- Phân tích và phân loại biển số (màu sắc, loại biển, mã tỉnh).
- API RESTful đơn giản dễ tích hợp.
- Xử lý bất đồng bộ giúp tăng hiệu suất.

## Cài đặt

1.  Clone repository:
    ```bash
    git clone https://github.com/yourusername/ANPR-backend.git
    cd ANPR-backend
    ```

2.  Tạo môi trường ảo Python:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # hoặc
    venv\Scripts\activate     # Windows
    ```

3.  Cài đặt các thư viện:
    ```bash
    pip install -r requirements.txt
    ```

4.  Tải model YOLO:
    Đặt file model YOLO (`best.pt` hoặc tên khác) vào thư mục `models/`, hoặc
    chỉnh sửa file `.env` để chỉ định đường dẫn model (`YOLO_MODEL_PATH`).

## Cấu hình

Chỉnh sửa file `.env` để thay đổi các cài đặt:

```dotenv
# Đường dẫn đến model YOLO (tương đối từ gốc dự án)
YOLO_MODEL_PATH=models/best.pt
# Ngưỡng tin cậy cho model YOLO
YOLO_CONF_THRESHOLD=0.4

# Cài đặt PaddleOCR
PADDLE_USE_GPU=false
PADDLE_LANGUAGE=en
PADDLE_USE_ANGLE_CLS=true

# Cài đặt tiền xử lý OCR (nếu muốn dùng trong image_utils)
ENABLE_OCR_PREPROCESSING=true
OCR_THRESH_BLOCK_SIZE=11
OCR_THRESH_C=5

# Cài đặt OpenAI Fallback
ENABLE_OPENAI_FALLBACK=true
OPENAI_API_KEY=sk-your_openai_api_key_here # Thay bằng key của bạn
OPENAI_MODEL=gpt-4o-mini

# Cài đặt API server
API_HOST=0.0.0.0
API_PORT=5000
LOG_LEVEL=debug # Hoặc info, warning, error
```

## Cấu trúc dự án (Sau Refactor)

```
ANPR-backend/
├── models/               # Chứa các file model (.pt)
│   └── best.pt
├── release/              # Chứa Dockerfile và các file liên quan đến deploy
│   └── Dockerfile
├── venv/                 # Thư mục môi trường ảo
├── .env                  # File cấu hình môi trường (cần tạo)
├── .env.example          # File cấu hình mẫu
├── .gitignore
├── config.py             # Load cấu hình từ .env, thiết lập logging
├── constants.py          # Chứa các hằng số (mã tỉnh, loại biển, màu sắc)
├── detection_processor.py # Xử lý việc chạy model YOLO
├── image_utils.py        # Các hàm tiện ích xử lý ảnh (decode, encode, download, vẽ, màu)
├── main.py               # Điểm vào chính, định nghĩa API endpoints, điều phối request
├── ocr_processor.py      # Xử lý việc chạy PaddleOCR, OpenAI OCR và logic fallback
├── analysis.py           # Phân tích chi tiết biển số từ text OCR cuối cùng
├── requirements.txt      # Danh sách các thư viện phụ thuộc
├── schemas.py            # Định nghĩa các Pydantic schema cho API và dữ liệu
```

**Mô tả các file chính sau refactor:**

-   **`config.py`**: Load cấu hình từ `.env`, thiết lập logging cơ bản.
-   **`constants.py`**: Lưu trữ các hằng số (mã tỉnh, loại biển, dải màu HSV).
-   **`detection_processor.py`**: Khởi tạo model YOLO và chứa hàm `run_detection` để phát hiện biển số.
-   **`image_utils.py`**: Chứa các hàm tiện ích xử lý ảnh (decode, encode, download, vẽ detection, xác định màu nền).
-   **`ocr_processor.py`**: Khởi tạo PaddleOCR, OpenAI client (nếu được bật). Chứa hàm `get_ocr_result` để thực hiện OCR (Paddle), kiểm tra kết quả, thực hiện fallback sang OpenAI nếu cần, và trả về kết quả OCR cuối cùng cùng engine đã sử dụng.
-   **`analysis.py`**: Chứa hàm `analyze_license_plate` để phân tích chi tiết (tỉnh, loại, định dạng) từ chuỗi text OCR cuối cùng nhận được.
-   **`schemas.py`**: Định nghĩa các Pydantic schema (request, response, cấu trúc dữ liệu nội bộ).
-   **`main.py`**: Điểm vào chính của ứng dụng FastAPI. Định nghĩa các API endpoints (`/process-image`, `/process-image-url`, `/health`). Điều phối luồng xử lý request bằng cách gọi các hàm từ `detection_processor`, `ocr_processor`, `analysis`, và `image_utils`.
-   **`requirements.txt`**: Danh sách các thư viện phụ thuộc.

## API Endpoints

(Giữ nguyên mô tả API endpoints `/process-image` và `/process-image-url` như trước, nhưng cập nhật ví dụ response để bao gồm `ocr_engine_used`)

### POST /process-image

Upload ảnh và nhận dạng biển số xe.

**Request:**

-   Form-data với key `file` chứa ảnh cần phân tích.

**Response:**

```json
{
  "detections": [
    {
      "plate_number": "88G133343", // Kết quả cuối cùng (có thể từ Paddle hoặc OpenAI)
      "confidence_detection": 0.92,
      "bounding_box": [89, 112, 120, 134],
      "plate_analysis": {
        "original": "88-G1 333.43", // Text gốc từ engine được chọn
        "normalized": "88G133343",
        "province_code": "88",
        "province_name": "Vĩnh Phúc",
        "serial": "G1",
        "number": "33343",
        "plate_type": "motorcycle",
        "plate_type_info": {
          "name": "Xe máy", 
          "description": "Biển số xe máy (thường là biển trắng, 2 dòng hoặc 1 dòng)"
        },
        "detected_color": "white",
        "is_valid_format": true,
        "format_description": "Biển số xe máy 2 dòng"
      },
      "ocr_engine_used": "openai"
    }
  ],
  "processed_image_url": "data:image/jpeg;base64,..."
}
```

### POST /process-image-url

Cung cấp URL ảnh và nhận dạng biển số xe.

**Request Body (JSON):**

```json
{
  "url": "https://example.com/image.jpg"
}
```

**Response:** (Tương tự như `/process-image`)

### GET /health

Kiểm tra trạng thái API và cấu hình OpenAI.

**Response:**

```json
{
  "status": "ok",
  "openai_enabled": true 
}
```

## Chạy ứng dụng

```bash
# Đứng từ thư mục gốc ANPR-backend
# Chạy với Uvicorn (khuyến nghị)
# Đặt log level debug qua command line
uvicorn main:app --host 0.0.0.0 --port 5000 --log-level debug --reload 

# Hoặc chạy trực tiếp (ít linh hoạt hơn về cấu hình server)
# python main.py
```

## Deploy với Docker

1.  Đảm bảo `release/Dockerfile` đã được cập nhật (thường không cần thay đổi nếu `main.py`, `requirements.txt` ở gốc).
2.  Build Docker image từ thư mục gốc `ANPR-backend`:
    ```bash
    docker build -t your-anpr-image:latest -f release/Dockerfile .
    ```
3.  Chạy container:
    ```bash
    docker run -p 5000:8000 --env-file .env your-anpr-image:latest
    ```
    Lưu ý: `-p 5000:8000` map cổng 8000 trong container (mặc định của CMD trong Dockerfile) ra cổng 5000 trên máy host. `--env-file .env` để truyền các biến môi trường vào container.

FROM python:3.9-slim

WORKDIR /app

# Cài đặt các dependencies cần thiết
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt với phiên bản mới nhất
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install --upgrade fastapi uvicorn[standard] python-multipart \
    opencv-python-headless numpy torch ultralytics paddlepaddle \
    paddleocr Pillow python-dotenv pydantic pydantic-settings \
    python-logging-loki requests

# Copy toàn bộ source code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Sử dụng Python 3.10
FROM python:3.10

# 1. Cài đặt các công cụ biên dịch và thư viện hệ thống
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libgeos-dev \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Biến môi trường để khởi động nhanh
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# 2. Cài đặt Torch CPU 2.1.0
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Cài đặt Detectron2 từ Git (Dùng Git để đảm bảo tương thích CPU)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# 4. Cài đặt  (Paddle 2.6.2 + OCR 2.7.0) 
# Ép dùng NumPy < 2 để tránh crash với Torch 2.1
RUN pip install --no-cache-dir \
    gradio \
    opencv-python \
    paddleocr==2.7.0 \
    paddlepaddle==2.6.2 \
    "numpy<2" \
    pillow \
    pyyaml

# 5. Thiết lập User cho Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 6. Copy code và model weights
COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
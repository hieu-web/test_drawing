# 📐 Engineering Drawing Detection & OCR System

Hệ thống tự động phát hiện và trích xuất thông tin từ bản vẽ kỹ thuật bằng Deep Learning. Hệ thống sử dụng **Detectron2** để nhận diện các thành phần (PartDrawing, Note, Table) và **PaddleOCR** để trích xuất nội dung văn bản.

---

## 🚀 Tính năng chính
- **Object Detection**: Phát hiện chính xác 3 loại thực thể trên bản vẽ:
  - `PartDrawing`: Khu vực chứa bản vẽ chi tiết.
  - `Note`: Các ghi chú kỹ thuật.
  - `Table`: Bảng thông số kỹ thuật (Title block, BOM).
- **Automated OCR**: Tự động nhận diện văn bản bên trong các block `Note` và `Table`.
- **Web Interface**: Giao diện trực quan được xây dựng bằng Gradio.
- **Dockerized**: Sẵn sàng triển khai trên Hugging Face Spaces hoặc Server riêng.

---

## 🛠 Cài đặt môi trường

### 1. Cài đặt trực tiếp (Local)
Yêu cầu: Python 3.10+

```bash
# Clone repository
git clone https://github.com/hieu-web/engineering_drawing.git
cd engineering_drawing

# Cài đặt PyTorch (Bản CPU được tối ưu cho deployment)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Cài đặt Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Cài đặt các thư viện khác (PaddleOCR, Gradio, OpenCV)
pip install -r requirements.txt
```

### 2. Sử dụng Docker
```bash
docker build -t drawing-ocr .
docker run -p 7860:7860 drawing-ocr
```

---

## 🏋️ Hướng dẫn Train Model

Quá trình training được thực hiện trên Google Colab để tận dụng GPU T4.

1.  **Chuẩn bị dữ liệu**: Cấu trúc dữ liệu theo định dạng COCO:
    ```text
    dataset/
    ├── images/
    └── result.json (COCO annotations)
    ```
2.  **Chạy Notebook**: Mở file `Untitled6.ipynb` trên Colab.
3.  **Cấu hình Training**:
    - Backbone: `faster_rcnn_R_50_FPN_3x`.
    - Số lượng class: 3 (`Note`, `PartDrawing`, `Table`).
    - Số vòng lặp: 3000 iterations.
    - Base Learning Rate: 0.00025.
4.  **Lưu kết quả**: Sau khi train xong, copy file `model_final.pth` vào thư mục gốc của dự án này.

---

## 🔍 Chạy Inference Pipeline

Để khởi động giao diện Web Demo cục bộ:
```bash
python app.py
```
Sau đó truy cập: `http://localhost:7860`

Giao diện cho phép:
- Upload ảnh bản vẽ (jpg, png, webp).
- Xem kết quả bounding box trực quan.
- Nhận kết quả JSON chi tiết và nội dung văn bản OCR chuyển đổi.

---

## 📦 Model Weights & Demo
- **Model Weights**: [Google Drive Link](https://drive.google.com/file/d/1hGeQ75RmGPqg8tSi7_MOtYrqaQWoNloG/view?usp=sharing) hoặc [HuggingFace Space](https://huggingface.co/spaces/hieu098/hieu1234/resolve/main/model_final.pth).
- **Web Demo URL**: [https://huggingface.co/spaces/hieu098/hieu1234](https://huggingface.co/spaces/hieu098/hieu1234)

---

## 📝 Báo cáo Kỹ thuật (Short Report)

### 1. Tiếp cận (Approach)
- **Detection**: Sử dụng Faster R-CNN với backbone ResNet-50 kết hợp FPN. Đây là kiến trúc cân bằng tốt giữa độ chính xác và tốc độ.
- **OCR**: Sử dụng PaddleOCR (bản 2.7.0) vì khả năng xử lý văn bản tiếng Anh/Số trong môi trường kỹ thuật rất mạnh mẽ, hỗ trợ tốt xoay góc văn bản.
- **Pipeline**: Ảnh đầu vào -> Detect các vùng -> Crop Note/Table -> Tiền xử lý -> OCR -> Tổng hợp kết quả JSON.

### 2. Các thử nghiệm (Experiments)
- **Xử lý NumPy 2.0**: Team đã giải quyết lỗi crash do xung đột giữa Torch 2.1 và NumPy 2.0 bằng cách giới hạn `numpy<2` trong Docker và requirements.
- **Tối ưu CPU**: Cấu hình Detectron2 chạy hoàn toàn trên CPU để giảm chi phí deployment mà vẫn đảm bảo tốc độ inference ~2-5s/ảnh.
- **Threshold**: Thử nghiệm `score_threshold` từ 0.3 đến 0.7; mức **0.5** cho kết quả tối ưu nhất, hạn chế dương tính giả (False Positive).

### 3. Kết quả đạt được
- Mô hình nhận diện tốt các khung tên (Table) và ghi chú (Note) ngay cả với bản vẽ có mật độ chi tiết dày đặc.
- OCR đọc được các thông số kích thước và bảng vật tư với độ chính xác cao.

### 4. Hướng cải thiện
- **Data Augmentation**: Thêm các phép xoay, nhiễu và làm mờ để mô hình bền bỉ hơn với ảnh chụp bản vẽ từ thực tế.
- **Fine-tuning OCR**: Train thêm một module OCR chuyên biệt cho font chữ kỹ thuật để giảm sai sót ở các ký tự đặc biệt (phi, độ, dung sai).
- **Optimized Inference**: Chuyển đổi mô hình sang định dạng ONNX hoặc TensorRT để tăng tốc độ inference lên gấp 3-5 lần.

---
*Dự án được thực hiện bởi Antigravity AI Assistant.*

# 🍅 Tomato Quality Detection System

Hệ thống nhận diện cà chua và phân tích chất lượng sử dụng AI và Computer Vision.

## ✨ Tính năng

- **Nhận diện cà chua**: Sử dụng YOLO để phát hiện cà chua trong ảnh
- **Phân tích chất lượng**: Tính toán RGB, Lycopene, và thời gian thu hoạch
- **Upload ảnh**: Hỗ trợ upload file ảnh hoặc chụp ảnh trực tiếp
- **Hiển thị kết quả**: Ảnh đã xử lý với bounding box và thông tin chi tiết
- **Xuất dữ liệu**: Tải về ảnh đã xử lý, dữ liệu CSV, và JSON

## 🏗️ Kiến trúc

### Backend (FastAPI)
- **API Endpoint**: `POST /api/v1/detect/tomato`
- **Input**: File ảnh (multipart/form-data)
- **Output**: JSON với processed_image_url và detections array
- **Xử lý**: YOLO detection, RGB analysis, Lycopene estimation, Harvest time prediction

### Frontend (React)
- **Upload**: Drag & drop hoặc click để chọn file
- **Camera**: Chụp ảnh trực tiếp từ webcam
- **Display**: Hiển thị ảnh gốc và ảnh đã xử lý
- **Analysis**: Thông tin chi tiết cho từng quả cà chua
- **Export**: Tải về 3 loại file (JPG, CSV, JSON)

## 🚀 Cài đặt và chạy

### Cách 1: Sử dụng Docker (Khuyến nghị)

#### Yêu cầu hệ thống
- Docker
- Docker Compose

#### Chạy ứng dụng
```bash
# Build và chạy container
docker-compose up --build

# Chạy ở background
docker-compose up -d

# Dừng container
docker-compose down
```

Ứng dụng sẽ chạy tại: `http://localhost:8000`

### Cách 2: Cài đặt thủ công

#### Yêu cầu hệ thống
- Python 3.8+
- Node.js 16+
- npm hoặc yarn

#### Backend Setup

1. **Cài đặt dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Chạy server**:
```bash
python main.py
```

Server sẽ chạy tại: `http://localhost:8000`

#### Frontend Setup

1. **Cài đặt dependencies**:
```bash
cd frontend
npm install
```

2. **Chạy ứng dụng**:
```bash
npm start
```

Ứng dụng sẽ chạy tại: `http://localhost:3000`

## 📊 API Documentation

### POST /api/v1/detect/tomato

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response**:
```json
{
  "success": true,
  "processed_image_url": "/static/processed_images/uuid.jpg",
  "detections": [
    {
      "box": [x1, y1, x2, y2],
      "confidence": 0.85,
      "rgb_avg": {"r": 200, "g": 50, "b": 30},
      "lycopene_estimate": 75.5,
      "harvest_time_label": "Ready to harvest"
    }
  ],
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET /api/v1/health

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔬 Thuật toán phân tích

### RGB Analysis
- Tính giá trị RGB trung bình trong bounding box
- Sử dụng để đánh giá màu sắc cà chua

### Lycopene Estimation
- Công thức giả lập: `(R * 0.8 - G * 0.3 - B * 0.2) / 255.0`
- Lycopene tỷ lệ thuận với màu đỏ, tỷ lệ nghịch với xanh lá và xanh dương

### Harvest Time Prediction
- Dựa trên độ đỏ: `redness = R / (R + G + B)`
- Phân loại:
  - `> 0.7`: Ready to harvest
  - `0.5-0.7`: Almost ready (2-3 days)
  - `0.3-0.5`: Not ready (1-2 weeks)
  - `< 0.3`: Too early (3+ weeks)

## 📁 Cấu trúc dự án

```
tomato/
├── backend/
│   ├── main.py              # FastAPI server
│   └── static/              # Static files
│       ├── processed_images/ # Processed images
│       └── uploads/         # Uploaded images
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── index.js         # React entry point
│   │   └── index.css        # Styles
│   └── package.json
├── requirements.txt         # Python dependencies
└── README.md
```

## 🎯 Sử dụng

1. **Mở ứng dụng**: Truy cập `http://localhost:3000`
2. **Upload ảnh**: Kéo thả file hoặc click để chọn
3. **Chụp ảnh**: Click "Chụp Ảnh" để sử dụng webcam
4. **Phân tích**: Click "Phân Tích" để xử lý ảnh
5. **Xem kết quả**: Ảnh đã xử lý và thông tin chi tiết
6. **Tải về**: Lưu ảnh, CSV, hoặc JSON

## 🔧 Tùy chỉnh

### Thay đổi thuật toán phân tích
Chỉnh sửa các method trong class `TomatoDetector`:
- `estimate_lycopene()`: Công thức tính Lycopene
- `estimate_harvest_time()`: Logic dự đoán thời gian thu hoạch

### Tích hợp YOLO thật
Thay thế method `detect_tomatoes()` để sử dụng YOLO model thực tế.

### Thay đổi giao diện
Chỉnh sửa file `frontend/src/index.css` và `frontend/src/App.js`.

## 🐛 Troubleshooting

### Backend không chạy được
- Kiểm tra Python version (>= 3.8)
- Cài đặt lại dependencies: `pip install -r requirements.txt`
- Kiểm tra port 8000 có bị chiếm không

### Frontend không kết nối được API
- Đảm bảo backend đang chạy tại port 8000
- Kiểm tra proxy trong `package.json`
- Kiểm tra CORS settings trong backend

### Lỗi xử lý ảnh
- Kiểm tra format ảnh (JPG, PNG)
- Đảm bảo kích thước ảnh hợp lý
- Kiểm tra quyền ghi file trong thư mục static

## 📝 Ghi chú

- Đây là phiên bản demo với YOLO simulation
- Công thức tính Lycopene và Harvest Time là giả lập
- Để sử dụng thực tế, cần tích hợp YOLO model thật và công thức chính xác
- Hệ thống hỗ trợ xử lý ảnh đa định dạng (JPG, PNG, etc.)

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

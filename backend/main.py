from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from PIL import Image
import io
import json
import os
import uuid
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any
import base64

app = FastAPI(title="Tomato Quality Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo thư mục để lưu ảnh
os.makedirs("static/processed_images", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class TomatoDetector:
    def __init__(self):
        # Giả lập YOLO model (thực tế sẽ load model YOLO)
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
    
    def detect_tomatoes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Giả lập YOLO detection cho cà chua
        Trong thực tế, đây sẽ là model YOLO thật
        """
        height, width = image.shape[:2]
        detections = []
        
        # Giả lập phát hiện 2-4 quả cà chua ngẫu nhiên
        num_tomatoes = np.random.randint(2, 5)
        
        for i in range(num_tomatoes):
            # Tạo box ngẫu nhiên
            x1 = np.random.randint(0, width//2)
            y1 = np.random.randint(0, height//2)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(50, 150)
            
            # Đảm bảo box không vượt quá kích thước ảnh
            x2 = min(x2, width)
            y2 = min(y2, height)
            
            confidence = np.random.uniform(0.6, 0.95)
            
            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(confidence)
            })
        
        return detections
    
    def calculate_rgb_average(self, image: np.ndarray, box: List[int]) -> Dict[str, float]:
        """Tính RGB trung bình trong box"""
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {"r": 0, "g": 0, "b": 0}
        
        # Chuyển từ BGR sang RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        avg_rgb = np.mean(roi_rgb, axis=(0, 1))
        
        return {
            "r": float(avg_rgb[0]),
            "g": float(avg_rgb[1]),
            "b": float(avg_rgb[2])
        }
    
    def estimate_lycopene(self, rgb: Dict[str, float]) -> float:
        """
        Ước tính Lycopene dựa trên RGB
        Công thức giả lập: Lycopene tỷ lệ thuận với R và tỷ lệ nghịch với G, B
        """
        r, g, b = rgb["r"], rgb["g"], rgb["b"]
        
        # Công thức giả lập (trong thực tế sẽ phức tạp hơn)
        lycopene = (r * 0.8 - g * 0.3 - b * 0.2) / 255.0
        lycopene = max(0, min(1, lycopene))  # Clamp between 0 and 1
        
        return round(lycopene * 100, 2)  # Trả về phần trăm
    
    def estimate_harvest_time(self, rgb: Dict[str, float]) -> str:
        """
        Ước tính thời gian thu hoạch dựa trên RGB
        Công thức giả lập dựa trên màu sắc
        """
        r, g, b = rgb["r"], rgb["g"], rgb["b"]
        
        # Tính độ đỏ
        redness = r / (r + g + b + 1e-6)
        
        if redness > 0.7:
            return "Ready to harvest"
        elif redness > 0.5:
            return "Almost ready (2-3 days)"
        elif redness > 0.3:
            return "Not ready (1-2 weeks)"
        else:
            return "Too early (3+ weeks)"
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Vẽ box và nhãn lên ảnh"""
        result_image = image.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["box"]
            confidence = detection["confidence"]
            
            # Vẽ box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Vẽ nhãn
            label = f"Tomato {i+1}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Vẽ background cho text
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Vẽ text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image

detector = TomatoDetector()

@app.post("/api/v1/detect/tomato")
async def detect_tomato(file: UploadFile = File(...)):
    """
    API endpoint để nhận diện cà chua và phân tích chất lượng
    """
    try:
        # Đọc ảnh
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Phát hiện cà chua
        detections = detector.detect_tomatoes(image)
        
        # Xử lý từng detection
        processed_detections = []
        for detection in detections:
            # Tính RGB trung bình
            rgb_avg = detector.calculate_rgb_average(image, detection["box"])
            
            # Ước tính Lycopene
            lycopene_estimate = detector.estimate_lycopene(rgb_avg)
            
            # Ước tính thời gian thu hoạch
            harvest_time_label = detector.estimate_harvest_time(rgb_avg)
            
            processed_detections.append({
                "box": detection["box"],
                "confidence": detection["confidence"],
                "rgb_avg": rgb_avg,
                "lycopene_estimate": lycopene_estimate,
                "harvest_time_label": harvest_time_label
            })
        
        # Vẽ kết quả lên ảnh
        result_image = detector.draw_detections(image, processed_detections)
        
        # Lưu ảnh đã xử lý
        unique_id = str(uuid.uuid4())
        processed_image_path = f"static/processed_images/{unique_id}.jpg"
        cv2.imwrite(processed_image_path, result_image)
        
        # Tạo URL cho ảnh đã xử lý
        processed_image_url = f"/static/processed_images/{unique_id}.jpg"
        
        return JSONResponse(content={
            "success": True,
            "processed_image_url": processed_image_url,
            "detections": processed_detections,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

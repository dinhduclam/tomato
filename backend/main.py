from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any
from ultralytics import YOLO

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

# Mount static files (images) - with a different path to avoid conflict with React
app.mount("/api/static", StaticFiles(directory="static"), name="api_static")

# We'll serve the frontend manually via the catch-all route
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "build")
backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend")

# Mount React build static assets
static_react_path = os.path.join(frontend_path, "static")
if os.path.exists(static_react_path):
    app.mount("/static", StaticFiles(directory=static_react_path), name="static")

class TomatoDetector:
    def __init__(self):
        # Giả lập YOLO model (thực tế sẽ load model YOLO)
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
    
    def detect_tomatoes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # Load mô hình đã train
        model = YOLO(os.path.join(backend_path, "model", "best.pt"))
        results = model.predict(source=image)
        result = results[0]
        boxes = result.boxes

        detections = []

        for box in boxes:
            # Tọa độ bounding box (pixel)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Độ tin cậy
            conf = float(box.conf[0])

            # ID lớp
            cls = int(box.cls[0])

            # Tên lớp
            label = result.names[cls]

            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": 1,
                "label_id": cls,
                "label_name": label
            })
        
        return detections
    
    def extract_rgb_with_grabcut(self, image: np.ndarray, box: List[int]) -> Dict[str, float]:
        """
        Grab cut để cut ra quả cà chua
        Tính RGB trung bình trong box
        """
        rect = box
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)

        # Xóa nền bằng GrabCut
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        result = image * mask2[:, :, np.newaxis]

        # Lấy pixel của đối tượng
        pixels = result[np.where(mask2 == 1)]
        r_mean, g_mean, b_mean = np.mean(pixels, axis=0)
        return {
            "r": r_mean,
            "g": g_mean,
            "b": b_mean
        }

    def estimate_lycopene(self, rgb: Dict[str, float]) -> float:
        """
        Ước tính Lycopene dựa trên RGB
        """
        r, g, b = rgb["r"], rgb["g"], rgb["b"]
        

        x = (2*r-g-b) / (r+g+b)
        lycopene = 1.1264*x - 0.0359
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
            rgb_avg = detector.extract_rgb_with_grabcut(image, detection["box"])
            
            # Ước tính Lycopene
            lycopene_estimate = detector.estimate_lycopene(rgb_avg)
            
            # Ước tính thời gian thu hoạch
            harvest_time_label = detector.estimate_harvest_time(rgb_avg)
            
            processed_detections.append({
                "box": detection["box"],
                "confidence": detection["confidence"],
                "label_id": detection["label_id"],
                "label_name": detection["label_name"],
                "rgb_avg": rgb_avg,
                "lycopene_estimate": lycopene_estimate,
                "harvest_time_label": harvest_time_label
            })
        
        # Vẽ kết quả lên ảnh
        result_image = detector.draw_detections(image, processed_detections)
        
        print("Response:", processed_detections)

        # Lưu ảnh đã xử lý
        unique_id = str(uuid.uuid4())
        processed_image_path = f"static/processed_images/{unique_id}.jpg"
        cv2.imwrite(processed_image_path, result_image)
        
        # Tạo URL cho ảnh đã xử lý
        processed_image_url = f"/api/static/processed_images/{unique_id}.jpg"
        
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

# Catch-all route for React app (must be last)
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str, request: Request):
    """Serve React app or static assets"""
    # Don't intercept API routes
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    
    # Try to serve the requested file from frontend build
    file_path = os.path.join(frontend_path, full_path)
    
    if os.path.isfile(file_path) and os.path.exists(file_path):
        return FileResponse(file_path)
    
    # Otherwise serve index.html for React routing
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return {"message": "Frontend not built. Please run 'npm run build' in the frontend directory."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

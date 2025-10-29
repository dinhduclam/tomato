import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { saveAs } from 'file-saver';
import './index.css';

// API base URL - automatically use current domain and port
const getApiBaseUrl = () => {
  if (process.env.NODE_ENV === 'production') {
    // In production (Docker), use relative URL
    return '';
  } else {
    // In development, use current window location
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
};
const API_BASE_URL = getApiBaseUrl();

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  
  const fileInputRef = useRef(null);
  const webcamRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      setSuccess(null);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setError(null);
      setSuccess(null);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const capturePhoto = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      // Convert data URL to File object
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], 'captured-photo.jpg', { type: 'image/jpeg' });
          setSelectedFile(file);
          setShowCamera(false);
          setError(null);
          setSuccess(null);
        });
    }
  }, [webcamRef]);

  const processImage = async () => {
    if (!selectedFile) {
      setError('Vui lòng chọn ảnh trước khi xử lý');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post(`${API_BASE_URL}/api/v1/detect/tomato`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setProcessedImage(response.data.processed_image_url);
        setDetections(response.data.detections);
        setSuccess(`Đã phát hiện ${response.data.detections.length} quả cà chua`);
      } else {
        setError('Có lỗi xảy ra khi xử lý ảnh');
      }
    } catch (err) {
      setError(`Lỗi: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const saveProcessedImage = () => {
    if (processedImage) {
      const link = document.createElement('a');
      link.href = `${API_BASE_URL}${processedImage}`;
      link.download = 'processed-tomato-image.jpg';
      link.click();
    }
  };

  const saveCSV = () => {
    if (detections.length === 0) return;

    const csvData = detections.map((detection, index) => ({
      'Tomato ID': index + 1,
      'Confidence': detection.confidence,
      'Label ID': detection.label_id,
      'Label Name': detection.label_name,
      'R': detection.rgb_avg.r,
      'G': detection.rgb_avg.g,
      'B': detection.rgb_avg.b,
      'Lycopene (%)': detection.lycopene_estimate,
      'Harvest Time': detection.harvest_time_label,
      'Box X1': detection.box[0],
      'Box Y1': detection.box[1],
      'Box X2': detection.box[2],
      'Box Y2': detection.box[3]
    }));

    const csv = convertToCSV(csvData);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, 'tomato-analysis.csv');
  };

  const saveJSON = () => {
    if (detections.length === 0) return;

    const jsonData = {
      timestamp: new Date().toISOString(),
      total_tomatoes: detections.length,
      detections: detections
    };

    const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
    saveAs(blob, 'tomato-analysis.json');
  };

  const convertToCSV = (data) => {
    const headers = Object.keys(data[0]);
    const csvContent = [
      headers.join(','),
      ...data.map(row => headers.map(header => row[header]).join(','))
    ].join('\n');
    return csvContent;
  };

  const getHarvestTimeClass = (harvestTime) => {
    if (harvestTime.includes('Ready to harvest')) return 'harvest-ready';
    if (harvestTime.includes('Almost ready')) return 'harvest-almost';
    return 'harvest-not-ready';
  };

  return (
    <div className="container">
      <div className="header">
        <h1>🍅 Tomato Quality Detection</h1>
        <p>Phân tích chất lượng cà chua dựa trên AI và Computer Vision</p>
      </div>

      <div className="main-content">
        <div className="card">
          <h2>📸 Upload Ảnh</h2>
          
          <div 
            className="upload-area"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-icon">📁</div>
            <div className="upload-text">
              {selectedFile ? selectedFile.name : 'Kéo thả ảnh vào đây hoặc click để chọn'}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
            />
          </div>

          <div className="button-group">
            <button 
              className="btn btn-primary" 
              onClick={() => setShowCamera(!showCamera)}
            >
              {showCamera ? 'Ẩn Camera' : '📷 Chụp Ảnh'}
            </button>
            <button 
              className="btn btn-secondary" 
              onClick={processImage}
              disabled={!selectedFile || loading}
            >
              {loading ? 'Đang xử lý...' : '🔍 Phân Tích'}
            </button>
          </div>

          {showCamera && (
            <div className="camera-section">
              <div className="webcam-container">
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  width="100%"
                  height="auto"
                />
                <button 
                  className="capture-button"
                  onClick={capturePhoto}
                >
                  📸 Chụp Ảnh
                </button>
              </div>
            </div>
          )}

          {selectedFile && (
            <div style={{ marginTop: '20px', textAlign: 'center' }}>
              <img 
                src={URL.createObjectURL(selectedFile)} 
                alt="Selected" 
                className="image-preview"
              />
            </div>
          )}
        </div>

        <div className="card">
          <h2>📊 Kết Quả Phân Tích</h2>
          
          {loading && (
            <div className="loading">
              <div className="loading-spinner"></div>
              <p>Đang xử lý ảnh và phân tích chất lượng cà chua...</p>
            </div>
          )}

          {error && (
            <div className="error">
              {error}
            </div>
          )}

          {success && (
            <div className="success">
              {success}
            </div>
          )}

          {processedImage && (
            <div style={{ marginBottom: '20px' }}>
              <img 
                src={`${API_BASE_URL}${processedImage}`} 
                alt="Processed" 
                className="image-preview"
              />
            </div>
          )}

          {detections.length > 0 && (
            <div className="results">
              <h3>Chi Tiết Phát Hiện:</h3>
              {detections.map((detection, index) => (
                <div key={index} className="detection-item">
                  <div className="detection-header">
                    <span className="tomato-label">Cà chua #{index + 1}</span>
                    <span className="confidence">
                      {(detection.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="detection-details">
                    <div className="detail-item">
                      <div className="detail-label">Màu RGB</div>
                      <div className="rgb-display">
                        <div 
                          className="rgb-color" 
                          style={{ backgroundColor: `rgb(${detection.rgb_avg.r}, ${detection.rgb_avg.g}, ${detection.rgb_avg.b})` }}
                        ></div>
                        <span>{Math.round(detection.rgb_avg.r)}, {Math.round(detection.rgb_avg.g)}, {Math.round(detection.rgb_avg.b)}</span>
                      </div>
                    </div>
                    
                    <div className="detail-item">
                      <div className="detail-label">Lycopene</div>
                      <div className="detail-value">{detection.lycopene_estimate}%</div>
                    </div>

                    <div className="detail-item">
                      <div className="detail-label">Label</div>
                      <div className="detail-value">{detection.label_name}</div>
                    </div>
                    
                    <div className="detail-item">
                      <div className="detail-label">Thời gian thu hoạch</div>
                      <div className={`detail-value harvest-time ${getHarvestTimeClass(detection.harvest_time_label)}`}>
                        {detection.harvest_time_label}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {detections.length > 0 && (
        <div className="save-section">
          <h2>💾 Lưu Kết Quả</h2>
          <div className="save-buttons">
            <button className="btn btn-success" onClick={saveProcessedImage}>
              📷 Tải ảnh đã xử lý
            </button>
            <button className="btn btn-success" onClick={saveCSV}>
              📊 Tải dữ liệu CSV
            </button>
            <button className="btn btn-success" onClick={saveJSON}>
              📄 Tải dữ liệu JSON
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

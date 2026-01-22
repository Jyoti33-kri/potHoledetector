# potHoledetector
🚧 Pothole Detector - Real-time Web Application
A complete MVP web application for detecting potholes in real-time using device cameras (laptop/phone/tablet). Built with FastAPI backend and YOLOv8 computer vision model.



🎯 Features
Real-time Detection: Live pothole detection from device camera
Cross-Device Support: Works on laptops, phones, and tablets via browser
Interactive UI: Start/stop controls with live statistics
Performance Monitoring: Real-time FPS and confidence metrics
Responsive Design: Mobile-friendly interface


🛠️ Tech Stack
Backend
FastAPI - Modern Python web framework
YOLOv8 (Ultralytics) - State-of-the-art object detection
OpenCV - Image processing
Uvicorn - ASGI server
Frontend
HTML5/CSS3/JavaScript - Vanilla web technologies
Canvas API - Bounding box rendering
MediaDevices API - Camera access

📦 Installation
Prerequisites
Python 3.11+
Modern web browser (Chrome, Firefox, Safari, Edge)
Setup Steps
Install Dependencies
pip install -r requirements.txt
Run the Application
uvicorn main:app --host 0.0.0.0 --port 5000
Access the App
Open your browser and navigate to: http://localhost:5000
On mobile: Use your computer's IP address (e.g., http://192.168.1.100:5000)
🚀 Usage
Start Camera: Click "Start Camera" and allow camera access
Begin Detection: Click "Start Detection" to begin real-time analysis
View Results: Detected potholes appear with bounding boxes and confidence scores
Monitor Stats: Track detection count, FPS, and average confidence
Stop: Click "Stop Detection" or "Stop Camera" when done
📁 Project Structure
pothole-detector/
├── main.py              # FastAPI backend server
├── index.html           # Frontend UI
├── static/
│   └── script.js        # Camera and detection logic
├── requirements.txt     # Python dependencies
└── README.md           # Documentation
🔧 API Endpoints
GET /
Returns the main HTML interface
POST /detect
Accepts: Image file (JPEG/PNG)
Returns: JSON with detection results
{
  "detections": [
    {
      "label": "pothole",
      "confidence": 0.89,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "count": 1,
  "image_size": {"width": 1280, "height": 720}
}
GET /health
Health check endpoint
Returns model status
GET /model-info
Returns information about loaded model
🤖 Detection Method
Current Implementation (MVP Demo)
Detection Method: OpenCV Edge Detection

Uses Canny edge detection and contour analysis
Simulates pothole detection for demonstration
Lightweight and works without GPU requirements
For Production Use - Real YOLOv8 Integration
To use actual YOLOv8 pothole detection:

Install PyTorch & YOLOv8 (requires ~2-3GB disk space):

pip install torch torchvision ultralytics
Get a Pothole-Trained Model:

Download from Roboflow Universe
Or train your own using Kaggle pothole datasets
Use YOLOv8 fine-tuned on pothole images
Update main.py:

Uncomment the YOLO import: from ultralytics import YOLO
Replace detect_with_edge_detection() with actual YOLO inference
Load your model: model = YOLO('path/to/pothole-model.pt')
Example YOLO Integration:

from ultralytics import YOLO

@app.on_event("startup")
async def load_model():
    global model
    model = YOLO('yolov8n-pothole.pt')  # Your trained model

# In detect_potholes function:
results = model(img_array, verbose=False)
for result in results:
    boxes = result.boxes
    # Process detections...
🔄 Customization
Change Model
Edit main.py line 43:

model = YOLO('your-pothole-model.pt')
Adjust Detection Frequency
Edit static/script.js line 103:

detectionInterval = setInterval(captureAndDetect, 333); // 333ms = ~3 FPS
Modify Camera Settings
Edit static/script.js line 34:

const constraints = {
    video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'environment' // 'user' for front camera
    }
};
🚀 Future Enhancements
 GPS tagging of detected potholes
 SQLite/PostgreSQL database for storage
 Heatmap visualization with Leaflet.js
 Multi-user support
 Detection history and analytics
 Export detection reports
🐛 Troubleshooting
Camera Access Denied

Ensure HTTPS connection (or localhost)
Check browser permissions
Try different browser
Model Not Loading

Check internet connection (downloads model on first run)
Verify Python dependencies installed
Check console for error messages
Slow Detection

Reduce detection frequency in script.js
Use smaller image resolution
Try YOLOv8n (nano) for faster inference
📝 License
MIT License - Free to use and modify

🤝 Contributing
Contributions welcome! Please feel free to submit issues and pull requests.

📧 Support
For issues and questions, please open an issue on the repository.

Built with ❤️ using FastAPI and YOLOv8

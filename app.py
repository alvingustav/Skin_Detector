from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import torch
import numpy as np
import base64
import os
import time
import threading
from io import BytesIO

# Global variables
camera = None
output_frame = None
frame_lock = threading.Lock()
detection_counts = {}
detection_active = False
camera_index = 0  # ✅ TAMBAHAN: Variable untuk index kamera

# Ensure templates directory exists
os.makedirs('templates', exist_ok=True)

# Copy index.html to templates if needed
if not os.path.exists('templates/index.html') and os.path.exists('index.html'):
    import shutil
    print("Copying index.html to templates directory...")
    shutil.copy('index.html', 'templates/index.html')

# Import YOLO with error handling
try:
    from ultralytics import YOLO
    print("Successfully imported YOLO from ultralytics")
except ImportError as e:
    print(f"Error importing YOLO: {e}")
    print("Please install ultralytics package: pip install ultralytics==8.0.145")
    
    # Provide a mock implementation for testing UI without model
    class MockYOLO:
        def __init__(self, model_path):
            print(f"Mock YOLO initialized with {model_path}")
        
        def __call__(self, img):
            # Return a simple mock result
            from types import SimpleNamespace
            result = SimpleNamespace()
            result.boxes = []
            result.names = {0: 'person', 1: 'bicycle', 2: 'car'}
            return [result]
    
    YOLO = MockYOLO

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'yolov8detection'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLOv8 model
model = None

def load_model():
    global model
    try:
        # Prioritas loading model
        model_path = os.environ.get('MODEL_PATH', 'my_model1.pt')
        
        # Cek apakah file model ada
        if os.path.exists(model_path):
            print(f"Loading custom model: {model_path}")
            model = YOLO(model_path)
            print(f"✅ Custom model loaded successfully: {model_path}")
        else:
            print(f"❌ Custom model {model_path} not found, trying fallback...")
            # Fallback ke model standard
            model = YOLO('yolov8n.pt')
            print("✅ Fallback model loaded: yolov8n.pt")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        # Final fallback
        try:
            model = YOLO('yolov8n.pt')
            print("✅ Final fallback model loaded")
        except Exception as e2:
            print(f"All model loading failed: {e2}")
            model = MockYOLO('fallback')

# Load model in a separate thread to prevent blocking
threading.Thread(target=load_model).start()

def detect_objects(frame):
    """Detect objects in a frame using YOLOv8"""
    global detection_counts
    
    if model is None:
        # Return original frame if model is not loaded
        return frame, {}
    
    try:
        # Perform detection
        results = model(frame)
        
        # Process results
        result = results[0]
        counts = {}
        
        # Draw bounding boxes and count objects
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get class name
            class_name = result.names[cls]
            
            # Update counts
            if class_name in counts:
                counts[class_name] += 1
            else:
                counts[class_name] = 1
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        detection_counts = counts
        return frame, counts
        
    except Exception as e:
        print(f"Error in detection: {e}")
        return frame, {}

def generate_frames():
    """Generate frames for video streaming"""
    global output_frame, detection_counts, detection_active
    
    while detection_active:
        # Wait until a new frame is available
        with frame_lock:
            if output_frame is None:
                continue
            frame = output_frame.copy()
        
        # Detect objects in the frame
        processed_frame, counts = detect_objects(frame)
        
        # Emit detection counts via Socket.IO
        socketio.emit('detection_update', {'counts': counts})
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def webcam_stream():
    """Capture frames from webcam"""
    global camera, output_frame, detection_active, camera_index
    
    try:
        # ✅ Gunakan camera_index yang dipilih user
        camera = cv2.VideoCapture(camera_index)
        
        # Set resolution
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while detection_active:
            success, frame = camera.read()
            if not success:
                break
            
            # Update the output frame
            with frame_lock:
                output_frame = frame.copy()
            
            time.sleep(0.03)  # ~30 FPS
            
    except Exception as e:
        print(f"Camera error: {e}")
    finally:
        # Release resources
        if camera is not None:
            camera.release()

# ✅ Tambahkan endpoint untuk camera selection
@app.route('/set_camera', methods=['POST'])
def set_camera():
    global camera_index
    data = request.get_json()
    camera_index = data.get('camera_index', 0)
    return jsonify({'status': 'success', 'camera_index': camera_index})

# ✅ TAMBAHAN: Endpoint untuk deteksi kamera yang tersedia
@app.route('/available_cameras', methods=['GET'])
def available_cameras():
    available = []
    for idx in range(5):  # Check up to 5 camera indices
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return jsonify({'cameras': available})

@app.route('/')
def index():
    """Serve the index page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    global detection_active
    
    if not detection_active:
        detection_active = True
        # Start webcam stream in a new thread
        threading.Thread(target=webcam_stream).start()
        # Start detection in a new thread
        threading.Thread(target=generate_frames).start()
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream', methods=['GET'])
def stop_stream():
    """Stop the webcam stream"""
    global detection_active, camera
    detection_active = False
    return jsonify({'status': 'success'})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and perform detection"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'No selected file'})
    
    try:
        # Read image as bytes
        img_bytes = file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform detection
        processed_img, counts = detect_objects(img)
        
        # Convert processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'image': img_base64,
            'counts': counts
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

# Tambahkan di bagian akhir file app.py
if __name__ == '__main__':
    # Determine port for Azure deployment
    port = int(os.environ.get('PORT', 8000))
    
    # Run the app - PERBAIKAN untuk Azure
    if 'WEBSITE_HOSTNAME' in os.environ:
        # Running on Azure - gunakan socketio.run
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    else:
        # Running locally
        socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)

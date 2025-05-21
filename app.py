import os
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
from ultralytics import YOLO
import numpy as np
import base64
import time
import threading

# Inisialisasi Flask App dan SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key_for_socketio' # Ganti dengan secret key yang kuat
socketio = SocketIO(app, async_mode='eventlet') # Atau 'gevent'

# Muat model YOLOv8
# Ganti 'my_model1.pt' dengan path ke model Anda, atau gunakan model standar YOLO
MODEL_PATH = 'my_model1.pt'
if not os.path.exists(MODEL_PATH):
    print(f"Peringatan: File model '{MODEL_PATH}' tidak ditemukan. Menggunakan 'yolov8n.pt' sebagai gantinya.")
    MODEL_PATH = 'yolov8n.pt' # Model default jika my_model1.pt tidak ada
try:
    model = YOLO(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model: {e}")
    print("Pastikan Anda memiliki file model .pt yang benar dan library Ultralytics terinstal.")
    exit()

# Variabel global untuk stream webcam
camera = None
camera_active = False
camera_lock = threading.Lock() # Untuk mengontrol akses ke variabel camera_active dan camera

def get_camera():
    """Menginisialisasi dan mengembalikan objek kamera."""
    global camera
    if camera is None:
        try:
            camera = cv2.VideoCapture(0) # Coba buka kamera default
            if not camera.isOpened():
                print("Error: Tidak bisa membuka webcam.")
                camera = None # Set kembali ke None jika gagal
        except Exception as e:
            print(f"Error saat mengakses kamera: {e}")
            camera = None
    return camera

def release_camera():
    """Melepaskan objek kamera."""
    global camera, camera_active
    with camera_lock:
        if camera:
            camera.release()
            camera = None
        camera_active = False
    print("Kamera dilepaskan.")

def generate_frames():
    """Generator untuk frame video dari webcam dengan deteksi objek."""
    global camera_active
    
    local_camera = get_camera()
    if not local_camera or not local_camera.isOpened():
        print("Gagal mengakses kamera untuk generate_frames.")
        # Emit pesan error ke client jika diinginkan, atau stream gambar placeholder
        # Untuk saat ini, kita biarkan stream kosong atau error di sisi client
        return

    with camera_lock:
        camera_active = True

    print("Memulai streaming webcam...")
    frame_count = 0
    fps_start_time = time.time()

    while True:
        with camera_lock:
            if not camera_active or not local_camera or not local_camera.isOpened():
                print("Menghentikan generate_frames karena kamera tidak aktif atau tidak tersedia.")
                break
        
        success, frame = local_camera.read()
        if not success:
            print("Gagal membaca frame dari kamera.")
            time.sleep(0.1) # Beri jeda sebelum mencoba lagi
            # Coba re-inisialisasi kamera jika gagal membaca frame
            local_camera.release()
            local_camera = cv2.VideoCapture(0)
            if not local_camera.isOpened():
                print("Gagal re-inisialisasi kamera.")
                break
            continue

        # Lakukan deteksi objek
        results = model(frame, verbose=False) # verbose=False untuk mengurangi output log YOLO

        # Gambar bounding box dan label pada frame
        annotated_frame = results[0].plot() # results[0].plot() akan menggambar langsung di frame

        # Hitung objek
        counts = {}
        if results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                counts[class_name] = counts.get(class_name, 0) + 1
        
        # Kirim update statistik via Socket.IO
        # Untuk mengurangi beban, kirim update setiap beberapa frame atau interval waktu
        frame_count += 1
        if frame_count % 5 == 0: # Kirim update setiap 5 frame
            socketio.emit('detection_update', {'counts': counts})

        # Hitung FPS (opsional, untuk debugging)
        if frame_count % 30 == 0: # Hitung FPS setiap 30 frame
            elapsed_time = time.time() - fps_start_time
            fps = 30 / elapsed_time if elapsed_time > 0 else 0
            # print(f"FPS: {fps:.2f}") # Bisa di-log jika perlu
            fps_start_time = time.time() # Reset timer

        # Encode frame ke JPEG
        try:
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("Gagal meng-encode frame ke JPEG.")
                continue
            frame_bytes = buffer.tobytes()
            # Kirim frame sebagai MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error saat encoding atau yielding frame: {e}")
            continue
        
        socketio.sleep(0.01) # Kontrol laju frame sedikit untuk mengurangi beban CPU

    # Pastikan kamera dilepaskan jika loop berhenti
    if local_camera and local_camera.isOpened():
        local_camera.release()
    print("Streaming webcam dihentikan dari generate_frames.")


@app.route('/')
def index():
    """Menyajikan halaman utama."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route untuk streaming video."""
    if not camera_active: # Hanya mulai jika belum aktif
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Jika sudah aktif, mungkin bisa return error atau biarkan client menunggu
        # Untuk saat ini, kita coba berikan response yang sama, tapi generate_frames punya lock sendiri
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_stream', methods=['GET', 'POST']) # Bisa GET atau POST dari JS
def stop_stream():
    """Menghentikan stream webcam."""
    print("Menerima permintaan untuk menghentikan stream...")
    release_camera()
    return jsonify({'status': 'stream_stopped'})

@app.route('/upload', methods=['POST'])
def upload_image():
    """Menerima upload gambar, melakukan deteksi, dan mengembalikan hasilnya."""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'No selected file'})

    if file:
        try:
            # Baca gambar dari file stream
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({'status': 'error', 'error': 'Could not decode image'})

            # Lakukan deteksi objek
            results = model(img, verbose=False)
            annotated_img = results[0].plot() # Dapatkan gambar dengan anotasi

            # Hitung objek
            counts = {}
            if results[0].boxes:
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    counts[class_name] = counts.get(class_name, 0) + 1
            
            # Encode gambar hasil ke base64
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                'status': 'success',
                'image': img_base64,
                'counts': counts
            })
        except Exception as e:
            print(f"Error processing uploaded image: {e}")
            return jsonify({'status': 'error', 'error': f'Error processing image: {str(e)}'})

    return jsonify({'status': 'error', 'error': 'Unknown error'})


# Socket.IO event handlers (jika diperlukan selain emit dari route)
@socketio.on('connect')
def handle_connect():
    print('Client terhubung')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client terputus')
    # Pertimbangkan untuk menghentikan stream jika tidak ada client yang terhubung
    # Namun, ini bisa kompleks jika ada multiple client, jadi stop_stream eksplisit lebih aman


if __name__ == '__main__':
    print("Memulai aplikasi Flask...")
    print(f"Akses aplikasi di http://127.0.0.1:5000")
    # socketio.run(app, debug=True, host='0.0.0.0') # Gunakan eventlet atau gevent untuk produksi
    # Untuk pengembangan, debug=True bisa berguna. Untuk deployment, gunakan server WSGI seperti gunicorn.
    # 'eventlet' atau 'gevent' diperlukan untuk performa SocketIO yang baik.
    socketio.run(app, host='0.0.0.0', port=5000, use_reloader=True, debug=True, allow_unsafe_werkzeug=True)
    # Catatan: allow_unsafe_werkzeug=True hanya untuk pengembangan agar auto-reloader berfungsi dengan baik
    # Pada production, Anda akan menggunakan Gunicorn atau server WSGI lainnya.

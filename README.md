# YOLOv8 Object Detection Web App

Aplikasi web deteksi objek berbasis YOLOv8 yang dapat di-deploy di Render.com. Aplikasi ini menyediakan antarmuka pengguna yang interaktif untuk deteksi objek baik menggunakan webcam maupun melalui upload gambar.

## Fitur

- 📹 Deteksi objek secara real-time melalui webcam
- 🖼️ Upload dan analisis gambar
- 📊 Statistik penghitungan objek
- 🚀 Antarmuka pengguna yang responsif dan menarik
- ☁️ Mudah di-deploy di platform Render.com

## Teknologi yang Digunakan

- **Backend**: Flask, Flask-SocketIO
- **Frontend**: HTML, CSS (Tailwind CSS), JavaScript
- **Machine Learning**: Ultralytics YOLOv8
- **Computer Vision**: OpenCV
- **Deployment**: Docker, Render.com

## Cara Menjalankan Aplikasi Secara Lokal

1. Clone repository:
   ```
   git clone https://github.com/yourusername/yolov8-detection-app.git
   cd yolov8-detection-app
   ```

2. Buat virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   # atau
   venv\Scripts\activate  # Untuk Windows
   ```

3. Install dependensi:
   ```
   pip install -r requirements.txt
   ```

4. Letakkan model YOLOv8 Anda di folder `models/`:
   ```
   mkdir -p models
   # Salin model ke folder models/
   cp path/to/my_model1.pt models/
   ```

5. Jalankan aplikasi:
   ```
   python app.py
   ```

6. Buka browser dan akses `http://localhost:5000`

## Deployment di Render.com

1. Fork repository ini ke GitHub Anda.

2. Login ke [Render.com](https://render.com/) dan buat Web Service baru.

3. Pilih "Build and deploy from a Git repository".

4. Hubungkan dengan repository GitHub Anda dan gunakan pengaturan berikut:
   - **Environment**: Docker
   - **Build Command**: `docker build -t yolov8-detection-app .`
   - **Start Command**: `gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:$PORT app:app`

5. Tambahkan Environment Variable:
   - `MODEL_PATH`: `models/my_model1.pt`

6. Klik "Create Web Service" dan tunggu proses deployment selesai.

## Catatan Penting

- Upload model YOLOv8 Anda (`my_model1.pt`) ke folder `models/` sebelum deployment.
- Jika model Anda berukuran besar, pertimbangkan untuk meng-host file model di layanan penyimpanan terpisah dan mengunduhnya saat runtime.
- Perhatikan batasan resource pada paket Render.com yang Anda gunakan.

## Kontribusi

Kontribusi selalu disambut baik! Silakan buat issue atau pull request jika Anda memiliki saran atau perbaikan.

## Lisensi

[MIT License](LICENSE)

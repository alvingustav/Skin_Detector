"""
Simple script to run the application locally.
This provides a more reliable way to start the app compared to running app.py directly,
especially when encountering issues with model loading or YOLO initialization.
"""

import os
import sys
import time
import subprocess

def ensure_directories():
    """Ensure all necessary directories exist"""
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

def setup_index_html():
    """Check if index.html exists in templates directory, if not copy it from project root"""
    if not os.path.exists('templates/index.html'):
        if os.path.exists('index.html'):
            import shutil
            print("Copying index.html to templates directory...")
            shutil.copy('index.html', 'templates/index.html')
        else:
            print("Warning: index.html not found in project root directory.")

def check_model():
    """Check if model file exists, if not download a default YOLOv8 model"""
    model_path = os.environ.get('MODEL_PATH', 'my_model1.pt')
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        user_choice = input("Do you want to download a default YOLOv8n model? (y/n): ").strip().lower()
        if user_choice == 'y':
            try:
                from ultralytics import YOLO
                print("Downloading YOLOv8n model...")
                YOLO('yolov8n.pt')
                os.environ['MODEL_PATH'] = 'yolov8n.pt'
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("You can manually download a model and place it in the project directory.")
        else:
            print("You'll need to provide a valid model file to use the application.")

def main():
    """Main function to set up and run the application"""
    print("Setting up YOLOv8 Object Detection App...")
    
    # Ensure necessary directories exist
    ensure_directories()
    
    # Setup index.html
    setup_index_html()
    
    # Check model file
    check_model()
    
    # Run the Flask application
    print("\nStarting the application...")
    try:
        os.environ['FLASK_APP'] = 'app.py'
        subprocess.run([sys.executable, '-m', 'flask', 'run', '--host=0.0.0.0'], check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError running the application: {e}")
        print("\nAlternative startup method...")
        try:
            subprocess.run([sys.executable, 'app.py'], check=True)
        except Exception as e2:
            print(f"Alternative startup also failed: {e2}")

if __name__ == "__main__":
    main()

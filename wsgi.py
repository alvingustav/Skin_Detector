"""
WSGI entry point for Azure App Service
"""
from app import app, socketio

if __name__ == "__main__":
    socketio.run(app)
import cv2
from ultralytics import YOLO
import socketio
import base64
import time

# Initialize Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    print("Connected to server")

@sio.event
def connect_error(data):
    print("Connection failed:", data)

@sio.event
def disconnect():
    print("Disconnected from server")

# Replace with your ngrok URL after starting ngrok
sio.connect('https://db9d-218-102-205-108.ngrok-free.app')  # Update to ngrok URL later

# Load YOLOv11 model
model = YOLO("best.pt")

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv11 inference
    results = model(frame)
    annotated_frame = results[0].plot()

    # Encode frame as JPEG and base64
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    # Send frame to server
    sio.emit('frame', frame_base64)
    time.sleep(0.05)  # ~20 FPS

cap.release()
sio.disconnect()

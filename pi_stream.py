import cv2
import time
import numpy as np
from ultralytics import YOLO
import threading
import socketio
import base64
import logging
import queue
from collections import deque
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a queue for frames - increased size for smoother streaming
frame_queue = queue.Queue(maxsize=10)

# Initialize Socket.IO client with optimized settings
sio = socketio.Client(reconnection=True, 
                      reconnection_attempts=10,
                      reconnection_delay=1,
                      reconnection_delay_max=5)

# Flag to control threads
running = True

# Thread synchronization objects and shared frame storage
frame_lock = threading.Lock()
current_frame = None
processing_frame = None
results_lock = threading.Lock()
current_results = None

# Global model reference
model = None

# For FPS calculation
class FPSCounter:
    def __init__(self, num_frames=30):
        self.frame_times = deque(maxlen=num_frames)
        self.last_frame_time = None
    
    def update(self):
        current_time = time.time()
        if self.last_frame_time is not None:
            self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time
    
    def get_fps(self):
        if not self.frame_times:
            return 0
        return len(self.frame_times) / sum(self.frame_times)

# Rate limiter for sending frames
class RateLimiter:
    def __init__(self, max_rate=5):  # Default to 5 FPS max send rate
        self.max_rate = max_rate
        self.last_send_time = 0
        
    def can_send(self):
        current_time = time.time()
        if current_time - self.last_send_time >= 1.0 / self.max_rate:
            self.last_send_time = current_time
            return True
        return False

@sio.event
def connect():
    logger.info("Connected to server!")

@sio.event
def connect_error(data):
    logger.error(f"Connection error: {data}")

@sio.event
def disconnect():
    logger.warning("Disconnected from server")

# YOLO model initialization with advanced options
def initialize_model():
    try:
        logger.info("Loading YOLO model...")
        model = YOLO("yolo11n.pt")
        # Set inference size smaller for faster processing
        model.overrides['imgsz'] = 320
        # Enable half-precision (FP16) - huge performance boost on compatible hardware
        model.overrides['half'] = True
        # Lower confidence threshold to improve FPS
        model.overrides['conf'] = 0.35
        # Limit to only common classes for faster inference (optional)
        # model.overrides['classes'] = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
        
        logger.info("Model loaded successfully with optimized settings")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return None

# Inference thread function - separates detection from capture
def inference_thread(model):
    global processing_frame, current_results, running
    
    logger.info("Inference thread started")
    
    while running:
        # Get a frame to process
        with frame_lock:
            if processing_frame is None:
                time.sleep(0.001)  # Small sleep to prevent CPU hogging
                continue
            frame_to_process = processing_frame.copy()
            processing_frame = None
        
        try:
            # Convert BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            
            # Run inference with no verbose output
            results = model(frame_rgb, verbose=False)
            
            # Update the results
            with results_lock:
                current_results = results
                
        except Exception as e:
            logger.error(f"Inference error: {e}")
    
    logger.info("Inference thread stopped")

# Frame encoding and sending thread
def send_frames_thread():
    global running, current_results, current_frame, model
    
    fps_counter = FPSCounter()
    last_send_time = time.time()
    frames_sent = 0
    
    # Create rate limiter - set to 5 FPS to prevent overwhelming the server
    rate_limiter = RateLimiter(max_rate=4)  # Only send up to 5 frames per second
    
    logger.info("Send thread started")
    
    try:
        while running:
            # Only send frames if connected - no extra try/except here
            if not sio.connected:
                time.sleep(0.5)
                continue
            
            # Apply rate limiting - only send at the specified rate
            if not rate_limiter.can_send():
                time.sleep(0.01)  # Short sleep to prevent CPU hogging
                continue
                
            # Get current frame and results
            local_frame = None
            local_results = None
            
            with frame_lock:
                if current_frame is not None:
                    local_frame = current_frame.copy()
            
            if local_frame is None:
                time.sleep(0.001)
                continue
                
            with results_lock:
                if current_results is not None:
                    local_results = current_results
            
            # Process the frame with YOLO results
            if local_frame is not None:
                # Draw bounding boxes if results are available
                if local_results is not None and model is not None:
                    for result in local_results:
                        if hasattr(result, 'boxes') and hasattr(result.boxes, 'xyxy'):
                            boxes = result.boxes.xyxy.cpu().numpy()
                            scores = result.boxes.conf.cpu().numpy()
                            classes = result.boxes.cls.cpu().numpy()
                            
                            for box, score, cls in zip(boxes, scores, classes):
                                # Skip low confidence detections
                                if score < 0.35:
                                    continue
                                    
                                x1, y1, x2, y2 = map(int, box)
                                # Make sure coordinates are within frame bounds
                                h, w = local_frame.shape[:2]
                                x1 = max(0, min(x1, w - 1))
                                y1 = max(0, min(y1, h - 1))
                                x2 = max(0, min(x2, w - 1))
                                y2 = max(0, min(y2, h - 1))
                                
                                # Draw bounding box and label
                                cls_id = int(cls)
                                if cls_id < len(model.names):
                                    label = f"{model.names[cls_id]}"
                                    cv2.rectangle(local_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                                    cv2.putText(local_frame, label, (x1, y1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Calculate FPS
                fps_counter.update()
                current_fps = fps_counter.get_fps()
                
                # Display FPS
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(
                    local_frame, 
                    fps_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                
                # Encode frame as JPEG with lower quality for faster transmission
                _, buffer = cv2.imencode('.jpg', local_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send the frame through Socket.IO - use simple try/except
                try:
                    sio.emit('frame', frame_base64)
                    frames_sent += 1
                except Exception as e:
                    logger.warning(f"Error sending frame: {e}")
                    continue
                
                # Log stats periodically
                current_time = time.time()
                if current_time - last_send_time > 5:
                    logger.info(f"Sent {frames_sent} frames in the last 5 seconds (FPS: {current_fps:.1f})")
                    frames_sent = 0
                    last_send_time = current_time
                
    except Exception as e:
        logger.exception(f"Error in send thread: {e}")
    finally:
        logger.info("Send thread stopped")

# Capture thread function
def capture_frames_thread():
    global running, current_frame, processing_frame
    
    logger.info("Initializing camera...")
    
    # Open the video device with OpenCV - simple initialization
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Error: Could not open camera.")
        running = False
        return
    
    # Allow camera to warm up
    time.sleep(2)
    
    # Confirm actual camera settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    logger.info(f"Camera initialized with resolution: {actual_width}x{actual_height}, "
               f"FPS: {actual_fps}, Format: {fourcc_str}")
    
    # Skip frames counter for processing (only process every N frames)
    skip_frames = 2  # Process every 2nd frame - back to original value
    frame_counter = 0
    
    # For FPS measurement
    fps_counter = FPSCounter()
    last_log_time = time.time()
    frames_captured = 0
    
    try:
        while running:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame")
                time.sleep(0.01)
                continue
            
            # Update FPS counter
            fps_counter.update()
            frames_captured += 1
            
            # Update the current frame for display
            with frame_lock:
                current_frame = frame.copy()
            
            # Submit frames for processing at regular intervals
            frame_counter += 1
            if frame_counter >= skip_frames:
                frame_counter = 0
                with frame_lock:
                    # Only update processing_frame if inference thread is ready for new frame
                    if processing_frame is None:
                        processing_frame = frame.copy()
            
            # Log stats periodically
            current_time = time.time()
            if current_time - last_log_time > 5:
                current_fps = fps_counter.get_fps()
                logger.info(f"Captured {frames_captured} frames in the last 5 seconds (FPS: {current_fps:.1f})")
                frames_captured = 0
                last_log_time = current_time
                
            # Yield to other threads
            time.sleep(0.001)
            
    except Exception as e:
        logger.exception(f"Error in capture thread: {e}")
    finally:
        cap.release()
        logger.info("Capture thread stopped")

# Connection management function - with backoff strategy
def maintain_connection(url):
    global running
    
    # Exponential backoff parameters
    base_wait = 1
    max_wait = 30
    wait_time = base_wait
    
    while running:
        try:
            if not sio.connected:
                logger.info(f"Connecting to {url}...")
                sio.connect(url)
                # Reset wait time on successful connection
                wait_time = base_wait
                
            # Stay connected for some time
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            
            # Apply exponential backoff
            logger.info(f"Waiting {wait_time} seconds before reconnecting...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, max_wait)
            
            # Disconnect if still connected
            if sio.connected:
                try:
                    sio.disconnect()
                except:
                    pass
    
# Main function
def main():
    global running, model
    
    try:
        # Try to set process priority (Linux only)
        try:
            if os.name == 'posix':
                os.system(f"renice -n -20 -p {os.getpid()}")
                logger.info("Set high process priority")
        except:
            pass
        
        # Initialize YOLO model
        model = initialize_model()
        if model is None:
            logger.error("Failed to initialize YOLO model")
            return
        
        url = 'https://1987-218-102-205-108.ngrok-free.app'
        
        # Start threads
        threads = []
        
        # Create and start connection thread
        conn_thread = threading.Thread(target=maintain_connection, args=(url,))
        conn_thread.daemon = True
        conn_thread.start()
        threads.append(conn_thread)
        
        # Wait for initial connection attempt
        time.sleep(3)
        
        # Create and start inference thread
        infer_thread = threading.Thread(target=inference_thread, args=(model,))
        infer_thread.daemon = True
        infer_thread.start()
        threads.append(infer_thread)
        
        # Create and start capture thread
        capture_thread = threading.Thread(target=capture_frames_thread)
        capture_thread.daemon = True
        capture_thread.start()
        threads.append(capture_thread)
        
        # Create and start send thread
        send_thread = threading.Thread(target=send_frames_thread)
        send_thread.daemon = True
        send_thread.start()
        threads.append(send_thread)
        
        # Keep main thread alive until interrupted
        while running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    finally:
        # Clean up
        running = False
        
        # Disconnect if connected
        if hasattr(sio, 'connected') and sio.connected:
            try:
                sio.disconnect()
            except:
                pass
            
        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=2)
            
        logger.info("Program terminated")

if __name__ == "__main__":
    main()

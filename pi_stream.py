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
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a queue for frames - increased size for smoother streaming
frame_queue = queue.Queue(maxsize=15)

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

# Recording variables
recording = False
record_lock = threading.Lock()
video_writer = None
last_detection_time = 0
recording_cooldown = 5.0  # seconds to continue recording after object disappears
record_start_time = None
record_min_duration = 3.0  # minimum recording duration in seconds
recording_dir = "recordings"
upload_queue = queue.Queue()

# Google Drive upload settings
SCOPES = ['https://www.googleapis.com/auth/drive']
PARENT_FOLDER_ID = "16gNhmALfjDGkLumAcNAPzHIkvSs1OSi7"
SERVICE_ACCOUNT_FILE = 'backend/credentials.json'

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
    def __init__(self, max_rate=10):  # Default to 5 FPS max send rate
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
        
        logger.info("Model loaded successfully with optimized settings")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return None

# Make sure recording directory exists
def ensure_recording_dir():
    if not os.path.exists(recording_dir):
        os.makedirs(recording_dir)
        logger.info(f"Created recording directory: {recording_dir}")

# Initialize or update video writer
# Initialize or update video writer
def get_video_writer(frame):
    global video_writer, record_start_time
    
    if video_writer is None:
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Generate unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(recording_dir, f"detection_{timestamp}.mp4")
        
        # Use mp4v codec - more widely supported
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 20, (w, h))
        record_start_time = time.time()
        
        logger.info(f"Started recording to: {video_path}")
        return video_path
    
    return None

# Stop recording and finalize video
def stop_recording():
    global video_writer, recording, record_start_time
    
    if video_writer is not None:
        # Find the most recent recording file by sorting the files by creation time
        recording_files = [f for f in os.listdir(recording_dir) if f.startswith('detection_')]
        
        if recording_files:
            # Sort files by creation time (newest first)
            recording_files.sort(key=lambda f: os.path.getctime(os.path.join(recording_dir, f)), reverse=True)
            latest_file = recording_files[0]
            video_path = os.path.join(recording_dir, latest_file)
            
            logger.info(f"Finalizing recording: {latest_file}")
        else:
            # Fallback if no files found (shouldn't happen)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(recording_dir, f"detection_{timestamp}_final.mp4")
            logger.warning(f"No existing recording files found, using: {video_path}")
        
        # Release the video writer
        video_writer.release()
        video_writer = None
        recording = False
        
        # Calculate duration
        duration = 0
        if record_start_time is not None:
            duration = time.time() - record_start_time
            record_start_time = None
        
        logger.info(f"Stopped recording. Duration: {duration:.2f}s")
        
        # Add to upload queue
        upload_queue.put(video_path)
        
        return video_path
    
    return None
    return None

# Simple Google Drive authentication
def authenticate_drive():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return credentials
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None

# Simplified Google Drive upload function
def upload_to_drive(file_path):
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            logger.error(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
            return False
            
        credentials = authenticate_drive()
        if credentials is None:
            return False
            
        service = build('drive', 'v3', credentials=credentials)
        
        file_name = os.path.basename(file_path)
        
        file_metadata = {
            'name': file_name,
            'parents': [PARENT_FOLDER_ID]
        }
        
        # Create proper MediaFileUpload with MIME type
        media = MediaFileUpload(
            file_path,
            mimetype='video/mp4',
            resumable=True
        )
        
        # Upload the file with proper MediaFileUpload
        logger.info(f"Uploading {file_name} to Google Drive...")
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,name,mimeType'
        ).execute()
        
        logger.info(f"Successfully uploaded file: {file.get('name')} (ID: {file.get('id')}, Type: {file.get('mimeType')})")
        
        # Set permissions to make file public for easier playback
        try:
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            service.permissions().create(
                fileId=file.get('id'),
                body=permission
            ).execute()
            logger.info(f"Set public read permissions for {file.get('name')}")
        except Exception as e:
            logger.warning(f"Failed to set permissions: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error uploading to Google Drive: {e}")
        return False

# Upload thread function
def upload_thread():
    global running
    
    logger.info("Upload thread started")
    
    while running:
        try:
            # Get file from queue with timeout
            try:
                file_path = upload_queue.get(timeout=5)
            except queue.Empty:
                continue
                
            # Upload file to Google Drive
            success = upload_to_drive(file_path)
            
            # Mark task as done
            upload_queue.task_done()
            
            if success:
                logger.info(f"Successfully uploaded: {os.path.basename(file_path)}")
            else:
                logger.warning(f"Failed to upload: {os.path.basename(file_path)}")
                
        except Exception as e:
            logger.exception(f"Error in upload thread: {e}")
    
    logger.info("Upload thread stopped")

# Inference thread function - separates detection from capture
def inference_thread(model):
    global processing_frame, current_results, running, recording, last_detection_time, record_start_time
    
    logger.info("Inference thread started")
    
    # Higher confidence threshold specifically for recording decisions
    RECORD_CONFIDENCE_THRESHOLD = 0.65  # Increased from 0.45 to 0.65
    
    # Counter for consecutive frames with high confidence detections
    # This helps prevent flickering recordings due to momentary detections
    high_confidence_frames = 0
    required_consecutive_frames = 3  # Require 3 consecutive frames with high confidence
    
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
            
            # Check if any objects were detected with high confidence
            high_conf_detections = 0
            
            for result in results:
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'conf'):
                    # Count detections with confidence above the recording threshold
                    high_conf_scores = result.boxes.conf.cpu().numpy()
                    high_conf_detections += sum(score >= RECORD_CONFIDENCE_THRESHOLD for score in high_conf_scores)
            
            # If we have high confidence detections, consider it for recording
            if high_conf_detections > 0:
                high_confidence_frames += 1
                last_detection_time = time.time()
                
                # Start recording if we have enough consecutive high confidence frames
                if high_confidence_frames >= required_consecutive_frames:
                    with record_lock:
                        if not recording:
                            logger.info(f"High confidence object detected ({high_conf_detections} objects with conf >= {RECORD_CONFIDENCE_THRESHOLD})")
                            recording = True
                            # Initialize record_start_time when starting recording (important fix)
                            record_start_time = time.time()
                            ensure_recording_dir()
            else:
                # Reset the counter if no high confidence detections in this frame
                high_confidence_frames = 0
                
            # Update the results (for display - this uses the regular confidence threshold)
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
    rate_limiter = RateLimiter(max_rate=6)  # Only send up to 6 frames per second
    
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
                                    label = f"{model.names[cls_id]} {score:.2f}"
                                    cv2.rectangle(local_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                                    cv2.putText(local_frame, label, (x1, y1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Add recording indicator
                with record_lock:
                    if recording:
                        # Draw red recording circle
                        cv2.circle(local_frame, (20, 20), 10, (0, 0, 255), -1)
                        # Display recording duration
                        if record_start_time is not None:
                            duration = time.time() - record_start_time
                            cv2.putText(local_frame, f"REC {duration:.1f}s", 
                                        (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Calculate FPS
                fps_counter.update()
                current_fps = fps_counter.get_fps()
                
                # Display FPS
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(
                    local_frame, 
                    fps_text, 
                    (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                
                # Encode frame as JPEG with lower quality for faster transmission
                _, buffer = cv2.imencode('.jpg', local_frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
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

# Recording management thread
def recording_manager_thread():
    global running, recording, last_detection_time, video_writer, record_start_time
    
    logger.info("Recording manager thread started")
    
    while running:
        try:
            # Check if we're recording
            with record_lock:
                if recording:
                    current_time = time.time()
                    
                    # Safety check: ensure record_start_time is not None
                    if record_start_time is None:
                        record_start_time = current_time
                        logger.warning("record_start_time was None, initializing it now")
                    
                    time_since_detection = current_time - last_detection_time
                    recording_duration = current_time - record_start_time
                    
                    # Stop recording if cooldown has expired and minimum duration met
                    if (time_since_detection > recording_cooldown and 
                        recording_duration > record_min_duration):
                        logger.info(f"No detections for {time_since_detection:.1f}s, stopping recording")
                        stop_recording()
            
            # Sleep to prevent CPU hogging
            time.sleep(0.1)
            
        except Exception as e:
            logger.exception(f"Error in recording manager: {e}")
    
    # Make sure to stop recording when thread exits
    with record_lock:
        if recording:
            stop_recording()
    
    logger.info("Recording manager thread stopped")

# Capture thread function
def capture_frames_thread():
    global running, current_frame, processing_frame, recording, video_writer
    
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
            
            # Add frame to recording if active
            with record_lock:
                if recording and current_frame is not None:
                    # Initialize video writer if needed
                    if video_writer is None:
                        get_video_writer(current_frame)
                    
                    # Write frame to video
                    if video_writer is not None:
                        video_writer.write(current_frame)
            
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

# Load ngrok URL from backend .env file
def load_ngrok_url_from_env():
    """Load the ngrok URL from the backend .env file"""
    try:
        env_file_path = os.path.join('backend', '.env')
        
        # Check if the file exists
        if not os.path.exists(env_file_path):
            logger.warning(f"Environment file not found at {env_file_path}")
            return None
            
        # Read the .env file
        with open(env_file_path, 'r') as file:
            for line in file:
                # Look for the REACT_APP_NGROK_URL variable
                if line.startswith('REACT_APP_NGROK_URL='):
                    # Extract the URL part
                    url = line.strip().split('=', 1)[1]
                    # Remove quotes if present
                    url = url.strip('"\'')
                    if url:
                        logger.info(f"Found ngrok URL in .env file: {url}")
                        return url
                        
        logger.warning("REACT_APP_NGROK_URL not found in .env file")
        return None
        
    except Exception as e:
        logger.error(f"Error reading .env file: {e}")
        return None

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
        # Create recordings directory
        ensure_recording_dir()
        
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
        
        # Load URL from .env file
        default_url = load_ngrok_url_from_env()
        
        # Fall back to user input if .env URL not found
        if default_url:
            url = input(f"Enter server URL (or press Enter for {default_url}): ").strip()
            if not url:
                url = default_url
        else:
            url = input("Enter server URL (or press Enter for default): ").strip()
            if not url:
                url = 'https://1c45ac72026a.ngrok.app'  # Default fallback
        
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
        
        # Create and start recording manager thread
        rec_manager_thread = threading.Thread(target=recording_manager_thread)
        rec_manager_thread.daemon = True
        rec_manager_thread.start()
        threads.append(rec_manager_thread)
        
        # Create and start upload thread
        upld_thread = threading.Thread(target=upload_thread)
        upld_thread.daemon = True
        upld_thread.start()
        threads.append(upld_thread)
        
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
        
        # Stop recording if active
        with record_lock:
            if recording and video_writer is not None:
                stop_recording()
                
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

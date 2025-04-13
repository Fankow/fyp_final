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
import serial
import subprocess
from serial.tools import list_ports

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

# PTZ Control variables
ptz_enabled = False
ptz_lock = threading.Lock()
ptz_controller = None
last_ptz_command_time = 0
ptz_command_cooldown = 0.5  # seconds between PTZ commands to prevent overloading

# Add these to your other global variables at the top
automatic_mode = True  # Default to automatic mode
ptz_manual_control = None  # Who has manual PTZ control
manual_recording_control = None  # Who has manual recording control

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
    def __init__(self, max_rate=10):  # Default to 10 FPS max send rate
        self.max_rate = max_rate
        self.last_send_time = 0
        
    def can_send(self):
        current_time = time.time()
        if current_time - self.last_send_time >= 1.0 / self.max_rate:
            self.last_send_time = current_time
            return True
        return False

# Class for PTZ camera control using PelcoD protocol
class PelcoD:
    def __init__(self, address=0x01, port=None, baudrate=9600):
        self.address = address
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        
    def scan_for_ports(self):
        """Scan and return available serial ports"""
        ports = []
        try:
            available_ports = list_ports.comports()
            if available_ports:
                for port in available_ports:
                    ports.append({
                        'device': port.device,
                        'description': port.description,
                        'hwid': port.hwid
                    })
                logger.info(f"Found {len(ports)} serial port(s)")
                return ports
            else:
                logger.warning("No serial ports found")
                return []
        except Exception as e:
            logger.error(f"Error scanning for serial ports: {e}")
            return []
    
    def test_connection(self, port, baudrate=9600, timeout=0.5):
        """Test if a port can be opened and potentially has a PTZ camera"""
        try:
            # Try to open the port
            ser = serial.Serial(port, baudrate, timeout=timeout)
            
            # Send a stop command (common to most PTZ protocols)
            stop_cmd = bytearray([0xFF, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01])
            ser.write(stop_cmd)
            
            # Try to read any response (might be none)
            response = ser.read(8)
            
            # Close the port
            ser.close()
            
            return True
            
        except serial.SerialException as e:
            logger.debug(f"Port {port} test failed: {e}")
            return False
        except Exception as e:
            logger.debug(f"Error testing port {port}: {e}")
            return False
    
    def connect(self, validate=True):
        """Try to connect to the PTZ camera with validation"""
        try:
            # If no port is specified, try auto-detection
            if not self.port:
                ports = self.scan_for_ports()
                if not ports:
                    logger.warning("No serial ports found for PTZ camera")
                    return False
                
                # Try each port until we find one that works
                if validate:
                    logger.info("Testing available ports for PTZ camera...")
                    for port_info in ports:
                        port = port_info['device']
                        logger.info(f"Testing port: {port} ({port_info['description']})")
                        
                        if self.test_connection(port):
                            logger.info(f"Port {port} appears to work. Will use this port.")
                            self.port = port
                            break
                    
                    if not self.port:
                        logger.warning("No responding PTZ camera found on any port")
                        return False
                else:
                    # Just use the first port without validation
                    self.port = ports[0]['device']
                    logger.info(f"Auto-selected serial port: {self.port} (validation skipped)")
            
            # Try to open the selected port
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self.connected = True
            
            # Send a stop command to initialize the camera
            self.stop_action()
            
            logger.info(f"PTZ camera connected on {self.port} at {self.baudrate} baud")
            return True
            
        except serial.SerialException as e:
            logger.error(f"Failed to connect to PTZ camera: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Error connecting to PTZ camera: {e}")
            self.connected = False
            return False
    
    def set_address(self, address):
        """Set the camera address"""
        self.address = address
    
    def send_command(self, command):
        """Send a command to the PTZ camera using PelcoD protocol"""
        if not self.connected or not self.serial:
            logger.debug("Cannot send PTZ command - not connected")
            return False
            
        try:
            # Construct the PelcoD message format
            # [Sync byte, Address, Command1, Command2, Data1, Data2, Checksum]
            msg = [0xFF, self.address] + command + [self.calculate_checksum(command)]
            
            # Convert to bytearray and send
            msg_bytes = bytearray(msg)
            logger.debug(f"Sending PTZ command: {msg_bytes.hex()}")
            self.serial.write(msg_bytes)
            return True
            
        except Exception as e:
            logger.error(f"Error sending PTZ command: {e}")
            return False
    
    def calculate_checksum(self, command):
        """Calculate the PelcoD checksum"""
        return (self.address + sum(command)) % 256
    
    def stop_action(self):
        """Stop all PTZ movement"""
        if self.connected:
            logger.info("PTZ: Stopping all movement")
            return self.send_command([0x00, 0x00, 0x00, 0x00])
        return False
    
    def pan_left(self, speed=0xFF):
        """Pan the camera left at the specified speed"""
        if self.connected:
            logger.info(f"PTZ: Panning LEFT at speed {speed}")
            return self.send_command([0x00, 0x04, speed, 0x00])
        return False
    
    def pan_right(self, speed=0xFF):
        """Pan the camera right at the specified speed"""
        if self.connected:
            logger.info(f"PTZ: Panning RIGHT at speed {speed}")
            return self.send_command([0x00, 0x02, speed, 0x00])
        return False
    
    def tilt_up(self, speed=0xFF):
        """Tilt the camera up at the specified speed"""
        if self.connected:
            logger.info(f"PTZ: Tilting UP at speed {speed}")
            return self.send_command([0x00, 0x08, 0x00, speed])
        return False
    
    def tilt_down(self, speed=0xFF):
        """Tilt the camera down at the specified speed"""
        if self.connected:
            logger.info(f"PTZ: Tilting DOWN at speed {speed}")
            return self.send_command([0x00, 0x10, 0x00, speed])
        return False
    
    def test_ptz_functionality(self):
        """Test if the PTZ camera responds to commands"""
        if not self.connected:
            logger.warning("Cannot test PTZ functionality - not connected")
            return False
        
        try:
            logger.info("Testing PTZ functionality...")
            
            # Test sequence: pan left briefly, then stop
            success = self.pan_left(0x40)  # Half speed
            time.sleep(0.5)
            self.stop_action()
            time.sleep(0.5)
            
            # Test pan right briefly, then stop
            success = success and self.pan_right(0x40)
            time.sleep(0.5)
            self.stop_action()
            time.sleep(0.5)
            
            # Test tilt up briefly, then stop
            success = success and self.tilt_up(0x40)
            time.sleep(0.5)
            self.stop_action()
            time.sleep(0.5)
            
            # Test tilt down briefly, then stop
            success = success and self.tilt_down(0x40)
            time.sleep(0.5)
            self.stop_action()
            
            if success:
                logger.info("PTZ functionality test completed successfully")
            else:
                logger.warning("PTZ functionality test failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error during PTZ functionality test: {e}")
            return False
    
    def close(self):
        """Close the serial connection"""
        if self.connected and self.serial:
            try:
                self.stop_action()
                self.serial.close()
                logger.info("PTZ camera connection closed")
            except Exception as e:
                logger.error(f"Error closing PTZ connection: {e}")
            finally:
                self.connected = False

@sio.event
def connect():
    logger.info("Connected to server!")

@sio.event
def connect_error(data):
    logger.error(f"Connection error: {data}")

@sio.event
def disconnect():
    logger.warning("Disconnected from server")

# Add these event handlers after your existing socket.io event handlers

@sio.event
def ptz_command(data):
    """Handle PTZ command from web client"""
    global ptz_controller, ptz_enabled, ptz_manual_control
    
    if not ptz_enabled or ptz_controller is None:
        logger.warning("PTZ command received but PTZ is not enabled")
        return
    
    # Check if the command is from authorized client
    client_id = data.get('clientId')
    if ptz_manual_control and client_id != ptz_manual_control.get('clientId'):
        logger.warning(f"Unauthorized PTZ command from {client_id}")
        return
    
    direction = data.get('direction')
    logger.info(f"Received PTZ command: {direction}")
    
    with ptz_lock:
        if direction == "up":
            ptz_controller.tilt_up()
            time.sleep(0.3)
            ptz_controller.stop_action()
        elif direction == "down":
            ptz_controller.tilt_down()
            time.sleep(0.3)
            ptz_controller.stop_action()
        elif direction == "left":
            ptz_controller.pan_left()
            time.sleep(0.3)
            ptz_controller.stop_action()
        elif direction == "right":
            ptz_controller.pan_right()
            time.sleep(0.3)
            ptz_controller.stop_action() 
        else:
            logger.warning(f"Unknown PTZ command: {direction}")

            

@sio.event
def recording_command(data):
    """Handle recording command from web client"""
    global recording, manual_recording_control
    
    # Check if the command is from authorized client
    client_id = data.get('clientId')
    if manual_recording_control and client_id != manual_recording_control.get('clientId'):
        logger.warning(f"Unauthorized recording command from {client_id}")
        return
    
    action = data.get('action')
    logger.info(f"Received recording command: {action}")
    
    with record_lock:
        if action == "start" and not recording:
            # Start manual recording
            recording = True
            record_start_time = time.time()
            ensure_recording_dir()
            logger.info("Manual recording started")
        elif action == "stop" and recording:
            # Stop recording
            stop_recording()
            logger.info("Manual recording stopped")

@sio.event
def manual_mode_command(data):
    """Handle manual mode command from web client"""
    global automatic_mode, ptz_manual_control, manual_recording_control
    
    enabled = data.get('enabled', False)
    client_id = data.get('clientId')
    
    if enabled:
        logger.info(f"Manual mode enabled by client: {client_id}")
        automatic_mode = False
        ptz_manual_control = {'clientId': client_id, 'timestamp': time.time()}
        manual_recording_control = {'clientId': client_id, 'timestamp': time.time()}
    else:
        logger.info("Manual mode disabled, returning to automatic operation")
        automatic_mode = True
        ptz_manual_control = None
        manual_recording_control = None

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

# Initialize PTZ controller with validation
def initialize_ptz():
    global ptz_enabled, ptz_controller
    
    try:
        # Create PTZ controller instance
        ptz = PelcoD()
        logger.info("Starting automatic PTZ detection...")
        
        # Scan for available ports
        available_ports = ptz.scan_for_ports()
        
        # If no ports found, continue without PTZ
        if not available_ports:
            logger.info("No serial ports detected - continuing without PTZ support")
            ptz_enabled = False
            return True
        
        # Set to auto-detect mode (port = None)
        ptz.port = None
        
        ptz.baudrate = 9600
        
        # Try to connect with auto-detection
        if ptz.connect():
            logger.info("PTZ camera auto-detected and connected successfully")
            
            # Test PTZ functionality without asking
            if ptz.test_ptz_functionality():
                logger.info("PTZ functionality test successful")
                ptz_controller = ptz
                ptz_enabled = True
                return True
            else:
                logger.warning("PTZ functionality test failed - continuing without PTZ")
                ptz.close()
                ptz_enabled = False
                return True
        else:
            logger.info("PTZ auto-detection failed - continuing without PTZ")
            ptz_enabled = False
            return True
                
    except Exception as e:
        logger.error(f"Error initializing PTZ controller: {e}")
        logger.info("Continuing without PTZ support")
        ptz_enabled = False
        return True

# Function to control PTZ movement based on object position
def control_ptz_by_object_position(frame, boxes, confidence_threshold=0.65):
    global ptz_enabled, ptz_controller, last_ptz_command_time
    
    # Return if PTZ is not enabled
    if not ptz_enabled or ptz_controller is None:
        return
        
    # Get current time for command rate limiting
    current_time = time.time()
    if current_time - last_ptz_command_time < ptz_command_cooldown:
        return
        
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Define movement zones - divide frame into 3x3 grid
    left_boundary = w / 3
    right_boundary = w * 2 / 3
    top_boundary = h / 3
    bottom_boundary = h * 2 / 3
    
    # Find the highest confidence box above threshold
    best_box = None
    best_confidence = confidence_threshold
    
    for box in boxes:
        if box.conf[0] >= best_confidence and box.cls[0] == 0:  # Class 0 is person
            best_box = box
            best_confidence = box.conf[0]
    
    # If no suitable box is found, do not move
    if best_box is None:
        return
        
    # Calculate the center of the detection box
    x1, y1, x2, y2 = best_box.xyxy[0]
    object_center_x = (x1 + x2) / 2
    object_center_y = (y1 + y2) / 2
    
    # Maximum speed for all PTZ movements
    MAX_SPEED = 0xFF  # 255 (maximum speed in PelcoD protocol)
    MOVE_DURATION = 0.2  # Duration for each movement
    
    # Determine movement direction based on object position
    with ptz_lock:
        if object_center_x < left_boundary:
            # Object is in the left zone
            if object_center_y < top_boundary:
                # Top-left zone
                logger.info("PTZ tracking: Object in top-left zone")
                ptz_controller.pan_left(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()
                time.sleep(0.05)  # Short delay between movements
                ptz_controller.tilt_up(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()
            elif object_center_y > bottom_boundary:
                # Bottom-left zone
                logger.info("PTZ tracking: Object in bottom-left zone")
                ptz_controller.pan_left(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()
                time.sleep(0.05)
                ptz_controller.tilt_down(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()
            else:
                # Middle-left zone
                logger.info("PTZ tracking: Object in middle-left zone")
                ptz_controller.pan_left(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()
        elif object_center_x > right_boundary:
            # Object is in the right zone
            if object_center_y < top_boundary:
                # Top-right zone
                logger.info("PTZ tracking: Object in top-right zone")
                ptz_controller.pan_right(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()
                time.sleep(0.05)
                ptz_controller.tilt_up(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()
            elif object_center_y > bottom_boundary:
                # Bottom-right zone
                logger.info("PTZ tracking: Object in bottom-right zone")
                ptz_controller.pan_right(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()
                time.sleep(0.05)
                ptz_controller.tilt_down(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()
            else:
                # Middle-right zone
                logger.info("PTZ tracking: Object in middle-right zone")
                ptz_controller.pan_right(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()
        else:
            # Object is in the middle horizontal zone
            if object_center_y < top_boundary:
                # Top-middle zone
                logger.info("PTZ tracking: Object in top-middle zone")
                ptz_controller.tilt_up(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()
            elif object_center_y > bottom_boundary:
                # Bottom-middle zone
                logger.info("PTZ tracking: Object in bottom-middle zone")
                ptz_controller.tilt_down(MAX_SPEED)
                time.sleep(MOVE_DURATION)
                ptz_controller.stop_action()

# Make sure recording directory exists
def ensure_recording_dir():
    if not os.path.exists(recording_dir):
        os.makedirs(recording_dir)
        logger.info(f"Created recording directory: {recording_dir}")

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

# Simple Google Drive authentication
def authenticate_drive():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return credentials
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None

# Fix the convert_to_web_format function and ensure it's correctly defined outside of upload_to_drive
def convert_to_web_format(input_path):
    """Convert a video to a web-friendly format using FFmpeg, without audio processing."""
    try:
        # Check if FFmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg not found, skipping video conversion")
            return input_path
        
        # Create web-compatible output path
        output_path = os.path.splitext(input_path)[0] + "_web.mp4"
        
        # Command for web compatibility - without audio processing
        cmd = [
            'ffmpeg',
            '-i', input_path,              # Input file
            '-an',                         # No audio (skip audio processing)
            '-c:v', 'libx264',             # H.264 video codec
            '-profile:v', 'baseline',      # Baseline profile for maximum compatibility
            '-level', '3.0',               # Compatible level
            '-pix_fmt', 'yuv420p',         # Pixel format for browser compatibility
            '-crf', '23',                  # Quality (lower is better)
            '-preset', 'ultrafast',        # Encoding speed (faster for RPi)
            '-r', '30',                    # Frame rate
            '-g', '30',                    # Keyframe interval
            '-movflags', '+faststart',     # Optimize for web streaming
            '-y',                          # Overwrite output file if exists
            output_path
        ]
        
        logger.info(f"Converting video to web format: {os.path.basename(input_path)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Conversion successful: {os.path.basename(output_path)}")
            return output_path
        else:
            logger.warning(f"Conversion failed: {result.stderr.decode('utf-8')}")
            return input_path
            
    except Exception as e:
        logger.error(f"Error in video conversion: {e}")
        return input_path

# Simplified Google Drive upload function
def upload_to_drive(file_path):
    """Uploads a video file to Google Drive with web format conversion."""
    try:
        # Basic validation
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        # Authenticate
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        
        # Optional: Convert to web format first
        web_path = convert_to_web_format(file_path)
        upload_path = web_path  # Use the converted file
        
        # Create the service
        service = build('drive', 'v3', credentials=credentials)
        
        # Upload
        file_name = os.path.basename(upload_path)
        file_metadata = {
            'name': file_name,
            'parents': [PARENT_FOLDER_ID],
            'mimeType': 'video/mp4'
        }
        
        media = MediaFileUpload(upload_path, mimetype='video/mp4', resumable=True)
        
        logger.info(f"Uploading {file_name} to Google Drive...")
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,name'
        ).execute()
        
        logger.info(f"Successfully uploaded: {file.get('name')} (ID: {file.get('id')})")
        
        # Set public permissions
        permission = {'type': 'anyone', 'role': 'reader'}
        service.permissions().create(
            fileId=file.get('id'),
            body=permission
        ).execute()
        
        # Clean up temp file if needed
        if web_path != file_path and os.path.exists(web_path):
            os.remove(web_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
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
# Modify your inference_thread function to respect manual mode
def inference_thread(model):
    global processing_frame, current_results, running, recording, last_detection_time, record_start_time, automatic_mode
    
    logger.info("Inference thread started")
    
    # Higher confidence threshold specifically for recording decisions
    RECORD_CONFIDENCE_THRESHOLD = 0.5  
    
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
            
            # Only handle automatic recording and PTZ if in automatic mode
            if automatic_mode:
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
                                # Initialize record_start_time when starting recording
                                record_start_time = time.time()
                                ensure_recording_dir()
                                
                                # Let clients know recording has started
                                try:
                                    sio.emit('recording_status', {
                                        'recording': True,
                                        'manual': False
                                    })
                                except Exception as e:
                                    logger.warning(f"Error sending recording status: {e}")
                                
                        # Control PTZ if enabled
                        if ptz_enabled and ptz_controller and result.boxes:
                            # Send PTZ status before moving
                            try:
                                sio.emit('ptz_status', {
                                    'moving': True,
                                    'manual': False
                                })
                            except Exception as e:
                                logger.warning(f"Error sending PTZ status: {e}")
                                
                            control_ptz_by_object_position(frame_to_process, result.boxes, RECORD_CONFIDENCE_THRESHOLD)
                            
                            # Send PTZ status after moving
                            try:
                                sio.emit('ptz_status', {
                                    'moving': False,
                                    'manual': False
                                })
                            except Exception as e:
                                logger.warning(f"Error sending PTZ status: {e}")
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
    global running, current_results, current_frame, model, ptz_enabled
    
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
                # Draw grid lines for PTZ zones if enabled
                if ptz_enabled:
                    h, w = local_frame.shape[:2]
                    
                    # Horizontal grid lines (at 1/3 and 2/3 of height)
                    cv2.line(local_frame, (0, int(h/3)), (w, int(h/3)), (0, 0, 255), 1)
                    cv2.line(local_frame, (0, int(2*h/3)), (w, int(2*h/3)), (0, 0, 255), 1)
                    
                    # Vertical grid lines (at 1/3 and 2/3 of width)
                    cv2.line(local_frame, (int(w/3), 0), (int(w/3), h), (0, 0, 255), 1)
                    cv2.line(local_frame, (int(2*w/3), 0), (int(2*w/3), h), (0, 0, 255), 1)
                
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
                
                # Add status indicators
                # PTZ status
                ptz_status = "PTZ: Enabled" if ptz_enabled else "PTZ: Disabled"
                cv2.putText(local_frame, ptz_status, (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Recording indicator
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
    global running, recording, last_detection_time, video_writer, record_start_time, automatic_mode
    
    logger.info("Recording manager thread started")
    
    while running:
        try:
            # Only manage automatic recordings if in automatic mode
            if automatic_mode:
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
                            
                            # Let clients know recording has stopped
                            try:
                                sio.emit('recording_status', {
                                    'recording': False,
                                    'manual': False
                                })
                            except Exception as e:
                                logger.warning(f"Error sending recording status: {e}")
            
            # Sleep to prevent CPU hogging
            time.sleep(0.1)
            
        except Exception as e:
            logger.exception(f"Error in recording manager: {e}")
    
    # Make sure to stop recording when thread exits
    with record_lock:
        if recording:
            stop_recording()
            
            # Let clients know recording has stopped
            try:
                sio.emit('recording_status', {
                    'recording': False,
                    'manual': False
                })
            except Exception as e:
                logger.warning(f"Error sending recording status: {e}")
    
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
    global running, model, ptz_enabled, automatic_mode, ptz_manual_control, manual_recording_control
    
    # Initialize control variables
    automatic_mode = True
    ptz_manual_control = None
    manual_recording_control = None
    
    try:
        # Create recordings directory
        ensure_recording_dir()
        
        # Initialize YOLO model
        model = initialize_model()
        if model is None:
            logger.error("Failed to initialize YOLO model")
            return
        
        # Initialize PTZ controller with automatic detection
        if not initialize_ptz():
            logger.warning("PTZ initialization failed")
            return
        
        server_url = 'https://fyp-web.ngrok.app/'

        
        # Start threads
        threads = []
        
        conn_thread = threading.Thread(target=maintain_connection, args=(server_url,))
        conn_thread.daemon = True
        conn_thread.start()
        threads.append(conn_thread)
        
        time.sleep(3)
        
        infer_thread = threading.Thread(target=inference_thread, args=(model,))
        infer_thread.daemon = True
        infer_thread.start()
        threads.append(infer_thread)
        
        rec_manager_thread = threading.Thread(target=recording_manager_thread)
        rec_manager_thread.daemon = True
        rec_manager_thread.start()
        threads.append(rec_manager_thread)
        
        upld_thread = threading.Thread(target=upload_thread)
        upld_thread.daemon = True
        upld_thread.start()
        threads.append(upld_thread)
        
        capture_thread = threading.Thread(target=capture_frames_thread)
        capture_thread.daemon = True
        capture_thread.start()
        threads.append(capture_thread)
        
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
        
        # Close PTZ controller if enabled
        if ptz_enabled and ptz_controller:
            try:
                ptz_controller.stop_action()
                ptz_controller.close()
                logger.info("PTZ controller closed")
            except Exception as e:
                logger.error(f"Error closing PTZ controller: {e}")
                
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
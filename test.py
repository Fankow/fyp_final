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
import shutil
import uuid

# ==============================================================================
# Configuration
# ==============================================================================

# Logging Setup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- System Settings ---
MAX_FRAME_QUEUE_SIZE = 15
SERVER_URL_FALLBACK = 'http://localhost:3000' # Fallback if .env/input fails

# --- YOLO Model Settings ---
MODEL_PATH = "yolo11n.pt"
MODEL_IMG_SIZE = 320
MODEL_HALF_PRECISION = True
MODEL_CONF_THRESHOLD_DISPLAY = 0.35 # For drawing boxes on stream
MODEL_CONF_THRESHOLD_RECORD = 0.65  # For triggering recording/PTZ

# --- Recording Settings ---
RECORDING_DIR = "recordings"
TEMP_FRAMES_BASE_DIR = "temp_frames" # Base directory for temporary frames
RECORDING_COOLDOWN_SEC = 5.0  # Stop recording after X seconds of no detection
RECORD_MIN_DURATION_SEC = 3.0 # Minimum length of a recording
RECORD_CONSECUTIVE_FRAMES = 3  # Frames needed to start recording
FFMPEG_INPUT_FRAMERATE = 20    # Assumed framerate of saved JPEG frames
FFMPEG_OUTPUT_FRAMERATE = 30   # Target framerate for output video
FFMPEG_CRF = 23                # Quality (lower is better, 18-28 typical)
FFMPEG_PRESET = 'ultrafast'    # Encoding speed (faster for RPi)

# --- PTZ Settings ---
PTZ_ENABLED_BY_DEFAULT = False # Set to True to attempt PTZ init without asking
PTZ_COMMAND_COOLDOWN_SEC = 0.5 # Min time between auto PTZ commands
PTZ_MANUAL_MOVE_DURATION_SEC = 0.3 # How long to move per manual command
PTZ_AUTO_MOVE_DURATION_SEC = 0.2 # How long to move per auto PTZ command
PTZ_AUTO_MOVE_SPEED = 0x20     # Speed for automatic movements
PTZ_MANUAL_MOVE_SPEED = 0x30   # Speed for manual movements

# --- Google Drive Settings ---
SCOPES = ['https://www.googleapis.com/auth/drive']
PARENT_FOLDER_ID = "16gNhmALfjDGkLumAcNAPzHIkvSs1OSi7" # Default folder ID
SERVICE_ACCOUNT_FILE = 'backend/credentials.json'

# --- Streaming Settings ---
STREAM_MAX_FPS = 6             # Max FPS to send to clients
STREAM_JPEG_QUALITY = 30       # Quality of JPEG frames sent (0-100)

# --- Connection Settings ---
SOCKETIO_RECONNECTION_ATTEMPTS = 10
SOCKETIO_RECONNECTION_DELAY_SEC = 1
SOCKETIO_RECONNECTION_DELAY_MAX_SEC = 5
CONNECTION_CHECK_INTERVAL_SEC = 5
CONNECTION_BACKOFF_BASE_SEC = 1
CONNECTION_BACKOFF_MAX_SEC = 30

# ==============================================================================
# Global State & Synchronization
# ==============================================================================

# --- Control Flags ---
running = True # Master flag to stop all threads
automatic_mode = True # True for auto PTZ/record, False for manual

# --- Shared Resources ---
frame_queue = queue.Queue(maxsize=MAX_FRAME_QUEUE_SIZE) # Not currently used, consider removing if capture->inference is direct
upload_queue = queue.Queue()
current_frame = None # Latest raw frame from camera (for streaming)
processing_frame = None # Frame designated for inference
current_results = None # Latest YOLO results
model = None # YOLO model instance
ptz_controller = None # PelcoD instance
temp_frames_current_dir = None # Specific dir for current recording's frames

# --- State Tracking ---
recording = False
record_start_time = None
last_detection_time = time.time() # Initialize to avoid immediate stop
last_ptz_command_time = 0
ptz_enabled = False # Will be set during initialization
ptz_manual_control = None # Info of client with manual PTZ control {'clientId': str, 'timestamp': float}
manual_recording_control = None # Info of client with manual record control {'clientId': str, 'timestamp': float}

# --- Thread Synchronization Locks ---
frame_lock = threading.Lock() # Protects current_frame, processing_frame
results_lock = threading.Lock() # Protects current_results
record_lock = threading.Lock() # Protects recording, record_start_time, temp_frames_current_dir, last_detection_time
ptz_lock = threading.Lock() # Protects ptz_controller access and last_ptz_command_time

# --- Socket.IO Client ---
sio = socketio.Client(reconnection=True,
                      reconnection_attempts=SOCKETIO_RECONNECTION_ATTEMPTS,
                      reconnection_delay=SOCKETIO_RECONNECTION_DELAY_SEC,
                      reconnection_delay_max=SOCKETIO_RECONNECTION_DELAY_MAX_SEC)

# ==============================================================================
# Utility Classes
# ==============================================================================

class FPSCounter:
    """Calculates frames per second over a moving window."""
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = None

    def update(self):
        current_time = time.time()
        if self.last_frame_time is not None:
            delta = current_time - self.last_frame_time
            if delta > 0:
                self.frame_times.append(delta)
        self.last_frame_time = current_time

    def get_fps(self):
        if not self.frame_times:
            return 0.0
        return len(self.frame_times) / sum(self.frame_times)

class RateLimiter:
    """Limits the frequency of an action."""
    def __init__(self, max_rate_hz):
        if max_rate_hz <= 0:
            raise ValueError("Max rate must be positive")
        self.min_interval = 1.0 / max_rate_hz
        self.last_action_time = 0

    def check(self):
        """Returns True if the action can be performed, False otherwise."""
        return time.time() - self.last_action_time >= self.min_interval

    def update(self):
        """Updates the last action time. Call this after performing the action."""
        self.last_action_time = time.time()

    def check_and_update(self):
        """Checks if the action can be performed and updates the time if it can."""
        if self.check():
            self.update()
            return True
        return False

# ==============================================================================
# PTZ Camera Control (PelcoD)
# ==============================================================================

class PelcoD:
    """Controls a PTZ camera using the Pelco-D protocol over serial."""
    def __init__(self, address=0x01, port=None, baudrate=9600):
        self.address = address
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        logger.info(f"PelcoD initialized (Address: {address})")

    @staticmethod
    def scan_for_ports():
        """Scans and returns available serial ports."""
        ports_found = []
        try:
            for port in list_ports.comports():
                ports_found.append({
                    'device': port.device,
                    'description': port.description,
                    'hwid': port.hwid
                })
            if ports_found:
                logger.info(f"Found {len(ports_found)} serial port(s): {[p['device'] for p in ports_found]}")
            else:
                logger.warning("No serial ports detected.")
            return ports_found
        except Exception as e:
            logger.error(f"Error scanning for serial ports: {e}", exc_info=True)
            return []

    @staticmethod
    def _test_port_connection(port, baudrate, timeout=0.5):
        """Attempts to open a serial port and send a basic stop command."""
        try:
            with serial.Serial(port, baudrate, timeout=timeout) as ser:
                # Send a stop command - relatively safe
                stop_cmd = bytearray([0xFF, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01])
                ser.write(stop_cmd)
                ser.read(1) # Attempt to read, even if it times out
            logger.debug(f"Port {port} opened successfully at {baudrate} baud.")
            return True
        except serial.SerialException as e:
            logger.debug(f"Failed to open port {port} at {baudrate}: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error testing port {port}: {e}")
        return False

    def connect(self):
        """
        Connects to the PTZ camera. Attempts auto-detection if port is not set.
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        global ptz_enabled

        if self.connected:
            logger.info("PTZ already connected.")
            return True

        selected_port = self.port
        selected_baudrate = self.baudrate

        # Auto-detection if port not specified
        if not selected_port:
            available_ports = self.scan_for_ports()
            if not available_ports: return False

            logger.info("Attempting to auto-detect PTZ camera port...")
            for port_info in available_ports:
                port_device = port_info['device']
                logger.debug(f"Testing port: {port_device} at {selected_baudrate} baud...")
                if self._test_port_connection(port_device, selected_baudrate):
                    selected_port = port_device
                    logger.info(f"Auto-detected potential PTZ port: {selected_port}")
                    break
            if not selected_port:
                logger.warning("Auto-detection failed: No responding PTZ camera found.")
                return False

        # Attempt to establish the serial connection
        try:
            logger.info(f"Connecting to PTZ on {selected_port} at {selected_baudrate} baud...")
            self.serial = serial.Serial(selected_port, selected_baudrate, timeout=1)
            self.port = selected_port # Store the successfully connected port
            self.connected = True
            self.stop_action() # Send initial stop command
            logger.info(f"PTZ camera connected successfully on {self.port}.")
            ptz_enabled = True
            return True
        except serial.SerialException as e:
            logger.error(f"Serial error connecting to PTZ on {selected_port}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to PTZ: {e}", exc_info=True)

        self.connected = False
        ptz_enabled = False
        return False

    def send_command(self, command_bytes):
        """Sends a specific Pelco-D command payload."""
        if not self.connected or not self.serial:
            logger.warning("PTZ command ignored: Not connected.")
            return False
        try:
            checksum = (self.address + sum(command_bytes)) % 256
            message = bytearray([0xFF, self.address] + command_bytes + [checksum])
            logger.debug(f"Sending PTZ command: {message.hex()}")
            self.serial.write(message)
            return True
        except serial.SerialException as e:
            logger.error(f"Serial error sending PTZ command: {e}")
            self.connected = False # Assume connection lost on serial error
            ptz_enabled = False
        except Exception as e:
            logger.error(f"Unexpected error sending PTZ command: {e}", exc_info=True)
        return False

    def stop_action(self):
        """Stops all PTZ movement."""
        logger.debug("Sending PTZ STOP command.")
        return self.send_command([0x00, 0x00, 0x00, 0x00])

    def pan_left(self, speed):
        logger.debug(f"Sending PTZ PAN LEFT command (Speed: {speed:#04x}).")
        return self.send_command([0x00, 0x04, speed, 0x00])

    def pan_right(self, speed):
        logger.debug(f"Sending PTZ PAN RIGHT command (Speed: {speed:#04x}).")
        return self.send_command([0x00, 0x02, speed, 0x00])

    def tilt_up(self, speed):
        logger.debug(f"Sending PTZ TILT UP command (Speed: {speed:#04x}).")
        return self.send_command([0x00, 0x08, 0x00, speed])

    def tilt_down(self, speed):
        logger.debug(f"Sending PTZ TILT DOWN command (Speed: {speed:#04x}).")
        return self.send_command([0x00, 0x10, 0x00, speed])

    def move_and_stop(self, move_func, duration, speed):
        """Executes a move command for a duration, then stops."""
        if not self.connected: return False
        # ptz_lock should be acquired by the caller (e.g., manual or auto control function)
        if move_func(speed):
            time.sleep(duration)
            self.stop_action()
            return True
        return False

    def close(self):
        """Closes the serial connection."""
        global ptz_enabled
        if self.connected and self.serial:
            try:
                self.stop_action() # Stop movement before closing
                self.serial.close()
                logger.info(f"PTZ connection on {self.port} closed.")
            except Exception as e:
                logger.error(f"Error closing PTZ connection: {e}", exc_info=True)
            finally:
                self.connected = False
                self.serial = None
                ptz_enabled = False

# ==============================================================================
# Socket.IO Event Handlers
# ==============================================================================

@sio.event
def connect():
    logger.info(f"Successfully connected to Socket.IO server (SID: {sio.sid})")

@sio.event
def connect_error(data):
    logger.error(f"Socket.IO connection error: {data}")

@sio.event
def disconnect():
    logger.warning("Disconnected from Socket.IO server.")
    # Server should handle releasing manual control on disconnect

@sio.event
def ptz_command(data):
    """Handles PTZ movement commands from the server."""
    global ptz_controller, ptz_enabled, ptz_manual_control, automatic_mode

    if not ptz_enabled or ptz_controller is None:
        logger.warning("PTZ command received but PTZ is disabled or not initialized.")
        return

    if automatic_mode:
         logger.warning("PTZ command ignored: System is in automatic mode.")
         return

    client_id = data.get('clientId')
    if not ptz_manual_control or client_id != ptz_manual_control.get('clientId'):
        logger.warning(f"Unauthorized PTZ command from client {client_id}.")
        return

    direction = data.get('direction')
    logger.info(f"Received manual PTZ command: '{direction}' from client {client_id}")

    move_actions = {
        "up": ptz_controller.tilt_up,
        "down": ptz_controller.tilt_down,
        "left": ptz_controller.pan_left,
        "right": ptz_controller.pan_right,
    }

    move_action = move_actions.get(direction)
    if move_action:
        with ptz_lock: # Ensure exclusive access for manual move
            ptz_controller.move_and_stop(move_action, PTZ_MANUAL_MOVE_DURATION_SEC, PTZ_MANUAL_MOVE_SPEED)
            # No need to update last_ptz_command_time for manual moves
    else:
        logger.warning(f"Unknown PTZ direction received: {direction}")

@sio.event
def recording_command(data):
    """Handles manual recording start/stop commands."""
    global recording, manual_recording_control, automatic_mode

    if automatic_mode:
         logger.warning("Recording command ignored: System is in automatic mode.")
         return

    client_id = data.get('clientId')
    if not manual_recording_control or client_id != manual_recording_control.get('clientId'):
        logger.warning(f"Unauthorized recording command from client {client_id}.")
        return

    action = data.get('action')
    logger.info(f"Received manual recording command: '{action}' from client {client_id}")

    with record_lock:
        if action == "start" and not recording:
            logger.info("Manual recording started.")
            start_recording_session() # Use helper function
            # Server should notify clients based on its state or this script can emit
            # sio.emit('recording_status', {'recording': True, 'manual': True})
        elif action == "stop" and recording:
            logger.info("Manual recording stopped.")
            stop_and_process_recording() # Use helper function
            # sio.emit('recording_status', {'recording': False, 'manual': True})
        else:
             logger.warning(f"Ignoring recording command '{action}': Recording state is already {recording}")

@sio.event
def manual_mode_command(data):
    """Handles commands from the server to switch between manual/automatic modes."""
    global automatic_mode, ptz_manual_control, manual_recording_control

    enabled = data.get('enabled', False) # True if switching TO manual
    client_id = data.get('clientId') # Client who initiated the switch

    if enabled:
        if not automatic_mode:
             logger.warning(f"Received manual mode enable, but already in manual mode (Controller: {ptz_manual_control.get('clientId') if ptz_manual_control else 'None'}). Updating controller.")
        logger.info(f"Switching to MANUAL mode. Control granted to client: {client_id}")
        automatic_mode = False
        control_info = {'clientId': client_id, 'timestamp': time.time()}
        ptz_manual_control = control_info
        manual_recording_control = control_info
        # Stop any automatic PTZ movement immediately
        if ptz_enabled and ptz_controller:
            with ptz_lock:
                ptz_controller.stop_action()
    else:
        if automatic_mode:
            logger.warning("Received manual mode disable, but already in automatic mode.")
        else:
            logger.info("Switching to AUTOMATIC mode.")
            automatic_mode = True
            ptz_manual_control = None
            manual_recording_control = None
            # Stop any manual recording if active
            with record_lock:
                if recording:
                    logger.info("Stopping manual recording due to switch to automatic mode.")
                    stop_and_process_recording()

# ==============================================================================
# Core Application Logic Helpers
# ==============================================================================

def initialize_model():
    """Loads and configures the YOLO model."""
    global model
    try:
        logger.info(f"Loading YOLO model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        model.overrides['imgsz'] = MODEL_IMG_SIZE
        model.overrides['half'] = MODEL_HALF_PRECISION
        model.overrides['conf'] = MODEL_CONF_THRESHOLD_DISPLAY # Default for display
        logger.info(f"YOLO model loaded. Settings: imgsz={MODEL_IMG_SIZE}, half={MODEL_HALF_PRECISION}, conf={MODEL_CONF_THRESHOLD_DISPLAY}")
        return True
    except Exception as e:
        logger.error(f"Fatal error loading YOLO model: {e}", exc_info=True)
        return False

def _get_user_choice(prompt, options, default_choice):
    """Helper for interactive setup to get a valid choice."""
    while True:
        choice_str = input(prompt).strip()
        choice_str = choice_str if choice_str else str(default_choice)
        try:
            choice_idx = int(choice_str)
            if 0 <= choice_idx < len(options):
                return choice_idx
            else:
                print(f"Invalid selection. Please enter a number between 0 and {len(options)-1}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def initialize_ptz_interactive():
    """Guides the user through PTZ setup via console interaction."""
    global ptz_controller, ptz_enabled

    ptz_instance = PelcoD()
    available_ports = ptz_instance.scan_for_ports()

    print("\n--- PTZ Camera Setup ---")
    if not available_ports:
        print("No serial ports detected.")
        confirm = input("Continue without PTZ control? (Y/n): ").strip().lower()
        if confirm == 'n': return False # User chose not to continue
        logger.info("Continuing without PTZ support (no ports found).")
        ptz_enabled = False
        return True

    enable_ptz = input("Enable PTZ control? (y/N): ").strip().lower()
    if enable_ptz != 'y':
        logger.info("PTZ control disabled by user.")
        ptz_enabled = False
        return True

    # --- Port Selection ---
    port_options = ["Auto-detect"] + [f"{p['device']} ({p.get('description', 'N/A')})" for p in available_ports]
    print("\nSelect the PTZ serial port:")
    for i, desc in enumerate(port_options): print(f"  {i}. {desc}")
    port_idx = _get_user_choice(f"Enter number (0-{len(port_options)-1}, default: 0): ", port_options, 0)

    if port_idx == 0:
        ptz_instance.port = None # Signal auto-detect
        logger.info("PTZ port set to auto-detect.")
    else:
        ptz_instance.port = available_ports[port_idx-1]['device']
        logger.info(f"PTZ port selected: {ptz_instance.port}")

    # --- Baud Rate Selection ---
    baud_rates = [2400, 4800, 9600, 19200, 38400, 57600, 115200]
    baud_options = [f"{rate} {'(Common)' if rate == 9600 else ''}" for rate in baud_rates]
    print("\nSelect the PTZ baud rate:")
    for i, desc in enumerate(baud_options): print(f"  {i}. {desc}")
    baud_idx = _get_user_choice(f"Enter number (0-{len(baud_options)-1}, default: 2 for 9600): ", baud_options, 2)
    ptz_instance.baudrate = baud_rates[baud_idx]
    logger.info(f"PTZ baud rate selected: {ptz_instance.baudrate}")

    # --- Attempt Connection ---
    print("\nAttempting to connect to PTZ camera...")
    if ptz_instance.connect():
        print("PTZ Connection Successful!")
        ptz_controller = ptz_instance # Assign to global
        ptz_enabled = True
        return True
    else:
        print("PTZ Connection Failed.")
        confirm = input("Continue without PTZ control? (Y/n): ").strip().lower()
        if confirm == 'n': return False # User chose not to continue
        logger.info("Continuing without PTZ support (connection failed).")
        ptz_enabled = False
        return True

def control_ptz_automatically(frame, boxes):
    """Automatically moves the PTZ camera to track the highest confidence person."""
    global ptz_controller, last_ptz_command_time

    if not ptz_enabled or ptz_controller is None or not ptz_controller.connected or not automatic_mode:
        return

    with ptz_lock: # Lock to check and update command time
        current_time = time.time()
        if current_time - last_ptz_command_time < PTZ_COMMAND_COOLDOWN_SEC:
            return # Rate limit

        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2
        # Define thresholds relative to center (e.g., move if target is > 1/6th of width/height away)
        move_threshold_x = w / 6
        move_threshold_y = h / 6

        best_box = None
        highest_conf = MODEL_CONF_THRESHOLD_RECORD
        for box in boxes:
            if int(box.cls[0]) == 0 and box.conf[0] >= highest_conf: # Class 0 = person
                best_box = box
                highest_conf = box.conf[0]

        if best_box is None: return # No suitable target

        x1, y1, x2, y2 = best_box.xyxy[0]
        target_x = (x1 + x2) / 2
        target_y = (y1 + y2) / 2

        delta_x = target_x - center_x
        delta_y = target_y - center_y

        move_horizontal = None
        move_vertical = None

        if delta_x < -move_threshold_x:
            move_horizontal = ptz_controller.pan_left
            logger.debug("Auto PTZ: Target left.")
        elif delta_x > move_threshold_x:
            move_horizontal = ptz_controller.pan_right
            logger.debug("Auto PTZ: Target right.")

        if delta_y < -move_threshold_y: # Y coordinates increase downwards
            move_vertical = ptz_controller.tilt_up
            logger.debug("Auto PTZ: Target up.")
        elif delta_y > move_threshold_y:
            move_vertical = ptz_controller.tilt_down
            logger.debug("Auto PTZ: Target down.")

        moved = False
        # Execute movement (prioritize?) - let's do horizontal then vertical
        if move_horizontal:
            if ptz_controller.move_and_stop(move_horizontal, PTZ_AUTO_MOVE_DURATION_SEC, PTZ_AUTO_MOVE_SPEED):
                 moved = True
                 time.sleep(0.1) # Small delay between moves
        if move_vertical:
            if ptz_controller.move_and_stop(move_vertical, PTZ_AUTO_MOVE_DURATION_SEC, PTZ_AUTO_MOVE_SPEED):
                 moved = True

        if moved:
            last_ptz_command_time = time.time() # Update timestamp only if moved
            # Optionally notify clients about PTZ movement
            # try: sio.emit('ptz_status', {'moving': False, 'manual': False})
            # except Exception as e: logger.warning(f"Error sending PTZ status: {e}")
        else:
            logger.debug("Auto PTZ: Target near center.")


def ensure_base_directories():
    """Creates base directories for recordings and temporary frames."""
    try:
        os.makedirs(RECORDING_DIR, exist_ok=True)
        os.makedirs(TEMP_FRAMES_BASE_DIR, exist_ok=True)
        logger.info(f"Ensured base directories exist: '{RECORDING_DIR}', '{TEMP_FRAMES_BASE_DIR}'")
        return True
    except OSError as e:
        logger.error(f"Fatal error creating base directories: {e}", exc_info=True)
        return False

def start_recording_session():
    """Sets up the temporary directory and start time for a new recording. Assumes record_lock is held."""
    global record_start_time, temp_frames_current_dir, recording

    if not recording: # Should only be called when starting
        logger.warning("start_recording_session called but not in recording state.")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = uuid.uuid4().hex[:8]
    temp_dir_name = f"recording_{timestamp}_{session_id}"
    temp_frames_current_dir = os.path.join(TEMP_FRAMES_BASE_DIR, temp_dir_name)

    try:
        os.makedirs(temp_frames_current_dir, exist_ok=True)
        record_start_time = time.time()
        logger.info(f"Recording session started. Temp frames in: '{temp_frames_current_dir}'")
        return True
    except OSError as e:
        logger.error(f"Failed to create temp frame directory '{temp_frames_current_dir}': {e}", exc_info=True)
        # Failed to start session, reset state
        recording = False
        record_start_time = None
        temp_frames_current_dir = None
        return False

def save_frame_for_recording(frame):
    """Saves the current frame as a JPEG in the temporary directory if recording."""
    # No lock needed here, reads globals that are set/reset under record_lock
    if not recording or record_start_time is None or temp_frames_current_dir is None:
        return

    try:
        # Use a monotonic clock for elapsed time if available and precise timing is needed,
        # but time.time() is usually sufficient here.
        elapsed_time = time.time() - record_start_time
        # Calculate frame number based on target input framerate for FFmpeg
        frame_number = int(elapsed_time * FFMPEG_INPUT_FRAMERATE)
        frame_filename = f"frame_{frame_number:05d}.jpg"
        frame_path = os.path.join(temp_frames_current_dir, frame_filename)

        # Save the frame as JPEG (high quality for intermediate storage)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        # logger.debug(f"Saved frame {frame_number} to {frame_path}")
    except Exception as e:
        logger.error(f"Failed to save frame to '{temp_frames_current_dir}': {e}", exc_info=True)

def stop_and_process_recording():
    """Stops recording, uses FFmpeg to create MP4, queues for upload, and cleans up. Assumes record_lock is held."""
    global recording, record_start_time, temp_frames_current_dir

    if temp_frames_current_dir is None or record_start_time is None:
        logger.warning("stop_and_process_recording called with invalid state.")
        recording = False # Ensure state is reset
        return

    local_temp_dir = temp_frames_current_dir # Copy path before resetting global
    local_start_time = record_start_time

    # Reset recording state immediately
    recording = False
    record_start_time = None
    temp_frames_current_dir = None
    logger.info(f"Finalizing recording from temp dir: {local_temp_dir}")

    try:
        if not os.path.isdir(local_temp_dir):
            logger.error(f"Temp directory '{local_temp_dir}' not found.")
            return
        # Use glob for potentially simpler file finding
        frame_files = sorted(os.path.join(local_temp_dir, f) for f in os.listdir(local_temp_dir) if f.startswith('frame_') and f.endswith('.jpg'))
        if not frame_files:
            logger.warning(f"No frame files found in '{local_temp_dir}'. Aborting video creation.")
            shutil.rmtree(local_temp_dir)
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"detection_{timestamp}.mp4"
        video_path = os.path.join(RECORDING_DIR, video_filename)

        duration = time.time() - local_start_time
        logger.info(f"Creating video '{video_filename}' from {len(frame_files)} frames (Duration: {duration:.2f}s)")

        # Construct FFmpeg command for web-compatible MP4
        ffmpeg_cmd = [
            'ffmpeg',
            '-y', # Overwrite output file
            '-framerate', str(FFMPEG_INPUT_FRAMERATE), # Input framerate
            '-i', os.path.join(local_temp_dir, 'frame_%05d.jpg'), # Input pattern
            '-c:v', 'libx264',
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-pix_fmt', 'yuv420p', # Crucial for browser compatibility
            '-crf', str(FFMPEG_CRF),
            '-preset', FFMPEG_PRESET,
            '-r', str(FFMPEG_OUTPUT_FRAMERATE), # Output framerate
            '-movflags', '+faststart', # Optimize for web streaming
            video_path
        ]

        logger.debug(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            logger.info(f"Successfully created video: {video_path}")
            upload_queue.put(video_path) # Queue the final video for upload
        else:
            logger.error(f"FFmpeg conversion failed for frames in '{local_temp_dir}'. Return code: {result.returncode}")
            logger.error(f"FFmpeg stderr:\n{result.stderr}")
            logger.warning(f"Video creation failed, deleting temp frames in {local_temp_dir}")

    except Exception as e:
        logger.error(f"Error during stop_and_process_recording for '{local_temp_dir}': {e}", exc_info=True)
    finally:
        # Clean up temporary frame directory regardless of success/failure
        if local_temp_dir and os.path.exists(local_temp_dir):
            try:
                shutil.rmtree(local_temp_dir)
                logger.info(f"Cleaned up temporary frames directory: {local_temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp frames directory '{local_temp_dir}': {e}")


def authenticate_drive():
    """Authenticates with Google Drive using service account credentials."""
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        logger.error(f"Google Drive authentication failed: Service account file not found at '{SERVICE_ACCOUNT_FILE}'")
        return None
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        logger.info("Google Drive authentication successful.")
        return credentials
    except Exception as e:
        logger.error(f"Google Drive authentication error: {e}", exc_info=True)
        return None

def upload_to_drive(file_path):
    """Uploads the specified file to Google Drive."""
    logger.info(f"Attempting Google Drive upload: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        logger.error(f"Drive upload failed: File not found '{file_path}'")
        return False
    if os.path.getsize(file_path) == 0:
         logger.error(f"Drive upload failed: File '{file_path}' is empty.")
         # Optionally delete the empty file here
         # try: os.remove(file_path) except OSError: pass
         return False

    credentials = authenticate_drive()
    if not credentials: return False

    try:
        service = build('drive', 'v3', credentials=credentials)
        file_name = os.path.basename(file_path)
        file_metadata = {'name': file_name, 'parents': [PARENT_FOLDER_ID], 'mimeType': 'video/mp4'}
        media = MediaFileUpload(file_path, mimetype='video/mp4', resumable=True)

        logger.info(f"Uploading '{file_name}' to Drive folder '{PARENT_FOLDER_ID}'...")
        start_time = time.time()
        uploaded_file = service.files().create(
            body=file_metadata, media_body=media, fields='id, name'
        ).execute()
        duration = time.time() - start_time
        file_id = uploaded_file.get('id')
        logger.info(f"Successfully uploaded '{uploaded_file.get('name')}' (ID: {file_id}) in {duration:.2f}s.")

        # Set Public Permissions
        try:
            permission = {'type': 'anyone', 'role': 'reader'}
            service.permissions().create(fileId=file_id, body=permission).execute()
            logger.info(f"Set public read permissions for '{uploaded_file.get('name')}'.")
        except Exception as perm_error:
            logger.warning(f"Could not set public permissions for '{uploaded_file.get('name')}' (ID: {file_id}): {perm_error}")

        # Optional: Delete local file after successful upload
        # try:
        #     os.remove(file_path)
        #     logger.info(f"Removed local file after upload: {file_path}")
        # except OSError as cleanup_error:
        #     logger.warning(f"Failed to remove local file '{file_path}': {cleanup_error}")

        return True

    except Exception as e:
        logger.error(f"Google Drive upload error for '{file_path}': {e}", exc_info=True)
        return False

def load_ngrok_url_from_env():
    """Loads the ngrok URL from the backend/.env file."""
    env_path = os.path.join('backend', '.env')
    try:
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('REACT_APP_NGROK_URL='):
                        ngrok_url = line.split('=', 1)[1].strip('"\'')
                        if ngrok_url:
                            logger.info(f"Found ngrok URL in {env_path}: {ngrok_url}")
                            return ngrok_url
            logger.warning(f"REACT_APP_NGROK_URL not found in {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}")
    except Exception as e:
        logger.error(f"Error reading .env file '{env_path}': {e}", exc_info=True)
    return None

# ==============================================================================
# Worker Threads
# ==============================================================================

def capture_frames_thread():
    """Captures frames, updates shared state, saves frames if recording."""
    global running, current_frame, processing_frame, recording

    logger.info("Capture thread started. Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("FATAL: Could not open video capture device.")
        running = False
        return

    logger.info("Camera opened. Allowing warmup...")
    time.sleep(2.0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Camera properties: {width}x{height} @ {fps:.2f} FPS (reported)")

    fps_counter = FPSCounter()
    last_log_time = time.time()
    frames_processed_since_log = 0
    # Process every frame for inference, let inference thread handle its own rate if needed
    # FRAMES_TO_SKIP_FOR_INFERENCE = 1

    while running:
        try:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to retrieve frame. Retrying...")
                time.sleep(0.05)
                continue

            fps_counter.update()
            frames_processed_since_log += 1
            frame_copy = frame.copy() # Make one copy early

            # Update latest frame for streaming
            with frame_lock:
                current_frame = frame_copy

            # Save frame if recording (uses the same copy)
            save_frame_for_recording(frame_copy)

            # Provide frame for inference if ready
            with frame_lock:
                if processing_frame is None:
                    processing_frame = frame_copy # Pass the copy

            # Log FPS periodically
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                actual_fps = fps_counter.get_fps()
                logger.info(f"Capture FPS: {actual_fps:.1f} (Processed {frames_processed_since_log} frames in ~5s)")
                frames_processed_since_log = 0
                last_log_time = current_time

            # Let camera driver handle timing, avoid artificial sleep unless necessary

        except Exception as e:
            logger.error(f"Error in capture loop: {e}", exc_info=True)
            running = False # Stop on capture errors

    cap.release()
    logger.info("Capture thread finished.")


def inference_thread():
    """Performs YOLO detection and triggers automatic actions."""
    global model, processing_frame, current_results, running, recording, last_detection_time, automatic_mode

    if model is None:
        logger.error("Inference thread cannot start: Model not loaded.")
        return

    logger.info("Inference thread started.")
    high_confidence_frame_count = 0

    while running:
        frame_to_process = None
        with frame_lock:
            if processing_frame is not None:
                frame_to_process = processing_frame
                processing_frame = None # Mark as taken

        if frame_to_process is None:
            time.sleep(0.005) # Wait briefly
            continue

        try:
            # Run inference (YOLO handles BGR)
            results = model(frame_to_process, verbose=False)

            # Update shared results for streaming thread
            with results_lock:
                current_results = results

            # --- Automatic Mode Logic ---
            if automatic_mode:
                detected_high_conf = False
                boxes = None
                if results and results[0].boxes:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        # Check for person (class 0) above recording threshold
                        if int(boxes.cls[i]) == 0 and boxes.conf[i] >= MODEL_CONF_THRESHOLD_RECORD:
                            detected_high_conf = True
                            break

                with record_lock: # Lock needed to access last_detection_time and recording state
                    if detected_high_conf:
                        last_detection_time = time.time()
                        high_confidence_frame_count += 1

                        # Start recording if conditions met
                        if high_confidence_frame_count >= RECORD_CONSECUTIVE_FRAMES and not recording:
                            logger.info(f"High confidence detection threshold met. Starting recording.")
                            recording = True # Set flag first
                            if start_recording_session(): # Then initialize session
                                # Notify clients
                                try: sio.emit('recording_status', {'recording': True, 'manual': False})
                                except Exception as e: logger.warning(f"Error sending recording status: {e}")
                            else:
                                recording = False # Reset if init failed

                    else: # No high confidence detection this frame
                        high_confidence_frame_count = 0

                # Trigger automatic PTZ control (outside record_lock)
                if detected_high_conf and boxes:
                    control_ptz_automatically(frame_to_process, boxes)

        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)

    logger.info("Inference thread finished.")


def send_frames_thread():
    """Encodes frames with overlays and sends via Socket.IO, respecting rate limits."""
    global running, current_frame, current_results, model, ptz_enabled

    logger.info("Send frames thread started.")
    fps_counter = FPSCounter()
    rate_limiter = RateLimiter(STREAM_MAX_FPS)

    while running:
        if not sio.connected:
            time.sleep(0.5)
            continue
        if not rate_limiter.check():
            time.sleep(0.01)
            continue

        frame_copy = None
        results_copy = None
        with frame_lock:
            if current_frame is not None:
                frame_copy = current_frame.copy()
        with results_lock:
            if current_results is not None:
                results_copy = current_results # Assume lightweight enough

        if frame_copy is None:
            time.sleep(0.005)
            continue

        try:
            output_frame = frame_copy
            h, w = output_frame.shape[:2]

            # Draw PTZ grid lines if enabled
            if ptz_enabled:
                color = (0, 0, 255); thickness = 1
                cv2.line(output_frame, (0, int(h/3)), (w, int(h/3)), color, thickness)
                cv2.line(output_frame, (0, int(2*h/3)), (w, int(2*h/3)), color, thickness)
                cv2.line(output_frame, (int(w/3), 0), (int(w/3), h), color, thickness)
                cv2.line(output_frame, (int(2*w/3), 0), (int(2*w/3), h), color, thickness)

            # Draw detection boxes
            if results_copy and model and results_copy[0].boxes:
                boxes = results_copy[0].boxes
                for i in range(len(boxes)):
                    if boxes.conf[i] >= MODEL_CONF_THRESHOLD_DISPLAY:
                        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                        cls_id = int(boxes.cls[i])
                        label = f"{model.names[cls_id]} {boxes.conf[i]:.2f}"
                        color = (0, 255, 0) # Green
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(output_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw Status Indicators
            fps_counter.update()
            fps = fps_counter.get_fps()
            status_color = (0, 255, 0) # Green
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)
            ptz_status_text = f"PTZ: {'On' if ptz_enabled else 'Off'}"
            cv2.putText(output_frame, ptz_status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)
            mode_text = f"Mode: {'Auto' if automatic_mode else 'Manual'}"
            cv2.putText(output_frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)

            # Recording Indicator
            with record_lock: # Check 'recording' state safely
                if recording:
                    rec_color = (0, 0, 255) # Red
                    cv2.circle(output_frame, (w - 20, 20), 8, rec_color, -1)
                    if record_start_time:
                         duration = time.time() - record_start_time
                         cv2.putText(output_frame, f"REC {int(duration)}s", (w - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 1, cv2.LINE_AA)

            # Encode and Send
            ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
            if ret:
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                sio.emit('frame', frame_base64)
                rate_limiter.update() # Update rate limiter time only after successful send
            else:
                logger.warning("Failed to encode frame to JPEG.")

        except Exception as e:
            logger.error(f"Error in send frames loop: {e}", exc_info=True)

    logger.info("Send frames thread finished.")


def recording_manager_thread():
    """Monitors detection times and stops automatic recordings after a cooldown."""
    global running, recording, last_detection_time, automatic_mode

    logger.info("Recording manager thread started.")
    while running:
        try:
            # Only manage recordings if in automatic mode
            if automatic_mode:
                with record_lock:
                    if recording and record_start_time is not None:
                        current_time = time.time()
                        time_since_last_detection = current_time - last_detection_time
                        current_recording_duration = current_time - record_start_time

                        # Check cooldown and minimum duration
                        if (time_since_last_detection > RECORDING_COOLDOWN_SEC and
                            current_recording_duration > RECORD_MIN_DURATION_SEC):
                            logger.info(f"Recording cooldown expired. Stopping recording.")
                            stop_and_process_recording() # Handles state reset and processing
                            # Notify clients
                            try: sio.emit('recording_status', {'recording': False, 'manual': False})
                            except Exception as e: logger.warning(f"Error sending recording status: {e}")

            time.sleep(0.5) # Check periodically

        except Exception as e:
            logger.error(f"Error in recording manager: {e}", exc_info=True)
            time.sleep(1)

    # Ensure recording stops if thread exits while recording
    with record_lock:
        if recording:
            logger.info("Recording manager stopping, finalizing active recording.")
            stop_and_process_recording()
            try: sio.emit('recording_status', {'recording': False, 'manual': False})
            except: pass # Ignore errors on shutdown

    logger.info("Recording manager thread finished.")


def upload_thread():
    """Monitors the upload queue and uploads files to Google Drive."""
    global running, upload_queue

    logger.info("Upload thread started.")
    while running:
        try:
            file_to_upload = upload_queue.get(timeout=5.0)
            logger.info(f"Dequeued for upload: {os.path.basename(file_to_upload)}")
            success = upload_to_drive(file_to_upload) # Handles its own logging
            if success:
                logger.info(f"Upload successful for: {os.path.basename(file_to_upload)}")
            else:
                logger.warning(f"Upload failed for: {os.path.basename(file_to_upload)}. Check logs.")
                # Consider retry logic or moving failed files
            upload_queue.task_done()
        except queue.Empty:
            continue # Normal when queue is empty
        except Exception as e:
            logger.error(f"Critical error in upload thread: {e}", exc_info=True)
            time.sleep(5) # Avoid rapid looping on errors
    logger.info("Upload thread finished.")


def maintain_connection(target_url):
    """Manages the Socket.IO connection with exponential backoff."""
    global running

    logger.info(f"Connection manager started. Target URL: {target_url}")
    wait_time = CONNECTION_BACKOFF_BASE_SEC

    while running:
        try:
            if not sio.connected:
                logger.info(f"Attempting connection to {target_url}...")
                sio.connect(target_url, transports=['websocket']) # Prefer websocket
                wait_time = CONNECTION_BACKOFF_BASE_SEC # Reset backoff on success
                logger.info("Connection successful.")
            else:
                # Check periodically if connected
                time.sleep(CONNECTION_CHECK_INTERVAL_SEC)
        except socketio.exceptions.ConnectionError as e:
            logger.error(f"Connection failed: {e}")
            logger.info(f"Waiting {wait_time:.1f}s before retry.")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, CONNECTION_BACKOFF_MAX_SEC)
        except Exception as e:
            logger.error(f"Unexpected error in connection manager: {e}", exc_info=True)
            # Ensure disconnect on error, handling potential exceptions during disconnect
            if sio.connected:
                try:
                    sio.disconnect()
                    logger.info("Disconnected due to error in connection manager.")
                except Exception as disconnect_err:
                    logger.warning(f"Error during disconnect attempt: {disconnect_err}")
            logger.info(f"Waiting {wait_time:.1f}s before retry.")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, CONNECTION_BACKOFF_MAX_SEC)

    logger.info("Connection manager thread finished.")

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    global running, automatic_mode, ptz_manual_control, manual_recording_control

    print("--- Surveillance System Starting ---")
    logger.info("Application starting...")

    # Reset global control states
    automatic_mode = True
    ptz_manual_control = None
    manual_recording_control = None

    # --- Initial Setup ---
    if not ensure_base_directories():
        print("FATAL: Could not create necessary directories. Exiting.")
        return

    # Set high process priority (Linux/POSIX only, requires permissions)
    if os.name == 'posix':
        try:
            pid = os.getpid()
            # Use nice value -10, less aggressive than -18/-20, less likely to need root
            ret = os.system(f"renice -n -10 -p {pid}")
            if ret == 0: logger.info(f"Set process priority (renice -10) for PID {pid}")
            else: logger.warning(f"Failed to set process priority (renice returned {ret}).")
        except Exception as e: logger.warning(f"Could not set process priority: {e}")

    # Load YOLO Model
    if not initialize_model():
        print("FATAL: Failed to load YOLO model. Exiting.")
        return

    # Initialize PTZ (Interactive)
    if not initialize_ptz_interactive():
        print("Setup cancelled by user during PTZ initialization. Exiting.")
        return

    # Determine Server URL
    server_url = load_ngrok_url_from_env()
    if not server_url:
        print(f"\nCould not find ngrok URL in backend/.env.")
        server_url_input = input(f"Enter Server URL (or press Enter for default: {SERVER_URL_FALLBACK}): ").strip()
        server_url = server_url_input if server_url_input else SERVER_URL_FALLBACK
    else:
        print(f"\nUsing Server URL from .env file: {server_url}")
        confirm_url = input("Press Enter to use this URL, or enter a different one: ").strip()
        if confirm_url: server_url = confirm_url

    logger.info(f"Final Server URL: {server_url}")

    # --- Start Worker Threads ---
    threads = []
    thread_configs = [
        ("ConnectionMgr", maintain_connection, (server_url,)),
        ("Capture", capture_frames_thread, ()),
        ("Inference", inference_thread, ()),
        ("StreamSend", send_frames_thread, ()),
        ("RecordMgr", recording_manager_thread, ()),
        ("Upload", upload_thread, ()),
    ]

    print("\nStarting worker threads...")
    for name, target, args in thread_configs:
        thread = threading.Thread(target=target, args=args, name=name, daemon=True)
        thread.start()
        threads.append(thread)
        logger.info(f"Thread '{name}' started.")

    print("\nSystem is running. Press Ctrl+C to stop.")

    # --- Main Loop (Keep main thread alive, monitor worker threads) ---
    try:
        while running:
            # Check if any critical threads have died unexpectedly
            for thread in threads:
                if not thread.is_alive() and running:
                     # Exclude ConnectionMgr as it might restart internally
                     if thread.name != "ConnectionMgr":
                         logger.error(f"Thread '{thread.name}' terminated unexpectedly! Shutting down.")
                         running = False
                         break
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Ctrl+C detected. Initiating shutdown...")
    except Exception as e:
        logger.critical(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        running = False # Signal all threads to stop
        logger.info("Starting cleanup...")

        # Ensure recording stops cleanly
        with record_lock:
            if recording:
                logger.info("Main cleanup: Finalizing active recording.")
                stop_and_process_recording()

        # Close PTZ connection
        if ptz_controller:
            ptz_controller.close()

        # Disconnect Socket.IO
        if sio.connected:
            logger.info("Disconnecting from Socket.IO server...")
            try: sio.disconnect()
            except Exception as e: logger.warning(f"Error during Socket.IO disconnect: {e}")

        # Wait for threads to finish
        logger.info("Waiting for threads to complete...")
        for thread in threads:
            try:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Thread '{thread.name}' did not terminate gracefully.")
            except Exception as e:
                 logger.warning(f"Error joining thread '{thread.name}': {e}")

        # Optional: Wait for upload queue to finish (can block shutdown)
        # logger.info("Waiting for remaining uploads...")
        # upload_queue.join()

        logger.info("Application shutdown complete.")
        print("--- System Stopped ---")

if __name__ == "__main__":
    main()
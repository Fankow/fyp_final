import cv2
import time
import numpy as np
from ultralytics import YOLO
import threading
import socketio
import base64
import logging
import queue
import os
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import serial
import subprocess
from serial.tools import list_ports
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
RECORDING_DIR = "recordings"
DETECTION_PREFIX = "detection_"
CONFIDENCE_THRESHOLD = 0.35  # Regular detection threshold
RECORD_THRESHOLD = 0.5      # Higher threshold for recording decisions
STREAM_FPS = 6              # Frame rate limit for sending to clients
SERVER_URL = 'https://fyp-web.ngrok.app/'
DRIVE_FOLDER_ID = "16gNhmALfjDGkLumAcNAPzHIkvSs1OSi7"
CREDENTIALS_FILE = 'backend/credentials.json'
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']

# Global state
running = True
automatic_mode = True
ptz_enabled = False
recording = False
current_frame = None
processing_frame = None
current_results = None
video_writer = None
last_detection_time = 0
record_start_time = None
ptz_controller = None
ptz_manual_control = None
manual_recording_control = None
upload_queue = queue.Queue()
last_ptz_command_time = 0  # Add this line

# Locks for thread synchronization
frame_lock = threading.Lock()
record_lock = threading.Lock()
results_lock = threading.Lock()
ptz_lock = threading.Lock()

# Initialize Socket.IO client
sio = socketio.Client(reconnection=True, reconnection_attempts=10,
                      reconnection_delay=1, reconnection_delay_max=5)


class FPSCounter:
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.last_time = None
    
    def update(self):
        now = time.time()
        if self.last_time:
            self.frame_times.append(now - self.last_time)
        self.last_time = now
    
    def get_fps(self):
        if not self.frame_times:
            return 0
        return len(self.frame_times) / sum(self.frame_times)


class RateLimiter:
    def __init__(self, max_rate=10):
        self.max_rate = max_rate
        self.last_time = 0
        
    def can_send(self):
        now = time.time()
        if now - self.last_time >= 1.0 / self.max_rate:
            self.last_time = now
            return True
        return False


class PelcoD:
    def __init__(self, address=0x01, port=None, baudrate=9600):
        self.address = address
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        
    def scan_for_ports(self):
        ports = []
        try:
            available_ports = list_ports.comports()
            for port in available_ports:
                ports.append({
                    'device': port.device,
                    'description': port.description,
                    'hwid': port.hwid
                })
            logger.info(f"Found {len(ports)} serial port(s)")
            return ports
        except Exception as e:
            logger.error(f"Error scanning ports: {e}")
            return []
    
    def test_connection(self, port, baudrate=9600, timeout=0.5):
        try:
            ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            ser.close()
            return True
        except Exception as e:
            return False
    
    def connect(self, validate=True):
        if self.port is None:
            ports = self.scan_for_ports()
            if not ports:
                return False
                
            for port_info in ports:
                port = port_info['device']
                if self.test_connection(port, self.baudrate):
                    self.port = port
                    break
            
            if self.port is None:
                return False
        
        try:
            self.serial = serial.Serial(
                self.port,
                baudrate=self.baudrate,
                timeout=1.0
            )
            
            if self.serial.is_open:
                self.connected = True
                if validate:
                    self.stop_action()
                return True
            return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False
    
    def send_command(self, command):
        if not self.connected or not self.serial:
            return False
            
        try:
            msg = [0xFF, self.address] + command + [self.calculate_checksum(command)]
            self.serial.write(bytearray(msg))
            return True
        except Exception as e:
            logger.error(f"Command error: {e}")
            return False
    
    def calculate_checksum(self, command):
        return (self.address + sum(command)) % 256
    
    def stop_action(self):
        if self.connected:
            return self.send_command([0x00, 0x00, 0x00, 0x00])
        return False
    
    def pan_left(self, speed=0xFF):
        if self.connected:
            return self.send_command([0x00, 0x04, speed, 0x00])
        return False
    
    def pan_right(self, speed=0xFF):
        if self.connected:
            return self.send_command([0x00, 0x02, speed, 0x00])
        return False
    
    def tilt_up(self, speed=0xFF):
        if self.connected:
            return self.send_command([0x00, 0x08, 0x00, speed])
        return False
    
    def tilt_down(self, speed=0xFF):
        if self.connected:
            return self.send_command([0x00, 0x10, 0x00, speed])
        return False
    
    def test_ptz_functionality(self):
        if not self.connected:
            return False
        
        try:
            success = self.pan_left(0x40)
            time.sleep(0.5)
            self.stop_action()
            time.sleep(0.5)
            
            success = success and self.pan_right(0x40)
            time.sleep(0.5)
            self.stop_action()
            time.sleep(0.5)
            
            success = success and self.tilt_up(0x40)
            time.sleep(0.5)
            self.stop_action()
            time.sleep(0.5)
            
            success = success and self.tilt_down(0x40)
            time.sleep(0.5)
            self.stop_action()
            
            return success
        except Exception as e:
            logger.error(f"PTZ test error: {e}")
            return False
    
    def close(self):
        if self.connected and self.serial:
            try:
                self.stop_action()
                self.serial.close()
            except Exception as e:
                logger.error(f"Close error: {e}")
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

@sio.event
def ptz_command(data):
    global ptz_enabled, ptz_controller, ptz_manual_control
    
    if not ptz_enabled or not ptz_controller:
        return
    
    client_id = data.get('clientId')
    if ptz_manual_control and client_id != ptz_manual_control.get('clientId'):
        return
    
    direction = data.get('direction')
    logger.info(f"PTZ command: {direction}")
    
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

@sio.event
def recording_command(data):
    global recording, record_start_time, manual_recording_control
    
    client_id = data.get('clientId')
    if manual_recording_control and client_id != manual_recording_control.get('clientId'):
        return
    
    action = data.get('action')
    logger.info(f"Recording command: {action}")
    
    with record_lock:
        if action == "start" and not recording:
            recording = True
            record_start_time = time.time()
            ensure_recording_dir()
            logger.info("Manual recording started")
        elif action == "stop" and recording:
            stop_recording()
            logger.info("Manual recording stopped")

@sio.event
def manual_mode_command(data):
    global automatic_mode, ptz_manual_control, manual_recording_control
    
    enable = data.get('enable', False)
    client_id = data.get('clientId')
    
    if enable:
        if ptz_manual_control and ptz_manual_control.get('clientId') != client_id:
            sio.emit('manual_mode_response', {
                'success': False,
                'message': 'Another user has manual control'
            }, room=client_id)
            return
                
        automatic_mode = False
        ptz_manual_control = {'clientId': client_id, 'timestamp': time.time()}
        manual_recording_control = {'clientId': client_id, 'timestamp': time.time()}
        
        sio.emit('manual_mode_response', {
            'success': True,
            'message': 'Manual control granted'
        }, room=client_id)
        
        sio.emit('control_status_update', {
            'status': 'manual',
            'controller': client_id,
            'isYou': False
        })
        
        sio.emit('control_status_update', {
            'status': 'manual',
            'controller': client_id,
            'isYou': True
        }, room=client_id)
    else:
        if ptz_manual_control and ptz_manual_control.get('clientId') != client_id:
            return
                
        automatic_mode = True
        ptz_manual_control = None
        manual_recording_control = None
        
        sio.emit('control_status_update', {
            'status': 'automatic'
        })
        
        sio.emit('manual_mode_response', {
            'success': True,
            'message': 'Returned to automatic mode'
        }, room=client_id)


def initialize_model():
    try:
        model = YOLO("yolo11n.pt")
        model.overrides['imgsz'] = 320
        model.overrides['half'] = True
        model.overrides['conf'] = CONFIDENCE_THRESHOLD
        return model
    except Exception as e:
        logger.error(f"Model error: {e}")
        return None


def initialize_ptz():
    global ptz_enabled, ptz_controller
    
    try:
        ptz = PelcoD()
        available_ports = ptz.scan_for_ports()
        
        if not available_ports:
            ptz_enabled = False
            return True
        
        ptz.port = None
        ptz.baudrate = 9600
        
        if ptz.connect():
            if ptz.test_ptz_functionality():
                ptz_controller = ptz
                ptz_enabled = True
                return True
            else:
                ptz.close()
                ptz_enabled = False
                return True
        else:
            ptz_enabled = False
            return True
                
    except Exception as e:
        logger.error(f"PTZ init error: {e}")
        ptz_enabled = False
        return True


def control_ptz_by_object_position(frame, boxes, confidence_threshold=0.65):
    global ptz_enabled, ptz_controller, last_ptz_command_time
    
    if not ptz_enabled or ptz_controller is None:
        return
        
    current_time = time.time()
    if current_time - last_ptz_command_time < 0.5:  # Command cooldown
        return
        
    h, w = frame.shape[:2]
    left_boundary = w / 3
    right_boundary = w * 2 / 3
    top_boundary = h / 3
    bottom_boundary = h * 2 / 3
    
    best_box = None
    best_confidence = confidence_threshold
    
    for box in boxes:
        if box.conf[0] >= best_confidence and box.cls[0] == 0:  # Class 0 is person
            best_box = box
            best_confidence = box.conf[0]
    
    if best_box is None:
        return
        
    x1, y1, x2, y2 = best_box.xyxy[0]
    object_center_x = (x1 + x2) / 2
    object_center_y = (y1 + y2) / 2
    
    max_speed = 0xFF  # Maximum speed
    move_duration = 0.5  # Movement duration
    
    with ptz_lock:
        if object_center_x < left_boundary:
            if object_center_y < top_boundary:
                # Top-left
                ptz_controller.pan_left(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
                time.sleep(0.05)
                ptz_controller.tilt_up(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
            elif object_center_y > bottom_boundary:
                # Bottom-left
                ptz_controller.pan_left(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
                time.sleep(0.05)
                ptz_controller.tilt_down(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
            else:
                # Middle-left
                ptz_controller.pan_left(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
        elif object_center_x > right_boundary:
            if object_center_y < top_boundary:
                # Top-right
                ptz_controller.pan_right(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
                time.sleep(0.05)
                ptz_controller.tilt_up(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
            elif object_center_y > bottom_boundary:
                # Bottom-right
                ptz_controller.pan_right(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
                time.sleep(0.05)
                ptz_controller.tilt_down(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
            else:
                # Middle-right
                ptz_controller.pan_right(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
        else:
            if object_center_y < top_boundary:
                # Top-middle
                ptz_controller.tilt_up(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
            elif object_center_y > bottom_boundary:
                # Bottom-middle
                ptz_controller.tilt_down(max_speed)
                time.sleep(move_duration)
                ptz_controller.stop_action()
    
    last_ptz_command_time = time.time()


def ensure_recording_dir():
    if not os.path.exists(RECORDING_DIR):
        os.makedirs(RECORDING_DIR)


def get_video_writer(frame):
    global video_writer, record_start_time
    
    if video_writer is None:
        h, w = frame.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(RECORDING_DIR, f"{DETECTION_PREFIX}{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 20, (w, h))
        record_start_time = time.time()
        logger.info(f"Recording to: {video_path}")
        return video_path
    
    return None


def stop_recording():
    global video_writer, recording, record_start_time
    
    if video_writer is not None:
        recording_files = [f for f in os.listdir(RECORDING_DIR) if f.startswith(DETECTION_PREFIX)]
        
        if recording_files:
            recording_files.sort(key=lambda f: os.path.getctime(os.path.join(RECORDING_DIR, f)), reverse=True)
            latest_file = recording_files[0]
            video_path = os.path.join(RECORDING_DIR, latest_file)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(RECORDING_DIR, f"{DETECTION_PREFIX}{timestamp}_final.mp4")
        
        video_writer.release()
        video_writer = None
        recording = False
        
        duration = 0
        if record_start_time is not None:
            duration = time.time() - record_start_time
            record_start_time = None
        
        logger.info(f"Recording stopped. Duration: {duration:.2f}s")
        upload_queue.put(video_path)
        return video_path
    
    return None


def authenticate_drive():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_FILE, scopes=DRIVE_SCOPES)
        return credentials
    except Exception as e:
        logger.error(f"Auth error: {e}")
        return None


def convert_to_web_format(input_path):
    try:
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except:
            return input_path
        
        output_path = os.path.splitext(input_path)[0] + "_web.mp4"
        
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-an',
            '-c:v', 'libx264',
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'ultrafast',
            '-r', '30',
            '-g', '30',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            return input_path
            
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return input_path


def upload_to_drive(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        web_format_path = convert_to_web_format(file_path)
        
        credentials = authenticate_drive()
        if not credentials:
            logger.error("Authentication failed")
            return False
        
        service = build('drive', 'v3', credentials=credentials)
        
        file_name = os.path.basename(web_format_path)
        file_metadata = {
            'name': file_name,
            'parents': [DRIVE_FOLDER_ID]
        }
        
        media = MediaFileUpload(
            web_format_path,
            mimetype='video/mp4',
            resumable=True
        )
        
        logger.info(f"Uploading {file_name} to Drive...")
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        service.permissions().create(
            fileId=file.get('id'),
            body={'type': 'anyone', 'role': 'reader'},
            fields='id'
        ).execute()
        
        logger.info(f"Upload success: {file.get('id')}")
        
        if web_format_path != file_path and os.path.exists(web_format_path):
            try:
                os.remove(web_format_path)
            except:
                pass
        
        return True
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return False


def upload_thread():
    global running
    
    logger.info("Upload thread started")
    
    while running:
        try:
            try:
                file_path = upload_queue.get(timeout=5)
            except queue.Empty:
                continue
                
            success = upload_to_drive(file_path)
            upload_queue.task_done()
            
            if success:
                logger.info(f"Upload complete: {os.path.basename(file_path)}")
            else:
                logger.warning(f"Upload failed: {os.path.basename(file_path)}")
                
        except Exception as e:
            logger.error(f"Upload thread error: {e}")
    
    logger.info("Upload thread stopped")


def inference_thread(model):
    global processing_frame, current_results, running, recording, last_detection_time
    
    logger.info("Inference thread started")
    high_confidence_frames = 0
    required_consecutive_frames = 3
    
    while running:
        with frame_lock:
            if processing_frame is None:
                time.sleep(0.001)
                continue
            frame_to_process = processing_frame.copy()
            processing_frame = None
        
        try:
            results = model(frame_to_process)
            
            with results_lock:
                current_results = results
                
            if automatic_mode:
                high_conf_detections = 0
                
                for result in results:
                    high_conf_scores = result.boxes.conf.cpu().numpy()
                    high_conf_detections += sum(score >= RECORD_THRESHOLD for score in high_conf_scores)
                
                if high_conf_detections > 0:
                    high_confidence_frames += 1
                    last_detection_time = time.time()
                    
                    if high_confidence_frames >= required_consecutive_frames:
                        with record_lock:
                            if not recording:
                                recording = True
                                try:
                                    sio.emit('recording_status', {
                                        'recording': True,
                                        'manual': False
                                    })
                                except:
                                    pass
                                
                        if ptz_enabled and ptz_controller:
                            try:
                                sio.emit('ptz_status', {
                                    'moving': True,
                                    'manual': False
                                })
                            except:
                                pass
                                
                            control_ptz_by_object_position(frame_to_process, result.boxes, RECORD_THRESHOLD)
                            
                            try:
                                sio.emit('ptz_status', {
                                    'moving': False,
                                    'manual': False
                                })
                            except:
                                pass
                else:
                    high_confidence_frames = 0
                
        except Exception as e:
            logger.error(f"Inference error: {e}")
    
    logger.info("Inference thread stopped")


def send_frames_thread():
    global running, current_results, current_frame
    
    fps_counter = FPSCounter()
    rate_limiter = RateLimiter(max_rate=STREAM_FPS)
    
    logger.info("Send frames thread started")
    
    while running:
        if not sio.connected:
            time.sleep(0.5)
            continue
        
        if not rate_limiter.can_send():
            time.sleep(0.01)
            continue
            
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
        
        if local_frame is not None:
            if ptz_enabled:
                h, w = local_frame.shape[:2]
                cv2.line(local_frame, (0, int(h/3)), (w, int(h/3)), (0, 0, 255), 1)
                cv2.line(local_frame, (0, int(2*h/3)), (w, int(2*h/3)), (0, 0, 255), 1)
                cv2.line(local_frame, (int(w/3), 0), (int(w/3), h), (0, 0, 255), 1)
                cv2.line(local_frame, (int(2*w/3), 0), (int(2*w/3), h), (0, 0, 255), 1)
            
            if local_results is not None:
                for result in local_results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, score, cls in zip(boxes, scores, classes):
                        if score < 0.35:
                            continue
                            
                        x1, y1, x2, y2 = map(int, box)
                        h, w = local_frame.shape[:2]
                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(0, min(x2, w - 1))
                        y2 = max(0, min(y2, h - 1))
                        
                        cls_id = int(cls)
                        label = f"{result.names[cls_id]} {score:.2f}"
                        cv2.rectangle(local_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(local_frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Add PTZ status
            ptz_status = "PTZ: Enabled" if ptz_enabled else "PTZ: Disabled"
            cv2.putText(local_frame, ptz_status, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Add recording indicator
            with record_lock:
                if recording:
                    cv2.circle(local_frame, (20, 20), 10, (0, 0, 255), -1)
                    if record_start_time is not None:
                        duration = time.time() - record_start_time
                        cv2.putText(local_frame, f"REC {duration:.1f}s", 
                                    (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            # Add FPS counter
            fps_counter.update()
            current_fps = fps_counter.get_fps()
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(local_frame, fps_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Add mode indicator
            mode_text = "AUTO" if automatic_mode else "MANUAL"
            cv2.putText(local_frame, mode_text, (local_frame.shape[1] - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if automatic_mode else (0, 0, 255), 2)
            
            # Encode and send frame
            _, buffer = cv2.imencode('.jpg', local_frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            try:
                sio.emit('frame', frame_base64)
            except Exception as e:
                pass
    
    logger.info("Send frames thread stopped")


def recording_manager_thread():
    global running, recording, last_detection_time, video_writer, record_start_time, automatic_mode
    
    logger.info("Recording manager thread started")
    recording_cooldown = 5.0  # seconds
    record_min_duration = 3.0  # seconds
    
    while running:
        try:
            if automatic_mode:
                with record_lock:
                    if recording:
                        current_time = time.time()
                        
                        if record_start_time is None:
                            record_start_time = current_time
                        
                        time_since_detection = current_time - last_detection_time if last_detection_time else 0
                        recording_duration = current_time - record_start_time
                        
                        if (time_since_detection > recording_cooldown and 
                            recording_duration > record_min_duration):
                            logger.info(f"No detections for {time_since_detection:.1f}s, stopping recording")
                            stop_recording()
                            
                            try:
                                sio.emit('recording_status', {
                                    'recording': False,
                                    'manual': False
                                })
                            except:
                                pass
            
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Recording manager error: {e}")
    
    with record_lock:
        if recording:
            stop_recording()
            
            try:
                sio.emit('recording_status', {
                    'recording': False,
                    'manual': False
                })
            except:
                pass
    
    logger.info("Recording manager thread stopped")


def capture_frames_thread():
    global running, current_frame, processing_frame, recording, video_writer
    
    logger.info("Starting camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        running = False
        return
    
    logger.info("Camera initialized")
    
    skip_frames = 2
    frame_counter = 0
    
    while running:
        ret, frame = cap.read()
        
        if not ret:
            logger.warning("Failed to grab frame")
            time.sleep(0.01)
            continue
        
        with frame_lock:
            current_frame = frame.copy()
        
        frame_counter += 1
        if frame_counter >= skip_frames:
            frame_counter = 0
            with frame_lock:
                if processing_frame is None:
                    processing_frame = frame.copy()
        
        with record_lock:
            if recording and current_frame is not None:
                if video_writer is None:
                    get_video_writer(current_frame)
                if video_writer is not None:
                    video_writer.write(current_frame)
    
    if cap.isOpened():
        cap.release()
    logger.info("Capture thread stopped")


def maintain_connection(url):
    global running
    
    base_wait = 1
    max_wait = 30
    wait_time = base_wait
    
    while running:
        try:
            if not sio.connected:
                logger.info(f"Connecting to {url}...")
                sio.connect(url)
                wait_time = base_wait
                
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            
            logger.info(f"Waiting {wait_time}s before reconnecting...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, max_wait)
            
            if sio.connected:
                try:
                    sio.disconnect()
                except:
                    pass


def main():
    global running, ptz_controller, ptz_enabled
    
    os.makedirs(RECORDING_DIR, exist_ok=True)
    
    model = initialize_model()
    if model is None:
        logger.error("Failed to initialize YOLO model. Exiting.")
        return
        
    initialize_ptz()
    
    threads = []
    
    connection_thread = threading.Thread(target=maintain_connection, args=(SERVER_URL,), daemon=True)
    connection_thread.start()
    threads.append(connection_thread)
    
    # Wait for connection
    connection_timeout = 10
    connection_wait = 0
    while not sio.connected and connection_wait < connection_timeout:
        time.sleep(1)
        connection_wait += 1
    
    infer_thread = threading.Thread(target=inference_thread, args=(model,), daemon=True)
    rec_manager_thread = threading.Thread(target=recording_manager_thread, daemon=True)
    upld_thread = threading.Thread(target=upload_thread, daemon=True)
    capture_thread = threading.Thread(target=capture_frames_thread, daemon=True)
    send_thread = threading.Thread(target=send_frames_thread, daemon=True)
    
    threads.extend([infer_thread, rec_manager_thread, upld_thread, capture_thread, send_thread])
    
    for thread in threads[1:]:
        thread.start()
        
    logger.info("All threads started")
    
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        running = False
        
    for thread in threads:
        try:
            thread.join(timeout=1)
        except:
            pass
            
    if sio.connected:
        sio.disconnect()
        
    logger.info("System shutdown complete")


if __name__ == "__main__":
    main()

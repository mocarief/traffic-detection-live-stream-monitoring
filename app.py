from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import logging
import time
from threading import Lock, Thread
from queue import Queue, Empty
import subprocess
import sys
import torch
import torch.serialization

# Monkey-patch default torch.load to disable weights_only
_original_load = torch.load
def unsafe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)

torch.load = unsafe_load
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class TrafficMonitor:
    def __init__(self):
        
        self.stream_url = None
        self.crossing_line = {"start": (0, 0), "end": (0, 0)} # Default to (0,0)
        self.target_classes = [1, 2, 3]  # car, motorbike, bicycle (YOLOv8 class IDs for COCO dataset)
        self.confidence_threshold = 0.35
        self.inference_resolution = (1920, 1080)
        self.target_output_fps = 12
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda:0':
            logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        else:
            logger.warning("CUDA is not available. YOLO model will run on CPU, which may be slower.")
        logger.info(f"YOLO model will use device: {self.device}")

        
        self.model = None
        self.tracker = None
        self._initialize_model() 

        
        self.cap = None 
        self.ffmpeg_process = None 
        self.ffmpeg_stderr_thread = None
        self.frame_queue = Queue(maxsize=15)
        self.stream_active = False 
        self.stream_thread = None 
        self.stream_connection_status = "Not Started" 

        # Tracking and event variables
        self.prev_centroids = {}
        self.crossing_events = []
        self.lock = Lock() # Lock for thread-safe access to crossing_events

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

    def _initialize_model(self):
        """Initialize YOLO model and tracker with error handling"""
        try:
            logger.info("Loading YOLOv8 model...")
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
            self.model = YOLO("yolo11l.pt").to(self.device)
            
            self.tracker = sv.ByteTrack()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _stop_current_stream(self):
        """Stops any active video stream gracefully."""
        logger.info("Attempting to stop current stream...")
        self.stream_active = False 

        
        if self.stream_thread and self.stream_thread.is_alive():
            logger.info("Waiting for stream thread to terminate...")
            self.stream_thread.join(timeout=5) # Wait for the thread to finish
            if self.stream_thread.is_alive():
                logger.warning("Stream thread did not terminate gracefully within timeout.")
            self.stream_thread = None # Clear the thread handle

        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("OpenCV VideoCapture released.")
        self.cap = None

        if self.ffmpeg_process:
            logger.info("Terminating FFmpeg process...")
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5) # Wait for FFmpeg process to terminate
                if self.ffmpeg_process.poll() is None:
                    logger.warning("FFmpeg process did not terminate gracefully within timeout.")
            except subprocess.TimeoutExpired:
                logger.error("FFmpeg process did not terminate. Killing it.")
                self.ffmpeg_process.kill()
            self.ffmpeg_process = None
            logger.info("FFmpeg process terminated.")
        
        if self.ffmpeg_stderr_thread and self.ffmpeg_stderr_thread.is_alive():
            logger.debug("FFmpeg STDERR thread is still alive, will terminate with main process.")

        
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        
        self.stream_connection_status = "Stopped"
        self.frame_count = 0
        self.start_time = time.time() 
        logger.info("Current stream stopped.")

    def configure_and_start_stream(self, stream_url, line_coords):
        
        self._stop_current_stream() 

        self.stream_url = stream_url
        self.crossing_line = line_coords
        self.prev_centroids = {} 
        self.crossing_events = [] 
        self.stream_connection_status = "Connecting..."

        # Start the stream processing in a new thread
        self.stream_active = True
        self.stream_thread = Thread(target=self._run_stream_generator, daemon=True)
        self.stream_thread.start()
        logger.info(f"Stream configuration updated and new stream initiated for URL: {stream_url}")
        return True

    def _run_stream_generator(self):
        """Helper to run the generator within a thread."""
        try:
            for _ in self.generate_frames():
                if not self.stream_active:
                    break
        except Exception as e:
            logger.critical(f"Error in stream generator thread: {e}")
        finally:
            
            if self.stream_active: 
                self._stop_current_stream()
            logger.info("Stream generator thread finished.")


    def _connect_to_stream(self, max_retries=5):
        """Connect to CCTV stream with retry logic and multiple fallback methods using OpenCV"""
        connection_methods = [
            self._try_opencv_with_options,
            #self._try_opencv_basic
        ]
        for method_idx, method in enumerate(connection_methods):
            for attempt in range(max_retries):
                try:
                    logger.info(f"Method {method_idx + 1}: Connecting to HLS stream (attempt {attempt + 1}/{max_retries})...")
                    cap = method()
                    if cap and cap.isOpened():
                        for test_attempt in range(5):
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                logger.info(f"Successfully connected using method {method_idx + 1} (frame size: {frame.shape})")
                                return cap
                            time.sleep(0.5)
                        logger.warning(f"Method {method_idx + 1}: Stream opened but cannot read frames reliably. Releasing and retrying.")
                        cap.release()
                    elif cap:
                        logger.warning(f"Method {method_idx + 1}: VideoCapture object created but not opened. Releasing and retrying.")
                        cap.release()
                    wait_time = min(2 ** attempt, 5)
                    time.sleep(wait_time)
                except Exception as e:
                    logger.error(f"Method {method_idx + 1}, attempt {attempt + 1} error: {e}")
                    time.sleep(1)
        raise ConnectionError("All OpenCV connection methods failed for HLS CCTV stream")

    def _try_opencv_basic(self):
        cap = cv2.VideoCapture(self.stream_url)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _try_opencv_with_options(self):
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        except Exception as e:
            logger.debug(f"Could not set CAP_PROP_FOURCC: {e}")
            pass
        if cap.open(self.stream_url):
            return cap
        return None

    def _read_ffmpeg_stderr(self):
        if self.ffmpeg_process and self.ffmpeg_process.stderr:
            for line in iter(self.ffmpeg_process.stderr.readline, b''):
                logger.error(f"FFmpeg STDERR: {line.decode('utf-8').strip()}")
        logger.info("FFmpeg STDERR reading thread terminated.")

    def _start_ffmpeg_stream(self):
        try:
            ffmpeg_cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',         
                '-c:v', 'h264_cuvid',        
                '-probesize', '32M',         
                '-analyzeduration', '15M',   
                '-fflags', 'nobuffer',       
                '-flags', 'low_delay',       
                '-i', self.stream_url,
                '-f', 'rawvideo',            
                '-pix_fmt', 'bgr24',         
                '-an',                       
                '-sn',                       
                '-'                          
            ]
            logger.info(f"Starting FFmpeg process with command: {' '.join(ffmpeg_cmd)}")
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            self.ffmpeg_stderr_thread = Thread(target=self._read_ffmpeg_stderr, daemon=True)
            self.ffmpeg_stderr_thread.start()
            time.sleep(2)
            if self.ffmpeg_process.poll() is not None:
                logger.error(f"FFmpeg process terminated immediately with exit code {self.ffmpeg_process.returncode}.")
                remaining_stderr = self.ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                if remaining_stderr:
                    logger.error(f"Remaining FFmpeg STDERR: {remaining_stderr.strip()}")
                self.ffmpeg_process = None
                return False
            logger.info("FFmpeg process started successfully and is running.")
            return True
        except FileNotFoundError:
            logger.error("FFmpeg executable not found. Please ensure FFmpeg is installed and in your system's PATH.")
            return False
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            return False

    def _read_ffmpeg_frames(self):
        if not self.ffmpeg_process:
            logger.error("FFmpeg process not running, cannot read frames.")
            return
        try:
            # Frame dimensions are now fixed by the FFmpeg scale filter
            frame_width = 1920
            frame_height = 1080
            frame_size_bytes = frame_width * frame_height * 3 # 3 channels for BGR
            
            while self.ffmpeg_process and self.ffmpeg_process.poll() is None and self.stream_active:
                raw_frame = self.ffmpeg_process.stdout.read(frame_size_bytes)
                
                if not raw_frame:
                    logger.warning("FFmpeg stdout stream ended or returned empty data. Breaking frame read loop.")
                    break
                
                if len(raw_frame) != frame_size_bytes:
                    logger.warning(f"Incomplete frame read from FFmpeg: expected {frame_size_bytes} bytes, got {len(raw_frame)}. Skipping frame.")
                    self.ffmpeg_process.stdout.flush() 
                    continue
                    
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((frame_height, frame_width, 3))
                
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except Exception:
                    pass 
                    
        except Exception as e:
            logger.error(f"FFmpeg frame reading error: {e}")
        finally:
            logger.info("FFmpeg frame reading thread terminated.")
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process = None

    def _connect_with_ffmpeg_fallback(self):
        if self._start_ffmpeg_stream():
            ffmpeg_frame_read_thread = Thread(target=self._read_ffmpeg_frames, daemon=True)
            ffmpeg_frame_read_thread.start()
            logger.info("Using FFmpeg as primary for stream connection")
            return None, True
        logger.warning("FFmpeg connection failed, attempting OpenCV fallback...")
        try:
            cap = self._connect_to_stream()
            if cap:
                logger.info("Using OpenCV as fallback for stream connection")
                return cap, False
        except ConnectionError:
            logger.error("OpenCV fallback also failed.")
            pass
        raise ConnectionError("Both FFmpeg and OpenCV methods failed to connect to the stream.")

    def _is_line_crossing(self, prev_point, curr_point, line_start, line_end):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(prev_point, line_start, line_end) != ccw(curr_point, line_start, line_end) and \
               ccw(prev_point, curr_point, line_start) != ccw(prev_point, curr_point, line_end)

    def _get_vehicle_type(self, class_id):
        vehicle_types = {
            1: "bicycle",
            2: "car",
            3: "motorbike",
            5: "bus",
            7: "truck"
        }
        return vehicle_types.get(class_id, "unknown")

    def _process_detections(self, results):
        if not results.boxes or len(results.boxes) == 0:
            return sv.Detections.empty()
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )
        mask = (
            np.isin(detections.class_id, self.target_classes) &
            (detections.confidence >= self.confidence_threshold)
        )
        return detections[mask]

    def _update_tracking(self, detections):
        tracks = self.tracker.update_with_detections(detections)
        crossings = []
        for i in range(len(tracks)):
            track_id = int(tracks.tracker_id[i])
            x1, y1, x2, y2 = map(int, tracks.xyxy[i])
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if track_id in self.prev_centroids:
                if self._is_line_crossing(
                    self.prev_centroids[track_id],
                    centroid,
                    self.crossing_line["start"],
                    self.crossing_line["end"]
                ):
                    crossing_event = {
                        "track_id": track_id,
                        "timestamp": time.time(),
                        "position": centroid,
                        "vehicle_type": self._get_vehicle_type(tracks.class_id[i])
                    }
                    crossings.append(crossing_event)
                    logger.info(f"Vehicle ID {track_id} ({crossing_event['vehicle_type']}) crossed the line at {centroid}")
            self.prev_centroids[track_id] = centroid
        current_track_ids = {int(t_id) for t_id in tracks.tracker_id} if tracks.tracker_id.size > 0 else set()
        self.prev_centroids = {k: v for k, v in self.prev_centroids.items() if k in current_track_ids}
        return tracks, crossings

    def _annotate_frame(self, frame, tracks):
        annotated_frame = frame.copy()
        cv2.line(
            annotated_frame,
            self.crossing_line["start"],
            self.crossing_line["end"],
            (0, 0, 255), 3
        )
        cv2.putText(
            annotated_frame, "Detection Line",
            (self.crossing_line["start"][0], self.crossing_line["start"][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        for i in range(len(tracks)):
            track_id = int(tracks.tracker_id[i])
            x1, y1, x2, y2 = map(int, tracks.xyxy[i])
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            vehicle_type = self._get_vehicle_type(tracks.class_id[i]) if tracks.class_id.size > i else "N/A"
            label = f"ID: {track_id} ({vehicle_type})"
            cv2.putText(
                annotated_frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.circle(annotated_frame, centroid, 5, (255, 0, 0), -1)
        fps = self.frame_count / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0
        cv2.putText(
            annotated_frame, f"FPS: {fps:.1f} | Objects: {len(tracks)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        return annotated_frame

    def generate_frames(self):
        """
        Main frame processing generator.
        Reads frames, performs detection and tracking, annotates, and yields JPEG bytes.
        Includes robust error handling and reconnection logic.
        """
        if self.stream_url is None:
            logger.warning("Stream URL not configured. Yielding blank frame.")
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Configure Stream and Press Start", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return

        try:
            self.cap, use_ffmpeg = self._connect_with_ffmpeg_fallback()
            self.stream_connection_status = "Connected"
        except ConnectionError as e:
            logger.error(f"Initial stream connection failed: {e}. Cannot start video feed.")
            self.stream_connection_status = f"Error: {e}"
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Stream Error: {e}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return

        consecutive_failures = 0
        max_consecutive_failures = 10

        try:
            while self.stream_active: # Loop controlled by self.stream_active flag
                frame_start_time = time.time()
                frame = None
                if use_ffmpeg:
                    try:
                        frame = self.frame_queue.get(timeout=1.0) 
                    except Empty:
                        consecutive_failures += 1
                        logger.warning(f"FFmpeg frame queue empty (failure #{consecutive_failures})")
                else:
                    success, frame = self.cap.read()
                    if not success or frame is None:
                        consecutive_failures += 1
                        logger.warning(f"OpenCV frame read failed (failure #{consecutive_failures})")

                if frame is None:
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive frame failures, attempting to reconnect...")
                        self._stop_current_stream() # Full stop and restart attempt
                        try:
                            # Re-initialize stream_active to True for reconnection attempt
                            self.stream_active = True 
                            self.cap, use_ffmpeg = self._connect_with_ffmpeg_fallback()
                            consecutive_failures = 0
                            self.stream_connection_status = "Reconnected"
                            logger.info("Reconnection successful.")
                            continue
                        except ConnectionError:
                            logger.error("Reconnection failed, stopping stream generation.")
                            self.stream_connection_status = "Reconnection Failed"
                            break
                    else:
                        time.sleep(0.1) 
                        continue # Skip processing for this iteration

                consecutive_failures = 0
                self.frame_count += 1

                try:
                    original_frame = frame.copy()
                    original_height, original_width = original_frame.shape[:2]

                    if original_width > self.inference_resolution[0] or original_height > self.inference_resolution[1]: 
                        scale = min(self.inference_resolution[0] / original_width, self.inference_resolution[1] / original_height) # Maintain aspect ratio
                        inference_width = int(original_width * scale)
                        inference_height = int(original_height * scale)
                        frame_for_inference = cv2.resize(frame, (inference_width, inference_height), interpolation=cv2.INTER_AREA)
                    else:
                        frame_for_inference = frame # Use original frame if already small enough
                    
                    results = self.model(frame_for_inference, verbose=False, device=self.device)[0]
                    detections = self._process_detections(results)

                    if frame_for_inference.shape[0] != original_height or frame_for_inference.shape[1] != original_width:
                        scale_x = original_width / frame_for_inference.shape[1]
                        scale_y = original_height / frame_for_inference.shape[0]
                        detections.xyxy[:, [0, 2]] *= scale_x
                        detections.xyxy[:, [1, 3]] *= scale_y

                    tracks, crossings = self._update_tracking(detections)

                    with self.lock:
                        self.crossing_events.extend(crossings)
                        if len(self.crossing_events) > 100:
                            self.crossing_events = self.crossing_events[-100:]

                    annotated_frame = self._annotate_frame(original_frame, tracks)
                    
                    if annotated_frame is None or annotated_frame.size == 0:
                        logger.error("Annotated frame is empty or None before JPEG encoding. Skipping and showing error frame.")
                        blank_frame_on_error = np.zeros((original_height, original_width, 3), dtype=np.uint8)
                        cv2.putText(blank_frame_on_error, "Frame Encoding Failed", (50, original_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        ret, error_buffer = cv2.imencode('.jpg', blank_frame_on_error, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ret:
                            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + error_buffer.tobytes() + b'\r\n')
                        continue

                    ret, buffer = cv2.imencode('.jpg', annotated_frame, 
                                                [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if not ret:
                        logger.warning("Failed to encode frame to JPEG. Skipping and showing error frame.")
                        blank_frame_on_error = np.zeros((original_height, original_width, 3), dtype=np.uint8)
                        cv2.putText(blank_frame_on_error, "Frame Encoding Failed", (50, original_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        ret, error_buffer = cv2.imencode('.jpg', blank_frame_on_error, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ret:
                            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + error_buffer.tobytes() + b'\r\n')
                        continue

                    # --- Frame Rate Limiting ---
                    processing_time = time.time() - frame_start_time
                    required_delay = 1.0 / self.target_output_fps
                    if processing_time < required_delay:
                        time.sleep(required_delay - processing_time)
                    # ---------------------------
                    
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    error_frame_proc = np.zeros((original_height if 'original_height' in locals() else 480, original_width if 'original_width' in locals() else 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame_proc, f"Processing Error: {str(e)[:50]}...", (50, error_frame_proc.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    ret, error_buffer_proc = cv2.imencode('.jpg', error_frame_proc, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + error_buffer_proc.tobytes() + b'\r\n')
                    continue 
                
            
                    
        except Exception as e:
            logger.critical(f"Critical stream processing error: {e}. Stopping generator.")
            self.stream_connection_status = f"Critical Error: {e}"
        finally:
            
            logger.info("Stream connection closed and resources cleaned up (via _run_stream_generator).")

    def get_crossing_stats(self):
        with self.lock:
            return {
                "total_crossings": len(self.crossing_events),
                "recent_crossings": self.crossing_events[-10:] if self.crossing_events else [],
                "stream_status": self.stream_connection_status
            }

# Initialize traffic monitor globally
traffic_monitor = TrafficMonitor()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/configure_monitoring', methods=['POST'])
def configure_monitoring():
    data = request.json
    stream_url = data.get('streamUrl')
    line_coords = data.get('crossingLine')

    if not stream_url or not line_coords:
        return jsonify({"status": "error", "message": "Missing stream URL or crossing line coordinates"}), 400

    try:
        start_point = tuple(line_coords['start'])
        end_point = tuple(line_coords['end'])
        
        traffic_monitor.configure_and_start_stream(stream_url, {"start": start_point, "end": end_point})
        return jsonify({"status": "success", "message": "Monitoring started successfully!"})
    except Exception as e:
        logger.error(f"Failed to configure and start monitoring: {e}")
        return jsonify({"status": "error", "message": f"Failed to start monitoring: {e}"}), 500

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """API endpoint to stop the monitoring stream."""
    try:
        traffic_monitor._stop_current_stream()
        return jsonify({"status": "success", "message": "Monitoring stopped successfully."})
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        return jsonify({"status": "error", "message": f"Failed to stop monitoring: {e}"}), 500

@app.route('/video_feed')
def video_feed():
    return Response(
        traffic_monitor.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stats')
def stats():
    return traffic_monitor.get_crossing_stats()

if __name__ == "__main__":
    logger.info("Starting CCTV Traffic Monitor Flask application...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


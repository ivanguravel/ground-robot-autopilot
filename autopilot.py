"""
Main autopilot script.

Processes video/camera, detects objects via RetinaNet,
sends to LLM and generates control commands.

Usage:
    python autopilot.py --input video.mp4
    python autopilot.py --camera 0
"""

import os
import cv2
import json
import time
import torch
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from model import create_model
from config import NUM_CLASSES, DEVICE, CLASSES, RESIZE_TO, INFERENCE_RESIZE
from llm_controller import LLMController, DetectedObject


class Autopilot:
    """
    Autopilot system: detection + LLM control.
    """
    
    def __init__(
        self,
        model_path: str = "outputs/best_model.pth",
        detection_threshold: float = 0.5,
        llm_interval: float = 2.0,
        use_llm: bool = True,
        debug_llm: bool = False,
        llm_provider: str = None,
        onnx_path: str = None,
    ):
        self.debug_llm = debug_llm
        print(f"Initializing autopilot on device: {DEVICE}")
        
        self.use_onnx = False
        self.use_half = False
        self.onnx_session = None
        self.model = None
        self._onnx_input_name = None
        
        self._init_onnx(onnx_path)
        
        if not self.use_onnx:
            if os.path.exists(model_path):
                print(f"Loading fine-tuned model: {model_path}")
                self.model = create_model(num_classes=NUM_CLASSES)
                checkpoint = torch.load(model_path, map_location=DEVICE)
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                print(f"Model {model_path} not found")
                print("Using pretrained COCO model (91 classes)")
                import torchvision
                from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
                self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                    weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
                )
            
            self.model.to(DEVICE).eval()
            
            if str(DEVICE) != "cpu":
                try:
                    self.model.half()
                    dummy = torch.randn(1, 3, INFERENCE_RESIZE, INFERENCE_RESIZE,
                                        dtype=torch.float16, device=DEVICE)
                    with torch.no_grad():
                        self.model(dummy)
                    self.use_half = True
                    print(f"Float16 enabled on {DEVICE}")
                except Exception as e:
                    self.model.float()
                    self.use_half = False
                    print(f"Float16 not available ({e}), using float32")
        
        self.detection_threshold = detection_threshold
        self.llm_interval = llm_interval
        self.use_llm = use_llm
        self.last_llm_call = 0
        
        # Initialize LLM controller
        if use_llm:
            try:
                self.llm_controller = LLMController(debug=self.debug_llm, provider=llm_provider)
                print("LLM controller initialized")
            except Exception as e:
                print(f"LLM initialization error: {e}")
                self.use_llm = False
                self.llm_controller = None
        else:
            self.llm_controller = None
        
        # Colors for visualization
        self.colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        
        # Last command (for display between LLM calls)
        self.last_command = None
        
        # Async LLM: fire-and-forget calls so inference isn't blocked
        self._llm_executor = ThreadPoolExecutor(max_workers=1)
        self._llm_future = None
        
        # Optical flow for movement vector
        self.prev_gray = None
        self.movement_vector = (0, 0)  # (dx, dy) average movement
        self.movement_angle = 0  # degrees, 0 = forward, positive = right
        self.movement_speed = 0  # magnitude
        
        # Logging
        self.log_dir = "autopilot_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir,
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
    
    def _init_onnx(self, onnx_path: str = None):
        """Try to load ONNX model for faster inference."""
        search_paths = [p for p in [onnx_path, "retinanet.onnx", "outputs/retinanet.onnx"] if p]
        for path in search_paths:
            if not os.path.exists(path):
                continue
            try:
                import onnxruntime as ort
                providers = self._get_onnx_providers()
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.onnx_session = ort.InferenceSession(path, sess_options, providers=providers)
                self._onnx_input_name = self.onnx_session.get_inputs()[0].name
                
                # Warmup at INFERENCE_RESIZE (RetinaNet is fully conv, any size works)
                dummy = np.random.randn(1, 3, INFERENCE_RESIZE, INFERENCE_RESIZE).astype(np.float32)
                self.onnx_session.run(None, {self._onnx_input_name: dummy})
                
                self.use_onnx = True
                actual_provider = self.onnx_session.get_providers()[0]
                print(f"ONNX Runtime: {path} | {actual_provider} | input={INFERENCE_RESIZE}px")
                return
            except ImportError:
                print("onnxruntime not installed, falling back to PyTorch")
                return
            except Exception as e:
                print(f"ONNX failed for {path}: {e}")
                self.onnx_session = None
                continue
    
    def _get_onnx_providers(self) -> list:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if str(DEVICE) == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # CoreML EP has compatibility issues with RetinaNet ops, skip it
        return ["CPUExecutionProvider"]
    
    def compute_movement_vector(self, frame: np.ndarray):
        """
        Computes movement vector using optical flow.
        Updates self.movement_vector, movement_angle, movement_speed.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))  # Downscale for speed
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Get average flow in the lower-center region (road area)
        h, w = flow.shape[:2]
        roi = flow[h//2:, w//4:3*w//4]  # Bottom half, center region
        
        avg_dx = np.mean(roi[:, :, 0])
        avg_dy = np.mean(roi[:, :, 1])
        
        self.movement_vector = (avg_dx, avg_dy)
        self.movement_speed = np.sqrt(avg_dx**2 + avg_dy**2)
        
        # Angle: 0 = forward (up), positive = right, negative = left
        # Note: optical flow shows scene movement, which is OPPOSITE to car direction
        # When car turns RIGHT, scene moves LEFT (negative avg_dx)
        # So we NEGATE avg_dx to get correct car direction
        if self.movement_speed > 0.5:  # Threshold for noise
            self.movement_angle = np.degrees(np.arctan2(-avg_dx, -avg_dy))
        else:
            self.movement_angle = 0
        
        self.prev_gray = gray
    
    def estimate_distance_meters(self, bbox: np.ndarray, frame_height: int) -> float:
        """
        Estimates distance to object in meters based on bbox size.
        This is a rough approximation assuming average object sizes.
        
        Args:
            bbox: [x1, y1, x2, y2]
            frame_height: height of frame
        
        Returns:
            estimated distance in meters
        """
        box_height = bbox[3] - bbox[1]
        relative_height = box_height / frame_height
        
        # Rough distance estimation:
        # - Object takes 50% of frame height ≈ 2 meters
        # - Object takes 25% of frame height ≈ 5 meters
        # - Object takes 10% of frame height ≈ 15 meters
        # Using inverse relationship: distance ≈ k / relative_height
        
        if relative_height > 0.01:
            distance = 1.0 / relative_height  # Simplified formula
            distance = min(distance, 50)  # Cap at 50 meters
        else:
            distance = 50  # Far away
        
        return round(distance, 1)
    
    def detect_objects(self, frame: np.ndarray) -> tuple:
        """
        Performs object detection on frame.
        Uses ONNX Runtime if available, otherwise PyTorch (with Float16 if enabled).
        Returns (boxes, labels, scores) as numpy arrays.
        """
        h, w = frame.shape[:2]
        resize = INFERENCE_RESIZE
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (resize, resize))
        image /= 255.0
        image_chw = image.transpose(2, 0, 1)
        
        if self.use_onnx:
            image_np = image_chw[np.newaxis].astype(np.float32)
            outputs = self.onnx_session.run(None, {self._onnx_input_name: image_np})
            # Export order: boxes, scores, labels (see export.py)
            boxes = outputs[0]
            scores = outputs[1]
            labels = outputs[2].astype(int)
        else:
            dtype = torch.float16 if self.use_half else torch.float32
            image_tensor = torch.tensor(image_chw, dtype=dtype).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = self.model(image_tensor)
            
            result = {k: v.cpu().numpy() for k, v in out[0].items()}
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"].astype(int)
        
        if len(boxes) == 0:
            return np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0)
        
        scale_x = w / resize
        scale_y = h / resize
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        mask = scores >= self.detection_threshold
        return boxes[mask], labels[mask], scores[mask]
    
    def get_control_command(
        self,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        image_width: int,
        image_height: int,
        force: bool = False,
        frame_num: int = 0
    ) -> dict:
        """
        Gets control command. LLM calls run async in background thread
        so detection pipeline is never blocked by network latency.
        """
        current_time = time.time()
        self.current_frame = frame_num
        
        # Collect result from background LLM call if ready
        if self._llm_future is not None and self._llm_future.done():
            try:
                self.last_command = self._llm_future.result()
            except Exception:
                pass
            self._llm_future = None
        
        if not self.use_llm or self.llm_controller is None:
            return self._rule_based_command(boxes, labels, scores, image_width, image_height)
        
        # Launch async LLM call if interval passed and no call in progress
        interval_ok = force or (current_time - self.last_llm_call) >= self.llm_interval
        if interval_ok and self._llm_future is None:
            self.last_llm_call = current_time
            
            detected_objects = self.llm_controller.prepare_detections(
                boxes=boxes.tolist(),
                labels=labels.tolist(),
                scores=scores.tolist(),
                image_width=image_width,
                image_height=image_height,
                threshold=self.detection_threshold
            )
            
            speed = self.movement_speed
            angle = self.movement_angle
            self._llm_future = self._llm_executor.submit(
                self._call_llm_background, detected_objects, speed, angle, frame_num
            )
        
        return self.last_command or self._default_command()
    
    def _call_llm_background(self, detected_objects, speed, angle, frame_num) -> dict:
        """Runs LLM call in background thread. Returns command dict."""
        command = self.llm_controller.generate_control_command(
            detected_objects, current_speed=speed, movement_angle=angle
        )
        command_dict = self.llm_controller.command_to_dict(command)
        self._log_command(detected_objects, command_dict, frame_num)
        return command_dict
    
    def _default_command(self) -> dict:
        """Default command."""
        return {
            "steering": "center",
            "angle": "0 degrees",
            "pedal": "gas",
            "pedal_percent": "30%",
            "reasoning": "Waiting for data"
        }
    
    def _rule_based_command(
        self,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        image_width: int,
        image_height: int
    ) -> dict:
        """
        Simple rules without LLM (for testing).
        """
        command = {
            "steering": "center",
            "angle": "0 degrees",
            "pedal": "gas",
            "pedal_percent": "50%",
            "reasoning": "Rule-based, no LLM"
        }
        
        for box, label, score in zip(boxes, labels, scores):
            class_name = CLASSES[label] if 0 <= label < len(CLASSES) else "unknown"
            
            # If pedestrian nearby — brake
            if class_name == "person":
                box_height = box[3] - box[1]
                if box_height / image_height > 0.2:
                    command["pedal"] = "brake"
                    command["pedal_percent"] = "80%"
                    command["reasoning"] = "Pedestrian detected nearby"
                    break
            
            # STOP sign — full stop
            if class_name == "stop sign":
                command["pedal"] = "brake"
                command["pedal_percent"] = "100%"
                command["reasoning"] = "STOP sign"
                break
            
            # Vehicle ahead nearby — slow down
            if class_name in ["car", "truck", "bus"]:
                center_x = (box[0] + box[2]) / 2
                if 0.3 < center_x / image_width < 0.7:  # in center
                    box_height = box[3] - box[1]
                    if box_height / image_height > 0.3:  # nearby
                        command["pedal"] = "brake"
                        command["pedal_percent"] = "50%"
                        command["reasoning"] = f"Vehicle ({class_name}) ahead"
        
        self.last_command = command
        return command
    
    def _log_command(self, detected_objects, command, frame_num: int = 0):
        """Logs command to file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "frame": frame_num,
            "detected_objects": [
                {
                    "class_name": obj.class_name,
                    "confidence": float(obj.confidence),
                    "position": obj.position,
                    "distance_estimate": obj.distance_estimate,
                    "distance_meters": float(obj.distance_meters)
                }
                for obj in detected_objects
            ],
            "command": command,
            "movement": {
                "angle": float(self.movement_angle),
                "speed": float(self.movement_speed)
            }
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def draw_overlay(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        command: dict,
        fps: float
    ) -> np.ndarray:
        """
        Draws visualization on frame.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw bounding boxes with distance
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.astype(int)
            class_name = CLASSES[label] if 0 <= label < len(CLASSES) else f"cls_{label}"
            color = self.colors[label % len(self.colors)]
            
            # Estimate distance
            distance = self.estimate_distance_meters(box, h)
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Label with distance
            label_text = f"{class_name} {distance:.0f}m"
            cv2.putText(
                overlay, label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Distance indicator bar (color coded)
            if distance < 5:
                dist_color = (0, 0, 255)  # Red - close
            elif distance < 15:
                dist_color = (0, 165, 255)  # Orange - medium
            else:
                dist_color = (0, 255, 0)  # Green - far
            
            bar_width = max(5, int(50 / distance * 5))
            cv2.rectangle(overlay, (x2 + 5, y1), (x2 + 5 + bar_width, y1 + 10), dist_color, -1)
        
        # Draw movement vector arrow (top-center of frame)
        arrow_center = (w // 2, 60)
        arrow_length = min(50, int(self.movement_speed * 10))
        if arrow_length > 5:
            angle_rad = np.radians(self.movement_angle)
            arrow_end = (
                int(arrow_center[0] + arrow_length * np.sin(angle_rad)),
                int(arrow_center[1] - arrow_length * np.cos(angle_rad))
            )
            cv2.arrowedLine(overlay, arrow_center, arrow_end, (0, 255, 255), 3, tipLength=0.3)
            cv2.putText(
                overlay, f"{self.movement_angle:.0f}deg",
                (arrow_center[0] + 30, arrow_center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )
        else:
            cv2.circle(overlay, arrow_center, 10, (0, 255, 255), 2)
            cv2.putText(overlay, "STOP", (arrow_center[0] - 20, arrow_center[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Control panel (semi-transparent)
        panel_height = 150
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Command text
        y_offset = h - panel_height + 25
        line_height = 22
        
        cv2.putText(
            frame, f"FPS: {fps:.1f} | Movement: {self.movement_angle:.0f}deg @ {self.movement_speed:.1f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        y_offset += line_height
        
        cv2.putText(
            frame, f"Steering: {command.get('steering', 'N/A')} | Angle: {command.get('angle', 'N/A')}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        y_offset += line_height
        
        cv2.putText(
            frame, f"Pedal: {command.get('pedal', 'N/A')} | Percent: {command.get('pedal_percent', 'N/A')} | Gear: {command.get('gear', 'N/A')}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        y_offset += line_height
        
        # Reasoning (may be long, truncate)
        reasoning = command.get('reasoning', '')[:80]
        cv2.putText(
            frame, f"Reason: {reasoning}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        # Visual steering indicator
        steering = command.get('steering', 'center')
        indicator_x = w // 2
        if steering == 'left':
            indicator_x = w // 4
        elif steering == 'right':
            indicator_x = 3 * w // 4
        
        cv2.circle(frame, (indicator_x, h - panel_height - 20), 15, (0, 255, 255), -1)
        cv2.putText(frame, "^", (indicator_x - 8, h - panel_height - 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return frame
    
    def process_video(
        self,
        input_path: str,
        output_path: str = None,
        show_preview: bool = True,
        max_seconds: float = None,
        skip_frames: int = 1
    ):
        """
        Processes video file.
        
        Args:
            max_seconds: maximum processing time (None = full video)
            skip_frames: process every N-th frame (1=all)
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: failed to open {input_path}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Time limit
        if max_seconds:
            max_frames = int(max_seconds * fps_video)
            total_frames = min(total_frames, max_frames)
            print(f"Video: {width}x{height} @ {fps_video:.1f} FPS")
            print(f"Limit: {max_seconds} sec ({total_frames} frames)")
        else:
            print(f"Video: {width}x{height} @ {fps_video:.1f} FPS, {total_frames} frames")
        
        if skip_frames > 1:
            print(f"⚡ Speed mode: processing every {skip_frames}th frame")
        
        # Setup output video
        if output_path is None:
            output_path = f"autopilot_output_{Path(input_path).stem}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print("Processing video... (press 'q' to exit)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check time limit
            if max_seconds and frame_count >= total_frames:
                print(f"\n✅ Limit reached: {max_seconds} seconds")
                break
            
            frame_start = time.time()
            
            # Frame skipping for speed
            if skip_frames > 1 and frame_count % skip_frames != 0:
                # Use last command, draw overlay without detection
                if self.last_command:
                    result_frame = self.draw_overlay(
                        frame, np.array([]), np.array([]), np.array([]), 
                        self.last_command, self._last_fps if hasattr(self, '_last_fps') else 0
                    )
                else:
                    result_frame = frame
                out.write(result_frame)
                frame_count += 1
                continue
            
            # Compute movement vector (optical flow)
            self.compute_movement_vector(frame)
            
            # Detection
            boxes, labels, scores = self.detect_objects(frame)
            
            # Get command
            command = self.get_control_command(
                boxes, labels, scores, width, height, frame_num=frame_count
            )
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            self._last_fps = fps
            
            # Draw overlay
            result_frame = self.draw_overlay(
                frame, boxes, labels, scores, command, fps
            )
            
            # Write
            out.write(result_frame)
            
            # Show preview
            if show_preview:
                cv2.imshow("Autopilot", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                progress = frame_count / total_frames * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {elapsed:.0f}s", flush=True)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print(f"\nDone!")
        print(f"Frames processed: {frame_count}")
        print(f"Time: {total_time:.1f} sec")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Result saved: {output_path}")
        print(f"Logs: {self.log_file}")
    
    def run_camera(self, camera_id = 0, show_preview: bool = True):
        """
        Starts real-time processing from camera or stream.
        
        Args:
            camera_id: int (0,1,2...) or string with URL (rtsp://, http://)
        """
        # Determine source type
        if isinstance(camera_id, str):
            print(f"📡 Connecting to stream: {camera_id}")
        else:
            print(f"📷 Connecting to camera: {camera_id}")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: failed to open {camera_id}")
            if isinstance(camera_id, str):
                print("Check URL and stream availability")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Connected: {width}x{height}")
        print("Press 'q' to exit, 's' to save screenshot")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            frame_count += 1
            
            # Compute movement vector
            self.compute_movement_vector(frame)
            
            # Detection
            boxes, labels, scores = self.detect_objects(frame)
            
            # Get command
            command = self.get_control_command(
                boxes, labels, scores, width, height, frame_num=frame_count
            )
            
            # FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            
            # Draw
            result_frame = self.draw_overlay(
                frame, boxes, labels, scores, command, fps
            )
            
            if show_preview:
                cv2.imshow("Autopilot - Camera", result_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_path, result_frame)
                    print(f"Screenshot saved: {screenshot_path}")
            
            frame_count += 1
            
            # Print command to console
            if frame_count % 30 == 0:
                print(f"Command: {json.dumps(command, ensure_ascii=False)}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nSession ended. Logs: {self.log_file}")


def main():
    parser = argparse.ArgumentParser(description="Autopilot with object detection and LLM")
    
    parser.add_argument(
        "--input", "-i",
        help="Path to input video"
    )
    parser.add_argument(
        "--camera", "-c",
        default=None,
        help="Camera ID (0,1,2...) or RTSP/HTTP stream URL"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save result"
    )
    parser.add_argument(
        "--model", "-m",
        default="outputs/best_model.pth",
        help="Path to model"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM (use rules)"
    )
    parser.add_argument(
        "--llm-interval",
        type=float,
        default=2.0,
        help="Interval between LLM calls (sec)"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Don't show preview"
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Maximum video processing time (sec)"
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=10,
        help="Process every N-th frame (1=all, 10=every tenth)"
    )
    parser.add_argument(
        "--debug-llm",
        action="store_true",
        help="Print LLM prompts/responses to console (shows text sent to LLM)"
    )
    parser.add_argument(
        "--onnx",
        default=None,
        help="Path to ONNX model (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Create autopilot
    autopilot = Autopilot(
        model_path=args.model,
        detection_threshold=args.threshold,
        llm_interval=args.llm_interval,
        use_llm=not args.no_llm,
        debug_llm=args.debug_llm,
        onnx_path=args.onnx,
    )
    
    # Run
    if args.camera is not None:
        # Determine camera type: number or URL
        camera_source = args.camera
        try:
            camera_source = int(args.camera)
        except ValueError:
            # This is URL (RTSP, HTTP, etc.)
            pass
        
        autopilot.run_camera(
            camera_id=camera_source,
            show_preview=not args.no_preview
        )
    elif args.input:
        autopilot.process_video(
            input_path=args.input,
            output_path=args.output,
            show_preview=not args.no_preview,
            max_seconds=args.max_seconds,
            skip_frames=args.skip_frames
        )
    else:
        print("Specify --input for video or --camera for camera")
        parser.print_help()


if __name__ == "__main__":
    main()

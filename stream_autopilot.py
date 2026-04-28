#!/usr/bin/env python3
"""
Streaming Autopilot.

Workflow:
  Video/Camera → RetinaNet → Azure OpenAI → Console + File

Usage:
    python stream_autopilot.py --input video.mp4
    python stream_autopilot.py --camera 0
    python stream_autopilot.py --camera "rtsp://..."

Output:
    - Console: real-time commands
    - File: commands_TIMESTAMP.jsonl
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

from model import create_model
from config import NUM_CLASSES, DEVICE, CLASSES, RESIZE_TO, INFERENCE_RESIZE
from llm_controller import LLMController, DetectedObject

# Console colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


class StreamAutopilot:
    """
    Streaming autopilot — data only, no GUI.
    """
    
    def __init__(
        self,
        model_path: str = "outputs/best_model.pth",
        detection_threshold: float = 0.5,
        llm_interval: float = 0.5,
        use_llm: bool = True,
        output_file: str = None,
        llm_provider: str = None
    ):
        self.llm_provider = llm_provider
        print(f"{Colors.HEADER}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}🚗 STREAM AUTOPILOT{Colors.END}")
        print(f"{Colors.HEADER}{'='*60}{Colors.END}")
        print(f"Device: {Colors.CYAN}{DEVICE}{Colors.END}")
        
        # Load model
        if os.path.exists(model_path):
            print(f"Model: {Colors.GREEN}fine-tuned{Colors.END} ({model_path})")
            self.model = create_model(num_classes=NUM_CLASSES)
            checkpoint = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"Model: {Colors.YELLOW}COCO pretrained{Colors.END}")
            import torchvision
            from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
            self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
            )
        
        self.model.to(DEVICE).eval()
        
        self.detection_threshold = detection_threshold
        self.llm_interval = llm_interval
        self.use_llm = use_llm
        self.last_llm_call = 0
        self.last_command = None
        
        # LLM controller
        if use_llm:
            try:
                self.llm_controller = LLMController(provider=self.llm_provider)
                print(f"LLM: {Colors.GREEN}connected{Colors.END}")
            except Exception as e:
                print(f"LLM: {Colors.RED}Error - {e}{Colors.END}")
                self.use_llm = False
                self.llm_controller = None
        else:
            self.llm_controller = None
            print(f"LLM: {Colors.YELLOW}disabled (rule-based){Colors.END}")
        
        # Log file
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"commands_{timestamp}.jsonl"
        self.output_file = output_file
        print(f"Output: {Colors.CYAN}{output_file}{Colors.END}")
        
        print(f"{Colors.HEADER}{'='*60}{Colors.END}\n")
    
    def detect_objects(self, frame: np.ndarray) -> tuple:
        """Detect objects in frame."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (INFERENCE_RESIZE, INFERENCE_RESIZE))
        image /= 255.0
        
        image_tensor = torch.tensor(
            image.transpose(2, 0, 1),
            dtype=torch.float
        ).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        outputs = [{k: v.cpu().numpy() for k, v in outputs[0].items()}]
        
        boxes = outputs[0]["boxes"]
        scores = outputs[0]["scores"]
        labels = outputs[0]["labels"].astype(int)
        
        # Scale bbox
        h, w = frame.shape[:2]
        scale_x = w / INFERENCE_RESIZE
        scale_y = h / INFERENCE_RESIZE
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        mask = scores >= self.detection_threshold
        return boxes[mask], labels[mask], scores[mask]
    
    def get_command(
        self,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        width: int,
        height: int
    ) -> dict:
        """Get command from LLM or rule-based."""
        current_time = time.time()
        
        # Check interval
        if (current_time - self.last_llm_call) < self.llm_interval:
            return self.last_command or self._default_command()
        
        if not self.use_llm or self.llm_controller is None:
            return self._rule_based(boxes, labels, scores, width, height)
        
        # Prepare objects for LLM
        detected_objects = self.llm_controller.prepare_detections(
            boxes=boxes.tolist(),
            labels=labels.tolist(),
            scores=scores.tolist(),
            image_width=width,
            image_height=height,
            threshold=self.detection_threshold
        )
        
        # Note: stream mode doesn't have optical flow, so movement_angle=0
        command = self.llm_controller.generate_control_command(
            detected_objects,
            movement_angle=0.0  # No optical flow in stream mode
        )
        command_dict = self.llm_controller.command_to_dict(command)
        
        self.last_llm_call = current_time
        self.last_command = command_dict
        
        return command_dict
    
    def _default_command(self) -> dict:
        return {
            "steering": "center",
            "angle": "0 degrees",
            "pedal": "gas",
            "pedal_percent": "30%",
            "reasoning": "waiting"
        }
    
    def _rule_based(self, boxes, labels, scores, width, height) -> dict:
        """Simple rules without LLM."""
        command = self._default_command()
        
        for box, label, score in zip(boxes, labels, scores):
            class_name = CLASSES[label] if 0 <= label < len(CLASSES) else "unknown"
            
            if class_name == "person":
                box_height = box[3] - box[1]
                if box_height / height > 0.2:
                    command = {
                        "steering": "center",
                        "angle": "0 degrees",
                        "pedal": "brake",
                        "pedal_percent": "80%",
                        "reasoning": "pedestrian nearby"
                    }
                    break
            
            if class_name == "stop sign":
                command = {
                    "steering": "center",
                    "angle": "0 degrees",
                    "pedal": "brake",
                    "pedal_percent": "100%",
                    "reasoning": "STOP sign"
                }
                break
        
        self.last_command = command
        return command
    
    def format_detections(self, labels: np.ndarray, scores: np.ndarray) -> str:
        """Format detected objects for console."""
        if len(labels) == 0:
            return f"{Colors.YELLOW}no objects{Colors.END}"
        
        objects = []
        for label, score in zip(labels, scores):
            class_name = CLASSES[label] if 0 <= label < len(CLASSES) else f"cls_{label}"
            objects.append(f"{class_name}({score:.0%})")
        
        return ", ".join(objects[:5])  # Max 5 objects
    
    def format_command(self, cmd: dict) -> str:
        """Format command for console."""
        pedal = cmd.get('pedal', 'gas')
        pedal_color = Colors.RED if pedal == 'brake' else Colors.GREEN
        
        steering = cmd.get('steering', 'center')
        if steering == 'left':
            arrow = '←'
        elif steering == 'right':
            arrow = '→'
        else:
            arrow = '↑'
        
        gear = cmd.get('gear', 3)
        return (
            f"{Colors.BOLD}{arrow} {steering.upper()}{Colors.END} "
            f"| {pedal_color}{pedal} {cmd.get('pedal_percent', '0%')}{Colors.END} "
            f"| G{gear}"
        )
    
    def write_log(self, timestamp: str, detections: list, command: dict, fps: float):
        """Write log to file."""
        entry = {
            "timestamp": timestamp,
            "fps": round(fps, 1),
            "detections": detections,
            "command": command
        }
        
        with open(self.output_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def run(
        self,
        source,  # int, str (path), str (URL), or "pipe:0"
        skip_frames: int = 1,
        max_seconds: float = None,
        pipe_width: int = 640,
        pipe_height: int = 480
    ):
        """
        Start streaming processing.
        
        Args:
            source: camera (0,1), file (path), stream (rtsp://), or pipe:0
            skip_frames: process every N-th frame
            max_seconds: time limit
            pipe_width: frame width when reading from pipe
            pipe_height: frame height when reading from pipe
        """
        # Check if reading from FFmpeg pipe or TCP
        use_pipe = False
        use_tcp = False
        tcp_socket = None
        
        if isinstance(source, str) and source.startswith('pipe:'):
            use_pipe = True
            print(f"📺 Source: FFmpeg pipe (stdin)")
            print(f"   Expected frame size: {pipe_width}x{pipe_height} BGR24")
            width, height = pipe_width, pipe_height
            fps_video = 30.0  # Assume 30fps for pipe
            cap = None
        elif isinstance(source, str) and source.startswith('tcp://'):
            # TCP server mode: tcp://localhost:5000
            use_tcp = True
            import socket
            parts = source.replace('tcp://', '').split(':')
            host = parts[0] if parts[0] else 'localhost'
            port = int(parts[1]) if len(parts) > 1 else 5000
            
            print(f"📡 Source: TCP server {host}:{port}")
            print(f"   Expected frame size: {pipe_width}x{pipe_height} BGR24")
            print(f"   Waiting for FFmpeg connection...")
            
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(1)
            tcp_socket, addr = server.accept()
            print(f"   ✅ Connected from {addr}")
            
            width, height = pipe_width, pipe_height
            fps_video = 30.0
            cap = None
        else:
            # Open source
            if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
                source = int(source) if isinstance(source, str) else source
                print(f"📷 Source: camera {source}")
            elif source.startswith(('rtsp://', 'http://', 'udp://')):
                print(f"📡 Source: stream {source}")
            else:
                print(f"🎬 Source: file {source}")
            
            cap = cv2.VideoCapture(source)
        
        if cap is not None and not cap.isOpened():
            print(f"{Colors.RED}❌ Failed to open source{Colors.END}")
            return
        
        # Get dimensions from cap or use pipe/tcp dimensions
        if not use_pipe and not use_tcp:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        print(f"📐 Resolution: {width}x{height} @ {fps_video:.0f} FPS")
        if skip_frames > 1:
            print(f"⚡ Frame skip: every {skip_frames}th")
        if max_seconds:
            print(f"⏱️ Limit: {max_seconds} sec")
        
        print(f"\n{Colors.HEADER}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}▶️  STREAMING...{Colors.END} (Ctrl+C to stop)")
        print(f"{Colors.HEADER}{'='*60}{Colors.END}\n")
        
        frame_count = 0
        start_time = time.time()
        max_frames = int(max_seconds * fps_video) if max_seconds else float('inf')
        
        # Frame size for reading from pipe (BGR24 = 3 bytes per pixel)
        frame_size = width * height * 3
        
        try:
            while True:
                # Read frame from pipe, TCP, or capture
                if use_pipe:
                    import sys
                    raw_frame = sys.stdin.buffer.read(frame_size)
                    if len(raw_frame) != frame_size:
                        break  # End of pipe
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                    ret = True
                elif use_tcp:
                    # Read exact frame_size bytes from TCP
                    raw_frame = b''
                    while len(raw_frame) < frame_size:
                        chunk = tcp_socket.recv(frame_size - len(raw_frame))
                        if not chunk:
                            break  # Connection closed
                        raw_frame += chunk
                    if len(raw_frame) != frame_size:
                        break  # End of stream
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                    ret = True
                else:
                    ret, frame = cap.read()
                    
                if not ret:
                    if not use_pipe and not use_tcp and isinstance(source, str) and not source.startswith(('rtsp', 'http', 'udp')):
                        break  # End of file
                    continue  # Stream — try again
                
                # Check limit
                if frame_count >= max_frames:
                    print(f"\n{Colors.GREEN}✅ Time limit reached{Colors.END}")
                    break
                
                # Skip frames
                if skip_frames > 1 and frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                
                frame_start = time.time()
                
                # Detection
                boxes, labels, scores = self.detect_objects(frame)
                
                # Command
                command = self.get_command(boxes, labels, scores, width, height)
                
                # FPS
                fps = 1.0 / (time.time() - frame_start) if (time.time() - frame_start) > 0 else 0
                
                # Timestamp
                timestamp = datetime.now().isoformat()
                elapsed = time.time() - start_time
                
                # Form detections list
                detections = []
                for box, label, score in zip(boxes, labels, scores):
                    class_name = CLASSES[label] if 0 <= label < len(CLASSES) else f"cls_{label}"
                    detections.append({
                        "class": class_name,
                        "confidence": round(float(score), 3),
                        "bbox": box.tolist()
                    })
                
                # Write to file
                self.write_log(timestamp, detections, command, fps)
                
                # Print to console
                time_str = f"{elapsed:>6.1f}s"
                fps_str = f"{fps:>4.1f} fps"
                det_str = self.format_detections(labels, scores)
                cmd_str = self.format_command(command)
                
                print(
                    f"{Colors.CYAN}{time_str}{Colors.END} | "
                    f"{fps_str} | "
                    f"[{det_str}] | "
                    f"{cmd_str}"
                )
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}⏹️  Stopped by user{Colors.END}")
        
        finally:
            if cap is not None:
                cap.release()
            if tcp_socket is not None:
                tcp_socket.close()
            
            total_time = time.time() - start_time
            print(f"\n{Colors.HEADER}{'='*60}{Colors.END}")
            print(f"{Colors.BOLD}📊 SUMMARY{Colors.END}")
            print(f"   Frames processed: {frame_count}")
            print(f"   Time: {total_time:.1f} sec")
            print(f"   Logs saved: {Colors.CYAN}{self.output_file}{Colors.END}")
            print(f"{Colors.HEADER}{'='*60}{Colors.END}")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming Autopilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stream_autopilot.py --input video.mp4
  python stream_autopilot.py --camera 0
  python stream_autopilot.py --camera "rtsp://192.168.1.1:8554/live"
  python stream_autopilot.py --input video.mp4 --skip-frames 3 --max-seconds 30
  
FFmpeg pipe examples:
  ffmpeg -i video.mp4 -f rawvideo -pix_fmt bgr24 - | python stream_autopilot.py --camera pipe:0
  ffmpeg -i rtsp://cam/live -f rawvideo -pix_fmt bgr24 - | python stream_autopilot.py --camera pipe:0 --pipe-width 1280 --pipe-height 720
        """
    )
    
    parser.add_argument("--input", "-i", help="Path to video file")
    parser.add_argument("--camera", "-c", help="Camera ID, stream URL, or 'pipe:0' for FFmpeg pipe")
    parser.add_argument("--output", "-o", help="Output file for commands (.jsonl)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--max-seconds", type=float, help="Time limit (sec)")
    parser.add_argument("--llm-interval", type=float, default=0.5, help="LLM interval (sec)")
    parser.add_argument("--no-llm", action="store_true", help="No LLM (rule-based)")
    parser.add_argument("--pipe-width", type=int, default=640, help="Frame width for FFmpeg pipe input")
    parser.add_argument("--pipe-height", type=int, default=480, help="Frame height for FFmpeg pipe input")
    parser.add_argument("--llm-provider", choices=["azure", "ollama", "openai"], 
                        help="LLM provider: azure, ollama (local), openai")
    
    args = parser.parse_args()
    
    # Determine source
    if args.input:
        source = args.input
    elif args.camera:
        try:
            source = int(args.camera)
        except ValueError:
            source = args.camera
    else:
        print("Specify --input or --camera")
        parser.print_help()
        return
    
    # Create and run
    autopilot = StreamAutopilot(
        detection_threshold=args.threshold,
        llm_interval=args.llm_interval,
        use_llm=not args.no_llm,
        output_file=args.output,
        llm_provider=args.llm_provider
    )
    
    autopilot.run(
        source=source,
        skip_frames=args.skip_frames,
        max_seconds=args.max_seconds,
        pipe_width=args.pipe_width,
        pipe_height=args.pipe_height
    )


if __name__ == "__main__":
    main()

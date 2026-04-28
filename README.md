# 🚗 Autopilot for the Toy Car / Ground Robot

A computer vision + LLM system that processes video/camera feeds, detects road objects with RetinaNet, and generates control commands via Azure OpenAI, Ollama, or OpenAI.

## 🎯 Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Video/    │───▶│  RetinaNet  │───▶│   LLM Layer │───▶│  Control    │
│   Camera    │    │  Detection  │    │    (LLM)    │    │  Commands   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 🎬 Demo

![Autopilot demo](./demo.gif)

### Output Format

```json
{
    "steering": "left|right|center",
    "angle": "0-90 degrees",
    "pedal": "brake|gas",
    "pedal_percent": "0-100%",
    "reasoning": "explanation of decision"
}
```

## 🏗️ Architecture

### 1. Object Detection (RetinaNet)

- **Model**: RetinaNet with ResNet50-FPN backbone
- **Pretrained on**: COCO dataset (80 classes)
- **Key classes for driving**: car, truck, bus, motorcycle, bicycle, person, traffic light, stop sign
- **Framework**: PyTorch + torchvision

RetinaNet uses **Focal Loss** to handle class imbalance between background and objects, making it excellent for detecting small objects on roads.

### 2. LLM Integration (Azure / Ollama / OpenAI)

- **Model**: Provider-specific (configurable)
- **Purpose**: Analyze detected objects and generate safe driving commands
- **Interval**: Configurable; `autopilot.py` default is 2.0s, `stream_autopilot.py` default is 0.5s
- **Fallback**: Rule-based system when LLM is unavailable

The LLM receives:
- List of detected objects with positions (left/center/right)
- Distance estimates (close/medium/far)
- Confidence scores

### 3. Processing Pipeline

1. **Frame Capture**: OpenCV reads frames from video/camera/RTSP stream
2. **Preprocessing**: Resize to `INFERENCE_RESIZE` (320 on CPU, 416 on MPS/CUDA by default), normalize to [0,1]
3. **Detection**: RetinaNet outputs bounding boxes, class labels, confidence scores
4. **Scene Analysis**: Convert detections to natural language description
5. **LLM Query**: Send scene to GPT for command generation
6. **Output**: JSON command + visualization overlay

## 📦 Installation

### Prerequisites

- Python 3.9+
- macOS / Linux / Windows
- (Optional) NVIDIA GPU with CUDA for faster inference

### Setup

```bash
# Clone or navigate to project
cd AutopilotForGroundRobot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### LLM Configuration

Edit `config.py` and choose your provider:

```python
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"  # Your deployment name

LLM_PROVIDER = "azure"  # "azure", "ollama", "openai"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:3b"
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4o-mini"
```

## 🚀 Usage

### Quick Start (No LLM)

Test without any LLM provider using rule-based commands:

```bash
python autopilot.py --input video.mp4 --no-llm
python stream_autopilot.py --input video.mp4 --no-llm
```

### With LLM

```bash
# Process video file
python autopilot.py --input video.mp4

# Process with time limit (30 seconds)
python autopilot.py --input video.mp4 --max-seconds 30

# Speed up processing (every 10th frame, default in autopilot.py)
python autopilot.py --input video.mp4 --skip-frames 3
```

### Camera / Stream

```bash
# Webcam
python autopilot.py --camera 0

# GoPro (USB webcam mode)
python autopilot.py --camera 1

# RTSP stream
python autopilot.py --camera "rtsp://192.168.1.100:8554/live"
```

### Streaming Mode (Console Output)

For headless operation with real-time console output:

```bash
python stream_autopilot.py --input video.mp4 --skip-frames 3
```

Output:
```
   1.2s | 12.5 fps | [car(92%), person(87%)] | ↑ CENTER | gas 30%
   1.7s | 11.8 fps | [car(94%), truck(76%)]  | ← LEFT   | gas 20%
   2.2s | 12.1 fps | [person(91%)]           | ↑ CENTER | brake 80%
```

## 📁 Project Structure

```
AutopilotForGroundRobot/
├── config.py              # Configuration (device, classes, Azure OpenAI)
├── model.py               # RetinaNet model definition
├── llm_controller.py      # Azure OpenAI integration
├── autopilot.py           # Main autopilot with GUI overlay
├── stream_autopilot.py    # Streaming mode (console + file output)
├── train.py               # Training script (for custom datasets)
├── datasets.py            # Custom dataset loader
├── custom_utils.py        # Utilities (transforms, metrics)
├── requirements.txt       # Python dependencies
└── outputs/               # Saved models and logs
    └── best_model.pth     # Best trained model
```

## ⚙️ Command Line Options

### autopilot.py

| Option | Description | Default |
|--------|-------------|---------|
| `--input, -i` | Input video file | - |
| `--camera, -c` | Camera ID or stream URL | - |
| `--output, -o` | Output video path | auto-generated |
| `--model, -m` | Path to model weights | `outputs/best_model.pth` |
| `--threshold, -t` | Detection confidence threshold | 0.5 |
| `--no-llm` | Disable LLM, use rules | False |
| `--llm-interval` | LLM call interval (seconds) | 2.0 |
| `--skip-frames` | Process every N-th frame | 10 |
| `--max-seconds` | Time limit for processing | None |
| `--no-preview` | Don't show preview window | False |

### stream_autopilot.py

Includes similar core options, plus stream-specific flags:
- `--pipe-width` (default: `640`)
- `--pipe-height` (default: `480`)
- `--llm-provider` (`azure|ollama|openai`)

Automatic JSONL logging is enabled by default to `commands_TIMESTAMP.jsonl`.

## 📊 Output Files

### Video Output

`autopilot_output_VIDEO.mp4` - Video with:
- Bounding boxes around detected objects
- Control panel showing steering, pedal, reasoning
- FPS counter

### Log Files

`commands_TIMESTAMP.jsonl` - JSON Lines format:

```json
{"timestamp": "2026-01-01T12:00:01.234", "fps": 12.5, "detections": [{"class": "car", "confidence": 0.92, "bbox": [100, 200, 300, 400]}], "command": {"steering": "center", "angle": "0 degrees", "pedal": "gas", "pedal_percent": "30%", "reasoning": "Road is clear"}}
```

## 🎓 Training Custom Model

To train on your own dataset:

1. Prepare data in format:
```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
└── valid/
    ├── images/
    └── labels/
```

2. Label format (per line): `class_id x_min y_min x_max y_max` (normalized 0-1)

3. Update `config.py` with your classes

4. Run training:
```bash
python train.py
```

## 🔧 Performance Tips

### For Faster Processing

```bash
# Skip frames (higher value = fewer processed frames)
python autopilot.py --input video.mp4 --skip-frames 3

# Lower detection threshold (more detections, but faster decision)
python autopilot.py --input video.mp4 --threshold 0.3

# Increase LLM interval (fewer API calls)
python autopilot.py --input video.mp4 --llm-interval 1.0
```

### For Better Accuracy

```bash
# Process all frames
python autopilot.py --input video.mp4 --skip-frames 1

# Higher threshold (fewer false positives)
python autopilot.py --input video.mp4 --threshold 0.7
```

## 🖥️ Hardware Support

| Device | Support | Notes |
|--------|---------|-------|
| Apple Silicon (M1/M2/M3) | ✅ MPS | Good performance |
| NVIDIA GPU | ✅ CUDA | Best performance |
| CPU | ✅ | Slower, but works |

Device is auto-detected. Check with:
```python
python -c "from config import DEVICE; print(DEVICE)"
```

## 🎬 FFmpeg Integration

Use FFmpeg for advanced streaming scenarios: read from any source, transcode, or stream output.

### Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Read from FFmpeg Pipe

Process any FFmpeg-supported source:

```bash
# Read from IP camera and pipe to autopilot
ffmpeg -i "rtsp://camera:554/stream" -f rawvideo -pix_fmt bgr24 - | \
  python stream_autopilot.py --camera pipe:0

# Read from YouTube/HLS stream
ffmpeg -i "https://example.com/stream.m3u8" -f rawvideo -pix_fmt bgr24 - | \
  python stream_autopilot.py --camera pipe:0

# Read from video with hardware decoding (NVIDIA)
ffmpeg -hwaccel cuda -i video.mp4 -f rawvideo -pix_fmt bgr24 - | \
  python stream_autopilot.py --camera pipe:0
```

### GoPro via FFmpeg

```bash
# GoPro Hero via WiFi (replace IP)
ffmpeg -i "udp://@:8554" -f rawvideo -pix_fmt bgr24 - | \
  python stream_autopilot.py --camera pipe:0

# GoPro with low latency settings
ffmpeg -fflags nobuffer -flags low_delay -i "udp://@:8554" \
  -f rawvideo -pix_fmt bgr24 - | python stream_autopilot.py --camera pipe:0
```

### Stream Output via FFmpeg

Send processed commands to another system:

```bash
# Process any video through FFmpeg
ffmpeg -i video.mp4 -f rawvideo -pix_fmt bgr24 - | \
  python stream_autopilot.py --camera pipe:0

# Stream to MQTT (requires mosquitto_pub)
python stream_autopilot.py --input video.mp4 2>/dev/null | \
  while read line; do
    echo "$line" | grep -E "^\{" && \
    mosquitto_pub -h mqtt.local -t "robot/commands" -m "$line"
  done
```

### Separate Terminals (screen/tmux)

Run FFmpeg and autopilot in separate terminals:

**Option 1: Named Pipe (FIFO)**

```bash
# Terminal 1: Create pipe and run Python
mkfifo /tmp/video_pipe
python stream_autopilot.py --camera /tmp/video_pipe --pipe-width 640 --pipe-height 480

# Terminal 2: FFmpeg writes to pipe
ffmpeg -i video.mp4 -f rawvideo -pix_fmt bgr24 -s 640x480 /tmp/video_pipe
```

**Option 2: TCP Socket (recommended)**

```bash
# Terminal 1: Python listens on TCP port
python stream_autopilot.py --camera tcp://localhost:5000 --pipe-width 640 --pipe-height 480

# Terminal 2: FFmpeg streams to TCP
ffmpeg -i video.mp4 -f rawvideo -pix_fmt bgr24 -s 640x480 tcp://localhost:5000
```

**Option 3: UDP (for real-time with GoPro)**

```bash
# Terminal 1: Python reads UDP (via OpenCV)
python stream_autopilot.py --camera "udp://@:5000"

# Terminal 2: FFmpeg streams UDP
ffmpeg -i "udp://@:8554" -f mpegts "udp://localhost:5000"
```

### Full Pipeline Example

```bash
# Complete pipeline: FFmpeg → RetinaNet → LLM → Robot
# 1. FFmpeg reads GoPro stream
# 2. Autopilot processes frames
# 3. Commands sent to robot via UDP

ffmpeg -i "udp://@:8554" -f rawvideo -pix_fmt bgr24 - 2>/dev/null | \
  python stream_autopilot.py --camera pipe:0 --skip-frames 3 2>&1 | \
  tee commands.log | \
  grep -oE '\{[^}]+\}' | \
  while read cmd; do echo "$cmd" | nc -u -w0 robot.local 9000; done
```

### FFmpeg Common Options

| Option | Description |
|--------|-------------|
| `-f rawvideo` | Output raw video frames |
| `-pix_fmt bgr24` | BGR format (OpenCV compatible) |
| `-fflags nobuffer` | Reduce latency |
| `-flags low_delay` | Low delay mode |
| `-hwaccel cuda` | NVIDIA hardware decoding |
| `-hwaccel videotoolbox` | macOS hardware decoding |

## 🔌 GoPro Integration

### USB Webcam Mode (Recommended)

1. On GoPro: Preferences → Connections → USB Connection → **GoPro Connect**
2. Connect USB cable
3. Run: `python autopilot.py --camera 1`

### WiFi RTSP (GoPro Labs)

1. Install GoPro Labs firmware
2. Enable RTSP streaming
3. Connect to GoPro WiFi
4. Run: `python autopilot.py --camera "rtsp://172.20.XXX.XXX:8554/live"`

## 📝 Safety Rules (LLM Prompt)

The LLM follows these safety rules:

1. **Pedestrian nearby** → ALWAYS brake
2. **STOP sign** → Full stop
3. **Red traffic light** → Stop
4. **Stay in lane** → Follow road markings
5. **Unclear situation** → Slow down

## 🐛 Troubleshooting

### "Model not found"

Using pretrained COCO model is fine for testing. For custom training, run `train.py` first.

### "LLM error"

Check `config.py` for selected provider credentials and settings. Test with:
```bash
python llm_controller.py
```

### Slow performance

- Use `--skip-frames 3` or higher
- Increase `--llm-interval`
- Use GPU if available

## 📜 Notes

This project is inspired by the [LearnOpenCV RetinaNet tutorial](https://learnopencv.com/finetuning-retinanet/) and extends it with:
- LLM-based driving command generation
- Optical-flow-based movement context
- Streaming/headless operation mode
- Gradio dashboard with live processing and chat


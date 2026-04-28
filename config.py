import torch

BATCH_SIZE = 4  # Reduced for macOS without GPU
RESIZE_TO = 640  # Image size for training
NUM_EPOCHS = 60  # Number of epochs
NUM_WORKERS = 2  # Reduced for macOS

# Determine device: MPS for Apple Silicon, otherwise CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# Inference resolution: smaller = faster on CPU (RetinaNet)
# Auto: 416 on CPU, 640 on GPU/MPS. Or set explicitly: 416, 480, 640
INFERENCE_RESIZE = 320 if str(DEVICE) == "cpu" else 416

# Data directories
TRAIN_DIR = "data/train"
VALID_DIR = "data/valid"

# ============================================
# COCO classes (80 classes + background)
# RetinaNet is pretrained on this dataset
# ============================================
COCO_CLASSES = [
    "__background__",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Use COCO classes (for pretrained model)
CLASSES = COCO_CLASSES

# Important classes for autopilot (COCO indices)
AUTOPILOT_CLASSES = {
    1: "person",        # pedestrian
    2: "bicycle",       # bicycle  
    3: "car",           # car
    4: "motorcycle",    # motorcycle
    6: "bus",           # bus
    8: "truck",         # truck
    10: "traffic light", # traffic light
    13: "stop sign",    # stop sign
}

NUM_CLASSES = len(CLASSES)

# Whether to visualize transformed images
VISUALIZE_TRANSFORMED_IMAGES = False

# Directory for saving model and plots
OUT_DIR = "outputs"

# ============================================
# LLM Provider Settings
# ============================================
# Options: "azure", "ollama", "openai"
LLM_PROVIDER = "azure"

# ============================================
# Azure OpenAI Settings (if LLM_PROVIDER = "azure")
# ============================================
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"  # deployment name
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

# ============================================
# Ollama Settings (if LLM_PROVIDER = "ollama")
# ============================================
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:3b"  # Options: llama3.2:3b, mistral, phi3:mini, qwen2.5:3b

# ============================================
# OpenAI Settings (if LLM_PROVIDER = "openai")
# ============================================
OPENAI_API_KEY = ""  # Your OpenAI API key
OPENAI_MODEL = "gpt-4o-mini"

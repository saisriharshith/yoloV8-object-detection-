"""Application configuration settings for YOLOv8 Real-Time Analytics."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Ensure runtime directories exist
for directory in [MODELS_DIR, DATA_DIR, SAMPLES_DIR, INPUTS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Standard COCO 80 class labels
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

@dataclass
class ModelConfig:
    """YOLOv8 Model Configuration."""
    model_name: str = "yolov8n.pt"  # nano model by default for M2 Mac 8GB
    device: str = "cpu"             # cpu or mps (Metal Performance Shaders)
    conf_threshold: float = 0.35    # Confidence threshold
    iou_threshold: float = 0.45     # NMS IoU threshold
    img_size: int = 640             # Inference resolution (width/height)
    half_precision: bool = False    # FP16 (usually for GPU/MPS if supported)
    tracker_type: str = "bytetrack" # bytetrack or botsort
    enabled_classes: Optional[List[int]] = None # Filter classes (None = all)

@dataclass
class UIConfig:
    """User Interface and Display Configuration."""
    show_boxes: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_track_id: bool = True
    show_fps_hud: bool = True
    show_counts_hud: bool = True
    box_thickness: int = 2
    font_scale: float = 0.6
    line_thickness: int = 2

@dataclass
class ExportConfig:
    """Video export parameters."""
    codec: str = "mp4v"             # Fallback codec compatible with OpenCV
    fps: float = 25.0
    output_dir: Path = OUTPUTS_DIR

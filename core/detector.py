"""YOLOv8 Detection Engine supporting configurable models, confidence, IoU, and class filters."""

import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

# Configure module logger
logger = logging.getLogger("detector")

class ObjectDetector:
    """Encapsulates Ultralytics YOLOv8 inference with memory and device management."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        enabled_classes: Optional[List[int]] = None
    ):
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.enabled_classes = enabled_classes
        
        self.model = None
        self.class_names: Dict[int, str] = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLOv8 model weights safely, handling device fallback."""
        from ultralytics import YOLO
        import torch

        # Validate requested device
        if self.device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS device requested but not available. Falling back to CPU.")
                self.device = "cpu"

        logger.info(f"Loading YOLOv8 model '{self.model_name}' on device '{self.device}'...")
        try:
            self.model = YOLO(self.model_name)
            # Fetch model class names
            self.class_names = self.model.names if hasattr(self.model, "names") else {}
            logger.info(f"Model loaded successfully with {len(self.class_names)} classes.")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Could not load YOLO model '{self.model_name}': {e}")

    def update_config(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        img_size: Optional[int] = None,
        enabled_classes: Optional[List[int]] = None
    ) -> None:
        """Dynamically update inference hyperparameters without reloading model weights."""
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
        if iou_threshold is not None:
            self.iou_threshold = float(iou_threshold)
        if img_size is not None:
            self.img_size = int(img_size)
        if enabled_classes is not None:
            self.enabled_classes = enabled_classes

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Execute object detection on a single frame.
        
        Returns:
            Tuple of (detections_list, inference_latency_ms)
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
            
        if frame is None or frame.size == 0:
            return [], 0.0

        t0 = time.perf_counter()
        
        try:
            # Perform inference
            results = self.model.predict(
                source=frame,
                device=self.device,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                classes=self.enabled_classes,
                verbose=False
            )
        except Exception as e:
            # If MPS fails during runtime, attempt fallback to CPU once
            if self.device == "mps":
                logger.warning(f"MPS inference failed ({e}). Falling back to CPU.")
                self.device = "cpu"
                results = self.model.predict(
                    source=frame,
                    device="cpu",
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    classes=self.enabled_classes,
                    verbose=False
                )
            else:
                logger.error(f"Inference error: {e}")
                raise e

        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(xyxy, confs, cls_ids):
                    detections.append({
                        "box": box.tolist(),
                        "class_id": int(cls_id),
                        "class_name": self.class_names.get(int(cls_id), f"class_{cls_id}"),
                        "confidence": float(conf),
                        "track_id": None
                    })

        return detections, latency_ms

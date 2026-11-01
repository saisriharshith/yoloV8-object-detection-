"""Object Tracking Engine utilizing YOLOv8 ByteTrack / BoT-SORT integration.
Inspired by Ultralytics and Muhammad Moin Faisal's DeepSORT tracking implementations:
Features persistent unique IDs, centroid trajectory history, motion direction, and boundary line crossing.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple, Set, Any
import numpy as np

logger = logging.getLogger("tracker")

class ObjectTracker:
    """Wraps YOLOv8 tracking capability with persistent ID management, trajectory history, and line crossing."""

    def __init__(
        self,
        model_name: str = "models/yolov8n.pt",
        device: str = "cpu",
        tracker_type: str = "bytetrack",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        enabled_classes: Optional[List[int]] = None,
        line_position_ratio: float = 0.55  # Position of counting line (fraction of frame height)
    ):
        self.model_name = model_name
        self.device = device
        self.tracker_type = f"{tracker_type}.yaml" if not tracker_type.endswith(".yaml") else tracker_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.enabled_classes = enabled_classes
        self.line_position_ratio = line_position_ratio

        self.model = None
        self.class_names: Dict[int, str] = {}
        # Track history stores centroids: track_id -> [(x, y), ...]
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        
        # Line crossing registers
        self.crossed_in_ids: Set[int] = set()
        self.crossed_out_ids: Set[int] = set()

        self._load_model()

    def _load_model(self) -> None:
        """Initialize YOLO model."""
        from ultralytics import YOLO
        import torch

        if self.device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS device requested but not available. Falling back to CPU.")
                self.device = "cpu"

        logger.info(f"Loading YOLOv8 tracker model '{self.model_name}' on '{self.device}'...")
        try:
            self.model = YOLO(self.model_name)
            self.class_names = self.model.names if hasattr(self.model, "names") else {}
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Could not load tracker model '{self.model_name}': {e}")

    def update_config(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        img_size: Optional[int] = None,
        enabled_classes: Optional[List[int]] = None,
        tracker_type: Optional[str] = None
    ) -> None:
        """Update tracker runtime configuration."""
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
        if iou_threshold is not None:
            self.iou_threshold = float(iou_threshold)
        if img_size is not None:
            self.img_size = int(img_size)
        if enabled_classes is not None:
            self.enabled_classes = enabled_classes
        if tracker_type is not None:
            self.tracker_type = f"{tracker_type}.yaml" if not tracker_type.endswith(".yaml") else tracker_type

    def track(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Execute object tracking on a video frame.
        Maintains state across consecutive frames using persist=True.
        """
        if self.model is None or frame is None or frame.size == 0:
            return [], 0.0

        h, w = frame.shape[:2]
        line_y = int(h * self.line_position_ratio)
        t0 = time.perf_counter()

        try:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_type,
                device=self.device,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                classes=self.enabled_classes,
                verbose=False
            )
        except Exception as e:
            if self.device == "mps":
                logger.warning(f"MPS tracking failed ({e}). Falling back to CPU.")
                self.device = "cpu"
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_type,
                    device="cpu",
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    classes=self.enabled_classes,
                    verbose=False
                )
            else:
                logger.error(f"Tracking error: {e}")
                raise e

        latency_ms = (time.perf_counter() - t0) * 1000.0

        tracked_detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                # Track IDs might be None if an object is newly detected and not yet confirmed
                track_ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else [None] * len(boxes)
                )

                for box, conf, cls_id, trk_id in zip(xyxy, confs, cls_ids, track_ids):
                    t_id = int(trk_id) if trk_id is not None else None
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)

                    direction = None
                    if t_id is not None:
                        # Append to history
                        if t_id not in self.track_history:
                            self.track_history[t_id] = []
                        self.track_history[t_id].append((cx, cy))
                        if len(self.track_history[t_id]) > 40:
                            self.track_history[t_id].pop(0)

                        # Calculate motion direction from trajectory history
                        hist = self.track_history[t_id]
                        if len(hist) >= 5:
                            prev_x, prev_y = hist[-5]
                            dx = cx - prev_x
                            dy = cy - prev_y
                            if abs(dy) > abs(dx) and abs(dy) > 6:
                                direction = "Down" if dy > 0 else "Up"
                            elif abs(dx) > 6:
                                direction = "Right" if dx > 0 else "Left"

                        # Check line crossing (top to bottom = IN, bottom to top = OUT)
                        if len(hist) >= 2:
                            _, prev_cy = hist[-2]
                            if prev_cy < line_y and cy >= line_y:
                                if t_id not in self.crossed_in_ids:
                                    self.crossed_in_ids.add(t_id)
                            elif prev_cy > line_y and cy <= line_y:
                                if t_id not in self.crossed_out_ids:
                                    self.crossed_out_ids.add(t_id)

                    det = {
                        "box": box.tolist(),
                        "class_id": int(cls_id),
                        "class_name": self.class_names.get(int(cls_id), f"class_{cls_id}"),
                        "confidence": float(conf),
                        "track_id": t_id,
                        "centroid": (cx, cy),
                        "direction": direction
                    }
                    tracked_detections.append(det)

        return tracked_detections, latency_ms

    def get_crossing_counts(self) -> Dict[str, int]:
        """Return directional counting metrics."""
        return {
            "in": len(self.crossed_in_ids),
            "out": len(self.crossed_out_ids),
            "total": len(self.crossed_in_ids) + len(self.crossed_out_ids)
        }

    def reset(self) -> None:
        """Reset internal tracker state, trajectories, and crossing counters."""
        self.track_history.clear()
        self.crossed_in_ids.clear()
        self.crossed_out_ids.clear()
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = None

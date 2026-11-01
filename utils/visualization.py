"""Visualization utilities for rendering bounding boxes, object class tags, and analytics HUD.
Clean, modern, high-contrast overlay without distracting motion paths or boundary lines.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# High-contrast color palette (BGR format for OpenCV)
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (255, 143, 0), (255, 69, 0), (200, 0, 100), (255, 50, 180),
    (180, 50, 255), (100, 50, 255), (50, 100, 255), (0, 165, 255)
]

def get_color_for_id(track_id: Optional[int], class_id: int) -> Tuple[int, int, int]:
    """Deterministically choose a color based on track ID or class ID."""
    if track_id is not None and track_id >= 0:
        return PALETTE[track_id % len(PALETTE)]
    return PALETTE[class_id % len(PALETTE)]

def draw_hud(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    current_count: int,
    cumulative_unique_count: int,
    device: str,
    class_counts: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """Render clean HUD showing FPS, latency, total count, and class breakdown (cars, bikes, persons)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent dark header banner
    hud_h = 75
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 24, 33), -1)
    alpha = 0.80
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Row 1: Pipeline performance metrics
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, 24), font, 0.60, (0, 255, 128), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (135, 24), font, 0.60, (255, 220, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total In-Frame: {current_count}", (310, 24), font, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Unique Tracked: {cumulative_unique_count}", (520, 24), font, 0.60, (100, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Dev: {device.upper()}", (w - 130, 24), font, 0.60, (200, 200, 200), 2, cv2.LINE_AA)

    # Row 2: Prominent Class Breakdown (Persons, Cars, Bikes, Trucks/Buses)
    if class_counts:
        persons = class_counts.get("person", 0)
        cars = class_counts.get("car", 0)
        bikes = class_counts.get("motorcycle", 0) + class_counts.get("bicycle", 0)
        trucks = class_counts.get("bus", 0) + class_counts.get("truck", 0)

        # Other categories
        handled = {"person", "car", "motorcycle", "bicycle", "bus", "truck"}
        others = {k: v for k, v in class_counts.items() if k not in handled}
        other_str = f" | Others: {sum(others.values())}" if others else ""

        tally_text = f"Persons: {persons}  |  Cars: {cars}  |  Bikes: {bikes}  |  Trucks/Buses: {trucks}{other_str}"
        cv2.putText(frame, tally_text, (16, 56), font, 0.55, (220, 240, 255), 2, cv2.LINE_AA)

    return frame

def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    show_boxes: bool = True,
    show_labels: bool = True,
    show_confidence: bool = True,
    show_track_id: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.55
) -> np.ndarray:
    """Draw clean bounding box rectangles and class names for each object without trails."""
    for det in detections:
        box = det["box"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box)
        class_id = det.get("class_id", 0)
        class_name = det.get("class_name", "object")
        conf = det.get("confidence", 0.0)
        track_id = det.get("track_id", None)

        color = get_color_for_id(track_id, class_id)

        # Draw bounding box rectangle
        if show_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)
            # Corner accents
            accent_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            if accent_len > 4:
                accent_thick = box_thickness + 1
                cv2.line(frame, (x1, y1), (x1 + accent_len, y1), (255, 255, 255), accent_thick)
                cv2.line(frame, (x1, y1), (x1, y1 + accent_len), (255, 255, 255), accent_thick)
                cv2.line(frame, (x2, y1), (x2 - accent_len, y1), (255, 255, 255), accent_thick)
                cv2.line(frame, (x2, y1), (x2, y1 + accent_len), (255, 255, 255), accent_thick)
                cv2.line(frame, (x1, y2), (x1 + accent_len, y2), (255, 255, 255), accent_thick)
                cv2.line(frame, (x1, y2), (x1, y2 - accent_len), (255, 255, 255), accent_thick)
                cv2.line(frame, (x2, y2), (x2 - accent_len, y2), (255, 255, 255), accent_thick)
                cv2.line(frame, (x2, y2), (x2, y2 - accent_len), (255, 255, 255), accent_thick)

        # Build label: e.g. "#1 person 0.88" or "car 0.94"
        label_parts = []
        if show_track_id and track_id is not None:
            label_parts.append(f"#{track_id}")
        if show_labels:
            label_parts.append(class_name)
        if show_confidence:
            label_parts.append(f"{conf:.2f}")

        if label_parts:
            label = " ".join(label_parts)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

            label_y1 = max(0, y1 - th - baseline - 4)
            label_y2 = y1
            if label_y1 < 0:
                label_y1 = y1
                label_y2 = y1 + th + baseline + 4

            # Tag background rectangle
            cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 8, label_y2), color, -1)
            # Text inside tag
            cv2.putText(
                frame,
                label,
                (x1 + 4, label_y2 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return frame

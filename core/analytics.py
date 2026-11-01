"""Analytics Engine for tracking FPS, inference latency, object counts, and timeline telemetry."""

import time
from collections import deque, Counter
from typing import Dict, List, Set, Optional, Any
import pandas as pd

class AnalyticsEngine:
    """Manages real-time statistics, frame rates, latencies, and object tracking counts."""

    def __init__(self, fps_window_size: int = 30):
        self.fps_window_size = fps_window_size
        self.frame_times = deque(maxlen=fps_window_size)
        self.inference_latencies = deque(maxlen=fps_window_size)
        
        # State counters
        self.total_frames_processed = 0
        self.start_time = time.time()
        self.current_fps = 0.0
        self.current_latency_ms = 0.0
        
        # Unique tracking registers (track_id -> class_name)
        self.seen_track_ids: Dict[int, str] = {}
        self.current_frame_counts: Dict[str, int] = {}
        
        # Time-series log for dashboard graphs and export
        self.history: List[Dict[str, Any]] = []

    def start_frame(self) -> float:
        """Mark start of frame processing."""
        return time.perf_counter()

    def update(
        self,
        frame_start_time: float,
        inference_latency_ms: float,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update metrics for the current frame."""
        now = time.perf_counter()
        frame_duration = now - frame_start_time
        
        self.frame_times.append(frame_duration)
        self.inference_latencies.append(inference_latency_ms)
        self.total_frames_processed += 1
        
        # Compute smooth FPS
        if self.frame_times:
            avg_duration = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_duration if avg_duration > 0 else 0.0
        
        # Compute smooth latency
        if self.inference_latencies:
            self.current_latency_ms = sum(self.inference_latencies) / len(self.inference_latencies)
        
        # Current in-frame object counts
        current_classes = [d.get("class_name", "unknown") for d in detections]
        self.current_frame_counts = dict(Counter(current_classes))
        
        # Register new unique track IDs
        for det in detections:
            track_id = det.get("track_id")
            class_name = det.get("class_name", "unknown")
            if track_id is not None and track_id >= 0:
                if track_id not in self.seen_track_ids:
                    self.seen_track_ids[track_id] = class_name

        # Calculate cumulative unique objects per class
        cumulative_counts = dict(Counter(self.seen_track_ids.values()))
        
        # Record time-series entry (sample every frame, or log up to 10,000 frames)
        time_elapsed = round(time.time() - self.start_time, 2)
        record = {
            "frame_idx": self.total_frames_processed,
            "timestamp": time_elapsed,
            "fps": round(self.current_fps, 1),
            "latency_ms": round(inference_latency_ms, 1),
            "current_count": len(detections),
            "unique_tracked_total": len(self.seen_track_ids),
            **{f"curr_{k}": v for k, v in self.current_frame_counts.items()}
        }
        self.history.append(record)

        return {
            "frame_idx": self.total_frames_processed,
            "fps": self.current_fps,
            "latency_ms": self.current_latency_ms,
            "current_count": len(detections),
            "current_classes": self.current_frame_counts,
            "cumulative_unique_total": len(self.seen_track_ids),
            "cumulative_classes": cumulative_counts,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Generate high-level session summary."""
        total_time = time.time() - self.start_time
        avg_fps = self.total_frames_processed / total_time if total_time > 0 else 0.0
        avg_latency = (
            sum(self.inference_latencies) / len(self.inference_latencies)
            if self.inference_latencies else 0.0
        )
        
        cumulative_classes = dict(Counter(self.seen_track_ids.values()))
        
        return {
            "total_frames": self.total_frames_processed,
            "duration_seconds": round(total_time, 2),
            "average_fps": round(avg_fps, 2),
            "average_latency_ms": round(avg_latency, 2),
            "total_unique_objects": len(self.seen_track_ids),
            "unique_by_class": cumulative_classes
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export session metrics history as a Pandas DataFrame."""
        if not self.history:
            return pd.DataFrame(columns=["frame_idx", "timestamp", "fps", "latency_ms", "current_count", "unique_tracked_total"])
        df = pd.DataFrame(self.history)
        df.fillna(0, inplace=True)
        return df

    def reset(self) -> None:
        """Reset all counters and history."""
        self.frame_times.clear()
        self.inference_latencies.clear()
        self.total_frames_processed = 0
        self.start_time = time.time()
        self.current_fps = 0.0
        self.current_latency_ms = 0.0
        self.seen_track_ids.clear()
        self.current_frame_counts.clear()
        self.history.clear()

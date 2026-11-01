"""Video export utility with proper codec selection and resource cleanup."""

import cv2
import os
import time
from pathlib import Path
from typing import Optional, Tuple

class VideoExporter:
    """Manages saving processed video frames with OpenCV."""

    def __init__(
        self,
        output_path: Path,
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = "mp4v"
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps if fps > 0 else 25.0
        self.frame_size = frame_size  # (width, height)
        self.codec = codec
        self.writer: Optional[cv2.VideoWriter] = None
        self.frames_written = 0
        self.start_time = time.time()
        self._initialize_writer()

    def _initialize_writer(self) -> None:
        """Initialize OpenCV VideoWriter with fallback codec options."""
        # Standard OpenCV fourcc codecs
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            self.frame_size
        )

        if not self.writer.isOpened():
            # Try fallback codecs: 'avc1' or 'XVID'
            for fallback in ["avc1", "XVID", "MJPG"]:
                fourcc = cv2.VideoWriter_fourcc(*fallback)
                self.writer = cv2.VideoWriter(
                    str(self.output_path),
                    fourcc,
                    self.fps,
                    self.frame_size
                )
                if self.writer.isOpened():
                    self.codec = fallback
                    break

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.output_path}")

    def write_frame(self, frame) -> bool:
        """Write an individual frame, resizing if dimensions differ."""
        if self.writer is None or not self.writer.isOpened():
            return False

        h, w = frame.shape[:2]
        if (w, h) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)

        self.writer.write(frame)
        self.frames_written += 1
        return True

    def release(self) -> dict:
        """Close video writer and return summary stats."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        elapsed = time.time() - self.start_time
        file_size_mb = 0.0
        if self.output_path.exists():
            file_size_mb = round(self.output_path.stat().st_size / (1024 * 1024), 2)

        return {
            "output_path": str(self.output_path),
            "frames_written": self.frames_written,
            "duration_sec": round(self.frames_written / self.fps, 2) if self.fps > 0 else 0,
            "elapsed_real_sec": round(elapsed, 2),
            "file_size_mb": file_size_mb,
            "codec": self.codec
        }

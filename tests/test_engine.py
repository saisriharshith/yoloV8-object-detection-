"""Unit and integration tests for YOLOv8 Detection, Tracking, and Analytics Engines."""

import sys
import os
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.detector import ObjectDetector
from core.tracker import ObjectTracker
from core.analytics import AnalyticsEngine
from utils.visualization import draw_detections, draw_hud
from utils.video_exporter import VideoExporter

class TestVisionPipeline(unittest.TestCase):
    """Test suite for the core Computer Vision pipeline."""

    @classmethod
    def setUpClass(cls):
        """Prepare sample image for testing."""
        cls.img_h, cls.img_w = 480, 640
        # Create a synthetic image with distinct colored blocks simulating objects
        cls.test_image = np.zeros((cls.img_h, cls.img_w, 3), dtype=np.uint8)
        # Background
        cls.test_image[:] = (40, 40, 40)
        # Add simulated rectangle
        cv2.rectangle(cls.test_image, (100, 100), (300, 350), (200, 200, 200), -1)
        cv2.putText(cls.test_image, "Test Subject", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def test_01_analytics_engine(self):
        """Verify analytics calculation, FPS, and unique tracking logic."""
        analytics = AnalyticsEngine(fps_window_size=10)
        t_start = analytics.start_frame()
        
        # Simulated detections in frame 1
        dets_frame1 = [
            {"box": [10, 10, 50, 50], "class_id": 0, "class_name": "person", "confidence": 0.85, "track_id": 1},
            {"box": [60, 60, 120, 120], "class_id": 2, "class_name": "car", "confidence": 0.90, "track_id": 2}
        ]
        metrics1 = analytics.update(t_start, inference_latency_ms=15.0, detections=dets_frame1)
        
        self.assertEqual(metrics1["current_count"], 2)
        self.assertEqual(metrics1["cumulative_unique_total"], 2)
        self.assertIn("person", metrics1["current_classes"])
        self.assertIn("car", metrics1["current_classes"])

        # Simulated frame 2: person #1 stays, car #2 leaves, new bicycle #3 appears
        t_start2 = analytics.start_frame()
        dets_frame2 = [
            {"box": [12, 12, 52, 52], "class_id": 0, "class_name": "person", "confidence": 0.87, "track_id": 1},
            {"box": [200, 100, 250, 180], "class_id": 1, "class_name": "bicycle", "confidence": 0.78, "track_id": 3}
        ]
        metrics2 = analytics.update(t_start2, inference_latency_ms=14.5, detections=dets_frame2)
        
        self.assertEqual(metrics2["current_count"], 2)
        # Cumulative unique should now be 3 (person #1, car #2, bicycle #3) - no duplicate count for person #1!
        self.assertEqual(metrics2["cumulative_unique_total"], 3)
        self.assertEqual(metrics2["cumulative_classes"]["person"], 1)
        self.assertEqual(metrics2["cumulative_classes"]["car"], 1)
        self.assertEqual(metrics2["cumulative_classes"]["bicycle"], 1)

        # Verify DataFrame export
        df = analytics.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn("fps", df.columns)
        self.assertIn("latency_ms", df.columns)

    def test_02_visualization(self):
        """Verify bounding box and HUD rendering."""
        frame = self.test_image.copy()
        dets = [
            {"box": [50, 50, 200, 200], "class_id": 0, "class_name": "person", "confidence": 0.92, "track_id": 1}
        ]
        rendered = draw_detections(frame, dets, show_boxes=True, show_labels=True, show_track_id=True)
        self.assertEqual(rendered.shape, (self.img_h, self.img_w, 3))
        
        # Test HUD
        hud_frame = draw_hud(
            rendered,
            fps=32.5,
            latency_ms=18.2,
            current_count=1,
            cumulative_unique_count=1,
            device="cpu",
            class_counts={"person": 1}
        )
        self.assertEqual(hud_frame.shape, (self.img_h, self.img_w, 3))

    def test_03_video_exporter(self):
        """Verify video writer creates a valid video file."""
        out_path = PROJECT_ROOT / "data" / "outputs" / "test_output.mp4"
        exporter = VideoExporter(output_path=out_path, fps=20.0, frame_size=(self.img_w, self.img_h), codec="mp4v")
        
        for _ in range(5):
            exporter.write_frame(self.test_image)
        
        summary = exporter.release()
        self.assertTrue(out_path.exists())
        self.assertEqual(summary["frames_written"], 5)
        self.assertGreater(summary["file_size_mb"], 0.0)
        
        # Cleanup test output
        if out_path.exists():
            out_path.unlink()

if __name__ == "__main__":
    unittest.main()

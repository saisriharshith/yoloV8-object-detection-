"""Comprehensive Benchmarking Script for YOLOv8 on macOS Apple M2.
Tests CPU vs MPS, image detection, video tracking, and exports real performance metrics.
"""

import sys
import os
import time
import cv2
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.detector import ObjectDetector
from core.tracker import ObjectTracker
from core.video_stream import VideoStream
from core.analytics import AnalyticsEngine
from utils.visualization import draw_detections, draw_hud
from utils.video_exporter import VideoExporter
from utils.system_info import get_system_metrics

def benchmark_image_detection():
    """Benchmark image inference on both CPU and MPS devices."""
    print("\n" + "="*60)
    print("STEP 1: BENCHMARKING IMAGE DETECTION (CPU vs MPS)")
    print("="*60)

    image_path = PROJECT_ROOT / "data" / "samples" / "bus.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing {image_path}")

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    print(f"Loaded input image: {image_path.name} ({w}x{h})")

    devices_to_test = ["cpu"]
    sys_metrics = get_system_metrics()
    if sys_metrics["mps_available"]:
        devices_to_test.append("mps")

    results = []

    for dev in devices_to_test:
        print(f"\n--- Testing on Device: {dev.upper()} ---")
        detector = ObjectDetector(
            model_name=str(PROJECT_ROOT / "models" / "yolov8n.pt"),
            device=dev,
            conf_threshold=0.35,
            iou_threshold=0.45,
            img_size=640
        )

        # Warm-up run
        detector.detect(img)

        # Timed runs
        iterations = 15
        latencies = []
        dets = []
        for _ in range(iterations):
            d, lat = detector.detect(img)
            latencies.append(lat)
            dets = d

        avg_lat = sum(latencies) / len(latencies)
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0
        detected_classes = [d["class_name"] for d in dets]
        
        print(f"Device: {dev.upper()}")
        print(f"Average Inference Latency: {avg_lat:.2f} ms")
        print(f"Calculated Inference FPS:  {fps:.1f} FPS")
        print(f"Objects Detected ({len(dets)}): {detected_classes}")
        for d in dets:
            print(f"  - {d['class_name']} ({d['confidence']:.2f}) at {list(map(int, d['box']))}")

        results.append({
            "device": dev,
            "avg_latency_ms": round(avg_lat, 2),
            "fps": round(fps, 1),
            "objects_detected": len(dets),
            "sample_detections": ", ".join(detected_classes)
        })

        # Save annotated image output
        annotated = draw_detections(img.copy(), dets, show_boxes=True, show_labels=True, show_confidence=True)
        out_img_path = PROJECT_ROOT / "data" / "outputs" / f"bus_detected_{dev}.jpg"
        cv2.imwrite(str(out_img_path), annotated)
        print(f"Saved annotated image: {out_img_path}")

    return results

def benchmark_video_tracking(max_frames: int = 150):
    """Benchmark video processing with ByteTrack tracking and export."""
    print("\n" + "="*60)
    print("STEP 2: BENCHMARKING VIDEO PROCESSING & OBJECT TRACKING")
    print("="*60)

    video_path = PROJECT_ROOT / "data" / "samples" / "sample_traffic.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing {video_path}")

    # Use CPU by default as primary stable edge configuration
    tracker = ObjectTracker(
        model_name=str(PROJECT_ROOT / "models" / "yolov8n.pt"),
        device="cpu",
        tracker_type="bytetrack",
        conf_threshold=0.35,
        iou_threshold=0.45,
        img_size=640
    )

    analytics = AnalyticsEngine(fps_window_size=30)
    stream = VideoStream(video_path)

    out_video_path = PROJECT_ROOT / "data" / "outputs" / "sample_traffic_tracked.mp4"
    exporter = VideoExporter(
        output_path=out_video_path,
        fps=stream.fps,
        frame_size=(stream.width, stream.height),
        codec="mp4v"
    )

    print(f"Source: {video_path.name} ({stream.width}x{stream.height} @ {stream.fps:.1f} FPS, total frames: {stream.total_frames})")
    print(f"Processing up to {max_frames} frames with real-time tracking...")

    frame_count = 0
    start_total_time = time.time()

    while frame_count < max_frames:
        t_frame_start = analytics.start_frame()
        ret, frame = stream.read()
        if not ret:
            break

        # Object tracking
        tracks, lat_ms = tracker.track(frame)

        # Update telemetry
        metrics = analytics.update(t_frame_start, lat_ms, tracks)

        # Draw annotations and HUD
        annotated = draw_detections(frame, tracks, show_boxes=True, show_labels=True, show_confidence=True, show_track_id=True)
        annotated = draw_hud(
            annotated,
            fps=metrics["fps"],
            latency_ms=metrics["latency_ms"],
            current_count=metrics["current_count"],
            cumulative_unique_count=metrics["cumulative_unique_total"],
            device="cpu",
            class_counts=metrics["current_classes"]
        )

        exporter.write_frame(annotated)
        frame_count += 1

        if frame_count % 30 == 0 or frame_count == max_frames:
            print(
                f"Frame [{frame_count}/{max_frames}] | "
                f"FPS: {metrics['fps']:.1f} | "
                f"Latency: {metrics['latency_ms']:.1f} ms | "
                f"In-frame: {metrics['current_count']} | "
                f"Unique Total: {metrics['cumulative_unique_total']}"
            )

    stream.release()
    export_summary = exporter.release()
    total_time = time.time() - start_total_time

    summary = analytics.get_summary()
    print("\n--- Video Tracking Benchmark Summary ---")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Wall-Clock Processing Time: {total_time:.2f} s")
    print(f"Overall Processing Speed: {frame_count / total_time:.1f} FPS")
    print(f"Average Model Inference Latency: {summary['average_latency_ms']:.2f} ms")
    print(f"Total Unique Objects Tracked: {summary['total_unique_objects']}")
    print(f"Unique Tracked Breakdown: {summary['unique_by_class']}")
    print(f"Saved Tracked Output Video: {export_summary['output_path']} ({export_summary['file_size_mb']} MB)")

    # Export metrics timeline to CSV
    csv_path = PROJECT_ROOT / "data" / "outputs" / "tracking_telemetry.csv"
    analytics.to_dataframe().to_csv(csv_path, index=False)
    print(f"Saved Telemetry Dataset: {csv_path}")

    return {
        "frames_processed": frame_count,
        "processing_fps": round(frame_count / total_time, 1),
        "avg_latency_ms": summary["average_latency_ms"],
        "unique_objects": summary["total_unique_objects"],
        "unique_breakdown": summary["unique_by_class"],
        "video_output": str(export_summary["output_path"]),
        "video_size_mb": export_summary["file_size_mb"]
    }

if __name__ == "__main__":
    print("SYSTEM ENVIRONMENT:")
    metrics = get_system_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    img_res = benchmark_image_detection()
    vid_res = benchmark_video_tracking(max_frames=120)
    print("\nBENCHMARK COMPLETE!")

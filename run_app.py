#!/usr/bin/env python3
"""Unified Launcher for YOLOv8 Edge Vision Analytics.
Supports both the High-Performance Web Dashboard (HTML5, CSS3, JavaScript, FastAPI)
and the Headless/CLI streaming mode for embedded edge devices.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def launch_web_server(host: str = "0.0.0.0", port: int = 8000):
    """Launch the FastAPI + Uvicorn web dashboard with HTML5/JS frontend."""
    import uvicorn
    print("=" * 65)
    print("YOLOv8 Edge Video Analytics Platform (FastAPI + HTML5/JS)")
    print(f"Server starting on http://localhost:{port} (Local network: http://{host}:{port})")
    print("Direct low-latency MJPEG streaming and REST APIs active.")
    print("=" * 65)
    
    # Run uvicorn server directly with our app
    uvicorn.run(
        "web.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

def run_cli_stream(source, model_path, device, conf, iou, track, export_path):
    """Run real-time vision pipeline directly in CLI/OpenCV mode."""
    import cv2
    from core.detector import ObjectDetector
    from core.tracker import ObjectTracker
    from core.video_stream import VideoStream
    from core.analytics import AnalyticsEngine
    from utils.visualization import draw_detections, draw_hud
    from utils.video_exporter import VideoExporter

    print("=" * 60)
    print("YOLOv8 Edge Vision Pipeline (CLI Mode)")
    print(f"Source: {source} | Device: {device} | Model: {model_path} | Tracking: {track}")
    print("=" * 60)

    # Convert numeric camera index if applicable
    src = int(source) if str(source).isdigit() else source
    stream = VideoStream(src)
    analytics = AnalyticsEngine(fps_window_size=20)

    if track:
        engine = ObjectTracker(
            model_name=model_path,
            device=device,
            conf_threshold=conf,
            iou_threshold=iou
        )
    else:
        engine = ObjectDetector(
            model_name=model_path,
            device=device,
            conf_threshold=conf,
            iou_threshold=iou
        )

    exporter = None
    if export_path:
        exporter = VideoExporter(
            output_path=Path(export_path),
            fps=stream.fps,
            frame_size=(stream.width, stream.height)
        )
        print(f"Exporting processed output to: {export_path}")

    frame_count = 0
    print("Streaming started. Press Ctrl+C in terminal to stop.")

    try:
        while True:
            t_start = analytics.start_frame()
            ret, frame = stream.read()
            if not ret:
                print("End of stream or no frame received.")
                break

            if track:
                dets, lat_ms = engine.track(frame)
            else:
                dets, lat_ms = engine.detect(frame)

            metrics = analytics.update(t_start, lat_ms, dets)

            annotated = draw_detections(
                frame,
                dets,
                show_boxes=True,
                show_labels=True,
                show_confidence=True,
                show_track_id=track
            )
            annotated = draw_hud(
                annotated,
                fps=metrics["fps"],
                latency_ms=metrics["latency_ms"],
                current_count=metrics["current_count"],
                cumulative_unique_count=metrics["cumulative_unique_total"],
                device=device,
                class_counts=metrics["current_classes"]
            )

            if exporter:
                exporter.write_frame(annotated)

            frame_count += 1
            if frame_count % 30 == 0:
                print(
                    f"Frame {frame_count:04d} | "
                    f"FPS: {metrics['fps']:4.1f} | "
                    f"Latency: {metrics['latency_ms']:5.1f} ms | "
                    f"In-Frame: {metrics['current_count']:2d} | "
                    f"Unique Total: {metrics['cumulative_unique_total']:3d}"
                )

    except KeyboardInterrupt:
        print("\nStreaming interrupted by user.")
    finally:
        stream.release()
        if exporter:
            meta = exporter.release()
            print(f"Saved video: {meta['output_path']} ({meta['file_size_mb']} MB)")
        summary = analytics.get_summary()
        print("\n--- Pipeline Session Summary ---")
        print(f"Total Frames: {summary['total_frames']}")
        print(f"Average FPS:  {summary['average_fps']:.1f}")
        print(f"Unique Tracked Objects: {summary['total_unique_objects']}")
        print(f"Class Breakdown: {summary['unique_by_class']}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Real-Time Edge Analytics")
    parser.add_argument("--cli", action="store_true", help="Run in CLI headless mode instead of web server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind web server")
    parser.add_argument("--port", type=int, default=8000, help="Port for web dashboard")
    parser.add_argument("--source", type=str, default="data/samples/sample_traffic.mp4", help="Video file or camera index (e.g. 0)")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"], help="Inference device")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--track", action="store_true", default=True, help="Enable ByteTrack tracking")
    parser.add_argument("--export", type=str, default="", help="Path to export processed MP4 video")
    
    args = parser.parse_args()

    if args.cli:
        run_cli_stream(
            source=args.source,
            model_path=args.model,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            track=args.track,
            export_path=args.export if args.export else None
        )
    else:
        launch_web_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()

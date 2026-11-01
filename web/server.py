"""FastAPI Web Server for YOLOv8 Edge Video Analytics.
Provides REST APIs and ultra-low-latency MJPEG video streaming for HTML5/JS frontend.
"""

import os
import sys
import time
import json
import base64
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.detector import ObjectDetector
from core.tracker import ObjectTracker
from core.video_stream import VideoStream, is_youtube_url, resolve_youtube_stream_url
from core.analytics import AnalyticsEngine
from utils.visualization import draw_detections, draw_hud
from utils.video_exporter import VideoExporter
from utils.system_info import get_system_metrics

from config.settings import MODELS_DIR, SAMPLES_DIR, INPUTS_DIR, OUTPUTS_DIR, COCO_CLASSES

app = FastAPI(title="YOLOv8 Edge Video Analytics", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Application State
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.model_name = str(PROJECT_ROOT / "models" / "yolov8n.pt")
        self.device = "cpu"
        self.conf_thresh = 0.35
        self.iou_thresh = 0.45
        self.img_size = 640
        self.enabled_classes = None
        self.tracker_type = "bytetrack"
        self.show_trails = True
        self.show_counting_line = True
        self.record_stream = False
        
        # Stream state
        self.is_streaming = False
        self.stream_source = 0
        self.stream_mode = "track" # "track" or "detect"
        self.video_stream: Optional[VideoStream] = None
        self.detector: Optional[ObjectDetector] = None
        self.tracker: Optional[ObjectTracker] = None
        self.analytics = AnalyticsEngine(fps_window_size=20)
        self.latest_frame_jpeg: Optional[bytes] = None
        self.current_frame_idx: int = 0
        self.total_stream_frames: int = 0
        self.video_exporter: Optional[VideoExporter] = None
        self.last_exported_video_url: Optional[str] = None
        self.last_exported_csv_url: Optional[str] = None
        self.export_completed: bool = False
        self.latest_metrics: Dict[str, Any] = {
            "fps": 0.0,
            "latency_ms": 0.0,
            "current_count": 0,
            "cumulative_unique_total": 0,
            "categories": {"persons": 0, "cars": 0, "bikes": 0, "trucks": 0},
            "current_classes": {},
            "cumulative_classes": {}
        }
        self.stream_thread: Optional[threading.Thread] = None

state = AppState()



# Mount static assets
STATIC_DIR = PROJECT_ROOT / "web" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Pydantic Request Models
class SettingsUpdate(BaseModel):
    model_name: Optional[str] = None
    device: Optional[str] = None
    conf_thresh: Optional[float] = None
    iou_thresh: Optional[float] = None
    img_size: Optional[int] = None
    enabled_classes: Optional[List[int]] = None
    tracker_type: Optional[str] = None
    show_trails: Optional[bool] = None
    show_counting_line: Optional[bool] = None


class StreamControl(BaseModel):
    action: str  # "start" or "stop"
    source: Optional[str] = "0"
    mode: Optional[str] = "track"
    record: Optional[bool] = False


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve single-page HTML5 dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.get("/app.js")
async def serve_app_js():
    return FileResponse(STATIC_DIR / "js" / "app.js", media_type="application/javascript")

@app.get("/style.css")
async def serve_style_css():
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css")


@app.get("/api/system")
async def get_system():
    """Return hardware specs, M2 processor metrics, and MPS readiness."""
    metrics = get_system_metrics()
    return JSONResponse(metrics)

@app.get("/api/settings")
async def get_settings():
    """Return current inference settings and available models."""
    available_models = [m.name for m in MODELS_DIR.glob("*.pt")]
    return JSONResponse({
        "model_name": Path(state.model_name).name,
        "available_models": available_models,
        "device": state.device,
        "conf_thresh": state.conf_thresh,
        "iou_thresh": state.iou_thresh,
        "img_size": state.img_size,
        "enabled_classes": state.enabled_classes,
        "tracker_type": state.tracker_type,
        "show_trails": state.show_trails,
        "show_counting_line": state.show_counting_line,
        "all_classes": COCO_CLASSES
    })

@app.post("/api/settings")
async def update_settings(update: SettingsUpdate):
    """Update runtime hyperparameters without restarting the server."""
    with state.lock:
        if update.model_name:
            full_path = MODELS_DIR / update.model_name
            if full_path.exists():
                state.model_name = str(full_path)
        if update.device in ["cpu", "mps"]:
            state.device = update.device
        if update.conf_thresh is not None:
            state.conf_thresh = float(update.conf_thresh)
        if update.iou_thresh is not None:
            state.iou_thresh = float(update.iou_thresh)
        if update.img_size is not None:
            state.img_size = int(update.img_size)
        state.enabled_classes = update.enabled_classes
        if update.tracker_type:
            state.tracker_type = update.tracker_type
        if update.show_trails is not None:
            state.show_trails = update.show_trails
        if update.show_counting_line is not None:
            state.show_counting_line = update.show_counting_line


        # Reconfigure active engines if running
        if state.detector:
            state.detector.update_config(
                conf_threshold=state.conf_thresh,
                iou_threshold=state.iou_thresh,
                img_size=state.img_size,
                enabled_classes=state.enabled_classes
            )
        if state.tracker:
            state.tracker.update_config(
                conf_threshold=state.conf_thresh,
                iou_threshold=state.iou_thresh,
                img_size=state.img_size,
                enabled_classes=state.enabled_classes,
                tracker_type=state.tracker_type
            )
    return JSONResponse({"status": "updated", "settings": update.dict()})

@app.get("/api/samples")
async def list_samples():
    """List available sample videos and images."""
    videos = [v.name for v in SAMPLES_DIR.glob("*.mp4")] + [v.name for v in SAMPLES_DIR.glob("*.avi")]
    images = [i.name for i in SAMPLES_DIR.glob("*.jpg")] + [i.name for i in SAMPLES_DIR.glob("*.png")]
    outputs = [o.name for o in OUTPUTS_DIR.glob("*.*")]
    return JSONResponse({
        "sample_videos": videos,
        "sample_images": images,
        "exported_outputs": outputs
    })

# Stream worker loop
def _stream_worker():
    """Background thread continuously acquiring, predicting, and encoding frames."""
    exporter = None
    out_video_path = None
    out_csv_path = None

    try:
        src = int(state.stream_source) if str(state.stream_source).isdigit() else state.stream_source
        stream = VideoStream(src)
        state.video_stream = stream
        state.total_stream_frames = stream.total_frames if not stream.is_camera else 0
        state.current_frame_idx = 0
        state.export_completed = False

        # Initialize VideoExporter if recording was requested
        if state.record_stream:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if is_youtube_url(str(state.stream_source)):
                src_stem = "youtube_live"
            elif stream.is_camera:
                src_stem = "camera"
            else:
                src_stem = Path(str(src)).stem
            out_video_path = OUTPUTS_DIR / f"{src_stem}_live_{timestamp}.mp4"
            out_csv_path = OUTPUTS_DIR / f"{src_stem}_live_{timestamp}_telemetry.csv"
            exporter = VideoExporter(out_video_path, fps=stream.fps, frame_size=(stream.width, stream.height), codec="mp4v")
            state.video_exporter = exporter

        # Initialize engine
        if state.stream_mode == "track":
            engine = ObjectTracker(
                model_name=state.model_name,
                device=state.device,
                tracker_type=state.tracker_type,
                conf_threshold=state.conf_thresh,
                iou_threshold=state.iou_thresh,
                img_size=state.img_size,
                enabled_classes=state.enabled_classes
            )
            state.tracker = engine
        else:
            engine = ObjectDetector(
                model_name=state.model_name,
                device=state.device,
                conf_threshold=state.conf_thresh,
                iou_threshold=state.iou_thresh,
                img_size=state.img_size,
                enabled_classes=state.enabled_classes
            )
            state.detector = engine

        analytics = AnalyticsEngine(fps_window_size=20)
        state.analytics = analytics
        consecutive_misses = 0

        while state.is_streaming:
            t_start = analytics.start_frame()
            ret, frame = stream.read()
            if not ret or frame is None:
                # Video file reached end
                if not stream.is_camera:
                    if exporter:
                        meta = exporter.release()
                        state.video_exporter = None
                        analytics.to_dataframe().to_csv(out_csv_path, index=False)
                        state.last_exported_video_url = f"/api/download/output/{out_video_path.name}"
                        state.last_exported_csv_url = f"/api/download/output/{out_csv_path.name}"
                        state.export_completed = True
                    break
                else:
                    consecutive_misses += 1
                    time.sleep(0.04)
                    if consecutive_misses > 75:  # ~3 seconds of continuous network dropout
                        print("Live stream disconnected after timeout.")
                        break
                    continue

            consecutive_misses = 0

            state.current_frame_idx += 1
            if state.stream_mode == "track":
                results, lat_ms = engine.track(frame)
            else:
                results, lat_ms = engine.detect(frame)

            metrics = analytics.update(t_start, lat_ms, results)
            
            # Aggregate friendly categories: persons, cars, bikes, trucks
            c_cls = metrics["current_classes"]
            metrics["categories"] = {
                "persons": c_cls.get("person", 0),
                "cars": c_cls.get("car", 0),
                "bikes": c_cls.get("motorcycle", 0) + c_cls.get("bicycle", 0),
                "trucks": c_cls.get("bus", 0) + c_cls.get("truck", 0)
            }
            state.latest_metrics = metrics

            # Render clean bounding boxes and names (no trails, no line)
            annotated = draw_detections(
                frame.copy(),
                results,
                show_boxes=True,
                show_labels=True,
                show_confidence=True,
                show_track_id=(state.stream_mode == "track")
            )
            annotated = draw_hud(
                annotated,
                fps=metrics["fps"],
                latency_ms=metrics["latency_ms"],
                current_count=metrics["current_count"],
                cumulative_unique_count=metrics["cumulative_unique_total"],
                device=state.device,
                class_counts=metrics["current_classes"]
            )


            # Write to recorded file if exporter active
            if exporter:
                exporter.write_frame(annotated)

            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode(".jpg", annotated, encode_param)
            state.latest_frame_jpeg = buffer.tobytes()

            # Throttle slightly to match playback speed for recorded video
            if not stream.is_camera and stream.fps > 0:
                time.sleep(max(0.005, (1.0 / stream.fps) * 0.7))
            else:
                time.sleep(0.005)

    except Exception as e:
        print(f"Error in stream worker: {e}")
    finally:
        if exporter:
            exporter.release()
            state.video_exporter = None
            if out_csv_path and state.analytics:
                state.analytics.to_dataframe().to_csv(out_csv_path, index=False)
            if out_video_path and out_video_path.exists():
                state.last_exported_video_url = f"/api/download/output/{out_video_path.name}"
                state.last_exported_csv_url = f"/api/download/output/{out_csv_path.name}"
                state.export_completed = True
        if state.video_stream:
            state.video_stream.release()
            state.video_stream = None
        state.is_streaming = False

@app.post("/api/upload/video")
async def upload_video_file(file: UploadFile = File(...)):
    """Upload custom video for live analysis."""
    dest_path = INPUTS_DIR / file.filename
    content = await file.read()
    dest_path.write_bytes(content)
    return JSONResponse({
        "status": "uploaded",
        "filename": file.filename,
        "filepath": str(dest_path),
        "size_mb": round(len(content) / (1024 * 1024), 2)
    })

@app.post("/api/stream/control")
async def stream_control(ctrl: StreamControl):
    """Start or stop the live streaming thread."""
    if ctrl.action == "start":
        if state.is_streaming:
            # Stop existing first
            state.is_streaming = False
            if state.stream_thread and state.stream_thread.is_alive():
                state.stream_thread.join(timeout=2.0)

        # Resolve video path if sample or uploaded
        source_val = ctrl.source
        if source_val and not str(source_val).isdigit():
            # Check samples dir or inputs dir
            sample_candidate = SAMPLES_DIR / source_val
            input_candidate = INPUTS_DIR / source_val
            if sample_candidate.exists():
                source_val = str(sample_candidate)
            elif input_candidate.exists():
                source_val = str(input_candidate)

        state.stream_source = source_val
        state.stream_mode = ctrl.mode or "track"
        state.record_stream = bool(ctrl.record)
        state.export_completed = False
        state.is_streaming = True
        state.stream_thread = threading.Thread(target=_stream_worker, daemon=True)
        state.stream_thread.start()
        return JSONResponse({"status": "started", "source": str(source_val), "mode": state.stream_mode})

    elif ctrl.action == "stop":
        state.is_streaming = False
        if state.stream_thread and state.stream_thread.is_alive():
            state.stream_thread.join(timeout=2.0)
        return JSONResponse({
            "status": "stopped",
            "export_completed": state.export_completed,
            "video_download_url": state.last_exported_video_url,
            "csv_download_url": state.last_exported_csv_url
        })

    raise HTTPException(status_code=400, detail="Invalid action")


@app.get("/api/stream/video_feed")
async def video_feed():
    """Multipart MJPEG video stream consumed directly by <img> or <canvas>."""
    def frame_generator():
        while True:
            if not state.is_streaming or state.latest_frame_jpeg is None:
                # Return placeholder empty frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Stream Inactive (Click Start Camera)", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                _, buf = cv2.imencode(".jpg", blank)
                frame_bytes = buf.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(0.5)
            else:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + state.latest_frame_jpeg + b"\r\n")
                time.sleep(0.02)  # ~50 FPS max transmission

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/stream/status")
async def stream_status():
    """Poll latest frame metrics, FPS, object count, and unique tracked census."""
    stream_title = None
    if state.video_stream and state.video_stream.stream_title:
        stream_title = state.video_stream.stream_title
    return JSONResponse({
        "is_streaming": state.is_streaming,
        "source": str(state.stream_source),
        "stream_title": stream_title,
        "mode": state.stream_mode,
        "frame_idx": state.current_frame_idx,
        "total_frames": state.total_stream_frames,
        "metrics": state.latest_metrics,
        "export_completed": state.export_completed,
        "video_download_url": state.last_exported_video_url,
        "csv_download_url": state.last_exported_csv_url
    })

@app.post("/api/stream/validate_url")
async def validate_stream_url(payload: Dict[str, Any]):
    """Validate and get title / info for a YouTube or network stream URL."""
    url = payload.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing url parameter.")
    if is_youtube_url(url):
        try:
            stream_url, is_live, title = resolve_youtube_stream_url(url)
            return JSONResponse({
                "status": "valid",
                "type": "youtube",
                "is_live": is_live,
                "title": title
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"YouTube extraction error: {e}")
    elif url.startswith("http://") or url.startswith("https://") or url.startswith("rtsp://"):
        return JSONResponse({
            "status": "valid",
            "type": "network_stream",
            "is_live": True,
            "title": "Network Stream Feed"
        })
    else:
        raise HTTPException(status_code=400, detail="Unsupported URL format. Must be YouTube or http/https/rtsp.")


@app.get("/api/telemetry/history")
async def telemetry_history():
    """Return recent telemetry time-series records for live Chart.js graphs."""
    history = state.analytics.history[-100:] if state.analytics else []
    return JSONResponse({"history": history})

@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """Upload an image, run YOLOv8, return annotated base64 JPEG and detections JSON."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    detector = ObjectDetector(
        model_name=state.model_name,
        device=state.device,
        conf_threshold=state.conf_thresh,
        iou_threshold=state.iou_thresh,
        img_size=state.img_size,
        enabled_classes=state.enabled_classes
    )

    detections, lat_ms = detector.detect(img)
    annotated = draw_detections(img.copy(), detections, show_boxes=True, show_labels=True, show_confidence=True)

    # Encode annotated image as base64
    _, buf = cv2.imencode(".jpg", annotated)
    b64_img = base64.b64encode(buf).decode("utf-8")

    # Save to outputs
    out_filename = f"detected_{Path(file.filename).stem}.jpg"
    out_path = OUTPUTS_DIR / out_filename
    cv2.imwrite(str(out_path), annotated)

    return JSONResponse({
        "filename": file.filename,
        "width": img.shape[1],
        "height": img.shape[0],
        "latency_ms": round(lat_ms, 2),
        "objects_count": len(detections),
        "detections": detections,
        "annotated_image_base64": f"data:image/jpeg;base64,{b64_img}",
        "saved_path": f"/api/download/output/{out_filename}"
    })

@app.post("/api/process/video")
async def process_video_file(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    max_frames: int = Form(150),
    mode: str = Form("track")
):
    """Process a video file, track objects, and export an MP4 with CSV telemetry."""
    if file:
        dest_path = INPUTS_DIR / file.filename
        content = await file.read()
        dest_path.write_bytes(content)
        video_path = dest_path
    elif sample_name:
        video_path = SAMPLES_DIR / sample_name
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Sample video not found.")
    else:
        raise HTTPException(status_code=400, detail="Must supply uploaded file or sample_name.")

    stream = VideoStream(video_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_stem = f"{video_path.stem}_processed_{timestamp}"
    out_video_path = OUTPUTS_DIR / f"{out_stem}.mp4"
    out_csv_path = OUTPUTS_DIR / f"{out_stem}_telemetry.csv"

    if mode == "track":
        engine = ObjectTracker(
            model_name=state.model_name,
            device=state.device,
            tracker_type=state.tracker_type,
            conf_threshold=state.conf_thresh,
            iou_threshold=state.iou_thresh,
            img_size=state.img_size,
            enabled_classes=state.enabled_classes
        )
    else:
        engine = ObjectDetector(
            model_name=state.model_name,
            device=state.device,
            conf_threshold=state.conf_thresh,
            iou_threshold=state.iou_thresh,
            img_size=state.img_size,
            enabled_classes=state.enabled_classes
        )

    analytics = AnalyticsEngine(fps_window_size=30)
    exporter = VideoExporter(
        output_path=out_video_path,
        fps=stream.fps,
        frame_size=(stream.width, stream.height),
        codec="mp4v"
    )

    frame_count = 0
    target_frames = min(stream.total_frames, max_frames) if max_frames > 0 else stream.total_frames
    t0 = time.time()

    while frame_count < target_frames:
        t_frame_start = analytics.start_frame()
        ret, frame = stream.read()
        if not ret:
            break

        if mode == "track":
            results, lat_ms = engine.track(frame)
        else:
            results, lat_ms = engine.detect(frame)

        metrics = analytics.update(t_frame_start, lat_ms, results)
        
        annotated = draw_detections(
            frame.copy(),
            results,
            show_boxes=True,
            show_labels=True,
            show_confidence=True,
            show_track_id=(mode == "track")
        )
        annotated = draw_hud(
            annotated,
            fps=metrics["fps"],
            latency_ms=metrics["latency_ms"],
            current_count=metrics["current_count"],
            cumulative_unique_count=metrics["cumulative_unique_total"],
            device=state.device,
            class_counts=metrics["current_classes"]
        )


        exporter.write_frame(annotated)
        frame_count += 1

    stream.release()
    export_meta = exporter.release()
    total_sec = time.time() - t0
    summary = analytics.get_summary()

    # Save CSV
    analytics.to_dataframe().to_csv(out_csv_path, index=False)

    return JSONResponse({
        "status": "completed",
        "frames_processed": frame_count,
        "duration_sec": round(total_sec, 2),
        "overall_fps": round(frame_count / total_sec, 1) if total_sec > 0 else 0,
        "avg_latency_ms": summary["average_latency_ms"],
        "total_unique_objects": summary["total_unique_objects"],
        "unique_by_class": summary["unique_by_class"],
        "video_download_url": f"/api/download/output/{out_video_path.name}",
        "csv_download_url": f"/api/download/output/{out_csv_path.name}",
        "file_size_mb": export_meta["file_size_mb"]
    })

@app.get("/api/download/output/{filename}")
async def download_output(filename: str):
    """Download exported MP4 videos or CSV logs."""
    target_path = OUTPUTS_DIR / filename
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=str(target_path),
        filename=filename,
        media_type="application/octet-stream"
    )

@app.get("/api/edge/deepstream-config")
async def get_deepstream_config():
    """
    Returns Jetson DeepStream & TensorRT deployment architecture
    based on the Seeed Studio YOLOv8 DeepStream TRT Jetson documentation.
    """
    return JSONResponse({
        "target_platforms": ["NVIDIA Jetson Orin Nano", "Jetson Orin NX", "Jetson Xavier NX", "reComputer J4012"],
        "deepstream_version": "6.2+ / JetPack 5.1.1+",
        "export_steps": [
            "1. Export PyTorch weights to ONNX: yolo export model=yolov8n.pt format=onnx dynamic=True",
            "2. Clone DeepStream-Yolo: git clone https://github.com/marcoslucianops/DeepStream-Yolo",
            "3. Compile custom TensorRT parser: CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo",
            "4. Build TensorRT .engine with trtexec or deepstream-app auto-build",
            "5. Execute GStreamer pipeline with deepstream-app -c deepstream_app_config.txt"
        ],
        "sample_deepstream_infer_config": """[property]
gpu-id=0
net-scale-factor=0.003921569793452
model-color-format=0
custom-network-config=yolov8n.onnx
model-engine-file=model_b1_gpu0_fp16.engine
labelfile-path=labels.txt
batch-size=1
network-mode=2 # 0=FP32, 1=INT8, 2=FP16
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
engine-create-func-name=NvDsInferYoloCudaEngineGet
""",
        "comparison_matrix": {
            "Apple M2 CPU (PyTorch)": {"FPS": "21.3 - 27.5 FPS", "Latency": "36 - 47 ms", "Power": "~15W"},
            "Apple M2 MPS (Metal)": {"FPS": "61.5 FPS", "Latency": "16.2 ms", "Power": "~20W"},
            "Jetson Orin Nano (TensorRT FP16)": {"FPS": "85+ FPS", "Latency": "11.5 ms", "Power": "7W - 15W"},
            "Jetson Orin NX (TensorRT INT8)": {"FPS": "140+ FPS", "Latency": "6.8 ms", "Power": "10W - 25W"}
        }
    })

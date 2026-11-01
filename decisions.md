# Technical Decisions & Architectural Trade-offs

---

## 1. Migration from Streamlit to Native HTML5/CSS3/JavaScript + FastAPI

### Context
Streamlit was initially considered for rapid prototyping. However, live video analytics and edge deployment requirements (highlighted by the [Seeed Studio YOLOv8 DeepStream TRT Jetson Guide](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/)) demonstrated significant limitations with Streamlit:
- High latency and low frame rates (~5–10 FPS UI render limits).
- The execution model re-runs the entire script on every user interaction, causing dropped frames and high CPU spikes.
- Inability to handle real-time multi-stream video pipelines with fine-grained DOM control.

### Decision
Migrate completely to a decoupled architecture:
1. **Backend:** Asynchronous **FastAPI** with `uvicorn`, exposing REST endpoints for inference and direct multipart/x-mixed-replace (MJPEG) streaming.
2. **Frontend:** Clean, responsive **HTML5 + CSS3 + Vanilla JavaScript + Chart.js**, providing 30–60 FPS video canvas rendering, instant control feedback, and zero bundle-bloat.

### Trade-offs & Benefits
- **Zero Framework Overhead:** The frontend requires no Node.js build step or heavy bundle dependencies; it runs out of the box in any browser.
- **Microsecond Latency:** The MJPEG stream delivers frames as soon as they are processed by the OpenCV/PyTorch pipeline.
- **Production Edge Realism:** Mimics industrial video management systems (VMS) and NVIDIA DeepStream web dashboards.

---

## 2. NVIDIA Jetson & DeepStream TRT Deployment Readiness

### Context
In real edge deployments (e.g., Seeed Studio reComputer J4012 with Jetson Orin Nano / NX), PyTorch inference is replaced with TensorRT to achieve 85–140+ FPS.

### Decision
Incorporate full architectural parity with the Seeed Studio DeepStream pipeline:
1. Provide export commands (`yolo export model=yolov8n.pt format=onnx dynamic=True`).
2. Provide the DeepStream custom parser integration (`nvdsinfer_custom_impl_Yolo`).
3. Embed the primary inference configuration file (`config_infer_primary_yolov8.txt`) into the application's Edge tab.
4. Keep the core Python pipeline modular so it can ingest GStreamer RTSP streams identical to DeepStream.

---

## 3. Multi-Object Tracking & Duplicate-Free Counting

### Context
Tracking objects across frames requires persistent IDs without double-counting when objects remain in the field of view.

### Decision
Utilize **ByteTrack** with Kalman filtering to associate detections across frames.
Maintain dual telemetry tallies:
- **`current_count`:** Objects currently detected in the active frame.
- **`cumulative_unique_total`:** Persistent hash table of verified `track_id` values, ensuring a car or person is counted exactly once regardless of dwell time.

"""Image detection and analysis component."""

import cv2
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

from core.detector import ObjectDetector
from utils.visualization import draw_detections
from config.settings import SAMPLES_DIR, INPUTS_DIR, OUTPUTS_DIR

def render_image_page():
    st.title("Static Image Detection & Analysis")
    st.caption("Upload images or select sample benchmarks to evaluate YOLOv8 bounding boxes and confidence scores")

    model_name = st.session_state.get("model_name", "models/yolov8n.pt")
    device = st.session_state.get("device", "cpu")
    conf_thresh = st.session_state.get("conf_thresh", 0.35)
    iou_thresh = st.session_state.get("iou_thresh", 0.45)
    img_size = st.session_state.get("img_size", 640)
    enabled_classes = st.session_state.get("enabled_classes", None)

    source_type = st.radio("Select Image Source", ["Sample Benchmark Images", "Upload Custom Image"], horizontal=True)
    img_array = None
    img_name = "image"

    if source_type == "Sample Benchmark Images":
        samples = list(SAMPLES_DIR.glob("*.jpg")) + list(SAMPLES_DIR.glob("*.png"))
        if samples:
            sample_names = [s.name for s in samples]
            selected = st.selectbox("Choose Sample Image", sample_names)
            img_path = SAMPLES_DIR / selected
            img_array = cv2.imread(str(img_path))
            img_name = img_path.stem
        else:
            st.warning("No sample images found in `data/samples/`.")
    else:
        uploaded = st.file_uploader("Upload Image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_name = Path(uploaded.name).stem

    if img_array is not None:
        h, w = img_array.shape[:2]
        st.write(f"**Image Dimensions:** {w} x {h} pixels")

        col1, col2 = st.columns([1, 4])
        with col1:
            run_btn = st.button("Run Inference", type="primary")

        if run_btn:
            detector = ObjectDetector(
                model_name=model_name,
                device=device,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                img_size=img_size,
                enabled_classes=enabled_classes
            )

            with st.spinner(f"Running inference on {device.upper()}..."):
                detections, latency_ms = detector.detect(img_array)

            annotated = draw_detections(
                img_array.copy(),
                detections,
                show_boxes=True,
                show_labels=True,
                show_confidence=True,
                show_track_id=False
            )

            # Metric cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Objects Detected", len(detections))
            m2.metric("Inference Latency", f"{latency_ms:.2f} ms")
            m3.metric("Effective FPS", f"{1000.0/latency_ms:.1f}" if latency_ms > 0 else "N/A")
            m4.metric("Device", device.upper())

            # Side-by-side view
            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True)
            with c_right:
                st.subheader("Annotated Output")
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)

            # Save annotated output
            out_path = OUTPUTS_DIR / f"{img_name}_detected.jpg"
            cv2.imwrite(str(out_path), annotated)

            # Detection table
            st.subheader("Detection Breakdown")
            if detections:
                table_rows = []
                for idx, d in enumerate(detections, 1):
                    x1, y1, x2, y2 = map(int, d["box"])
                    box_w = x2 - x1
                    box_h = y2 - y1
                    table_rows.append({
                        "Index": idx,
                        "Class": d["class_name"],
                        "Confidence": f"{d['confidence']*100:.1f}%",
                        "Box [X1, Y1, X2, Y2]": f"[{x1}, {y1}, {x2}, {y2}]",
                        "Width x Height": f"{box_w} x {box_h} px",
                        "Area": f"{box_w * box_h} px²"
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
            else:
                st.info("No objects detected above the current confidence threshold.")

            # Download button
            with open(out_path, "rb") as f:
                img_bytes = f.read()
            st.download_button(
                label="Download Annotated Image",
                data=img_bytes,
                file_name=out_path.name,
                mime="image/jpeg"
            )

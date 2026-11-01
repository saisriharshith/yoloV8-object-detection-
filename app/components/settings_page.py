"""Configuration and Hyperparameters Settings Component."""

import streamlit as st
from pathlib import Path
from config.settings import MODELS_DIR, COCO_CLASSES
from utils.system_info import get_system_metrics

def render_settings_page():
    st.title("Configuration & Hyperparameters")
    st.caption("Customize YOLOv8 weights, execution backend, NMS thresholds, class filters, and UI rendering")

    sys_info = get_system_metrics()

    # Model Weights & Backend
    st.subheader("1. Model & Hardware Acceleration")
    col1, col2 = st.columns(2)

    with col1:
        available_models = [m.name for m in MODELS_DIR.glob("*.pt")]
        if not available_models:
            available_models = ["yolov8n.pt"]
        current_model = st.session_state.get("model_name", "models/yolov8n.pt")
        selected_model = st.selectbox(
            "YOLOv8 Model Weights",
            available_models,
            index=0 if Path(current_model).name not in available_models else available_models.index(Path(current_model).name)
        )
        st.session_state["model_name"] = str(MODELS_DIR / selected_model)

    with col2:
        device_options = ["cpu"]
        if sys_info["mps_available"]:
            device_options.append("mps")
        
        current_device = st.session_state.get("device", "cpu")
        selected_device = st.selectbox(
            "Compute Device",
            device_options,
            index=device_options.index(current_device) if current_device in device_options else 0,
            help="CPU is recommended for memory-constrained stability. MPS uses Apple Silicon GPU acceleration."
        )
        st.session_state["device"] = selected_device

    # Detection Thresholds
    st.subheader("2. Detection & Filtering Thresholds")
    c_th1, c_th2, c_th3 = st.columns(3)

    with c_th1:
        conf = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=1.00,
            value=st.session_state.get("conf_thresh", 0.35),
            step=0.05,
            help="Minimum confidence score required for a detection box to be accepted."
        )
        st.session_state["conf_thresh"] = conf

    with c_th2:
        iou = st.slider(
            "IoU (NMS) Threshold",
            min_value=0.10,
            max_value=0.95,
            value=st.session_state.get("iou_thresh", 0.45),
            step=0.05,
            help="Intersection-over-Union threshold for Non-Maximum Suppression duplicate suppression."
        )
        st.session_state["iou_thresh"] = iou

    with c_th3:
        img_size = st.selectbox(
            "Inference Resolution (px)",
            [320, 480, 640],
            index=[320, 480, 640].index(st.session_state.get("img_size", 640)),
            help="Input frame dimension resized for neural network feed-forward. 320px yields ultra-high FPS."
        )
        st.session_state["img_size"] = img_size

    # Class Filtering
    st.subheader("3. Class Selection & Filtering")
    st.caption("Filter specific object categories (e.g. only vehicle or pedestrian tracking).")
    
    all_classes_selected = st.checkbox("Enable All 80 COCO Classes", value=st.session_state.get("all_classes", True))
    
    if all_classes_selected:
        st.session_state["enabled_classes"] = None
        st.session_state["all_classes"] = True
        st.info("All 80 COCO classes are currently enabled.")
    else:
        st.session_state["all_classes"] = False
        default_selected = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]
        selected_classes = st.multiselect(
            "Select Active Classes",
            COCO_CLASSES,
            default=st.session_state.get("selected_class_names", default_selected)
        )
        st.session_state["selected_class_names"] = selected_classes
        if selected_classes:
            st.session_state["enabled_classes"] = [COCO_CLASSES.index(c) for c in selected_classes]
        else:
            st.session_state["enabled_classes"] = []
            st.warning("No classes selected. Detection will produce zero outputs.")

    # Tracking & UI Preferences
    st.subheader("4. Tracking Algorithm & Visual Display")
    col_trk, col_ui = st.columns(2)

    with col_trk:
        tracker_type = st.radio(
            "Tracking Algorithm",
            ["bytetrack", "botsort"],
            index=0 if st.session_state.get("tracker_type", "bytetrack") == "bytetrack" else 1,
            horizontal=True
        )
        st.session_state["tracker_type"] = tracker_type

    with col_ui:
        st.session_state["show_hud"] = st.checkbox("Display Heads-Up Display (HUD) Banner", value=st.session_state.get("show_hud", True))
        st.session_state["show_track_id"] = st.checkbox("Display Unique Tracking IDs", value=st.session_state.get("show_track_id", True))

    st.divider()

    # Reset
    if st.button("Reset Settings to Defaults"):
        st.session_state["model_name"] = "models/yolov8n.pt"
        st.session_state["device"] = "cpu"
        st.session_state["conf_thresh"] = 0.35
        st.session_state["iou_thresh"] = 0.45
        st.session_state["img_size"] = 640
        st.session_state["enabled_classes"] = None
        st.session_state["all_classes"] = True
        st.session_state["tracker_type"] = "bytetrack"
        st.session_state["show_hud"] = True
        st.session_state["show_track_id"] = True
        st.success("Settings restored to factory defaults.")
        st.rerun()

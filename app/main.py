"""Real-Time Object Detection and Live Video Analytics Using YOLOv8.
Main Streamlit Application Entrypoint.
"""

import sys
from pathlib import Path
import streamlit as st

# Configure sys.path so imports work cleanly from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components import (
    render_home_page,
    render_live_camera_page,
    render_video_page,
    render_image_page,
    render_analytics_page,
    render_settings_page,
)
from utils.system_info import get_system_metrics

# Streamlit Page Configuration
st.set_page_config(
    page_title="YOLOv8 Edge Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Professional Styling
st.markdown("""
<style>
    /* Clean container spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 6px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.6rem;
        font-weight: 600;
    }

    /* Clean divider styling */
    hr {
        margin: 1.5rem 0;
        border: 0;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Sidebar footer */
    .sidebar-footer {
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        font-size: 0.8rem;
        color: rgba(128, 128, 128, 0.9);
    }
</style>
""", unsafe_allow_html=True)

# Default session state variables
if "model_name" not in st.session_state:
    st.session_state["model_name"] = str(PROJECT_ROOT / "models" / "yolov8n.pt")
if "device" not in st.session_state:
    st.session_state["device"] = "cpu"
if "conf_thresh" not in st.session_state:
    st.session_state["conf_thresh"] = 0.35
if "iou_thresh" not in st.session_state:
    st.session_state["iou_thresh"] = 0.45
if "img_size" not in st.session_state:
    st.session_state["img_size"] = 640
if "enabled_classes" not in st.session_state:
    st.session_state["enabled_classes"] = None
if "all_classes" not in st.session_state:
    st.session_state["all_classes"] = True
if "tracker_type" not in st.session_state:
    st.session_state["tracker_type"] = "bytetrack"
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Home"

# Sidebar Navigation
with st.sidebar:
    st.title("YOLOv8 Edge Engine")
    st.caption("Live Vision Analytics Platform")

    pages = [
        "Home",
        "Live Camera",
        "Video Analysis",
        "Image Detection",
        "Analytics Dashboard",
        "Settings",
    ]

    current_idx = pages.index(st.session_state["nav_selection"]) if st.session_state["nav_selection"] in pages else 0
    selected_page = st.radio("Navigation", pages, index=current_idx)
    st.session_state["nav_selection"] = selected_page

    st.markdown("---")
    
    # Active configuration badge
    model_name_disp = Path(st.session_state['model_name']).name
    st.markdown(f"**Engine Configuration**")
    st.markdown(f"- Model: `{model_name_disp}`")
    st.markdown(f"- Device: `{st.session_state['device'].upper()}`")
    st.markdown(f"- Resolution: `{st.session_state['img_size']}px`")
    st.markdown(f"- Conf / IoU: `{st.session_state['conf_thresh']} / {st.session_state['iou_thresh']}`")

    # Real-time System telemetry in sidebar
    metrics = get_system_metrics()
    st.markdown("---")
    st.markdown(f"**M2 Hardware Telemetry**")
    st.markdown(f"- Architecture: `{metrics['architecture']}`")
    st.markdown(f"- RAM Usage: `{metrics['ram_usage_percent']}% ({metrics['available_ram_gb']} GB Free)`")
    st.markdown(f"- MPS Accel: `{'Ready' if metrics['mps_available'] else 'N/A'}`")

# Route to respective component
if selected_page == "Home":
    render_home_page()
elif selected_page == "Live Camera":
    render_live_camera_page()
elif selected_page == "Video Analysis":
    render_video_page()
elif selected_page == "Image Detection":
    render_image_page()
elif selected_page == "Analytics Dashboard":
    render_analytics_page()
elif selected_page == "Settings":
    render_settings_page()

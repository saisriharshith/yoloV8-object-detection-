"""Telemetry and Analytics Dashboard visualizing tracking timelines and performance distributions."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from config.settings import OUTPUTS_DIR

def render_analytics_page():
    st.title("Telemetry & Analytics Dashboard")
    st.caption("Quantitative telemetry analysis: frame-level FPS stability, inference latency distributions, and object count timelines")

    # Locate CSV telemetry files in outputs
    csv_files = list(OUTPUTS_DIR.glob("*.csv"))
    latest_csv = st.session_state.get("latest_telemetry_csv", None)

    if not csv_files:
        st.info("No telemetry logs found. Run a video or camera session first to generate performance data.")
        return

    csv_paths = {p.name: p for p in csv_files}
    default_idx = 0
    if latest_csv:
        latest_name = Path(latest_csv).name
        if latest_name in csv_paths:
            default_idx = list(csv_paths.keys()).index(latest_name)

    selected_csv_name = st.selectbox("Select Session Telemetry Log", list(csv_paths.keys()), index=default_idx)
    selected_path = csv_paths[selected_csv_name]

    # Load dataframe
    try:
        df = pd.read_csv(selected_path)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    if df.empty:
        st.warning("Selected telemetry log is empty.")
        return

    # High-level Metrics Row
    st.subheader("Session Summary KPIs")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Frames", len(df))
    with kpi2:
        avg_fps = df["fps"].mean() if "fps" in df.columns else 0.0
        st.metric("Average FPS", f"{avg_fps:.1f}")
    with kpi3:
        avg_lat = df["latency_ms"].mean() if "latency_ms" in df.columns else 0.0
        st.metric("Mean Latency", f"{avg_lat:.1f} ms")
    with kpi4:
        max_objs = df["current_count"].max() if "current_count" in df.columns else 0
        uniq_total = df["unique_tracked_total"].max() if "unique_tracked_total" in df.columns else 0
        st.metric("Peak Concurrent", f"{max_objs}", delta=f"{uniq_total} Unique Total")

    st.divider()

    # Time-series Charts
    st.subheader("Temporal Dynamics")
    tab1, tab2, tab3 = st.tabs(["Object Count Timeline", "FPS & Latency Stability", "Class Breakdown"])

    with tab1:
        if "frame_idx" in df.columns and "current_count" in df.columns:
            fig_count = go.Figure()
            fig_count.add_trace(go.Scatter(
                x=df["frame_idx"],
                y=df["current_count"],
                mode="lines",
                name="In-Frame Objects",
                line=dict(color="#1f77b4", width=2)
            ))
            if "unique_tracked_total" in df.columns:
                fig_count.add_trace(go.Scatter(
                    x=df["frame_idx"],
                    y=df["unique_tracked_total"],
                    mode="lines",
                    name="Cumulative Unique Objects",
                    line=dict(color="#2ca02c", width=2, dash="dash")
                ))
            fig_count.update_layout(
                title="Object Count vs Frame Index",
                xaxis_title="Frame Index",
                yaxis_title="Count",
                template="plotly_white",
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_count, use_container_width=True)

    with tab2:
        c_left, c_right = st.columns(2)
        with c_left:
            if "fps" in df.columns:
                fig_fps = px.line(
                    df,
                    x="frame_idx",
                    y="fps",
                    title="Real-Time FPS Over Frames",
                    labels={"frame_idx": "Frame Index", "fps": "Frames Per Second"},
                    template="plotly_white"
                )
                fig_fps.add_hline(y=avg_fps, line_dash="dot", annotation_text=f"Mean: {avg_fps:.1f} FPS")
                st.plotly_chart(fig_fps, use_container_width=True)
        with c_right:
            if "latency_ms" in df.columns:
                fig_lat = px.line(
                    df,
                    x="frame_idx",
                    y="latency_ms",
                    title="Inference Latency (ms) Over Frames",
                    labels={"frame_idx": "Frame Index", "latency_ms": "Latency (ms)"},
                    template="plotly_white"
                )
                fig_lat.add_hline(y=avg_lat, line_dash="dot", annotation_text=f"Mean: {avg_lat:.1f} ms")
                st.plotly_chart(fig_lat, use_container_width=True)

    with tab3:
        # Detect curr_* columns
        class_cols = [c for c in df.columns if c.startswith("curr_")]
        if class_cols:
            totals = {}
            for col in class_cols:
                class_name = col.replace("curr_", "")
                totals[class_name] = df[col].sum()
            
            df_classes = pd.DataFrame(list(totals.items()), columns=["Class", "Total Instances"])
            df_classes = df_classes.sort_values(by="Total Instances", ascending=False)

            fig_bar = px.bar(
                df_classes,
                x="Class",
                y="Total Instances",
                title="Frequency of Detected Classes across Video",
                template="plotly_white",
                color="Total Instances",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No class breakdown columns available in this log.")

    # Raw Data Explorer
    st.subheader("Raw Telemetry Log")
    st.dataframe(df, use_container_width=True)

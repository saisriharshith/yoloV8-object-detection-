# Deployment Guide: YOLOv8 Real-Time Video Analytics on Render

This guide outlines how to deploy the **YOLOv8 Real-Time Video Analytics** web application to **[Render](https://render.com)** using **Docker**.

---

## 🚀 Quick Deployment to Render (Recommended)

### Step 1: Push Code to GitHub
Ensure all project files (including [`Dockerfile`](file:///Users/konthamsaisriharshith/Desktop/PROJECT%20K/project/Dockerfile), [`render.yaml`](file:///Users/konthamsaisriharshith/Desktop/PROJECT%20K/project/render.yaml), and [`.dockerignore`](file:///Users/konthamsaisriharshith/Desktop/PROJECT%20K/project/.dockerignore)) are committed and pushed to your GitHub repository:

```bash
git add .
git commit -m "Add Docker and Render configuration"
git push origin main
```

---

### Step 2: Deploy on Render Dashboard

1. Log in to **[Render Dashboard](https://dashboard.render.com/)**.
2. Click **New +** at the top right and select **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your GitHub repository.
4. Render will automatically detect the **Dockerfile** in the repository root.
5. Configure the service settings:
   - **Name:** `yolov8-video-analytics` (or your preferred name)
   - **Region:** Pick the region closest to you (e.g., `Oregon (US West)` or `Frankfurt (EU)`)
   - **Branch:** `main`
   - **Runtime:** `Docker`
   - **Instance Type:**
     - **Free:** Good for static testing and lightweight inference.
     - **Starter ($7/mo, 1 CPU, 2GB RAM):** Recommended for live video decoding, YouTube Live streaming, and multi-object tracking.
6. Under **Advanced Settings**:
   - Verify environment variables:
     - `PYTHONUNBUFFERED`: `1`
     - `PORT`: `10000` (Render sets this automatically)
7. Click **Create Web Service**.

---

### Step 3: Access Your Live Application
- Render will pull the slim base image, install CPU-optimized PyTorch, cache the model weights, and start the FastAPI web server.
- Once the build finishes and shows `Build succeeded` and `Service is live`, click your custom Render URL (e.g., `https://yolov8-video-analytics.onrender.com`).

---

## 🛠️ Alternative: Deploy via Render Blueprint (`render.yaml`)

If you prefer automated Infrastructure-as-Code:
1. Go to **Render Dashboard** > **New +** > **Blueprint**.
2. Select your repository.
3. Render reads [`render.yaml`](file:///Users/konthamsaisriharshith/Desktop/PROJECT%20K/project/render.yaml) automatically.
4. Click **Apply**.

---

## 💻 Local Testing with Docker

Before deploying to the cloud, you can test the exact Docker container locally on your machine:

### Using Docker CLI:
```bash
# 1. Build the Docker image
docker build -t yolo-analytics .

# 2. Run the container on port 8000
docker run -p 8000:8000 -e PORT=8000 yolo-analytics
```

### Using Docker Compose:
```bash
docker compose up --build
```

Then visit **`http://localhost:8000`** in your browser to verify:
- ✅ Webcam & YouTube Live streaming
- ✅ Video upload & analytics
- ✅ Object tracking (ByteTrack)
- ✅ Live counters (Cars, Persons, Bikes, Trucks)

---

## ⚙️ Architecture & Optimization Details for Cloud Containers

| Component | Cloud Container Configuration | Benefit |
| :--- | :--- | :--- |
| **Base Image** | `python:3.11-slim` (Debian Bookworm) | Small footprint, security updates |
| **PyTorch Wheel** | `https://download.pytorch.org/whl/cpu` | Shrinks image from ~5 GB to ~900 MB, fast deployment |
| **FFmpeg Backend** | Installed via `apt-get install -y ffmpeg` | Enables smooth YouTube Live HLS and RTSP stream decoding |
| **Port Binding** | Reads `${PORT:-8000}` dynamically | Fully compatible with Render's dynamic reverse proxy |
| **Model Pre-caching** | Weights initialized during Docker build | Container boots in <2 seconds with zero network delay |
| **Health Check** | `curl -f http://localhost:${PORT:-8000}/` | Automatic failover and health monitoring |

"""Robust video capture abstraction supporting webcam, recorded videos, network streams, and YouTube Live."""

import cv2
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Union, List

logger = logging.getLogger("video_stream")

def is_youtube_url(url: str) -> bool:
    """Check if the provided string is a YouTube video or live stream link."""
    if not isinstance(url, str):
        return False
    u = url.lower().strip()
    return "youtube.com" in u or "youtu.be" in u

def resolve_youtube_stream_url(url: str) -> Tuple[str, bool, str]:
    """
    Extract direct m3u8 or video stream URL from YouTube using yt-dlp.
    Returns: (stream_url, is_live, video_title)
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is required for YouTube streaming. Install with: pip install yt-dlp")

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        is_live = bool(info.get('is_live', False))
        title = info.get('title', 'YouTube Stream')
        formats = [f for f in info.get('formats', []) if f.get('vcodec') != 'none']
        
        # Prefer ~720p for fast edge inference and clear bounding boxes
        chosen = None
        for f in formats:
            res_str = str(f.get('resolution', ''))
            if '720' in res_str:
                chosen = f
                break
        if not chosen and formats:
            chosen = formats[-1]
            
        stream_url = chosen['url'] if (chosen and 'url' in chosen) else info.get('url')
        if not stream_url:
            raise RuntimeError(f"Could not extract direct stream URL for {url}")
            
        return stream_url, is_live, title


class VideoStream:
    """Manages video frame acquisition with robust error handling and FPS control."""

    def __init__(self, source: Union[int, str, Path]):
        self.raw_source = source
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 25.0
        self.total_frames: int = 0
        self.is_camera: bool = False
        self.is_opened: bool = False
        self.stream_title: Optional[str] = None
        self._open_source()

    def _open_source(self) -> None:
        """Open video capture stream with verification."""
        src_str = str(self.raw_source).strip()

        # 1. Check if source is a local webcam device index (e.g. 0, 1)
        if isinstance(self.raw_source, int) or src_str.isdigit():
            src = int(src_str)
            self.is_camera = True
            self.source = src
        # 2. Check if source is a YouTube link (Live or recorded)
        elif is_youtube_url(src_str):
            logger.info(f"Resolving YouTube stream URL for: {src_str}")
            stream_url, is_live, title = resolve_youtube_stream_url(src_str)
            self.source = stream_url
            self.is_camera = is_live
            self.stream_title = title
            src = stream_url
        # 3. Check if source is an RTSP or HTTP/HTTPS network stream
        elif src_str.startswith("http://") or src_str.startswith("https://") or src_str.startswith("rtsp://"):
            self.source = src_str
            self.is_camera = True  # Network streams treated as live feeds
            src = src_str
        # 4. Local video file
        else:
            path = Path(src_str)
            if not path.exists():
                raise FileNotFoundError(f"Video file does not exist: {path}")
            self.source = str(path)
            self.is_camera = False
            src = str(path)

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.is_opened = False
            msg = (
                f"Cannot open webcam (device index {src}). Please check camera permissions or connectivity."
                if self.is_camera
                else f"Cannot open video source: {src}. File or stream may be unreachable."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        self.is_opened = True
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or self.fps > 120:
            self.fps = 25.0  # safe default

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else -1
        logger.info(
            f"Opened source '{self.stream_title or self.raw_source}': {self.width}x{self.height} @ {self.fps:.1f} FPS "
            f"({'live/camera' if self.is_camera else f'{self.total_frames} frames'})"
        )

    def read(self) -> Tuple[bool, Optional[any]]:
        """Read the next video frame safely with network retry logic."""
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret and self.is_camera:
            # For live streams, perform brief retry for network buffer recovery
            for _ in range(3):
                time.sleep(0.04)
                ret, frame = self.cap.read()
                if ret:
                    break

        if not ret:
            return False, None

        return True, frame

    def get_progress(self) -> float:
        """Get current playback progress (0.0 to 1.0) for video files."""
        if self.is_camera or self.total_frames <= 0 or self.cap is None:
            return 0.0
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return min(1.0, max(0.0, current_pos / self.total_frames))

    def release(self) -> None:
        """Safely release camera or video file resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        logger.info(f"Released video source: {self.raw_source}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @staticmethod
    def get_available_cameras(max_tested: int = 4) -> List[int]:
        """Scan and list indices of available, working webcams."""
        available = []
        for idx in range(max_tested):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                cap.release()
        return available

# Export as module-level function for core.__init__
get_available_cameras = VideoStream.get_available_cameras

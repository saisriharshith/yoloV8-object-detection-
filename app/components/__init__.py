"""Dashboard UI components."""

from .home_page import render_home_page
from .live_camera import render_live_camera_page
from .video_page import render_video_page
from .image_page import render_image_page
from .analytics_page import render_analytics_page
from .settings_page import render_settings_page

__all__ = [
    "render_home_page",
    "render_live_camera_page",
    "render_video_page",
    "render_image_page",
    "render_analytics_page",
    "render_settings_page",
]

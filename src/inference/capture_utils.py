"""OpenCV capture helpers for file paths and stream URLs (RTSP / HTTP)."""
from __future__ import annotations

import shutil
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def open_video_capture(uri: str, *, buffer_reduce: bool = False):
    """Return ``cv2.VideoCapture`` configured for URIs vs local paths.

    - Local files: unchanged default backend.
    - ``rtsp`` / ``http`` / ``https``: request smaller buffer when supported
      and set common OpenCAP properties (best-effort).
    """
    import cv2

    u = str(uri).strip()
    is_stream = (
        "://" in u
        or u.lower().startswith("rtsp:")
        or u.lower().startswith("udp:")
        or u.lower().startswith("tcp:")
    )
    if is_stream:
        cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
        if buffer_reduce:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
    else:
        p = Path(u)
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {u}")
        cap = cv2.VideoCapture(str(p))

    if not cap.isOpened():
        hint = "; try VLC/playable FFmpeg build" if is_stream else ""
        raise RuntimeError(f"Cannot open capture: {u}{hint}")
    return cap

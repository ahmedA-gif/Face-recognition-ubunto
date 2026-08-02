from __future__ import annotations

import os
import time
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

# go2rtc 1.9+ only serves RTSP over TCP; force OpenCV to use TCP transport
# instead of UDP (which causes "method SETUP failed: 461 Unsupported transport").
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def _is_network_source(source: Union[str, int]) -> bool:
    """True for RTSP/HTTP live URLs that need the FFmpeg backend + TCP transport."""
    if isinstance(source, int):
        return False
    s = str(source).strip().lower()
    return s.startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://"))


def _is_file_source(source: Union[str, int]) -> bool:
    if isinstance(source, int):
        return False
    s = str(source).strip().lower()
    if _is_network_source(s):
        return False
    return (
        s.startswith("file://")
        or any(s.endswith(ext) for ext in (".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".mpg", ".mpeg"))
        or "/" in s
        or "\\" in s
    )


class CameraStream:
    """Low-latency OpenCV capture from USB index, video file, or RTSP (go2rtc)."""

    def __init__(
        self,
        source: Union[str, int] = 0,
        buffer_size: int = 1,
        width: Optional[int] = None,
        height: Optional[int] = None,
        *,
        reconnect: bool = True,
        reconnect_delay_sec: float = 1.0,
        max_reconnect_attempts: int = 5,
        drop_stale: bool = True,
    ) -> None:
        self.source = source
        self.buffer_size = buffer_size
        self.width = width
        self.height = height
        self.reconnect = reconnect
        self.reconnect_delay_sec = reconnect_delay_sec
        self.max_reconnect_attempts = max_reconnect_attempts
        self.drop_stale = drop_stale
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_live = self._detect_live(source)
        self._fail_streak = 0

    @staticmethod
    def _detect_live(source: Union[str, int]) -> bool:
        if isinstance(source, int):
            return True
        s = str(source).strip().lower()
        return s.startswith(("rtsp://", "rtsps://", "http://", "https://")) or s.isdigit()

    def open(self) -> None:
        self.release()
        src = self.source
        if isinstance(src, str) and src.strip().isdigit():
            src = int(src.strip())

        if _is_network_source(src):
            # RTSP/HTTP: FFmpeg + OPENCV_FFMPEG_CAPTURE_OPTIONS (rtsp_transport;tcp)
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            # Local files + USB: default backend (AVFoundation on macOS works for .mp4;
            # CAP_FFMPEG often fails to open local files on opencv-python mac wheels).
            self.cap = cv2.VideoCapture(src)
            if (self.cap is None or not self.cap.isOpened()) and _is_file_source(src):
                # Fallback for Linux / builds where only FFmpeg decodes the container
                if self.cap is not None:
                    self.cap.release()
                self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

        if self.buffer_size is not None and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        if self.width and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")
        self._fail_streak = 0

    def _try_reconnect(self) -> bool:
        if not self.reconnect or not self._is_live:
            return False
        for attempt in range(1, self.max_reconnect_attempts + 1):
            time.sleep(self.reconnect_delay_sec)
            try:
                self.open()
                return True
            except RuntimeError:
                continue
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Fetch one frame. For live sources, optionally drop buffered stale frames."""
        if self.cap is None:
            self.open()
        assert self.cap is not None

        # Low-latency path: drain buffer, keep newest frame only
        if self.drop_stale and self._is_live and self.buffer_size and self.buffer_size <= 2:
            grabbed = False
            for _ in range(2):
                grabbed = self.cap.grab()
                if not grabbed:
                    break
            if grabbed:
                ok, frame = self.cap.retrieve()
            else:
                ok, frame = False, None
        else:
            ok, frame = self.cap.read()

        if ok and frame is not None:
            self._fail_streak = 0
            return True, frame

        self._fail_streak += 1
        # File EOF: stop cleanly without reconnect thrash
        if not self._is_live:
            return False, None

        if self._fail_streak >= 3 and self._try_reconnect():
            assert self.cap is not None
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self._fail_streak = 0
                return True, frame
        return False, None

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                break
            yield frame

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

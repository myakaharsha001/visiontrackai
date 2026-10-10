"""video_source.py - Webcam / video-file capture built on OpenCV.

One VideoSource owns exactly one cv2.VideoCapture and must be release()d.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config import SUPPORTED_VIDEO_EXTENSIONS
from logger import get_logger

log = get_logger(__name__)

WEBCAM = "webcam"
VIDEO_FILE = "video_file"


class SourceError(RuntimeError):
    """Base class for capture problems (message is safe to show to the user)."""


class CameraUnavailableError(SourceError):
    pass


class InvalidVideoError(SourceError):
    pass


@dataclass(frozen=True)
class SourceSpec:
    kind: str = WEBCAM                 # "webcam" | "video_file"
    camera_index: int = 0
    path: Optional[str] = None

    def describe(self) -> str:
        if self.kind == WEBCAM:
            return f"Webcam #{self.camera_index}"
        return f"Video: {Path(self.path).name if self.path else '(none)'}"


def validate_video_path(path: Optional[str]) -> Path:
    """Check a video path BEFORE opening it; raise InvalidVideoError with advice."""
    if not path or not str(path).strip():
        raise InvalidVideoError("No video file selected. Upload a file or type its full path.")
    p = Path(str(path).strip().strip('"'))
    if not p.exists():
        raise InvalidVideoError(f"Video file not found: {p}")
    if not p.is_file():
        raise InvalidVideoError(f"That path is a folder, not a video file: {p}")
    if p.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise InvalidVideoError(
            f"Unsupported file type '{p.suffix}'. Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}")
    return p


class VideoSource:
    def __init__(self, spec: SourceSpec):
        self.spec = spec
        self._cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.size: Tuple[int, int] = (0, 0)

    @property
    def source_type(self) -> str:
        return self.spec.kind

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> "VideoSource":
        if self.is_open:                     # never open the same device twice
            return self
        if self.spec.kind == WEBCAM:
            self._open_webcam()
        elif self.spec.kind == VIDEO_FILE:
            self._open_file()
        else:
            raise SourceError(f"Unknown source kind: {self.spec.kind!r}")
        return self

    def _open_webcam(self) -> None:
        idx = self.spec.camera_index
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            raise CameraUnavailableError(
                f"Could not open webcam #{idx}. Close other apps using the camera "
                "(Zoom, Teams, browser tabs), check Windows Settings > Privacy > Camera "
                "(allow desktop apps), or try a different camera index.")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Some cameras return empty frames for the first few reads while warming up.
        ok = False
        for _ in range(10):
            ok, _frame = cap.read()
            if ok:
                break
        if not ok:
            cap.release()
            raise CameraUnavailableError(
                f"Webcam #{idx} opened but returned no image. Another program may be using it.")
        self._cap = cap
        self._read_props()
        log.info("Webcam %d opened (%dx%d)", idx, *self.size)

    def _open_file(self) -> None:
        path = validate_video_path(self.spec.path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise InvalidVideoError(
                f"OpenCV could not decode '{path.name}'. The codec may be unsupported - "
                "try converting it to H.264 .mp4.")
        self._cap = cap
        self._read_props()
        log.info("Video file opened: %s (%.1f fps, %d frames)", path.name, self.fps, self.frame_count)

    def _read_props(self) -> None:
        assert self._cap is not None
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else (30.0 if self.spec.kind == VIDEO_FILE else 0.0)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_open:
            return False, None
        ok, frame = self._cap.read()
        return (True, frame) if ok and frame is not None else (False, None)

    def rewind(self) -> None:
        """Jump back to the first frame (video files only)."""
        if self._cap is not None and self.spec.kind == VIDEO_FILE:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
                log.info("Capture released (%s)", self.spec.describe())

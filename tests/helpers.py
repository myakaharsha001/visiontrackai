"""Fakes used by several test modules (no webcam, no YOLO weights, no MySQL)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from detector import Detection  # noqa: E402


def det(name="person", conf=0.9, box=(10, 20, 110, 220), cid=0):
    return Detection(cid, name, conf, *box)


class FakeDetector:
    """Returns a fixed list of detections for every frame."""
    def __init__(self, dets):
        self.dets = dets
        self.calls = 0

    def detect(self, frame, confidence, selected_classes=None, imgsz=640):
        self.calls += 1
        return list(self.dets)


class FakeSource:
    """Produces N black frames then ends."""
    def __init__(self, n_frames=5, kind="webcam"):
        from video_source import SourceSpec
        self.spec = SourceSpec(kind=kind, path="x.mp4" if kind == "video_file" else None)
        self.n = n_frames
        self.i = 0
        self.opened = False
        self.released = False

    @property
    def source_type(self):
        return self.spec.kind

    def open(self):
        self.opened = True
        return self

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.zeros((240, 320, 3), dtype=np.uint8)

    def rewind(self):
        self.i = 0

    def release(self):
        self.released = True


class ListSink:
    def __init__(self, accept=True):
        self.events = []
        self.accept = accept

    def submit(self, event):
        if self.accept:
            self.events.append(event)
        return self.accept

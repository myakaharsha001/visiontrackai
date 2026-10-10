"""pipeline.py - One detection run: read frame -> YOLO -> filter -> draw -> log.

DetectionSession contains NO Streamlit code, so it can be unit-tested with a
fake detector, fake source and fake sink.
"""
from __future__ import annotations

import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Protocol

import numpy as np

from config import DetectionSettings
from detector import (CooldownFilter, Detection, ObjectDetector, detection_to_event,
                      draw_detections, filter_detections)
from logger import get_logger
from video_source import VideoSource

log = get_logger(__name__)

IDLE, RUNNING, PAUSED, STOPPED, ENDED, ERROR = "idle", "running", "paused", "stopped", "ended", "error"


class EventSink(Protocol):
    def submit(self, event: Dict[str, Any]) -> bool: ...


@dataclass
class FrameResult:
    frame: np.ndarray                    # annotated BGR frame (or raw if boxes are off)
    detections: List[Detection]          # filtered detections drawn on this frame
    fps: float
    logged: int                          # events handed to the sink for this frame
    ran_inference: bool


class DetectionSession:
    def __init__(self, detector: ObjectDetector, source: VideoSource,
                 settings: DetectionSettings, sink: Optional[EventSink] = None,
                 session_id: Optional[str] = None):
        self.detector = detector
        self.source = source
        self.settings = settings
        self.sink = sink
        self.session_id = session_id or str(uuid.uuid4())
        self.status = IDLE
        self.error: Optional[str] = None
        self.cooldown = CooldownFilter(settings.cooldown_seconds)
        self.frames_read = 0
        self.frames_inferred = 0
        self.events_submitted = 0
        self.events_rejected = 0         # sink refused (queue full)
        self.class_counts: Counter = Counter()
        self.recent_events: Deque[dict] = deque(maxlen=50)
        self.last_result: Optional[FrameResult] = None
        self._last_dets: List[Detection] = []
        self._fps = 0.0
        self._t_prev: Optional[float] = None

    # ---- lifecycle ---------------------------------------------------
    def start(self) -> None:
        self.source.open()               # raises SourceError with a helpful message
        self.status = RUNNING
        self._t_prev = None

    def pause(self) -> None:
        if self.status == RUNNING:
            self.status = PAUSED

    def resume(self) -> None:
        if self.status == PAUSED:
            self.status = RUNNING
            self._t_prev = None

    def stop(self) -> None:
        self.source.release()
        if self.status not in (ENDED, ERROR):
            self.status = STOPPED

    def __del__(self):                   # safety net if a browser tab is closed mid-run
        try:
            self.source.release()
        except Exception:
            pass

    @property
    def active(self) -> bool:
        return self.status in (RUNNING, PAUSED)

    @property
    def fps(self) -> float:
        return self._fps

    # ---- one iteration -----------------------------------------------
    def step(self) -> Optional[FrameResult]:
        """Process one frame. Returns None when paused/ended/failed."""
        if self.status != RUNNING:
            return None
        s = self.settings
        ok, frame = self.source.read()
        if not ok:
            if s.loop_video and self.source.spec.kind == "video_file":
                self.source.rewind()
                ok, frame = self.source.read()
            if not ok:
                self.source.release()
                self.status = ENDED
                return None
        self.frames_read += 1

        ran = (self.frames_read - 1) % max(1, s.process_every_n) == 0
        logged = 0
        if ran:
            try:
                raw = self.detector.detect(frame, s.confidence, s.selected_classes, s.imgsz)
            except Exception as exc:     # inference failure ends the run visibly
                log.exception("Inference failed")
                self.error = f"Inference failed: {exc}"
                self.source.release()
                self.status = ERROR
                return None
            self.frames_inferred += 1
            # SAME filtered list feeds drawing, metrics AND logging.
            self._last_dets = filter_detections(raw, s.confidence, s.selected_classes)
            logged = self._log(self._last_dets)
        dets = self._last_dets

        now = time.perf_counter()
        if self._t_prev is not None:
            inst = 1.0 / max(now - self._t_prev, 1e-6)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._t_prev = now

        result = FrameResult(draw_detections(frame, dets, s.show_boxes), list(dets),
                             self._fps, logged, ran)
        self.last_result = result
        return result

    def _log(self, dets: List[Detection]) -> int:
        s = self.settings
        if not dets:
            return 0
        self.cooldown.cooldown_seconds = s.cooldown_seconds
        fresh = self.cooldown.filter(dets)
        count = 0
        for d in fresh:
            event = detection_to_event(d, self.source.source_type, self.session_id,
                                       datetime.now())
            self.class_counts[d.class_name] += 1
            self.recent_events.appendleft(event)     # in-memory list shown on Live page
            if s.logging_enabled and self.sink is not None:
                if self.sink.submit(event):
                    self.events_submitted += 1
                    count += 1
                else:
                    self.events_rejected += 1
        return count

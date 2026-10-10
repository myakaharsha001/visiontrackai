"""detector.py - YOLO model loading, inference, filtering and box drawing.

Coordinate convention (documented choice): every box is stored in ORIGINAL
FRAME PIXELS. YOLO gives corners (x1, y1, x2, y2); the database stores
(bbox_x, bbox_y, bbox_w, bbox_h) = (x1, y1, x2 - x1, y2 - y1).

Filtering rule (applied identically to the drawn feed, the on-screen metrics
and the database log, because they all use the SAME filtered list):
    keep a detection  <=>  confidence >= threshold  AND
                           (no class selected OR class name is selected)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from logger import get_logger
from utils import clamp_box, color_for_class, confidence_percent, xyxy_to_xywh

log = get_logger(__name__)


class ModelLoadError(RuntimeError):
    """The YOLO weights could not be loaded (missing package, no internet...)."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float          # 0.0 - 1.0
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox_xywh(self):
        return xyxy_to_xywh(self.x1, self.y1, self.x2, self.y2)


def filter_detections(detections: Iterable[Detection], min_confidence: float,
                      selected_classes: Optional[Sequence[str]] = None) -> List[Detection]:
    """Apply the confidence threshold and class filter (empty/None = all classes)."""
    wanted = set(selected_classes) if selected_classes else None
    return [d for d in detections
            if d.confidence >= min_confidence and (wanted is None or d.class_name in wanted)]


def detection_to_event(det: Detection, source_type: str, session_id: Optional[str],
                       timestamp: Optional[datetime] = None) -> dict:
    """Format a Detection as a database row dict (keys == table columns)."""
    x, y, w, h = det.bbox_xywh
    return {
        "timestamp": timestamp or datetime.now(),
        "object_class": det.class_name,
        "confidence": round(float(det.confidence), 4),
        "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
        "source_type": source_type,
        "session_id": session_id,
    }


class CooldownFilter:
    """Per-class duplicate-event suppression (a practical heuristic).

    After a class is logged, further detections of that SAME CLASS are
    suppressed for ``cooldown_seconds``. This is NOT object tracking: it cannot
    tell two different people apart or know that a person left and came back.
    """

    def __init__(self, cooldown_seconds: float = 1.0,
                 clock: Callable[[], float] = time.monotonic):
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._last_logged: Dict[str, float] = {}

    def reset(self) -> None:
        self._last_logged.clear()

    def filter(self, detections: Iterable[Detection]) -> List[Detection]:
        """Return the detections that may be logged now.

        If a class is allowed, ALL its detections in this frame pass (so two
        people in one frame give two rows) and the class is then locked.
        """
        now = self._clock()
        allowed_classes = set()
        out: List[Detection] = []
        for d in detections:
            if d.class_name in allowed_classes:
                out.append(d)
                continue
            last = self._last_logged.get(d.class_name)
            if last is None or (now - last) >= self.cooldown_seconds:
                allowed_classes.add(d.class_name)
                out.append(d)
        for name in allowed_classes:
            self._last_logged[name] = now
        return out


class ObjectDetector:
    """Thin wrapper around an Ultralytics YOLO model.

    ``model`` can be injected (used by tests) so no weights are needed.
    """

    def __init__(self, model_name: str = "yolov8n.pt", model=None):
        self.model_name = model_name
        self._model = model
        self._names: Dict[int, str] = {}
        if model is not None:
            self._names = self._read_names(model)

    # ---- loading ---------------------------------------------------
    def load(self) -> "ObjectDetector":
        if self._model is not None:
            return self
        try:
            from ultralytics import YOLO  # imported lazily: heavy import
        except ImportError as exc:
            raise ModelLoadError(
                "The 'ultralytics' package is not installed. Activate your virtual "
                "environment and run: pip install -r requirements.txt") from exc
        try:
            log.info("Loading YOLO model %s ...", self.model_name)
            self._model = YOLO(self.model_name)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load YOLO weights '{self.model_name}': {exc}\n"
                "On first run Ultralytics downloads the weights (~6 MB for yolov8n.pt) "
                "from GitHub, so an internet connection is needed once. If you are offline, "
                "download yolov8n.pt manually and place it in the project folder.") from exc
        self._names = self._read_names(self._model)
        log.info("Model loaded with %d classes", len(self._names))
        return self

    @staticmethod
    def _read_names(model) -> Dict[int, str]:
        names = getattr(model, "names", None) or {}
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        return {int(k): str(v) for k, v in names.items()}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> List[str]:
        """Class names straight from the loaded model (nothing hardcoded)."""
        return [self._names[k] for k in sorted(self._names)]

    def class_ids_for(self, selected: Optional[Sequence[str]]) -> Optional[List[int]]:
        if not selected:
            return None
        wanted = set(selected)
        return [i for i, n in self._names.items() if n in wanted]

    # ---- inference -------------------------------------------------
    def detect(self, frame: np.ndarray, confidence: float,
               selected_classes: Optional[Sequence[str]] = None,
               imgsz: int = 640) -> List[Detection]:
        if self._model is None:
            raise ModelLoadError("Model is not loaded yet - call load() first.")
        h, w = frame.shape[:2]
        results = self._model.predict(frame, conf=confidence, imgsz=imgsz,
                                      classes=self.class_ids_for(selected_classes),
                                      verbose=False)
        detections: List[Detection] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = clamp_box(float(x1), float(y1), float(x2), float(y2), w, h)
                detections.append(Detection(int(k), self._names.get(int(k), str(k)),
                                            float(c), x1, y1, x2, y2))
        # Explicit second pass: guarantees the documented rule no matter what
        # the model/library already did.
        return filter_detections(detections, confidence, selected_classes)


def draw_detections(frame: np.ndarray, detections: Sequence[Detection],
                    show_boxes: bool = True) -> np.ndarray:
    """Return a COPY of ``frame`` with boxes + 'name 87.5%' labels drawn on it."""
    out = frame.copy()
    if not show_boxes:
        return out
    for d in detections:
        color = color_for_class(d.class_name)
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        label = f"{d.class_name} {confidence_percent(d.confidence)}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(p1[1] - th - base - 4, 0)
        cv2.rectangle(out, (p1[0], top), (p1[0] + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(out, label, (p1[0] + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (15, 15, 15), 1, cv2.LINE_AA)
    return out

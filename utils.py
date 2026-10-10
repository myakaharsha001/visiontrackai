"""utils.py - Small, dependency-light helpers shared across modules."""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Iterable, Optional, Sequence, Tuple


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int, int, int]:
    """Convert YOLO corner format (x1, y1, x2, y2) to database format (x, y, w, h).

        bbox_x = x1      bbox_y = y1
        bbox_w = x2 - x1 bbox_h = y2 - y1

    Values are original-frame PIXELS (not normalised), rounded to integers.
    Width/height never go below 0 even if a model returns swapped corners.
    """
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))


def clamp_box(x1: float, y1: float, x2: float, y2: float,
              width: int, height: int) -> Tuple[float, float, float, float]:
    """Clip a box so it lies fully inside a ``width`` x ``height`` frame."""
    return (min(max(x1, 0), width), min(max(y1, 0), height),
            min(max(x2, 0), width), min(max(y2, 0), height))


def confidence_percent(confidence: float, digits: int = 1) -> str:
    """0.8532 -> '85.3%'."""
    return f"{confidence * 100:.{digits}f}%"


def color_for_class(name: str) -> Tuple[int, int, int]:
    """Stable BGR colour per class name, kept in a bright cyan/blue/violet range."""
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:6], 16)
    palette = [(238, 211, 34), (246, 130, 59), (246, 92, 139), (250, 165, 96),
               (153, 211, 52), (36, 191, 251), (212, 182, 6), (241, 102, 99)]
    return palette[h % len(palette)]


def escape_like(text: str) -> str:
    """Escape SQL LIKE wildcards so user search text is matched literally."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def format_timestamp(ts: Optional[datetime]) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-"


def unique_sorted(items: Iterable[str]) -> Sequence[str]:
    return sorted(set(items))

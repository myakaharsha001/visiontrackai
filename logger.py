"""logger.py - Central logging configuration (console + rotating file)."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure the root logger once (safe to call on every Streamlit rerun)."""
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)
    if log_file:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(fh)
        except OSError as exc:  # logging to file is optional
            root.warning("Could not open log file %s: %s", log_file, exc)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

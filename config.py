"""config.py - Environment variables and application-wide configuration.

All configuration is read ONCE from the environment (optionally populated from a
``.env`` file by python-dotenv). No other module reads ``os.environ`` directly,
which keeps configuration logic in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

APP_NAME = "VisionTrack AI"
APP_TAGLINE = "Real-Time Object Detection & Event Monitoring Platform"
APP_VERSION = "1.0.0"

# Allowed range for the confidence slider (Section 5 of the assignment brief).
MIN_CONFIDENCE = 0.10
MAX_CONFIDENCE = 0.95

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v")

PLACEHOLDER_PASSWORD = "replace_with_your_local_password"


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str = field(repr=False)  # never printed in logs/tracebacks
    pool_size: int = 3


@dataclass(frozen=True)
class AppConfig:
    database: DatabaseConfig
    model_name: str = "yolov8n.pt"
    default_confidence: float = 0.50
    default_cooldown_seconds: float = 1.0
    default_imgsz: int = 640
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/visiontrack.log"


def _require(env: Mapping[str, str], key: str, missing: list) -> str:
    value = env.get(key)
    if value is None or (value.strip() == "" and key != "DB_PASSWORD"):
        missing.append(key)
        return ""
    return value.strip() if key != "DB_PASSWORD" else value


def _parse_float(env, key, default, lo, hi, problems):
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = float(raw)
    except ValueError:
        problems.append(f"{key} must be a number, got {raw!r}")
        return default
    if not lo <= val <= hi:
        problems.append(f"{key} must be between {lo} and {hi}, got {val}")
        return default
    return val


def _parse_int(env, key, default, lo, hi, problems):
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = int(raw)
    except ValueError:
        problems.append(f"{key} must be a whole number, got {raw!r}")
        return default
    if not lo <= val <= hi:
        problems.append(f"{key} must be between {lo} and {hi}, got {val}")
        return default
    return val


def build_config(env: Mapping[str, str]) -> AppConfig:
    """Validate a mapping of environment variables and build an AppConfig.

    Pure function (no file or OS access) so it is easy to unit test.
    Raises ConfigError with a beginner-friendly message listing every problem.
    """
    missing: list = []
    problems: list = []

    host = _require(env, "DB_HOST", missing)
    name = _require(env, "DB_NAME", missing)
    user = _require(env, "DB_USER", missing)
    password = _require(env, "DB_PASSWORD", missing)
    port = _parse_int(env, "DB_PORT", 3306, 1, 65535, problems)

    if password == PLACEHOLDER_PASSWORD:
        problems.append("DB_PASSWORD still has the placeholder value from .env.example - "
                        "put your real local MySQL password in your .env file")

    model_name = (env.get("MODEL_NAME") or "yolov8n.pt").strip()
    if not model_name.lower().endswith(".pt"):
        problems.append(f"MODEL_NAME must be a .pt weights file (e.g. yolov8n.pt), got {model_name!r}")

    default_conf = _parse_float(env, "DEFAULT_CONFIDENCE", 0.50, MIN_CONFIDENCE, MAX_CONFIDENCE, problems)
    cooldown = _parse_float(env, "DETECTION_COOLDOWN_SECONDS", 1.0, 0.0, 60.0, problems)
    imgsz = _parse_int(env, "INFERENCE_IMGSZ", 640, 160, 1280, problems)
    pool = _parse_int(env, "DB_POOL_SIZE", 3, 1, 10, problems)

    log_level = (env.get("LOG_LEVEL") or "INFO").strip().upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        problems.append(f"LOG_LEVEL must be DEBUG/INFO/WARNING/ERROR/CRITICAL, got {log_level!r}")
        log_level = "INFO"
    log_file = (env.get("LOG_FILE") or "logs/visiontrack.log").strip() or None

    if missing or problems:
        lines = ["Configuration problem(s) found:"]
        if missing:
            lines.append("  Missing required variable(s): " + ", ".join(missing))
        lines.extend(f"  - {p}" for p in problems)
        lines.append("Copy .env.example to .env, fill in your values, then restart the app.")
        raise ConfigError("\n".join(lines))

    return AppConfig(
        database=DatabaseConfig(host=host, port=port, name=name, user=user,
                                password=password, pool_size=pool),
        model_name=model_name, default_confidence=default_conf,
        default_cooldown_seconds=cooldown, default_imgsz=imgsz,
        log_level=log_level, log_file=log_file,
    )


def load_config(env_path: Optional[Path] = None) -> AppConfig:
    """Load ``.env`` (if present) and return a validated AppConfig."""
    load_dotenv(env_path or PROJECT_ROOT / ".env", override=False)
    return build_config(os.environ)


@dataclass
class DetectionSettings:
    """Runtime settings the user changes in the dashboard.

    ``selected_classes`` is a list of class NAMES; an empty list means
    "detect all classes".
    """
    confidence: float = 0.50
    selected_classes: list = field(default_factory=list)
    logging_enabled: bool = True
    show_boxes: bool = True
    imgsz: int = 640
    cooldown_seconds: float = 1.0
    process_every_n: int = 1      # run YOLO on every Nth frame (performance)
    max_display_fps: int = 24     # cap on how fast frames are pushed to the browser
    loop_video: bool = False

    def validate(self) -> None:
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(f"confidence must be in [{MIN_CONFIDENCE}, {MAX_CONFIDENCE}]")
        if self.process_every_n < 1:
            raise ValueError("process_every_n must be >= 1")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")

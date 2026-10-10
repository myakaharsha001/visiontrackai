"""ui.py - Streamlit dashboard: widgets, pages and the live-video render loop.

This module only DRAWS and collects input. Detection lives in pipeline.py /
detector.py, storage in database.py, and start/stop orchestration in main.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, List, Optional

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

import theme
from config import (APP_NAME, APP_TAGLINE, APP_VERSION, MAX_CONFIDENCE, MIN_CONFIDENCE,
                    PROJECT_ROOT, SUPPORTED_VIDEO_EXTENSIONS, AppConfig, DetectionSettings)
from database import Database, DatabaseError, HistoryFilter
from detector import ObjectDetector
from pipeline import ENDED, ERROR, IDLE, PAUSED, RUNNING, STOPPED
from video_source import VIDEO_FILE, WEBCAM, SourceSpec

PAGES = ["📊 Overview", "🎥 Live Detection Studio", "🗂 Detection History",
         "📈 Analytics", "⚙️ Settings", "ℹ️ About & Help"]
LIVE_PAGE = PAGES[1]
SOURCE_LABELS = {WEBCAM: "Laptop webcam", VIDEO_FILE: "Local video file"}
UPLOAD_DIR = PROJECT_ROOT / ".uploads"
MAX_DISPLAY_WIDTH = 960
IMGSZ_OPTIONS = [320, 416, 480, 640, 800, 960, 1280]

CFG_KEYS = ["cfg_confidence", "cfg_classes", "cfg_logging", "cfg_boxes", "cfg_imgsz",
            "cfg_cooldown", "cfg_every_n", "cfg_max_fps", "cfg_loop",
            "cfg_source_kind", "cfg_camera_index", "cfg_video_path"]


@dataclass
class AppContext:
    config: AppConfig
    db: Database
    db_ok: bool
    db_msg: str
    detector: Optional[ObjectDetector]
    model_error: Optional[str]
    controller: Any                      # main.LiveController
    retry_model: Callable[[], None]


# --------------------------------------------------------------------------
# State helpers (settings live in st.session_state under the cfg_* keys)
# --------------------------------------------------------------------------
def init_state(config: AppConfig) -> None:
    defaults = {
        "cfg_confidence": round(round(float(config.default_confidence) / 0.05) * 0.05, 2), "cfg_classes": [],
        "cfg_logging": True, "cfg_boxes": True,
        "cfg_imgsz": min(IMGSZ_OPTIONS, key=lambda o: abs(o - int(config.default_imgsz))),
        "cfg_cooldown": float(config.default_cooldown_seconds), "cfg_every_n": 1,
        "cfg_max_fps": 24, "cfg_loop": False, "cfg_source_kind": WEBCAM,
        "cfg_camera_index": 0, "cfg_video_path": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    # Streamlit deletes the state of widgets that are not drawn on the current
    # page. Re-assigning the value each run keeps settings alive across pages.
    for k in CFG_KEYS:
        st.session_state[k] = st.session_state[k]


def settings_from_state() -> DetectionSettings:
    ss = st.session_state
    return DetectionSettings(
        confidence=float(ss.cfg_confidence), selected_classes=list(ss.cfg_classes),
        logging_enabled=bool(ss.cfg_logging), show_boxes=bool(ss.cfg_boxes),
        imgsz=int(ss.cfg_imgsz), cooldown_seconds=float(ss.cfg_cooldown),
        process_every_n=int(ss.cfg_every_n), max_display_fps=int(ss.cfg_max_fps),
        loop_video=bool(ss.cfg_loop))


def source_spec_from_state() -> SourceSpec:
    ss = st.session_state
    if ss.cfg_source_kind == VIDEO_FILE:
        return SourceSpec(kind=VIDEO_FILE, path=(ss.cfg_video_path or "").strip() or None)
    return SourceSpec(kind=WEBCAM, camera_index=int(ss.cfg_camera_index))


def sanitize_class_selection(class_names: List[str]) -> None:
    st.session_state.cfg_classes = [c for c in st.session_state.cfg_classes if c in class_names]


# --------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        f'<div class="vt-hero">{theme.LOGO_SVG}<div><div class="vt-title">{APP_NAME}</div>'
        f'<div class="vt-sub">{APP_TAGLINE}</div></div></div>', unsafe_allow_html=True)


def cards(items: List[tuple]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(theme.card(*item), unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="vt-section">{theme.esc(title)}</div>', unsafe_allow_html=True)


def show_config_error(message: str) -> None:
    hero()
    st.error("The application could not start because of a configuration problem.")
    st.code(message, language="text")
    st.info("Fix the .env file (see README, step 'Create the .env file') and refresh this page.")


def db_banner(ctx: AppContext) -> None:
    if not ctx.db_ok:
        st.warning(f"**MySQL is not reachable.** Detection still works, but events cannot be saved "
                   f"and history/analytics are unavailable.\n\n`{ctx.db_msg}`")


def _plot(fig) -> None:
    st.plotly_chart(theme.style_figure(fig), config={"displayModeBar": False})


def _show_image(slot, rgb) -> None:
    """Render a frame, stretched to the column width on any Streamlit version."""
    global _STRETCH_KW
    attempts = [_STRETCH_KW] if _STRETCH_KW is not None else [
        {"width": "stretch"}, {"use_container_width": True}, {}]
    last_exc: Optional[Exception] = None
    for kw in attempts:
        try:
            slot.image(rgb, channels="RGB", output_format="JPEG", **kw)
            _STRETCH_KW = kw
            return
        except Exception as exc:          # Streamlit control-flow exceptions are BaseException
            last_exc = exc
    raise last_exc  # type: ignore[misc]


_STRETCH_KW: Optional[dict] = None


def sidebar(ctx: AppContext) -> str:
    with st.sidebar:
        st.markdown(f'<div class="vt-hero">{theme.LOGO_SVG}<div><div class="vt-title" '
                    f'style="font-size:1.25rem">{APP_NAME}</div>'
                    f'<div class="vt-sub">v{APP_VERSION}</div></div></div>', unsafe_allow_html=True)
        page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        sess = ctx.controller.session
        status = sess.status if sess else IDLE
        kind = {RUNNING: "green", PAUSED: "amber", ERROR: "red"}.get(status, "blue")
        st.markdown("**Status**")
        st.markdown(theme.pill(status.upper(), kind, live=status == RUNNING) + " "
                    + theme.pill("MySQL OK" if ctx.db_ok else "MySQL DOWN", "green" if ctx.db_ok else "red")
                    + " " + theme.pill("MODEL OK" if ctx.detector else "NO MODEL",
                                       "green" if ctx.detector else "red"), unsafe_allow_html=True)
        st.caption("Model: " + ctx.config.model_name)
    return page


# --------------------------------------------------------------------------
# Shared controls (used on Live Studio and Settings)
# --------------------------------------------------------------------------
def detection_controls(class_names: List[str]) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Minimum confidence", MIN_CONFIDENCE, MAX_CONFIDENCE, step=0.05,
                  key="cfg_confidence",
                  help="Detections below this value are not drawn, counted or logged.")
        st.multiselect("Object classes (empty = detect all)", class_names, key="cfg_classes",
                       help="Options come from the loaded YOLO model.")
    with c2:
        st.toggle("Log detections to MySQL", key="cfg_logging")
        st.toggle("Draw bounding boxes", key="cfg_boxes")
        st.number_input("Per-class logging cooldown (seconds)", 0.0, 60.0, step=0.5, key="cfg_cooldown",
                        help="After a class is logged, the same class is not logged again for this long. "
                             "A duplicate-suppression heuristic, NOT object tracking.")
    with st.expander("Performance options"):
        p1, p2, p3 = st.columns(3)
        p1.select_slider("Inference image size", IMGSZ_OPTIONS, key="cfg_imgsz",
                         help="Smaller = faster on CPU, larger = better for small objects.")
        p2.number_input("Run YOLO every Nth frame", 1, 10, key="cfg_every_n")
        p3.number_input("Max display FPS", 5, 60, key="cfg_max_fps")


def source_controls() -> None:
    st.radio("Video source", list(SOURCE_LABELS), key="cfg_source_kind", horizontal=True,
             format_func=lambda k: SOURCE_LABELS[k])
    if st.session_state.cfg_source_kind == WEBCAM:
        st.number_input("Camera index (0 = default laptop webcam)", 0, 10, key="cfg_camera_index")
    else:
        up = st.file_uploader("Upload a video", type=[e.strip(".") for e in SUPPORTED_VIDEO_EXTENSIONS])
        if up is not None:
            marker = (up.name, up.size)
            if st.session_state.get("cfg_last_upload") != marker:
                UPLOAD_DIR.mkdir(exist_ok=True)
                target = UPLOAD_DIR / Path(up.name).name
                target.write_bytes(up.getbuffer())
                st.session_state.cfg_video_path = str(target)
                st.session_state.cfg_last_upload = marker
        st.text_input("...or full path of a video on this computer", key="cfg_video_path",
                      placeholder=r"C:\Users\you\Videos\sample.mp4")
        st.checkbox("Loop video", key="cfg_loop")


# --------------------------------------------------------------------------
# Page: Overview
# --------------------------------------------------------------------------
def page_overview(ctx: AppContext) -> None:
    hero()
    db_banner(ctx)
    sess = ctx.controller.session
    settings = settings_from_state()
    spec = source_spec_from_state()
    summary = None
    if ctx.db_ok:
        try:
            summary = ctx.db.summary()
        except DatabaseError as exc:
            st.error(f"Could not read statistics: {exc}")

    section("Database totals (real records)")
    if summary:
        top = summary["top_class"] or "-"
        cards([("Total detection events", f"{summary['total']:,}", "rows in detection_events", "cyan"),
               ("Unique object classes", summary["unique_classes"], "distinct classes logged", "blue"),
               ("Most frequent object", top, f"{summary['top_class_count']:,} events" if summary["top_class"] else "no data yet", "violet"),
               ("Average confidence", f"{summary['avg_confidence']*100:.1f}%" if summary["avg_confidence"] is not None else "-", "of logged events", "green")])
    else:
        st.markdown(theme.empty_state("No database statistics available",
                                      "Connect MySQL (see Settings) to see totals."), unsafe_allow_html=True)

    section("Current state")
    cards([("Processing status", (sess.status if sess else IDLE).title(), "detection engine", "cyan"),
           ("Confidence threshold", f"{settings.confidence*100:.0f}%", "minimum to draw & log", "blue"),
           ("Video source", SOURCE_LABELS[spec.kind], spec.describe(), "violet"),
           ("Database", "Connected" if ctx.db_ok else "Offline", ctx.config.database.name,
            "green" if ctx.db_ok else "red")])

    if not summary:
        return
    if summary["total"] == 0:
        st.markdown(theme.empty_state("No detections logged yet",
                                      "Open the Live Detection Studio and start the webcam or a video."),
                    unsafe_allow_html=True)
        return

    left, right = st.columns(2)
    with left:
        section("Object distribution")
        df = ctx.db.class_counts(limit=15)
        fig = px.bar(df, x="detections", y="object_class", orientation="h")
        fig.update_traces(marker_color=theme.CYAN)
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(title="Detections")
        _plot(fig)
    with right:
        section("Detection activity over time")
        act = ctx.db.activity_over_time("minute")
        fig = px.area(act, x="bucket", y="detections")
        fig.update_traces(line_color=theme.VIOLET, fillcolor="rgba(139,92,246,.25)")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="Detections / minute")
        _plot(fig)

    section("Recent detections")
    _history_table(ctx.db.fetch_recent(15))


def _history_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.markdown(theme.empty_state("No records match."), unsafe_allow_html=True)
        return
    view = df.copy()
    view["confidence"] = (view["confidence"] * 100).round(1)
    view = view.rename(columns={"confidence": "confidence_%"})
    st.dataframe(view, hide_index=True)


# --------------------------------------------------------------------------
# Page: Live Detection Studio
# --------------------------------------------------------------------------
def page_live(ctx: AppContext) -> None:
    hero()
    db_banner(ctx)
    ctrl = ctx.controller
    sess = ctrl.session

    if ctx.detector is None:
        st.error(f"The YOLO model is not loaded.\n\n{ctx.model_error}")
        st.button("Retry loading model", on_click=ctx.retry_model)
        return

    status = sess.status if sess else IDLE
    active = status in (RUNNING, PAUSED)

    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    start = b1.button("▶ Start", type="primary", disabled=active, key="btn_start")
    toggle_label = "⏯ Resume" if status == PAUSED else "⏸ Pause"
    toggle = b2.button(toggle_label, disabled=not active, key="btn_pause")
    stop = b3.button("⏹ Stop", disabled=not active, key="btn_stop")

    settings = settings_from_state()
    # After a state change we rerun once so the buttons above are redrawn with the
    # correct enabled/disabled state (the video loop below never redraws them).
    if start:
        problem = ctrl.start(settings, source_spec_from_state())
        if problem:
            st.error(problem)
        else:
            st.rerun()
    elif toggle and sess is not None:
        ctrl.resume() if status == PAUSED else ctrl.pause()
        st.rerun()
    elif stop:
        ctrl.stop()
        st.rerun()
    sess = ctrl.session
    if sess is not None:
        ctrl.sync_settings(settings)

    video_col, side_col = st.columns([3, 2])
    with video_col:
        frame_slot = st.empty()
    with side_col:
        stats_slot = st.empty()
        warn_slot = st.empty()
        st.markdown('<div class="vt-section">Recent events (this run)</div>', unsafe_allow_html=True)
        events_slot = st.empty()

    with st.expander("⚙ Source & detection controls", expanded=not active):
        if active:
            st.caption("Changing the source takes effect the next time you press Start. "
                       "Confidence, classes and the other detection options apply immediately.")
        source_controls()
        detection_controls(ctx.detector.class_names)

    _render_side(sess, ctrl, stats_slot, warn_slot, events_slot)

    if sess is None or sess.status == IDLE:
        frame_slot.markdown(theme.empty_state("Ready", "Press Start to open the selected source."),
                            unsafe_allow_html=True)
        return
    if sess.last_result is not None:
        _show_image(frame_slot, _to_display(sess.last_result.frame))
    if sess.status == RUNNING:
        _run_loop(sess, ctrl, frame_slot, stats_slot, warn_slot, events_slot)
    _after_loop(sess, ctrl)


def _to_display(frame):
    h, w = frame.shape[:2]
    if w > MAX_DISPLAY_WIDTH:
        frame = cv2.resize(frame, (MAX_DISPLAY_WIDTH, int(h * MAX_DISPLAY_WIDTH / w)),
                           interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _run_loop(sess, ctrl, frame_slot, stats_slot, warn_slot, events_slot) -> None:
    """Pull frames until paused/stopped/ended.

    A widget click makes Streamlit interrupt this loop and rerun the script;
    the DetectionSession (and its single capture) survives in session_state, so
    the next run simply continues from the next frame.
    """
    last_side = 0.0
    while sess.status == RUNNING:
        t0 = time.perf_counter()
        result = sess.step()
        if result is None:
            break
        _show_image(frame_slot, _to_display(result.frame))
        if t0 - last_side > 0.5:
            _render_side(sess, ctrl, stats_slot, warn_slot, events_slot)
            last_side = t0
        cap = max(5, sess.settings.max_display_fps)
        if sess.source.spec.kind == VIDEO_FILE and sess.source.fps:
            cap = min(cap, sess.source.fps)               # don't play video faster than real time
        remaining = 1.0 / cap - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
    _render_side(sess, ctrl, stats_slot, warn_slot, events_slot)


def _after_loop(sess, ctrl) -> None:
    if sess.status in (ENDED, ERROR):
        ctrl.finalize()
        if st.session_state.get("ended_synced") != sess.session_id:
            st.session_state.ended_synced = sess.session_id
            st.rerun()                      # redraw the buttons once (Start enabled again)
        if sess.status == ERROR:
            st.error(sess.error or "The detection run failed.")
        else:
            st.success("The video finished. Press Start to run it again.")
    elif sess.status == STOPPED:
        st.info("Stopped. The webcam / video file has been released.")


def _render_side(sess, ctrl, stats_slot, warn_slot, events_slot) -> None:
    if sess is None:
        return
    res = sess.last_result
    n_obj = len(res.detections) if res else 0
    w = ctrl.writer_stats()
    with stats_slot.container():
        cards([("Status", sess.status.title(), "engine", "cyan"),
               ("FPS", f"{sess.fps:.1f}", "processing speed", "blue")])
        cards([("Objects in frame", n_obj, "no objects detected" if (res and not n_obj) else "after filters", "violet"),
               ("Saved to MySQL", w["inserted"], f"{w['pending']} pending", "green")])
    msgs = []
    if w["failed"]:
        msgs.append(f"{w['failed']} event(s) could NOT be saved to MySQL and were skipped (not retried). "
                    f"Last error: {w['last_error']}")
    if w["dropped"] or sess.events_rejected:
        msgs.append(f"{w['dropped'] + sess.events_rejected} event(s) dropped because the write queue was full.")
    if msgs:
        warn_slot.warning("\n\n".join(msgs))
    else:
        warn_slot.empty()
    if sess.recent_events:
        df = pd.DataFrame(list(sess.recent_events)[:10])[["timestamp", "object_class", "confidence"]]
        df["timestamp"] = df["timestamp"].dt.strftime("%H:%M:%S")
        df["confidence"] = (df["confidence"] * 100).round(1)
        events_slot.dataframe(df.rename(columns={"confidence": "confidence_%"}), hide_index=True)
    else:
        events_slot.markdown(theme.empty_state("No events yet", "Events appear when objects pass the filters."),
                             unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Page: Detection History
# --------------------------------------------------------------------------
def page_history(ctx: AppContext) -> None:
    hero()
    db_banner(ctx)
    if not ctx.db_ok:
        return
    try:
        known = ctx.db.distinct_classes()
    except DatabaseError as exc:
        st.error(str(exc))
        return
    f1, f2, f3 = st.columns([2, 2, 1])
    search = f1.text_input("Search object class", key="h_search", placeholder="e.g. person")
    classes = f2.multiselect("Filter by class", known, key="h_classes")
    source = f3.selectbox("Source", ["Any", "webcam", "video_file"], key="h_source")
    use_dates = st.checkbox("Filter by date", key="h_use_dates")
    d_from = d_to = None
    if use_dates:
        d1, d2 = st.columns(2)
        d_from = d1.date_input("From", value=date.today(), key="h_from")
        d_to = d2.date_input("To", value=date.today(), key="h_to")
    flt = HistoryFilter(search=search, classes=classes, date_from=d_from, date_to=d_to,
                        source_type=None if source == "Any" else source)

    try:
        total = ctx.db.count_history(flt)
        p1, p2, p3 = st.columns([1, 1, 3])
        size = p1.selectbox("Rows per page", [25, 50, 100, 250], index=1, key="h_size")
        pages = max(1, -(-total // size))
        if st.session_state.get("h_page", 1) > pages:
            st.session_state.h_page = pages
        page = p2.number_input("Page", 1, pages, key="h_page")
        p3.markdown(f"<div class='vt-sub' style='padding-top:30px'>{total:,} matching records · page {page} of {pages}</div>",
                    unsafe_allow_html=True)
        df = ctx.db.fetch_history(flt, limit=size, offset=(int(page) - 1) * size)
    except DatabaseError as exc:
        st.error(str(exc))
        return
    if total == 0:
        st.markdown(theme.empty_state("No records match these filters",
                                      "Run a detection, or relax the filters."), unsafe_allow_html=True)
        return
    _history_table(df)

    section("Export")
    e1, e2 = st.columns([1, 3])
    if e1.button("Prepare CSV export", key="h_prepare"):
        try:
            export = ctx.db.export_history(flt)
            st.session_state.h_csv = export.to_csv(index=False).encode("utf-8")
            st.session_state.h_csv_rows = len(export)
        except DatabaseError as exc:
            st.error(str(exc))
    if st.session_state.get("h_csv"):
        e2.download_button(f"⬇ Download CSV ({st.session_state.h_csv_rows:,} rows)",
                           st.session_state.h_csv, "visiontrack_detections.csv", "text/csv")
        e2.caption("The file contains the rows matching the filters above (up to 100,000).")


# --------------------------------------------------------------------------
# Page: Analytics
# --------------------------------------------------------------------------
def page_analytics(ctx: AppContext) -> None:
    hero()
    db_banner(ctx)
    if not ctx.db_ok:
        return
    try:
        s = ctx.db.summary()
        if s["total"] == 0:
            st.markdown(theme.empty_state("No analytics yet", "Log some detections first."),
                        unsafe_allow_html=True)
            return
        counts = ctx.db.class_counts(limit=20)
        conf = ctx.db.confidence_values()
        bucket = st.radio("Activity bucket", ["minute", "hour", "day"], horizontal=True, key="a_bucket")
        act = ctx.db.activity_over_time(bucket)
    except DatabaseError as exc:
        st.error(str(exc))
        return

    cards([("Total events", f"{s['total']:,}", "all time", "cyan"),
           ("Unique classes", s["unique_classes"], "distinct", "blue"),
           ("Average confidence", f"{s['avg_confidence']*100:.1f}%", "logged events", "green"),
           ("First → last event", f"{s['first_ts']:%d %b %H:%M}", f"to {s['last_ts']:%d %b %H:%M}", "violet")])

    a, b = st.columns(2)
    with a:
        section("Detections by object class")
        fig = px.bar(counts, x="object_class", y="detections")
        fig.update_traces(marker_color=theme.CYAN)
        fig.update_xaxes(title=None)
        _plot(fig)
    with b:
        section("Confidence distribution")
        fig = px.histogram(conf.assign(pct=conf["confidence"] * 100), x="pct", nbins=20)
        fig.update_traces(marker_color=theme.BLUE)
        fig.update_xaxes(title="Confidence (%)", range=[0, 100])
        fig.update_yaxes(title="Events")
        _plot(fig)
    section(f"Detection activity per {bucket}")
    fig = px.line(act, x="bucket", y="detections", markers=True)
    fig.update_traces(line_color=theme.VIOLET)
    fig.update_xaxes(title=None)
    _plot(fig)
    section("Most frequently detected objects")
    top = counts.head(10).copy()
    top["avg_confidence"] = (top["avg_confidence"] * 100).round(1)
    st.dataframe(top.rename(columns={"avg_confidence": "avg_confidence_%"}), hide_index=True)


# --------------------------------------------------------------------------
# Page: Settings
# --------------------------------------------------------------------------
def page_settings(ctx: AppContext) -> None:
    hero()
    db_banner(ctx)
    st.subheader("Detection settings")
    if ctx.detector is not None:
        detection_controls(ctx.detector.class_names)
    else:
        st.error(f"Model not loaded: {ctx.model_error}")
        st.button("Retry loading model", on_click=ctx.retry_model, key="retry_settings")
    st.subheader("Video source")
    source_controls()

    st.subheader("Database")
    cfg = ctx.config.database
    st.markdown(theme.pill("CONNECTED" if ctx.db_ok else "OFFLINE", "green" if ctx.db_ok else "red"),
                unsafe_allow_html=True)
    st.write(f"Host `{cfg.host}:{cfg.port}` · database `{cfg.name}` · user `{cfg.user}` (password hidden)")
    if not ctx.db_ok:
        st.code(ctx.db_msg, language="text")
    c1, c2 = st.columns(2)
    if c1.button("Test connection", key="s_test"):
        ok, msg = ctx.db.check_connection()
        (st.success if ok else st.error)(msg)
    if c2.button("Create table if missing", key="s_init"):
        try:
            ctx.db.init_schema()
            st.success("Table detection_events is ready.")
        except DatabaseError as exc:
            st.error(str(exc))

    st.subheader("Model")
    det = ctx.detector
    st.write(f"Weights: `{ctx.config.model_name}` · "
             + (f"loaded, {len(det.class_names)} classes" if det else "not loaded"))
    st.subheader("Logging")
    st.write(f"Level `{ctx.config.log_level}` · file `{ctx.config.log_file or 'console only'}`")

    st.subheader("Changing environment variables")
    st.markdown(
        "These values come from the `.env` file in the project folder (copy `.env.example`). "
        "Edit the file, then **stop the app (Ctrl+C) and run `streamlit run main.py` again**.")
    st.dataframe(pd.DataFrame([
        ("DB_HOST / DB_PORT", "MySQL address", "localhost / 3306"),
        ("DB_NAME / DB_USER / DB_PASSWORD", "Database and login", "vision_platform / vision_user / (yours)"),
        ("MODEL_NAME", "YOLO weights file", "yolov8n.pt"),
        ("DEFAULT_CONFIDENCE", "Starting threshold (0.10-0.95)", "0.50"),
        ("DETECTION_COOLDOWN_SECONDS", "Per-class log cooldown", "1.0"),
        ("INFERENCE_IMGSZ", "YOLO input size", "640"),
        ("LOG_LEVEL / LOG_FILE", "Logging", "INFO / logs/visiontrack.log")],
        columns=["Variable", "Meaning", "Default"]), hide_index=True)


# --------------------------------------------------------------------------
# Page: About & Help
# --------------------------------------------------------------------------
ABOUT_MD = f"""
**{APP_NAME}** reads frames from your webcam or a video file with **OpenCV**, runs a pretrained
**Ultralytics YOLO** model on each frame, draws boxes with confidence scores, and logs qualifying
detections to **MySQL**. This dashboard is built with **Streamlit**.

#### How it works
```text
Webcam / Video file --(OpenCV)--> frame --(YOLO)--> detections
      --> confidence + class filter --> annotated frame (live view)
                                    --> per-class cooldown --> background writer --> MySQL
MySQL --> History, Analytics, Overview, CSV export
```

#### Good to know
* Boxes are stored in **original-frame pixels**: `bbox_x=x1, bbox_y=y1, bbox_w=x2-x1, bbox_h=y2-y1`.
* The cooldown prevents hundreds of duplicate rows. It is a **heuristic, not object tracking**:
  the app cannot tell individual people/objects apart.
* If MySQL is down, video keeps running; unsaved events are **skipped and counted** (never silently lost).
* Leaving the Live Detection Studio page stops the run and releases the camera.
"""

TROUBLE = [
    ("Webcam will not open", "Close Zoom/Teams/browser tabs using the camera, allow desktop apps in Windows "
     "Settings > Privacy & security > Camera, or try camera index 1."),
    ("MySQL: Access denied / Can't connect", "Check DB_USER/DB_PASSWORD in .env, make sure the MySQL80 "
     "service is running (Services app), and that database_schema.sql was run as the admin user."),
    ("Model fails to load", "The first run downloads yolov8n.pt (~6 MB) from GitHub - connect to the internet "
     "once, or place the file in the project folder."),
    ("Video is slow", "Lower 'Inference image size', raise 'Run YOLO every Nth frame', or use yolov8n.pt."),
    ("Changed .env but nothing changed", "Stop the app with Ctrl+C and start it again."),
]


def page_about(ctx: AppContext) -> None:
    hero()
    st.markdown(ABOUT_MD)
    st.subheader("Troubleshooting")
    for title, body in TROUBLE:
        with st.expander(title):
            st.write(body)

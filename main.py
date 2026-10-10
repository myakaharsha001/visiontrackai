"""main.py - Application entry point.   Run with:  streamlit run main.py

Wires configuration, database, detector and UI together and owns the
lifecycle of the single active detection run (LiveController).
"""
import streamlit as st

# Must be the first Streamlit call in the script.
st.set_page_config(page_title="VisionTrack AI", page_icon="🎯", layout="wide",
                   initial_sidebar_state="expanded")

from typing import Optional  # noqa: E402

import ui  # noqa: E402
from config import ConfigError, DetectionSettings, load_config  # noqa: E402
from database import Database, DatabaseError, EventWriter  # noqa: E402
from detector import ModelLoadError, ObjectDetector  # noqa: E402
from logger import get_logger, setup_logging  # noqa: E402
from pipeline import DetectionSession  # noqa: E402
from video_source import SourceError, SourceSpec, VideoSource  # noqa: E402

log = get_logger("main")


@st.cache_resource(show_spinner="Loading YOLO model (the first run downloads ~6 MB)...")
def _load_detector(model_name: str):
    """Load the model once per server process. Returns (detector | None, error | None)."""
    try:
        return ObjectDetector(model_name).load(), None
    except ModelLoadError as exc:
        return None, str(exc)


@st.cache_resource
def _get_database() -> Database:
    """One Database object (with its connection pool) shared by every session."""
    return Database(load_config().database)


@st.cache_data(ttl=5, show_spinner=False)
def _db_status(_db: Database):
    return _db.check_connection()


class LiveController:
    """Owns the one active DetectionSession + EventWriter of this browser session.

    Both are kept in st.session_state, so Streamlit reruns never open a second
    camera: start() refuses to run while a session is still active.
    """

    def __init__(self, db: Database, detector: Optional[ObjectDetector]):
        self.db = db
        self.detector = detector

    @property
    def session(self) -> Optional[DetectionSession]:
        return st.session_state.get("det_session")

    @property
    def writer(self) -> Optional[EventWriter]:
        return st.session_state.get("det_writer")

    @property
    def active(self) -> bool:
        return bool(self.session and self.session.active)

    def start(self, settings: DetectionSettings, spec: SourceSpec) -> Optional[str]:
        """Open the source and begin. Returns an error message, or None on success."""
        if self.detector is None:
            return "The YOLO model is not loaded."
        if self.active:
            return None                                   # already running: never open twice
        try:
            settings.validate()
        except ValueError as exc:
            return str(exc)
        self.finalize()                                   # flush the previous run's writer
        writer = EventWriter(self.db).start()
        source = VideoSource(spec)
        session = DetectionSession(self.detector, source, settings, sink=writer)
        try:
            session.start()
        except SourceError as exc:
            writer.stop()
            return str(exc)
        except Exception as exc:                          # unexpected: show it, do not crash the app
            log.exception("Could not start detection")
            source.release()
            writer.stop()
            return f"Unexpected error while starting: {exc}"
        st.session_state.det_session = session
        st.session_state.det_writer = writer
        st.session_state.det_finalized = False
        return None

    def pause(self) -> None:
        if self.session:
            self.session.pause()

    def resume(self) -> None:
        if self.session:
            self.session.resume()

    def stop(self) -> None:
        if self.session:
            self.session.stop()                           # releases the capture
        self.finalize()

    def finalize(self) -> None:
        """Flush and stop the background writer (idempotent)."""
        if self.writer is not None and not st.session_state.get("det_finalized", True):
            self.writer.stop()
            st.session_state.det_finalized = True

    def sync_settings(self, settings: DetectionSettings) -> None:
        if self.session:
            self.session.settings = settings

    def writer_stats(self) -> dict:
        if self.writer is not None:
            return self.writer.snapshot()
        return {"submitted": 0, "inserted": 0, "failed": 0, "dropped": 0, "pending": 0, "last_error": None}


def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        ui.inject_css()
        ui.show_config_error(str(exc))
        st.stop()
        return
    setup_logging(config.log_level, config.log_file)
    ui.inject_css()
    ui.init_state(config)

    db = _get_database()
    db_ok, db_msg = _db_status(db)
    if db_ok and not st.session_state.get("schema_ready"):
        try:
            db.init_schema()
            st.session_state.schema_ready = True
        except DatabaseError as exc:
            log.warning("init_schema failed: %s", exc)
            st.warning(f"Connected to MySQL but could not create the table automatically: {exc}\n\n"
                       "Run database_schema.sql as the MySQL admin user.")

    detector, model_error = _load_detector(config.model_name)
    if detector is not None:
        ui.sanitize_class_selection(detector.class_names)

    controller = LiveController(db, detector)
    ctx = ui.AppContext(config=config, db=db, db_ok=db_ok, db_msg=db_msg, detector=detector,
                        model_error=model_error, controller=controller,
                        retry_model=_load_detector.clear)

    page = ui.sidebar(ctx)

    # Only the Live page runs the video loop, so leaving it releases the camera.
    if page != ui.LIVE_PAGE and controller.active:
        controller.stop()
        st.toast("Detection stopped and the video source was released (you left the Live Studio).")

    {
        ui.PAGES[0]: ui.page_overview,
        ui.PAGES[1]: ui.page_live,
        ui.PAGES[2]: ui.page_history,
        ui.PAGES[3]: ui.page_analytics,
        ui.PAGES[4]: ui.page_settings,
        ui.PAGES[5]: ui.page_about,
    }[page](ctx)


main()
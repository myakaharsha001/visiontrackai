"""check_setup.py - Verify your installation step by step.   Run:  python check_setup.py

Prints PASS/FAIL for Python, packages, configuration, MySQL and (optionally) the YOLO weights.
It never prints your password.
"""
import importlib
import sys

OK, BAD = "PASS", "FAIL"


def line(status, name, detail=""):
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


def check_python():
    v = sys.version_info
    line(OK if v[:2] == (3, 12) else "WARN", "Python version", f"{v.major}.{v.minor}.{v.micro} (3.12 recommended)")


def check_packages():
    ok = True
    for mod, label in [("streamlit", "Streamlit"), ("cv2", "OpenCV"), ("ultralytics", "Ultralytics"),
                       ("mysql.connector", "mysql-connector-python"), ("dotenv", "python-dotenv"),
                       ("pandas", "Pandas"), ("numpy", "NumPy"), ("plotly", "Plotly"), ("pytest", "pytest")]:
        try:
            m = importlib.import_module(mod)
            line(OK, label, getattr(m, "__version__", "installed"))
        except Exception as exc:  # noqa: BLE001
            ok = False
            line(BAD, label, f"{exc}  -> run: pip install -r requirements.txt")
    return ok


def check_config():
    try:
        from config import load_config
        cfg = load_config()
        line(OK, ".env configuration", f"user={cfg.database.user} db={cfg.database.name} model={cfg.model_name}")
        return cfg
    except Exception as exc:  # noqa: BLE001
        line(BAD, ".env configuration", "\n" + str(exc))
        return None


def check_mysql(cfg):
    from database import Database
    db = Database(cfg.database)
    ok, msg = db.check_connection()
    line(OK if ok else BAD, "MySQL connection", msg)
    if ok:
        try:
            db.init_schema()
            line(OK, "Table detection_events", "exists / created")
        except Exception as exc:  # noqa: BLE001
            line(BAD, "Table detection_events", str(exc))


def check_model(cfg):
    from detector import ModelLoadError, ObjectDetector
    try:
        det = ObjectDetector(cfg.model_name).load()
        line(OK, "YOLO model", f"{cfg.model_name}, {len(det.class_names)} classes")
    except ModelLoadError as exc:
        line(BAD, "YOLO model", str(exc))


def check_camera():
    import cv2
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    line(OK if ok else "WARN", "Webcam #0", "frame received" if ok else "not available (video files still work)")


if __name__ == "__main__":
    check_python()
    if check_packages():
        cfg = check_config()
        if cfg:
            check_mysql(cfg)
            check_model(cfg)
        check_camera()

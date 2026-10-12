# VisionTrack AI — Real-Time Object Detection & Event Monitoring Platform

A Streamlit dashboard that reads your **webcam or a video file** with **OpenCV**, detects objects with a
pretrained **Ultralytics YOLO** model, draws labelled boxes live, and logs qualifying detections to **MySQL**.
Browse the history, see analytics, and export CSV — all from one dashboard.

> Assignment 5 project · Python 3.12 · OpenCV · Ultralytics YOLO (`yolov8n.pt`) · MySQL · Streamlit · Pandas · Plotly

**Suggested GitHub repository name:** `visiontrack-ai-object-detection`
**Suggested "About" text:** *Real-time object detection with YOLO + OpenCV, logged to MySQL, shown in a Streamlit dashboard.*

---

## 1. Verification status (read this first — honest summary)

| What | Status |
|---|---|
| Unit tests (65 tests: threshold & class filtering, bbox conversion, cooldown, config validation, SQL parameter handling, event formatting, pipeline logic, real OpenCV video read/release) | **Executed** where the code was generated (Python 3.13, OpenCV 5.0, `unittest`): 62 passed, 3 integration tests skipped by design. **Not yet run on your laptop** — please run them (section 9). |
| Streamlit UI in a real browser | **Not executed** — Streamlit could not be installed where this was generated. The UI code was checked only against a stand-in Streamlit for typos/logic errors. You must run it and walk the acceptance checklist (`docs/TESTING.md`). |
| YOLO inference with real weights | **Not executed** (weights/torch not available there). Inference code is tested with a fake model that mimics Ultralytics' output format. |
| Real MySQL server | **Not executed.** SQL building/parameter handling and the insert/rollback/pool logic were tested with a fake connector. `database_schema.sql` has not been run against a real server. |
| Webcam | **Not executed.** The "camera unavailable" error path was exercised with real OpenCV on a machine with no camera. |

Run `python check_setup.py` and the manual checklist on your laptop before the demo; fix anything they reveal.
Do not describe a feature as "verified" in your video/article until you have seen it work yourself.

## 2. Features

* Live annotated video from the **laptop webcam** or a **local video file** (upload or path)
* Pretrained YOLO (`yolov8n.pt`, CPU-friendly); class list comes from the loaded model (all 80 COCO classes, nothing hardcoded)
* Adjustable **confidence threshold (10–95 %)** and **class filter** — the *same* filtered list drives the drawn boxes, the on-screen counts and the database log
* **Start / Pause / Resume / Stop** with proper release of the camera
* **Per-class cooldown** so a continuously visible object doesn't create hundreds of rows (a heuristic, *not* tracking)
* MySQL logging through a **background writer** (video never waits for the database), pooled connections, parameterised SQL
* Visible warnings if MySQL fails — failed events are **counted and reported, never claimed as saved**
* Pages: Overview · Live Detection Studio · Detection History (search, filters, pagination, CSV export) · Analytics · Settings · About & Help
* Dark navy/cyan/violet UI; every number comes from real records or the live session

## 3. Architecture

```text
 Webcam / Video file ──OpenCV──► frame ──YOLO──► detections
                                                     │
                       confidence + class filter  ◄──┘
                              │                 │
                              ▼                 ▼
                 annotated frame (live view)   per-class cooldown ──► EventWriter thread ──► MySQL
                              │                                                │
                              ▼                                                ▼
                       Streamlit dashboard ◄──── History / Analytics / Overview / CSV
```

| File | Responsibility |
|---|---|
| `main.py` | Entry point: loads config, DB, model; owns the single active run (`LiveController`); page routing |
| `ui.py` | All Streamlit pages, controls and the live-frame loop |
| `theme.py` | Colours, CSS, HTML cards, Plotly styling |
| `detector.py` | YOLO loading & inference, filtering, cooldown, box drawing, event formatting |
| `video_source.py` | OpenCV capture for webcam/video with validation and clean release |
| `pipeline.py` | `DetectionSession`: read → infer → filter → draw → log (no Streamlit code; unit-tested) |
| `database.py` | Connection pool, parameterised queries, `EventWriter` background thread |
| `config.py` | `.env` loading/validation, runtime `DetectionSettings` |
| `logger.py`, `utils.py` | Logging setup; coordinate/format helpers |
| `database_schema.sql` | Database, least-privilege user, table |
| `check_setup.py` | Step-by-step installation checker |
| `tests/` | Unit tests (+ opt-in `tests/integration/`) |
| `docs/` | Architecture, testing checklist, GitHub guide, demo script, Medium draft, viva prep, submission checklist |

Design choices worth knowing:

* **Coordinates:** original-frame **pixels**. YOLO gives `x1,y1,x2,y2`; stored as `bbox_x=x1, bbox_y=y1, bbox_w=x2−x1, bbox_h=y2−y1`.
* **Streamlit video loop:** a plain `while` loop inside the script run pushes frames to an `st.empty()` placeholder. A click on any widget makes Streamlit interrupt and rerun the script; the `DetectionSession` (and its single `VideoCapture`) lives in `st.session_state`, so the rerun just continues. `Start` is ignored while a run is active, so reruns can never open a second camera.
* **Leaving the Live page stops the run** and releases the camera (only that page runs the video loop).
* **Pause** keeps the camera open but stops reading; **Stop** releases it.
* **Logging:** events are queued and inserted in batches by a daemon thread. If MySQL is down, those events are skipped (no retry) and shown as "could NOT be saved".
* **Extra columns** beyond the required ones: `source_type` (`webcam`/`video_file`), `session_id` (UUID of the run), `created_at`.
* **Timestamps** are the laptop's local time (`DATETIME(3)`).

## 4. Setup on Windows (copy-paste)

### 4.1 Install Python 3.12
Download from <https://www.python.org/downloads/windows/> → tick **"Add python.exe to PATH"** → Install. Verify in PowerShell:
```powershell
py -3.12 --version
```

### 4.2 Get the project folder and open it in VS Code
```powershell
cd $HOME\Documents
# (if you cloned it from GitHub)   git clone https://github.com/myakaharsha001/visiontrackai.git
cd visiontrack-ai-object-detection        # or the folder where you unzipped visiontrack-ai
code .
```

### 4.3 Create and activate a virtual environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell says *running scripts is disabled*: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. Your prompt should start with `(.venv)`.

### 4.4 Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This downloads PyTorch (~200 MB) via `ultralytics`; allow a few minutes. Afterwards record exact versions: `pip freeze > requirements-lock.txt`.

### 4.5 Install MySQL (skip if you already have MySQL 8.x)
1. Download **MySQL Installer** from <https://dev.mysql.com/downloads/installer/> (the larger "offline" file is easiest).
2. Choose **Server only** (Workbench is optional). Keep the defaults (port **3306**, service name `MySQL80`, start at boot).
3. Set a **root password** and remember it (this is *not* the app's password).

Check the service: `Get-Service MySQL80` (should say *Running*; if not: `Start-Service MySQL80` in an admin PowerShell).

### 4.6 Create the database, user and table
1. Open `database_schema.sql` and replace `CHANGE_ME_TO_A_STRONG_PASSWORD` with a password of your choice (don't commit that edit — or copy the file to `database_schema.local.sql` and edit the copy).
2. Run it as root. The folder name may differ (e.g. `MySQL Server 8.4`):
```powershell
& "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" -u root -p -e "source database_schema.sql"
```
(Enter the **root** password when asked.) If `source` can't find the file, use the full path with forward slashes, e.g. `source C:/Users/you/Documents/visiontrack-ai/database_schema.sql`.

What it does (SQL — *not* PowerShell): creates database `vision_platform`, user `vision_user`, and grants only `SELECT, INSERT, CREATE, INDEX` on that one database — no `DELETE`, `DROP` or access to other databases.

### 4.7 Create the `.env` file
```powershell
Copy-Item .env.example .env
notepad .env
```
Set `DB_PASSWORD` to the password you put in the SQL file. Never commit `.env` (it is in `.gitignore`).

### 4.8 Verify everything
```powershell
python check_setup.py
```
It checks Python, every package, the `.env`, the MySQL connection (and creates the table if missing), the YOLO weights (downloads `yolov8n.pt` ≈ 6 MB on first use — internet needed once) and the webcam.
Quick manual checks: `python --version` · `pip --version` · `streamlit --version` · `python -c "import cv2; print(cv2.__version__)"` · `python -c "import ultralytics; print(ultralytics.__version__)"`.

### 4.9 Run the app
```powershell
streamlit run main.py
```
Run it **from the project folder** (so `.streamlit/config.toml` and `.env` are found). The browser opens at <http://localhost:8501>. Stop with `Ctrl+C`.

## 5. Using the dashboard

1. **Live Detection Studio** → pick *Laptop webcam* (or upload/enter a video path) → **Start**.
2. Move the **Minimum confidence** slider or pick classes (e.g. only `person`, `cell phone`) — changes apply immediately.
3. Watch *Objects in frame*, *FPS*, *Saved to MySQL* and the recent-events table.
4. **Stop** releases the camera. Open **Detection History** to search/filter, then *Prepare CSV export* → *Download*.
5. **Analytics** and **Overview** show charts computed from the rows in MySQL.

Verify rows in MySQL (SQL, in the `mysql` client):
```sql
USE vision_platform;
SELECT log_id, `timestamp`, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h FROM detection_events ORDER BY log_id DESC LIMIT 10;
```

## 6. Configuration

All via `.env` (see `.env.example`): `DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD` (required) and optional `MODEL_NAME, DEFAULT_CONFIDENCE, DETECTION_COOLDOWN_SECONDS, INFERENCE_IMGSZ, DB_POOL_SIZE, LOG_LEVEL, LOG_FILE`. Invalid or missing values show a readable error screen. **Restart the app after editing `.env`.**

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| Configuration error screen | Compare `.env` with `.env.example`; the message lists the exact variables. Placeholder password is rejected on purpose. |
| `Access denied for user` | Wrong `DB_PASSWORD`, or the SQL file wasn't run (or you edited the password after running it: re-create with `ALTER USER 'vision_user'@'localhost' IDENTIFIED BY 'new';`). |
| `Can't connect to MySQL server` | Service not running (`Start-Service MySQL80`), or wrong `DB_HOST`/`DB_PORT`. |
| `Unknown database 'vision_platform'` | Run `database_schema.sql` as root (step 4.6). |
| Webcam won't open / black frames | Close Zoom/Teams/browser tabs using the camera; *Windows Settings → Privacy & security → Camera → allow desktop apps*; try camera index 1; test with a video file instead. |
| Model fails to load | First run needs internet to fetch `yolov8n.pt`; otherwise download it manually into the project folder. |
| Slow / low FPS | Lower *Inference image size* (320–480), raise *Run YOLO every Nth frame*, keep `yolov8n.pt`. |
| `ModuleNotFoundError` | Virtual environment not active (`.\.venv\Scripts\Activate.ps1`) or `pip install -r requirements.txt` not run. |
| Video file won't play | Convert to H.264 `.mp4`; check the path has no stray quotes. |
| Uploaded video not found | Uploads are saved in `.uploads\` inside the project folder. |

## 8. What is *not* implemented (honest limitations)

* No object **tracking** — the cooldown is a per-class heuristic; two different people are indistinguishable and a person leaving/returning isn't recognised.
* No retry queue for failed database inserts (they are skipped and counted).
* Webcam capture uses OpenCV (DirectShow on Windows); browser-camera streaming is not supported.
* CPU inference only by default; FPS depends on your laptop.

## 9. Testing

```powershell
python -m pytest tests -v          # or:  python -m unittest discover -s tests -t . -v
```
Unit tests need **no** webcam, YOLO weights or MySQL. Opt-in integration tests (real resources):
```powershell
$env:VT_TEST_DB="1";     python -m pytest tests/integration -v   # live MySQL via .env
$env:VT_TEST_MODEL="1";  python -m pytest tests/integration -v   # real yolov8n.pt
$env:VT_TEST_CAMERA="1"; python -m pytest tests/integration -v   # real webcam
```
The 15-point manual acceptance checklist is in [`docs/TESTING.md`](docs/TESTING.md).

## 10. Screenshots

Add real screenshots after running the app (don't include passwords or personal faces you don't want public):

| Overview | Live Studio | History | Analytics |
|---|---|---|---|
| `screenshots/overview.png` | `screenshots/live.png` | `screenshots/history.png` | `screenshots/analytics.png` |

## 11. Demo video

`<DEMO VIDEO LINK — add after recording; see docs/DEMO_SCRIPT.md>`

## 12. Documentation

[Architecture](docs/ARCHITECTURE.md) · [Testing & acceptance checklist](docs/TESTING.md) · [GitHub guide](docs/GITHUB_GUIDE.md) · [Demo script](docs/DEMO_SCRIPT.md) · [Medium draft](docs/MEDIUM_ARTICLE_DRAFT.md) · [Viva prep](docs/VIVA_PREP.md) · [Submission checklist](docs/SUBMISSION_CHECKLIST.md)

## 13. Author

* Name: `<MYAKALA HARSHAVARDHAN>` · Roll no.: `<1530>` · College/Course: `<COLLEGE, B.Tech CSE (AI & ML)>`
* GitHub: `https://github.com/myakaharsha001/visiontrackai>` · Medium article: `<LINK AFTER PUBLISHING>`

## 14. License

MIT (see `LICENSE` — replace `MYAKALA HARSHAVARDHAN`). MIT is a common, permissive choice for coursework. Note that **Ultralytics YOLO is licensed under AGPL-3.0** (with a separate commercial licence); fine for an academic project, but check their terms before any commercial use.
 
 
#   v i s i o n t r a c k - a i  
 
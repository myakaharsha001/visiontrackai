# VisionTrack AI — Real-Time Object Detection & Event Monitoring Platform

A Streamlit dashboard that reads your **webcam or a video file** with **OpenCV**, detects objects with a
pretrained **Ultralytics YOLO** model, draws labelled boxes live, and logs qualifying detections to **MySQL**.
Browse the history, see analytics, and export CSV, all from one dashboard.

> Assignment 5 project · Python 3.12 · OpenCV · Ultralytics YOLO (`yolov8n.pt`) · MySQL · Streamlit · Pandas · Plotly

*Real-time object detection with YOLO + OpenCV, logged to MySQL, shown in a Streamlit dashboard.*

---

## 1. Verification status

| What | Status |
|---|---|
| Unit tests (65 tests: threshold and class filtering, bbox conversion, cooldown, config validation, SQL parameter handling, event formatting, pipeline logic, real OpenCV video read/release) | **Executed** with `unittest`: 65 ran, 62 passed, 3 integration tests skipped by design (they need a real camera, model or database). Re-run them on your own machine with the command in section 9. |
| Streamlit UI in a real browser | **Tested by the author** on a Windows laptop: all six pages load. |
| YOLO inference with real weights | **Tested by the author** with `yolov8n.pt` on CPU: live boxes and labels (for example `person 92%`) appear on the webcam feed. |
| Real MySQL 8.0 server | **Tested by the author:** detections are saved, and Overview, History and Analytics read them back. |
| Webcam | **Tested by the author:** laptop webcam #0 works through OpenCV. |
| 15-point manual checklist in `docs/TESTING.md` | Work through it on your own machine before relying on any feature. |

Detection quality depends on lighting, the camera and the model size. The small `yolov8n` model can mislabel
bright or blurry areas at low confidence, so use the confidence slider and the class filter (section 5).

## 2. Features

* Live annotated video from the **laptop webcam** or a **local video file** (upload or path)
* Pretrained YOLO (`yolov8n.pt`, CPU-friendly); the class list comes from the loaded model (all 80 COCO classes, nothing hardcoded)
* Adjustable **confidence threshold (10-95 %)** and **class filter**. The *same* filtered list drives the drawn boxes, the on-screen counts and the database log
* **Start / Pause / Resume / Stop** with proper release of the camera
* **Per-class cooldown** so a continuously visible object does not create hundreds of rows (a heuristic, *not* tracking)
* MySQL logging through a **background writer** (video never waits for the database), pooled connections, parameterised SQL
* Visible warnings if MySQL fails. Failed events are **counted and reported, never claimed as saved**
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
| `detector.py` | YOLO loading and inference, filtering, cooldown, box drawing, event formatting |
| `video_source.py` | OpenCV capture for webcam/video with validation and clean release |
| `pipeline.py` | `DetectionSession`: read, infer, filter, draw, log (no Streamlit code; unit-tested) |
| `database.py` | Connection pool, parameterised queries, `EventWriter` background thread |
| `config.py` | `.env` loading/validation, runtime `DetectionSettings` |
| `logger.py`, `utils.py` | Logging setup; coordinate/format helpers |
| `database_schema.sql` | Database, least-privilege user, table |
| `check_setup.py` | Step-by-step installation checker |
| `tests/` | Unit tests (+ opt-in `tests/integration/`) |
| `docs/` | Architecture, testing checklist, GitHub guide, demo script, Medium draft, viva prep, submission checklist |

Design choices worth knowing:

* **Coordinates:** original-frame **pixels**. YOLO gives `x1,y1,x2,y2`; stored as `bbox_x=x1, bbox_y=y1, bbox_w=x2-x1, bbox_h=y2-y1`.
* **Streamlit video loop:** a plain `while` loop inside the script run pushes frames to an `st.empty()` placeholder. A click on any widget makes Streamlit interrupt and rerun the script; the `DetectionSession` (and its single `VideoCapture`) lives in `st.session_state`, so the rerun just continues. `Start` is ignored while a run is active, so reruns can never open a second camera.
* **Leaving the Live page stops the run** and releases the camera (only that page runs the video loop).
* **Pause** keeps the camera open but stops reading; **Stop** releases it.
* **Logging:** events are queued and inserted in batches by a daemon thread. If MySQL is down, those events are skipped (no retry) and shown as "could NOT be saved".
* **Extra columns** beyond the required ones: `source_type` (`webcam`/`video_file`), `session_id` (UUID of the run), `created_at`.
* **Timestamps** are the laptop's local time (`DATETIME(3)`).

## 4. Setup on Windows (copy-paste)

### 4.1 Install Python 3.12
Download from <https://www.python.org/downloads/windows/>, tick **"Add python.exe to PATH"**, then install. Verify in PowerShell:
```powershell
python --version
```

### 4.2 Get the project folder and open it in VS Code
```powershell
cd $HOME\Documents
git clone https://github.com/myakaharsha001/visiontrackai.git
cd visiontrackai
code .
```

### 4.3 Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell says *running scripts is disabled*: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. Your prompt should start with `(.venv)`.

### 4.4 Install dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
This downloads PyTorch (~200 MB) via `ultralytics`; allow a few minutes.

### 4.5 Install MySQL (skip if you already have MySQL 8.x)
1. Download **MySQL Installer** from <https://dev.mysql.com/downloads/installer/> (the larger "offline" file is easiest).
2. Choose **Server only**. Keep the defaults (port **3306**, service name `MySQL80`, start at boot).
3. Set a **root password** and remember it (this is *not* the app's password).

Check the service: `Get-Service MySQL80` (should say *Running*; if not, run `Start-Service MySQL80` in an Administrator PowerShell).

### 4.6 Create the database, user and table
1. Copy `database_schema.sql` to `database_schema.local.sql`, open the copy, and replace `CHANGE_ME_TO_A_STRONG_PASSWORD` with a password of your choice (letters and digits only). Do not commit the edited copy.
2. Run it as root. The folder name may differ (for example `MySQL Server 8.4`):
```powershell
& "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" -u root -p -e "source database_schema.local.sql"
```
(Enter the **root** password when asked.) If `source` cannot find the file, use the full path with forward slashes, for example `source C:/Users/you/Documents/visiontrackai/database_schema.local.sql`.

The script creates database `vision_platform`, user `vision_user`, and grants only `SELECT, INSERT, CREATE, INDEX` on that one database: no `DELETE`, `DROP` or access to other databases.

### 4.7 Create the `.env` file
```powershell
Copy-Item .env.example .env
notepad .env
```
Set `DB_PASSWORD` to the password you put in the SQL file. If `localhost` gives login errors, set `DB_HOST=127.0.0.1`. Never commit `.env` (it is in `.gitignore`).

### 4.8 Verify everything
```powershell
python check_setup.py
```
It checks Python, every package, the `.env`, the MySQL connection (and creates the table if missing), the YOLO weights (downloads `yolov8n.pt`, about 6 MB, on first use; internet is needed once) and the webcam.

### 4.9 Run the app
```powershell
streamlit run main.py
```
Run it **from the project folder** (so `.streamlit/config.toml` and `.env` are found). The browser opens at <http://localhost:8501>. Stop with `Ctrl+C`.

## 5. Using the dashboard

1. **Live Detection Studio**: pick *Laptop webcam* (or upload/enter a video path), then press **Start**.
2. Set **Minimum confidence** to about 0.70 and pick only the classes you need (for example `person`, `cell phone`, `bottle`). Changes apply immediately.
3. Watch *Objects in frame*, *FPS*, *Saved to MySQL* and the recent-events table.
4. **Stop** releases the camera. Open **Detection History** to search and filter, then use *Prepare CSV export* and *Download*.
5. **Analytics** and **Overview** show charts computed from the rows in MySQL.

Tips for better detections: face a light source instead of sitting in front of a window or white wall, hold objects close to the camera, and raise the confidence slider if random labels such as "airplane" or "kite" appear.

Verify rows in MySQL (SQL, in the `mysql` client):
```sql
USE vision_platform;
SELECT log_id, `timestamp`, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h FROM detection_events ORDER BY log_id DESC LIMIT 10;
```

## 6. Configuration

All via `.env` (see `.env.example`): `DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD` (required) and optional `MODEL_NAME, DEFAULT_CONFIDENCE, DETECTION_COOLDOWN_SECONDS, INFERENCE_IMGSZ, DB_POOL_SIZE, LOG_LEVEL, LOG_FILE`. Invalid or missing values show a readable error screen. **Fully restart the app (`Ctrl+C`, then `streamlit run main.py`) after editing `.env`.**

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| Configuration error screen | Compare `.env` with `.env.example`; the message lists the exact variables. The placeholder password is rejected on purpose. |
| `Access denied for user` | Wrong `DB_PASSWORD`, or the SQL file was not run (or the password was changed after running it). Re-create with `ALTER USER 'vision_user'@'localhost' IDENTIFIED BY 'new';` and put the same value in `.env`. |
| App still shows `MySQL DOWN` after the password is fixed | Streamlit keeps the old connection in memory. Stop it with `Ctrl+C`, start it again, and open a new browser tab. |
| `Can't connect to MySQL server` | Service not running (`Start-Service MySQL80`), or wrong `DB_HOST`/`DB_PORT`. |
| `Unknown database 'vision_platform'` | Run the schema file as root (step 4.6). |
| Webcam will not open / black frames | Close Zoom/Teams/browser tabs using the camera; *Windows Settings → Privacy & security → Camera → allow desktop apps*; try camera index 1; test with a video file instead. |
| Model fails to load | The first run needs internet to fetch `yolov8n.pt`; otherwise download it manually into the project folder. |
| Slow / low FPS | Lower *Inference image size* (320-480), raise *Run YOLO every Nth frame*, keep `yolov8n.pt`. On a CPU-only laptop a few FPS is normal. |
| `ModuleNotFoundError` | The virtual environment is not active (`.\.venv\Scripts\Activate.ps1`) or `python -m pip install -r requirements.txt` was not run. |
| `pip` gives a "Fatal error in launcher" | Use `python -m pip ...` instead of `pip ...`. |
| Video file will not play | Convert to H.264 `.mp4`; check that the path has no stray quotes. |
| Uploaded video not found | Uploads are saved in `.uploads\` inside the project folder. |

## 8. What is *not* implemented (honest limitations)

* No object **tracking**. The cooldown is a per-class heuristic; two different people are indistinguishable and a person leaving and returning is not recognised.
* No retry queue for failed database inserts (they are skipped and counted).
* Webcam capture uses OpenCV (DirectShow on Windows); browser-camera streaming is not supported.
* CPU inference only by default; FPS depends on your laptop.
* The app needs a local webcam and a local MySQL server, so it cannot run on cloud hosts such as Streamlit Community Cloud as it is.

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

| Overview | Live Studio |
|---|---|
| ![Overview](screenshots/overview.png) | ![Live Studio](screenshots/live.png) |

| History | Analytics |
|---|---|
| ![History](screenshots/history.png) | ![Analytics](screenshots/analytics.png) |

## 11. Demo video

Demo video: ADD YOUR VIDEO LINK HERE (see `docs/DEMO_SCRIPT.md`)

## 12. Documentation

[Architecture](docs/ARCHITECTURE.md) · [Testing & acceptance checklist](docs/TESTING.md) · [GitHub guide](docs/GITHUB_GUIDE.md) · [Demo script](docs/DEMO_SCRIPT.md) · [Medium draft](docs/MEDIUM_ARTICLE_DRAFT.md) · [Viva prep](docs/VIVA_PREP.md) · [Submission checklist](docs/SUBMISSION_CHECKLIST.md)

## 13. Author

* Name: MYAKALA HARSHAVARDHAN · Roll no.: 1530 · Course: B.Tech CSE (AI & ML) · College: SANDIP UNIVERSITY
* GitHub: <https://github.com/myakaharsha001/visiontrackai>
* Medium article: ADD YOUR MEDIUM LINK HERE

## 14. MYAKALA HARSHAVARDHAN

MIT (see `MYAKALA HARSHAVARDHAN`). MIT is a common, permissive choice for coursework. Note that **Ultralytics YOLO is licensed under AGPL-3.0** (with a separate commercial licence); that is fine for an academic project, but check their terms before any commercial use.

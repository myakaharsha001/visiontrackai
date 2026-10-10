# Testing

## Automated tests (no webcam / weights / MySQL needed)
```powershell
python -m pytest tests -v
```
| File | Covers |
|---|---|
| `tests/test_config.py` | `.env` validation: missing vars, placeholder password, bad port/confidence/model/log level, password hidden in `repr`, runtime settings limits |
| `tests/test_detector.py` | Threshold filtering (85 % kept / 52 % dropped at 70 %), class filter, all-classes default, xyxy→xywh, clamping, event formatting, cooldown, fake-model inference & class-id mapping, drawing |
| `tests/test_database.py` | WHERE builder uses placeholders only, SQL-injection strings stay in params, LIKE escaping, insert param order, commit/rollback, pool reuse, limit capping, `EventWriter` success / failure counting / full-queue drop |
| `tests/test_pipeline.py` | Start/pause/resume/stop + capture release, end of video, filters apply to feed *and* log, cooldown stops flooding, logging off, every-Nth-frame, inference error state, looping, real OpenCV video read & corrupt-file handling |

Integration tests (opt-in, real resources): see `tests/integration/test_integration.py` — set `VT_TEST_DB=1`, `VT_TEST_MODEL=1` or `VT_TEST_CAMERA=1`.

**Record your own results here after running them:**
`pytest result on my laptop: ____ passed / ____ failed (date: ____)`

## Manual acceptance checklist (tick each yourself)

| # | Check | How | ✔ |
|---|---|---|---|
| 1 | App launches without errors | `streamlit run main.py`, terminal shows no traceback | ☐ |
| 2 | Dashboard loads | All 6 sidebar pages open | ☐ |
| 3 | YOLO model loads | Sidebar shows "MODEL OK"; Settings lists class count (80 for yolov8n) | ☐ |
| 4 | Webcam input works | Live Studio → Laptop webcam → Start → video appears | ☐ |
| 5 | Local video works | Upload an .mp4 → Start → plays, ends with "video finished" | ☐ |
| 6 | Boxes & labels appear | Show a person/phone/bottle; label like `person 87.3%` | ☐ |
| 7 | Confidence threshold works | Raise to 90 %: weak boxes vanish and stop being logged; lower to 20 %: more appear | ☐ |
| 8 | Class filtering works | Select only `cell phone`: nothing else drawn or logged | ☐ |
| 9 | Events saved in MySQL | `SELECT * FROM detection_events ORDER BY log_id DESC LIMIT 10;` shows new rows | ☐ |
| 10 | Recent records in UI | Overview → Recent detections matches the SQL result | ☐ |
| 11 | Analytics use real records | Totals on Overview/Analytics equal `SELECT COUNT(*)` | ☐ |
| 12 | CSV export works | History → Prepare CSV → Download → open file, rows match filters | ☐ |
| 13 | DB failure handled visibly | While running: `Stop-Service MySQL80` (admin) → warning "N event(s) could NOT be saved", video keeps going; restart service afterwards | ☐ |
| 14 | Stop releases webcam | Press Stop → camera light turns off; open the Camera app to confirm it can use the camera | ☐ |
| 15 | No credentials on GitHub | `git status` shows no `.env`; search the GitHub repo for your password | ☐ |

Extra checks worth doing: cooldown (stand still for 10 s with cooldown 1 s → about 10 rows, not hundreds); leaving the Live page releases the camera; starting twice quickly doesn't open two cameras; wrong `DB_PASSWORD` shows a readable message.

## Which tests need what
| Test | Needs |
|---|---|
| unit tests | nothing but Python packages |
| `RealModel` | `yolov8n.pt` (internet once) |
| `RealCamera` | a webcam |
| `LiveMySQL` | running MySQL, schema applied, valid `.env` |

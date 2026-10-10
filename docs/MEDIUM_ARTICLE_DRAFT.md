# Building VisionTrack AI: A Real-Time Object Detection Platform with YOLO, OpenCV, MySQL, and Streamlit

> **Draft for you to edit.** Text in `[[double brackets]]` is a placeholder only YOU can fill after running the app. Delete any sentence you have not personally verified. Do not publish invented numbers.

![Cover: dashboard screenshot]([[screenshots/live.png]])

## 1. Introduction
Cameras are everywhere, but a camera alone only records. I wanted to see what happens when a laptop webcam is turned into a small *monitoring system*: it recognises objects as they appear and keeps a searchable record. For my Assignment 5 in [[course name]] I built **VisionTrack AI**, a dashboard that detects objects in real time with YOLO and OpenCV, stores events in MySQL, and shows everything in Streamlit.

## 2. Project objectives
* Capture video from a webcam or a video file.
* Detect objects with a pretrained YOLO model and draw labelled boxes.
* Let the user choose a confidence threshold and which classes to detect.
* Log qualifying detections to MySQL, and browse/export them.
* Keep the code modular, tested and easy to run on a student laptop (CPU only).

## 3. Technology stack
Python 3.12 · OpenCV · Ultralytics YOLO (`yolov8n.pt`) · Streamlit · MySQL + `mysql-connector-python` · Pandas · Plotly · pytest · Git/GitHub.

## 4. OpenCV and real-time video
OpenCV's `VideoCapture` gives one frame at a time as a NumPy array. A real-time program is a loop: read a frame → process → display → repeat. Two details mattered here: frames must be read quickly (cameras buffer old frames), and the camera must be **released** when you stop, otherwise other apps can't use it. In the project, `video_source.py` owns exactly one capture and releases it on stop, on end-of-video and on errors.

## 5. How YOLO object detection works
YOLO ("You Only Look Once") looks at the whole image in one pass of a neural network and predicts many boxes, each with a class and a **confidence score**. I used pretrained weights (`yolov8n.pt`, the "nano" model trained on the 80-class COCO dataset), so no training was needed. Boxes come out as corners `(x1, y1, x2, y2)`; I store them as `x, y, width, height` in original-frame pixels (`w = x2 − x1`, `h = y2 − y1`).

## 6. Modular architecture
`detector.py` (model + filtering), `video_source.py` (capture), `pipeline.py` (the per-frame workflow), `database.py` (MySQL), `ui.py` (dashboard), `config.py` (settings). The detection pipeline contains **no Streamlit code**, which let me unit-test it with fake cameras and fake models. [[Add your architecture diagram here — see docs/ARCHITECTURE.md]]

## 7. Confidence thresholds and object filtering
A detection is kept only if `confidence ≥ threshold` **and** its class is selected (or no class is selected). The same filtered list is used for drawing, counting and logging, so the screen and the database can never disagree. Example: at 70 %, an 85 % detection is logged and a 52 % one is not (this is an automated test in the repo).

To avoid logging the same visible object 30 times per second, a **per-class cooldown** (default 1 s) suppresses repeats. This is a heuristic, **not** object tracking — it can't tell two people apart.

## 8. MySQL integration
Events go into a `detection_events` table (`log_id`, `timestamp`, `object_class`, `confidence`, `bbox_x/y/w/h`, plus `source_type`, `session_id`, `created_at`). All queries are **parameterised** to prevent SQL injection, connections come from a small pool, and inserts happen on a background thread so a slow database can't freeze the video. If an insert fails, the dashboard says so instead of pretending it saved. The app user has only `SELECT, INSERT, CREATE, INDEX` on one database.

## 9. Streamlit dashboard design
Six pages: Overview, Live Detection Studio, Detection History, Analytics, Settings, About & Help, in a dark navy theme with cyan/blue/violet accents. All numbers come from real session data or MySQL rows. [[Screenshots: overview.png, live.png, history.png, analytics.png]]

## 10. Challenges and solutions
> Write only what really happened to you. Prompts to help you remember:
* [[A real error message you hit during setup (e.g. MySQL access denied, webcam busy) and how you fixed it]]
* [[What you changed to get a usable frame rate on your laptop — settings and the FPS you actually measured]]
* [[Anything surprising about YOLO's detections — e.g. a class it confused]]

Design problems the code was written to address (explain in your own words; these are design decisions, not stories about me): Streamlit reruns the script on every click, so the camera object is kept in `session_state` and `Start` is ignored while running; the video loop and the database must not block each other, hence the background writer.

## 11. Testing and validation
The repo includes unit tests for threshold/class filtering, bounding-box conversion, cooldown, configuration validation, SQL parameter handling and the detection pipeline, plus opt-in integration tests for a real MySQL server, real weights and a real webcam. [[State what YOU ran and the result, e.g. "pytest: __ passed on my laptop on <date>"]] and the manual checklist items you completed: [[list]].

## 12. Screenshots
[[Insert 3–4 real screenshots with captions. Blur anything private.]]

## 13. GitHub repository
[[https://github.com/<YOUR-USERNAME>/visiontrack-ai-object-detection]]

## 14. Demonstration video
[[Unlisted YouTube / Google Drive link]]

## 15. Key learnings
[[Your genuine takeaways — e.g. what you learned about confidence thresholds, threading, databases, Streamlit state. Keep it honest and specific.]]

## 16. Conclusion
VisionTrack AI turns a webcam into a small monitoring tool: detect, filter, log, review. Limitations: no object tracking, CPU-bound speed, no retry for failed inserts. Next steps could include tracking (e.g. ByteTrack), alerts, and a custom-trained model. [[Edit to match what you verified.]]

*Tags: YOLO, OpenCV, Streamlit, MySQL, Python, Computer Vision*

---
**Publishing on Medium:** medium.com → profile → Write → paste this text (headings → "Title/Subtitle" styles), upload images, add 5 tags, Publish. Copy the public URL into the submission form and README.

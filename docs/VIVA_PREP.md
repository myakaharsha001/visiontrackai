# Faculty presentation & viva preparation

## Abstract
VisionTrack AI is a real-time object-detection and event-logging platform. It captures video from a webcam or file with OpenCV, detects objects with a pretrained YOLOv8-nano model, draws labelled boxes, filters detections by confidence and class, and stores qualifying events in MySQL. A Streamlit dashboard shows the live feed, history, analytics and CSV export.

## Problem statement
Surveillance-style cameras produce video that nobody can search. Manually reviewing it is slow, and storing every frame's detections floods a database with duplicates.

## Objectives
Detect objects in real time on an ordinary laptop · let users tune confidence and classes · log only meaningful events · present data in a usable dashboard · build it modularly with tests and documentation.

## Existing vs proposed approach
| Existing | Proposed |
|---|---|
| Passive recording; manual review | Automatic detection with searchable event records |
| Scripts printing detections to a console | Integrated UI + database + analytics |
| Log every frame | Threshold + class filter + per-class cooldown |

## System architecture and data flow
See `docs/ARCHITECTURE.md` (Mermaid diagram). In words: **frame capture (OpenCV) → YOLO inference → confidence & class filtering → annotated display and cooldown-filtered logging → background writer → MySQL → dashboard**.

```text
Webcam/Video → OpenCV → YOLO → Filter ─┬→ Draw boxes → Streamlit live view
                                       └→ Cooldown → Writer thread → MySQL → History/Analytics/CSV
```

## Concepts to explain
* **OpenCV:** open-source computer-vision library; here used for capture (`VideoCapture`), drawing (`rectangle`, `putText`) and colour conversion.
* **YOLO & pretrained weights:** a single-pass CNN detector predicting boxes, classes and confidences; `yolov8n.pt` was trained on COCO (80 classes) by Ultralytics, so we use transfer-free inference without training.
* **Confidence threshold & filtering:** keep detection iff `conf ≥ T` and class selected.
* **Bounding boxes:** `(x1,y1,x2,y2)` → `x=x1, y=y1, w=x2−x1, h=y2−y1`, in original-frame pixels.
* **MySQL table:** `detection_events(log_id, timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h, + source_type, session_id, created_at)`.
* **Streamlit:** Python framework turning scripts into web apps; reruns the script on interaction.

## Advantages
Runs on CPU · modular & tested · least-privilege DB user · safe SQL · visible failure handling · no duplicate flooding.

## Limitations
No tracking (cooldown is a heuristic) · FPS depends on laptop CPU · failed inserts are skipped, not retried · COCO classes only · accuracy limited by lighting/occlusion and the nano model.

## Future enhancements
Object tracking (ByteTrack/DeepSORT) · alert rules (email/Telegram) · GPU support · custom-trained model · multi-camera · authentication · retry queue/outbox · Docker packaging.

## Likely viva questions (with short answers)
1. **What does YOLO stand for / how does it work?** "You Only Look Once"; one forward pass predicts boxes, classes and confidences for the whole image, unlike older two-stage detectors.
2. **Why YOLOv8n?** Smallest, fastest model; runs on a laptop CPU; accuracy is lower than larger variants.
3. **What are pretrained weights?** Parameters already learned on COCO; we didn't train anything.
4. **What is a confidence score?** The model's estimated probability that the box contains the predicted class — not a guarantee of correctness.
5. **What does the threshold do?** Discards detections below it from drawing, counting and logging (e.g. 85 % kept, 52 % dropped at 70 %).
6. **How are boxes converted for the database?** x=x1, y=y1, w=x2−x1, h=y2−y1, in original-frame pixels.
7. **Why not store every frame's detections?** Floods the DB with near-duplicates; we use a per-class cooldown (default 1 s).
8. **Is the cooldown object tracking?** No. It can't distinguish individual objects; real tracking needs an algorithm like ByteTrack/DeepSORT with IDs.
9. **How does the live video work in Streamlit?** A `while` loop reads frames with OpenCV, runs YOLO and updates an `st.empty()` image placeholder; session state holds the capture across reruns.
10. **What happens when you click a widget during playback?** Streamlit interrupts the loop and reruns the script; the session continues because the session object is in `session_state`.
11. **How do you prevent two webcams being opened?** `Start` is ignored while a session is active; capture is released on Stop, end-of-video, errors and when leaving the Live page.
12. **How do you prevent SQL injection?** All values go through `%s` placeholders; only fixed strings are in the SQL text; the bucket name is whitelisted.
13. **Why a connection pool?** Reuses connections instead of reconnecting for every insert/query.
14. **Why a background writer thread?** A slow/dead DB must not freeze the video; the queue is bounded so memory can't grow forever.
15. **What if MySQL goes down?** Video continues; failed events are counted and a warning shows; they are skipped, not retried.
16. **Why a least-privilege DB user?** Limits damage if credentials leak: only SELECT/INSERT/CREATE/INDEX on one database.
17. **Why `.env`?** Keeps secrets out of source code and Git; `.env.example` documents the variables.
18. **How do you test without a webcam/MySQL/model?** Fakes: fake detector/model, fake video source, fake MySQL connector; real OpenCV on a generated video file.
19. **Why `DECIMAL(5,4)` for confidence?** Exact 0.0000–1.0000 values with a CHECK constraint, no floating-point surprises.
20. **How would you add tracking?** Pass detections to a tracker (ByteTrack via `model.track`), store `track_id`, and log on first appearance/disappearance instead of cooldown.
21. **How do you measure FPS?** Exponential moving average of the inverse time between processed frames.
22. **What is the licence concern?** Ultralytics YOLO is AGPL-3.0 — fine for coursework; commercial use needs a licence or compliance.

# Demo video (target 1:45, max 2:00)

Rehearse once first. Have ready: app running, MySQL client open on `SELECT … FROM detection_events ORDER BY log_id DESC LIMIT 10;`, an object in hand (phone/bottle/book), the GitHub repo in a tab. Hide `.env`, passwords and the MySQL login from the recording. Set confidence to 50 %, class filter empty, cooldown 1 s.

| Time | Screen | Say / do |
|---|---|---|
| 0:00–0:10 | Overview page | "This is VisionTrack AI — real-time object detection with OpenCV and YOLO, logging to MySQL, in a Streamlit dashboard. Built for Assignment 5." |
| 0:10–0:25 | Live Studio | Choose Laptop webcam → **Start**. "OpenCV captures frames from my webcam." |
| 0:25–0:45 | Live feed | Show yourself, the phone, the bottle. Point at labels like `cell phone 82%`. "YOLOv8n runs on CPU; boxes show class and confidence; FPS is here." |
| 0:45–1:00 | Controls | Raise confidence to 80 % (weak boxes vanish). Select only `cell phone` (everything else disappears). "Only detections passing both filters are drawn and logged." |
| 1:00–1:20 | Side panel + Overview | Show *Saved to MySQL* counter and recent events. |
| 1:20–1:40 | MySQL client | Run the SELECT; show new rows with timestamp, class, confidence, bbox. Re-run to show it growing. "These are real rows, inserted live." |
| 1:40–1:52 | History page | Filter by class → Prepare CSV → Download. Quick glance at Analytics. |
| 1:52–2:00 | GitHub tab | "Source, SQL schema and docs are on GitHub." **Stop** the run first (releases camera). |

## Record on Windows
* **Xbox Game Bar:** press `Win + G` → Capture → record (records one app window; can capture the mic). Files save to *Videos\Captures*. Switching between app windows may stop it — if you need browser + MySQL window, use OBS instead.
* **OBS Studio** (free, <https://obsproject.com>): Sources → *Display Capture* + *Audio Input Capture* → Start Recording; output folder in Settings → Output.
* **Clipchamp** (built into Windows 11): Record → Screen (and camera/mic) → export 1080p.
Trim dead time so the final video is ≤ 2:00.

## Upload and share
**YouTube (Unlisted):** studio.youtube.com → Create → Upload videos → *Visibility: Unlisted* → Save → copy link. Verify: open the link in a private window while logged out.
**Google Drive:** drive.google.com → New → File upload → right-click → Share → *General access: Anyone with the link → Viewer* → Copy link. Verify in a private window; it must play without asking for access.
Paste the link into `README.md` ("Demo video") and the submission form. Don't claim the video exists until you have the working link.

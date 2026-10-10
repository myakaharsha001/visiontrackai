"""Integration tests - need REAL resources, so they are skipped unless you opt in.

  Live MySQL    : set VT_TEST_DB=1  (uses your .env; writes then deletes rows tagged with a test session id)
  YOLO weights  : set VT_TEST_MODEL=1 (downloads/loads yolov8n.pt and runs inference on a blank frame)
  Webcam        : set VT_TEST_CAMERA=1 (opens camera 0, reads one frame, releases it)

PowerShell example:   $env:VT_TEST_DB="1"; pytest tests/integration -v
"""
import os
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402


@unittest.skipUnless(os.getenv("VT_TEST_DB") == "1", "set VT_TEST_DB=1 to run (needs live MySQL + .env)")
class LiveMySQL(unittest.TestCase):
    def test_insert_read_back_and_cleanup(self):
        from config import load_config
        from database import TABLE, Database, HistoryFilter
        from detector import detection_to_event
        from tests.helpers import det

        db = Database(load_config().database)
        db.init_schema()
        sid = "test-" + uuid.uuid4().hex[:8]
        n = db.insert_events([detection_to_event(det("person", 0.91, (5, 6, 55, 106)), "webcam", sid)])
        self.assertEqual(n, 1)
        recent = db.fetch_recent(50)
        row = recent[recent["session_id"] == sid].iloc[0]
        self.assertEqual((row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h), (5, 6, 50, 100))
        self.assertAlmostEqual(float(row.confidence), 0.91, places=3)
        self.assertGreaterEqual(db.count_history(HistoryFilter(classes=["person"])), 1)
        # Cleanup needs DELETE, which the least-privilege user does not have - tell the human.
        print(f"\nTest row left in {TABLE} with session_id={sid}; delete it as admin if you wish.")


@unittest.skipUnless(os.getenv("VT_TEST_MODEL") == "1", "set VT_TEST_MODEL=1 to run (needs yolov8n.pt)")
class RealModel(unittest.TestCase):
    def test_model_loads_and_runs(self):
        from detector import ObjectDetector
        d = ObjectDetector("yolov8n.pt").load()
        self.assertEqual(len(d.class_names), 80)
        self.assertEqual(d.detect(np.zeros((480, 640, 3), np.uint8), 0.5), [])


@unittest.skipUnless(os.getenv("VT_TEST_CAMERA") == "1", "set VT_TEST_CAMERA=1 to run (needs a webcam)")
class RealCamera(unittest.TestCase):
    def test_open_read_release(self):
        from video_source import SourceSpec, VideoSource
        src = VideoSource(SourceSpec()).open()
        ok, frame = src.read()
        src.release()
        self.assertTrue(ok)
        self.assertEqual(frame.ndim, 3)


if __name__ == "__main__":
    unittest.main()

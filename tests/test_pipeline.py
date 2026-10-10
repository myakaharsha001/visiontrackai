import os
import tempfile
import unittest

import cv2
import numpy as np

from config import DetectionSettings
from pipeline import ENDED, ERROR, IDLE, PAUSED, RUNNING, STOPPED, DetectionSession
from tests.helpers import FakeDetector, FakeSource, ListSink, det
from video_source import (InvalidVideoError, SourceSpec, VideoSource, validate_video_path)


def make_session(dets, frames=5, settings=None, sink=None, kind="webcam"):
    sink = sink if sink is not None else ListSink()
    s = DetectionSession(FakeDetector(dets), FakeSource(frames, kind),
                         settings or DetectionSettings(cooldown_seconds=0.0), sink)
    return s, sink


class SessionTests(unittest.TestCase):
    def test_lifecycle(self):
        s, _ = make_session([det()])
        self.assertEqual(s.status, IDLE)
        s.start()
        self.assertEqual(s.status, RUNNING)
        self.assertTrue(s.source.opened)
        s.pause()
        self.assertEqual(s.status, PAUSED)
        self.assertIsNone(s.step())               # paused -> no frame processed
        s.resume()
        self.assertIsNotNone(s.step())
        s.stop()
        self.assertEqual(s.status, STOPPED)
        self.assertTrue(s.source.released)         # capture released on stop

    def test_run_ends_and_releases_at_end_of_video(self):
        s, _ = make_session([det()], frames=2)
        s.start()
        while s.step() is not None:
            pass
        self.assertEqual(s.status, ENDED)
        self.assertTrue(s.source.released)

    def test_threshold_example_85_logged_52_not(self):
        settings = DetectionSettings(confidence=0.70, cooldown_seconds=0.0)
        s, sink = make_session([det(conf=0.85), det("car", 0.52, cid=2)], 1, settings)
        s.start()
        res = s.step()
        self.assertEqual([d.confidence for d in res.detections], [0.85])   # feed
        self.assertEqual(len(sink.events), 1)                                # database
        self.assertEqual(sink.events[0]["confidence"], 0.85)

    def test_class_filter_applies_to_feed_and_log(self):
        settings = DetectionSettings(selected_classes=["car"], cooldown_seconds=0.0)
        s, sink = make_session([det("person"), det("car", 0.8, cid=2)], 1, settings)
        s.start()
        res = s.step()
        self.assertEqual([d.class_name for d in res.detections], ["car"])
        self.assertEqual([e["object_class"] for e in sink.events], ["car"])

    def test_cooldown_prevents_per_frame_flood(self):
        settings = DetectionSettings(cooldown_seconds=60.0)
        s, sink = make_session([det()], 20, settings)
        s.start()
        while s.step() is not None:
            pass
        self.assertEqual(len(sink.events), 1)      # 20 frames -> 1 event

    def test_logging_disabled_sends_nothing(self):
        settings = DetectionSettings(logging_enabled=False, cooldown_seconds=0.0)
        s, sink = make_session([det()], 3, settings)
        s.start()
        res = s.step()
        self.assertEqual(sink.events, [])
        self.assertEqual(len(res.detections), 1)   # still drawn/counted

    def test_no_detections_state(self):
        s, sink = make_session([], 2)
        s.start()
        res = s.step()
        self.assertEqual(res.detections, [])
        self.assertEqual(res.logged, 0)

    def test_process_every_n_skips_inference(self):
        settings = DetectionSettings(process_every_n=3, cooldown_seconds=0.0)
        s, _ = make_session([det()], 6, settings)
        s.start()
        while s.step() is not None:
            pass
        self.assertEqual(s.detector.calls, 2)      # frames 1 and 4 of 6

    def test_rejected_events_are_counted(self):
        s, _ = make_session([det()], 1, sink=ListSink(accept=False))
        s.start()
        s.step()
        self.assertEqual(s.events_rejected, 1)
        self.assertEqual(s.events_submitted, 0)

    def test_inference_error_sets_error_state_and_releases(self):
        class Boom:
            def detect(self, *a, **k):
                raise RuntimeError("kaput")
        s = DetectionSession(Boom(), FakeSource(3), DetectionSettings(), ListSink())
        s.start()
        self.assertIsNone(s.step())
        self.assertEqual(s.status, ERROR)
        self.assertIn("kaput", s.error)
        self.assertTrue(s.source.released)

    def test_video_loop_rewinds(self):
        settings = DetectionSettings(loop_video=True, cooldown_seconds=0.0)
        s, _ = make_session([], 2, settings, kind="video_file")
        s.start()
        for _ in range(5):
            self.assertIsNotNone(s.step())
        self.assertEqual(s.status, RUNNING)


class VideoSourceTests(unittest.TestCase):
    def test_validate_path_errors(self):
        for bad in (None, "", "   ", "/no/such/file.mp4"):
            with self.assertRaises(InvalidVideoError):
                validate_video_path(bad)
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "notes.txt")
            with open(txt, "w") as fh:
                fh.write("hi")
            with self.assertRaises(InvalidVideoError) as ctx:
                validate_video_path(txt)
            self.assertIn("Unsupported", str(ctx.exception))
            with self.assertRaises(InvalidVideoError):
                validate_video_path(d)                 # folder

    def test_real_video_file_roundtrip(self):
        """Uses real OpenCV: writes a tiny video, reads it back, releases it."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.avi")
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            if not w.isOpened():
                self.skipTest("this OpenCV build cannot write MJPG video")
            for i in range(6):
                w.write(np.full((48, 64, 3), i * 30, dtype=np.uint8))
            w.release()
            src = VideoSource(SourceSpec(kind="video_file", path=path)).open()
            self.assertTrue(src.is_open)
            n = 0
            while True:
                ok, frame = src.read()
                if not ok:
                    break
                n += 1
                self.assertEqual(frame.shape, (48, 64, 3))
            self.assertEqual(n, 6)
            src.release()
            self.assertFalse(src.is_open)
            src.release()                              # double release is safe

    def test_corrupt_video_gives_friendly_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.mp4")
            with open(path, "wb") as fh:
                fh.write(b"not a video")
            with self.assertRaises(InvalidVideoError):
                VideoSource(SourceSpec(kind="video_file", path=path)).open()


if __name__ == "__main__":
    unittest.main()

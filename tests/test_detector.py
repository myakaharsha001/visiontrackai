import unittest

import numpy as np

from detector import (CooldownFilter, ObjectDetector, detection_to_event, draw_detections,
                      filter_detections)
from tests.helpers import det
from utils import clamp_box, confidence_percent, escape_like, xyxy_to_xywh


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)

    def __len__(self):
        return len(self.conf.a)


class _T:
    def __init__(self, a):
        self.a = np.array(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    names = {0: "person", 1: "bicycle", 2: "car"}

    def __init__(self):
        self.kwargs = None

    def predict(self, frame, **kw):
        self.kwargs = kw
        return [FakeResult(FakeBoxes([[10, 20, 110, 220], [-5, 0, 700, 500], [0, 0, 5, 5]],
                                     [0.85, 0.52, 0.99], [0, 2, 1]))]


class FilterTests(unittest.TestCase):
    def test_threshold_keeps_85_drops_52_at_70(self):
        out = filter_detections([det(conf=0.85), det(conf=0.52)], 0.70)
        self.assertEqual([d.confidence for d in out], [0.85])

    def test_threshold_is_inclusive(self):
        self.assertEqual(len(filter_detections([det(conf=0.70)], 0.70)), 1)

    def test_class_filter_selected_only(self):
        dets = [det("person"), det("car", cid=2), det("dog", cid=16)]
        out = filter_detections(dets, 0.1, ["car", "dog"])
        self.assertEqual({d.class_name for d in out}, {"car", "dog"})

    def test_empty_or_none_selection_means_all(self):
        dets = [det("person"), det("car", cid=2)]
        self.assertEqual(len(filter_detections(dets, 0.1, [])), 2)
        self.assertEqual(len(filter_detections(dets, 0.1, None)), 2)

    def test_combined_filter(self):
        dets = [det("person", 0.9), det("person", 0.4), det("car", 0.95)]
        out = filter_detections(dets, 0.5, ["person"])
        self.assertEqual(len(out), 1)


class BoxTests(unittest.TestCase):
    def test_xyxy_to_xywh(self):
        self.assertEqual(xyxy_to_xywh(10, 20, 110, 220), (10, 20, 100, 200))

    def test_xyxy_rounding_and_swapped_corners(self):
        self.assertEqual(xyxy_to_xywh(10.4, 20.6, 110.4, 220.6), (10, 21, 100, 200))
        self.assertEqual(xyxy_to_xywh(110, 220, 10, 20), (10, 20, 100, 200))

    def test_clamp(self):
        self.assertEqual(clamp_box(-5, -1, 700, 500, 640, 480), (0, 0, 640, 480))

    def test_event_formatting(self):
        ev = detection_to_event(det("person", 0.853219, (10, 20, 110, 220)), "webcam", "sid-1")
        self.assertEqual((ev["bbox_x"], ev["bbox_y"], ev["bbox_w"], ev["bbox_h"]), (10, 20, 100, 200))
        self.assertEqual(ev["confidence"], 0.8532)
        self.assertEqual(ev["object_class"], "person")
        self.assertEqual(ev["source_type"], "webcam")
        self.assertEqual(set(ev), {"timestamp", "object_class", "confidence", "bbox_x", "bbox_y",
                                   "bbox_w", "bbox_h", "source_type", "session_id"})

    def test_percent_and_like_escape(self):
        self.assertEqual(confidence_percent(0.8532), "85.3%")
        self.assertEqual(escape_like("50%_x"), "50\\%\\_x")


class CooldownTests(unittest.TestCase):
    def test_same_class_suppressed_within_cooldown(self):
        t = [0.0]
        cd = CooldownFilter(1.0, clock=lambda: t[0])
        self.assertEqual(len(cd.filter([det("person")])), 1)
        t[0] = 0.5
        self.assertEqual(len(cd.filter([det("person")])), 0)
        t[0] = 1.0
        self.assertEqual(len(cd.filter([det("person")])), 1)

    def test_other_class_not_blocked(self):
        t = [0.0]
        cd = CooldownFilter(1.0, clock=lambda: t[0])
        cd.filter([det("person")])
        self.assertEqual(len(cd.filter([det("car", cid=2)])), 1)

    def test_all_instances_in_allowed_frame_pass(self):
        cd = CooldownFilter(1.0, clock=lambda: 0.0)
        self.assertEqual(len(cd.filter([det("person"), det("person", 0.8)])), 2)

    def test_zero_cooldown_logs_everything(self):
        cd = CooldownFilter(0.0, clock=lambda: 0.0)
        cd.filter([det()])
        self.assertEqual(len(cd.filter([det()])), 1)


class ObjectDetectorTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.d = ObjectDetector(model=self.model)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_class_names_come_from_model(self):
        self.assertEqual(self.d.class_names, ["person", "bicycle", "car"])

    def test_detect_filters_converts_and_clamps(self):
        out = self.d.detect(self.frame, 0.6, None, 320)
        self.assertEqual({x.class_name for x in out}, {"person", "bicycle"})  # 0.52 car dropped
        self.assertEqual(self.model.kwargs["imgsz"], 320)
        self.assertAlmostEqual(self.model.kwargs["conf"], 0.6)
        self.assertIsNone(self.model.kwargs["classes"])

    def test_class_selection_maps_names_to_ids(self):
        self.d.detect(self.frame, 0.1, ["car", "person"])
        self.assertEqual(sorted(self.model.kwargs["classes"]), [0, 2])
        out = self.d.detect(self.frame, 0.1, ["car"])
        self.assertEqual({x.class_name for x in out}, {"person", "bicycle", "car"} & {x.class_name for x in out})
        self.assertTrue(all(x.class_name == "car" for x in out))  # explicit 2nd-pass filter

    def test_box_clamped_to_frame(self):
        out = self.d.detect(self.frame, 0.1)
        wide = [x for x in out if x.class_name == "car"][0]
        self.assertEqual((wide.x1, wide.x2, wide.y2), (0, 640, 480))

    def test_unloaded_model_raises(self):
        from detector import ModelLoadError
        with self.assertRaises(ModelLoadError):
            ObjectDetector("yolov8n.pt").detect(self.frame, 0.5)


class DrawTests(unittest.TestCase):
    def test_draw_changes_pixels_and_keeps_input(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = draw_detections(frame, [det("person", 0.9, (50, 50, 150, 150))])
        self.assertEqual(frame.sum(), 0)          # original untouched
        self.assertGreater(out.sum(), 0)
        self.assertEqual(out.shape, frame.shape)

    def test_boxes_off_returns_plain_copy(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = draw_detections(frame, [det()], show_boxes=False)
        self.assertEqual(out.sum(), 0)


if __name__ == "__main__":
    unittest.main()

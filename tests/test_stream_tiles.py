"""Controlled view fusion must not confuse person coverage with item tiling."""

import unittest

from anomaly_eval.training.stream_tiles import TILES, fuse, global_box, tile_pixels


def detection(label, score, box=(0.1, 0.1, 0.2, 0.2)):
    return {"label": label, "confidence": score, "box": list(box)}


class StreamTilesTests(unittest.TestCase):
    def test_global_coordinates_use_actual_rounded_tile_origin(self):
        region = tile_pixels(TILES[4], 101, 79)
        self.assertEqual(region, (40, 32, 101, 79))
        self.assertEqual(
            global_box([0, 0, 61, 47], region, 101, 79), [40 / 101, 32 / 79, 1.0, 1.0]
        )

    def test_item_only_arm_never_imports_tile_persons(self):
        views = [{"tile": t, "detections": []} for t in TILES]
        views[0]["detections"] = [detection(1, 0.7)]
        views[1]["detections"] = [
            detection(1, 0.99, (0.5, 0.5, 0.8, 0.8)),
            detection(1001, 0.8),
        ]
        self.assertEqual([r["label"] for r in fuse(views, "item_tiles")], [1001, 1])
        self.assertEqual(sum(r["label"] == 1 for r in fuse(views, "all_tiles")), 2)

    def test_classwise_nms_keeps_overlapping_person_and_item(self):
        views = [{"tile": t, "detections": []} for t in TILES]
        views[0]["detections"] = [detection(1, 0.9), detection(1001, 0.8)]
        views[1]["detections"] = [detection(1001, 0.95)]
        result = fuse(views, "all_tiles")
        self.assertEqual(
            [(r["label"], r["confidence"]) for r in result], [(1001, 0.95), (1, 0.9)]
        )

    def test_full_arm_ignores_tile_detections(self):
        views = [{"tile": t, "detections": []} for t in TILES]
        views[4]["detections"] = [detection(1001, 0.99)]
        self.assertEqual(fuse(views, "full"), [])

    def test_partial_or_duplicate_view_sets_fail(self):
        views = [{"tile": TILES[0], "detections": []} for _ in TILES]
        with self.assertRaises(ValueError):
            fuse(views, "full")
        with self.assertRaises(ValueError):
            fuse(views[:1], "all_tiles")


if __name__ == "__main__":
    unittest.main()

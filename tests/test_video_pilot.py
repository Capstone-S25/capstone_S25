"""Temporal causality, ownership prerequisites and event-accounting regressions."""

import unittest

import numpy as np

from anomaly_eval.training.video import (
    CausalTracker,
    event_metrics,
    make_alerts,
    track_features,
    unattended_state,
)


def person(x=0.1):
    return {"label": 1, "confidence": 0.9, "box": [x, 0.1, x + 0.15, 0.7]}


def bag(x=0.2):
    return {"label": 27, "confidence": 0.85, "box": [x, 0.55, x + 0.05, 0.65]}


class VideoPilotTests(unittest.TestCase):
    def test_strict_float64_threshold_above_float32_max_emits_no_alert(self):
        threshold = np.nextafter(1.0, np.inf)
        self.assertEqual(
            make_alerts(
                [0.0, 0.5, 1.0], np.ones(3, dtype=np.float32), threshold, persistence=0
            ),
            [],
        )

    def test_explicit_film_parts_share_a_canonical_source(self):
        from scripts.video.guard_video_sources import canonical_title

        self.assertEqual(
            canonical_title("hmdb_title:The_Departed__Part_1"),
            canonical_title("hmdb_title:The_Departed__Part_2"),
        )
        self.assertNotEqual(
            canonical_title("hmdb_title:The_Departed"),
            canonical_title("hmdb_title:Other_Movie"),
        )

    def test_unattended_rule_cannot_combine_different_items(self):
        stable = {"kind": "item", "owner": 1, "stationary": 40.0, "owner_absence": 0.0}
        absent = {"kind": "item", "owner": 2, "stationary": 2.0, "owner_absence": 10.0}
        self.assertFalse(unattended_state([stable, absent]))
        self.assertTrue(unattended_state([{**stable, "owner_absence": 1.0}]))
        self.assertFalse(
            unattended_state([{**stable, "owner": None, "owner_absence": 10.0}])
        )

    def test_cold_start_item_has_no_invented_owner(self):
        tracker = CausalTracker()
        for t in range(10):
            tracks = tracker.update(t, [bag()])
        self.assertIsNone(tracks[0]["owner"])
        self.assertEqual(tracks[0]["owner_absence"], 0)
        self.assertEqual(tracks[0]["stationary"], 9)

    def test_owner_departure_requires_history_and_preserves_snapshot(self):
        tracker = CausalTracker()
        for t in range(3):
            old = tracker.update(t, [person(), bag()])
        old_copy = [dict(r) for r in old]
        self.assertIsNotNone(next(r for r in old if r["kind"] == "item")["owner"])
        for t in range(3, 8):
            tracks = tracker.update(t, [bag()])
        item = next(r for r in tracks if r["kind"] == "item")
        self.assertEqual(item["owner_absence"], 4)
        self.assertEqual(old, old_copy)
        self.assertTrue(np.isfinite(track_features(tracks)).all())

    def test_gap_discards_identity_and_low_scores_cannot_start_track(self):
        tracker = CausalTracker()
        old = tracker.update(0, [person()])[0]["id"]
        new = tracker.update(4, [person()])[0]["id"]
        self.assertNotEqual(old, new)
        self.assertEqual(CausalTracker().update(0, [{**bag(), "confidence": 0.3}]), [])

    def test_confidence_can_recover_existing_track(self):
        tracker = CausalTracker()
        old = tracker.update(0, [person()])[0]["id"]
        self.assertEqual(
            tracker.update(0.5, [{**person(0.11), "confidence": 0.3}])[0]["id"], old
        )

    def test_tracker_rejects_future_reordering_and_bad_boxes(self):
        tracker = CausalTracker()
        tracker.update(1, [])
        with self.assertRaises(ValueError):
            tracker.update(1, [])
        with self.assertRaises(ValueError):
            CausalTracker().update(0, [{**bag(), "box": [1, 0, 0, 1]}])

    def test_alert_is_not_backdated_and_unknown_gap_resets_persistence(self):
        self.assertEqual(make_alerts([0, 0.5, 1, 1.5, 2], [1] * 5, 0.5), [2.0])
        self.assertEqual(make_alerts([0, 0.5, 3, 3.5, 4, 4.5, 5], [1] * 7, 0.5), [5.0])
        self.assertEqual(
            make_alerts([0, 0.5, 1, 1.5, 2], [1, 1, np.nan, 1, 1], 0.5), []
        )

    def test_event_matching_counts_early_and_duplicate_alerts(self):
        m = event_metrics([4, 6, 7, 20], [(5, 10)], [(0, 5), (10, 15)])
        self.assertEqual(m["detected_events"], 1)
        self.assertEqual(m["normal_false_alerts"], 1)
        self.assertEqual(m["duplicate_alerts"], 1)
        self.assertEqual(m["unknown_time_alerts"], 1)
        self.assertEqual(m["normal_seconds"], 10)
        self.assertEqual(m["delays_seconds"], [1])

    def test_overlapping_normal_intervals_are_not_double_counted(self):
        m = event_metrics([], [], [(0, 10), (5, 15)])
        self.assertEqual(m["normal_seconds"], 15)
        self.assertIsNone(m["event_recall"])
        self.assertIsNone(m["median_delay_seconds"])

    def test_padded_future_features_cannot_change_temporal_score(self):
        import torch

        from anomaly_eval.training.temporal import TemporalHead

        torch.manual_seed(7)
        head = TemporalHead(6, 2).eval()
        visual = torch.randn(1, 4, 6)
        geometry = torch.randn(1, 4, 2)
        timing = torch.randn(1, 4, 2)
        lengths = torch.tensor([2])
        with torch.no_grad():
            original = head(visual, geometry, timing, lengths)
            visual[:, 2:] = 1e4
            geometry[:, 2:] = -1e4
            timing[:, 2:] = 1e4
            changed = head(visual, geometry, timing, lengths)
        torch.testing.assert_close(original, changed)


if __name__ == "__main__":
    unittest.main()

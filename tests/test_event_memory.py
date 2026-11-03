"""Regression tests for causal motion evidence, alert identity and dev budgets."""

import unittest

import numpy as np

from anomaly_eval.training.event_memory import (
    ItemEventMemory,
    RobustStationarityTracker,
    poisson_upper_rate,
)
from scripts.video.run_event_improvements import source_budget_threshold


def item(key=1, x=0.2, **kwargs):
    return {
        "id": key,
        "kind": "item",
        "box": [x, 0.4, x + 0.05, 0.5],
        "owner": 0,
        "stationary": 35.0,
        "owner_absence": 5.0,
        **kwargs,
    }


def detection(x=0.2):
    return {"label": 27, "confidence": 0.8, "box": [x, 0.4, x + 0.05, 0.5]}


class EventMemoryTests(unittest.TestCase):
    def test_isolated_detection_jitter_does_not_reset_stationarity(self):
        tracker = RobustStationarityTracker()
        for t in np.arange(0, 10, 0.5):
            live = tracker.update(float(t), [detection(0.23 if t == 5 else 0.2)])
        self.assertEqual(live[0]["stationary"], 9.5)

    def test_sustained_movement_resets_stationary_evidence(self):
        tracker = RobustStationarityTracker()
        for t in np.arange(0, 10, 0.5):
            live = tracker.update(float(t), [detection(0.23 if t >= 7 else 0.2)])
        self.assertLess(live[0]["stationary"], 3)

    def test_pending_motion_blocks_emission(self):
        memory = ItemEventMemory(persistence=0)
        self.assertEqual(memory.update(0, [item(motion_pending=True)]), [])
        self.assertEqual(len(memory.update(0.5, [item()])), 1)

    def test_track_fragmentation_retains_alarm_memory(self):
        memory = ItemEventMemory(persistence=0)
        self.assertEqual(len(memory.update(0, [item()])), 1)
        memory.update(0.5, [])
        self.assertEqual(memory.update(2, [item(key=999)]), [])

    def test_two_simultaneous_items_are_independent_events(self):
        memory = ItemEventMemory(persistence=0)
        self.assertEqual(len(memory.update(0, [item(), item(2, 0.7)])), 2)
        self.assertEqual(memory.update(0.5, [item(), item(2, 0.7)]), [])

    def test_item_absence_is_not_owner_return(self):
        memory = ItemEventMemory(persistence=0)
        memory.update(0, [item()])
        memory.update(10, [])
        self.assertEqual(memory.update(20, [item()]), [])

    def test_sustained_owner_return_rearms_same_item(self):
        memory = ItemEventMemory(persistence=0)
        memory.update(0, [item()])
        for t in range(1, 8):
            memory.update(t, [item(owner_absence=0.0)])
        self.assertEqual(len(memory.update(8, [item()])), 1)

    def test_short_gaps_reset_pending_persistence(self):
        memory = ItemEventMemory()
        memory.update(0, [item()])
        self.assertEqual(memory.update(3, [item()]), [])
        self.assertEqual(len(memory.update(5, [item()])), 0)  # gap again
        for t in [5.5, 6, 6.5]:
            self.assertEqual(memory.update(t, [item()]), [])
        self.assertEqual(len(memory.update(7, [item()])), 1)

    def test_invalid_time_is_rejected(self):
        m = ItemEventMemory()
        m.update(0, [])
        with self.assertRaises(ValueError):
            m.update(0, [])

    def test_dev_budget_is_satisfied_per_source_and_with_ties(self):
        rows = [{"source": s, "target": 0} for s in ["a"] * 20 + ["b"] * 20]
        scores = np.array([1.0] * 20 + [2.0] * 20)
        cutoff = source_budget_threshold(rows, scores)
        self.assertGreater(cutoff, 2.0)
        self.assertFalse((scores >= cutoff).any())
        with self.assertRaises(ValueError):
            source_budget_threshold([{"source": "x", "target": 1}], [1.0])

    def test_zero_false_alerts_still_has_nonzero_rate_upper_bound(self):
        self.assertAlmostEqual(poisson_upper_rate(0, 3600), -np.log(0.05))
        self.assertGreater(poisson_upper_rate(0, 165.7388667), 65.0)
        with self.assertRaises(ValueError):
            poisson_upper_rate(0, 0)


if __name__ == "__main__":
    unittest.main()

"""Rule-engine, stitching and metric checks on synthetic observations (no YOLO needed)."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

from stage1_bench import GTSequence, evaluate_run  # noqa: E402
from stage1_gate import GateConfig, Observation, RuleEngine, RunResult, merge_windows  # noqa: E402

FPS = 10.0


def person(raw_id, box, hist=None, kpts=None):
    return {"raw_id": raw_id, "box": list(map(float, box)), "conf": 0.9, "hist": hist, "kpts": kpts}


def obs(i, persons=(), items=(), blobs=()):
    return Observation(i, i / FPS, (720, 1280), list(persons), list(items), list(blobs), {"total_ms": 1.0})


class FakeSeq:
    seq_id = "s"


def run(cfg, observations):
    eng = RuleEngine(cfg, FPS)
    return [e for o in observations for e in eng.step(o)], eng


class RuleTests(unittest.TestCase):
    def test_loiter_fires_once_for_stationary_person(self):
        cfg = GateConfig(rules=("loiter",), loiter_s=3.0)
        events, _ = run(cfg, [obs(i, [person(1, (100, 100, 150, 300))]) for i in range(60)])
        self.assertEqual([e["event_type"] for e in events], ["sustained_presence"])
        self.assertAlmostEqual(events[0]["timestamp_s"], 3.0, places=1)

    def test_loiter_repeats_while_person_stays(self):
        cfg = GateConfig(rules=("loiter",), loiter_s=3.0, loiter_repeat_s=3.0)
        events, _ = run(cfg, [obs(i, [person(1, (100, 100, 150, 300))]) for i in range(100)])
        self.assertEqual([round(e["timestamp_s"]) for e in events], [3, 6, 9])

    def test_loiter_works_at_one_fps(self):
        cfg = GateConfig(rules=("loiter",), loiter_s=5.0)
        eng = RuleEngine(cfg, 1 / 1.04)
        frames = [Observation(i, i * 1.04, (720, 1280), [person(1, (100, 100, 150, 300))], [], [], {}) for i in range(10)]
        self.assertEqual(len([e for o in frames for e in eng.step(o)]), 1)

    def test_walking_person_does_not_loiter(self):
        cfg = GateConfig(rules=("loiter",), loiter_s=3.0)
        events, _ = run(cfg, [obs(i, [person(1, (10 + 15 * i, 100, 60 + 15 * i, 300))]) for i in range(60)])
        self.assertEqual(events, [])

    def test_fast_motion(self):
        cfg = GateConfig(rules=("speed",), speed_h_per_s=1.8)
        # 200 px tall person moving 50 px per frame at 10 fps = 2.5 heights/s
        events, _ = run(cfg, [obs(i, [person(1, (50 * i, 100, 50 * i + 60, 300))]) for i in range(10)])
        self.assertEqual([e["event_type"] for e in events], ["fast_motion"])

    def test_hand_to_face_needs_repeated_hits(self):
        k = np.zeros((17, 3), np.float32)
        k[0] = (125, 120, 0.9)   # nose
        k[9] = (128, 135, 0.9)   # wrist at mouth
        cfg = GateConfig(rules=("smoking",), smoke_min_hits=2)
        events, _ = run(cfg, [obs(0, [person(1, (100, 100, 150, 300), kpts=k)])])
        self.assertEqual(events, [])
        events, _ = run(cfg, [obs(i, [person(1, (100, 100, 150, 300), kpts=k)]) for i in range(3)])
        self.assertEqual([e["event_type"] for e in events], ["hand_to_face"])

    def test_unattended_requires_owner_to_leave(self):
        cfg = GateConfig(rules=("unattended",), unattended_s=1.0)
        bag = {"raw_id": 9, "box": [160.0, 260.0, 200.0, 300.0], "conf": 0.5, "cls": "backpack"}
        stay = [obs(i, [person(1, (100, 100, 150, 300))], [bag]) for i in range(20)]
        self.assertEqual(run(cfg, stay)[0], [])
        leave = stay[:5] + [obs(i, [person(1, (900, 100, 950, 300))], [bag]) for i in range(5, 30)]
        self.assertEqual([e["event_type"] for e in run(cfg, leave)[0]], ["unattended_object"])

    def test_preexisting_object_without_owner_is_ignored(self):
        cfg = GateConfig(rules=("unattended",), unattended_s=1.0)
        bag = {"raw_id": 9, "box": [160.0, 260.0, 200.0, 300.0], "conf": 0.5, "cls": "backpack"}
        self.assertEqual(run(cfg, [obs(i, [], [bag]) for i in range(40)])[0], [])


    def test_near_camera_needs_calibration_and_cools_down(self):
        tall = [obs(i, [person(1, (100, 50, 200, 600))]) for i in range(40)]
        self.assertEqual(run(GateConfig(rules=("near_camera",)), tall)[0], [])
        events, _ = run(GateConfig(rules=("near_camera",), near_camera_px=400, near_camera_cooldown_s=3.0), tall)
        self.assertEqual([round(e["timestamp_s"]) for e in events], [0, 3])
        short = [obs(i, [person(1, (100, 300, 150, 600))]) for i in range(40)]
        self.assertEqual(run(GateConfig(rules=("near_camera",), near_camera_px=400), short)[0], [])


    def test_in_view_catches_pacing_but_not_crossing(self):
        cfg = GateConfig(rules=("in_view",), inview_s=5.0, inview_area_h=2.0, inview_repeat_s=100)
        pacing = [obs(i, [person(1, (100 + 40 * ((i // 10) % 2), 100, 150 + 40 * ((i // 10) % 2), 300))]) for i in range(80)]
        events, _ = run(cfg, pacing)
        self.assertEqual([e["event_type"] for e in events], ["long_presence"])
        crossing = [obs(i, [person(1, (10 + 30 * i, 100, 60 + 30 * i, 300))]) for i in range(80)]
        self.assertEqual(run(cfg, crossing)[0], [])
        anywhere = GateConfig(rules=("in_view",), inview_s=5.0, inview_area_h=None, inview_repeat_s=100)
        self.assertEqual(len(run(anywhere, crossing)[0]), 1)

    def test_in_view_timer_survives_short_occlusion(self):
        hist = np.full(256, 1 / 128, np.float32)
        cfg = GateConfig(rules=("in_view",), inview_s=5.0, inview_area_h=None)
        seq = [obs(i, [person(1, (100, 100, 150, 300), hist)]) for i in range(30)]
        seq += [obs(i) for i in range(30, 45)]                                 # 1.5 s hidden, new tracker id after
        seq += [obs(i, [person(9, (105, 100, 155, 300), hist)]) for i in range(45, 60)]
        events, _ = run(cfg, seq)
        self.assertEqual([round(e["timestamp_s"]) for e in events], [5])

    def test_person_dedup_merges_rules_and_split_boxes(self):
        k = np.zeros((17, 3), np.float32)
        k[13] = (125, 250, 0.9); k[9] = (128, 260, 0.9)                         # wrist below knee
        cfg = GateConfig(rules=("reach_down", "near_camera"), near_camera_px=150)
        events, _ = run(cfg, [obs(0, [person(1, (100, 100, 150, 300), kpts=k), person(2, (100, 120, 150, 300))])])
        self.assertEqual(len(events), 1)


    def test_carried_bag_is_never_unattended(self):
        cfg = GateConfig(rules=("unattended",), unattended_s=1.0)
        seq = []
        for i in range(40):
            x = 100 + 20 * i
            bag = {"raw_id": 9, "box": [x + 10.0, 150.0, x + 40.0, 200.0], "conf": 0.5, "cls": "backpack"}
            seq.append(obs(i, [person(1 + i, (x, 100, x + 50, 300))], [bag]))     # tracker id keeps changing
        self.assertEqual(run(cfg, seq)[0], [])

    def test_thrown_bag_appears_alone_then_left(self):
        cfg = GateConfig(rules=("unattended",), unattended_s=3.0)
        bag = {"raw_id": 9, "box": [600.0, 260.0, 640.0, 300.0], "conf": 0.5, "cls": "backpack"}
        seq = [obs(i, [person(1, (100, 100, 150, 300))]) for i in range(30)]               # thrower, far from the bag
        seq += [obs(i, [person(1, (100 - 2 * (i - 30), 100, 150 - 2 * (i - 30), 300))], [bag]) for i in range(30, 80)]
        events, _ = run(cfg, seq)
        self.assertEqual([e["event_type"] for e in events], ["object_appeared", "unattended_object"])

    def test_scene_objects_from_the_start_do_not_fire(self):
        cfg = GateConfig(rules=("unattended",), unattended_s=1.0)
        bag = {"raw_id": 9, "box": [600.0, 260.0, 640.0, 300.0], "conf": 0.5, "cls": "suitcase"}
        self.assertEqual(run(cfg, [obs(i, [], [bag]) for i in range(60)])[0], [])

    def test_reach_down_skips_edge_people_and_unsure_keypoints(self):
        k = np.zeros((17, 3), np.float32)
        k[13] = (125, 250, 0.9); k[9] = (128, 260, 0.9)
        cfg = GateConfig(rules=("reach_down",))
        self.assertEqual(len(run(cfg, [obs(0, [person(1, (100, 100, 150, 300), kpts=k)])])[0]), 1)
        unsure = k.copy(); unsure[9, 2] = 0.4
        self.assertEqual(run(cfg, [obs(0, [person(1, (100, 100, 150, 300), kpts=unsure)])])[0], [])
        k_edge = k.copy(); k_edge[:, 0] -= 100
        self.assertEqual(run(cfg, [obs(i, [person(1, (0, 100, 50, 300), kpts=k_edge)]) for i in range(3)])[0], [])


class StitchTests(unittest.TestCase):
    def test_reborn_track_keeps_identity_after_occlusion(self):
        hist = np.full(256, 1 / 128, np.float32)
        cfg = GateConfig(rules=("loiter",), loiter_s=3.0)
        seq = [obs(i, [person(1, (100, 100, 150, 300), hist)]) for i in range(10)]
        seq += [obs(i) for i in range(10, 25)]                        # 1.5 s fully occluded
        seq += [obs(i, [person(7, (105, 100, 155, 300), hist)]) for i in range(25, 50)]
        events, eng = run(cfg, seq)
        self.assertEqual(eng.stitcher.n_stitched, 1)
        self.assertEqual(eng.last_cids, [1])
        self.assertEqual([e["entity_id"] for e in events], ["person1"])

    def test_different_clothes_are_not_stitched(self):
        a = np.zeros(256, np.float32); a[:128] = 1 / 128
        b = np.zeros(256, np.float32); b[128:] = 1 / 128
        seq = [obs(i, [person(1, (100, 100, 150, 300), a)]) for i in range(5)]
        seq += [obs(i, [person(7, (105, 100, 155, 300), b)]) for i in range(10, 15)]
        _, eng = run(GateConfig(rules=()), seq)
        self.assertEqual(eng.last_cids, [7])

    def test_stitching_can_be_disabled(self):
        hist = np.full(256, 1 / 128, np.float32)
        seq = [obs(0, [person(1, (100, 100, 150, 300), hist)]), obs(5, [person(7, (100, 100, 150, 300), hist)])]
        _, eng = run(GateConfig(rules=(), stitch=False), seq)
        self.assertEqual(eng.last_cids, [7])


class MetricTests(unittest.TestCase):
    def _result(self, n, events):
        observations = [obs(i, [person(1, (0, 0, 10, 10))]) for i in range(n)]
        return RunResult("s", "", FPS, n, n / FPS, events, observations, 1.0)

    def _event(self, t, roi, etype="reach_down"):
        return {"event_type": etype, "timestamp_s": t, "clip_window_s": [t - 0.5, t + 0.5], "roi_box": roi}

    def test_merge_windows(self):
        self.assertEqual(merge_windows([[3, 4], [0, 1], [0.5, 2]]), [[0, 2], [3, 4]])

    def test_event_and_crop_recall(self):
        gt = {i: [[100, 100, 150, 200]] for i in range(40, 50)}          # one event, 4.0-4.9 s
        g = GTSequence("x", FakeSeq(), gt)
        right = evaluate_run(g, self._result(100, [self._event(4.5, [100, 100, 150, 200])]))
        self.assertEqual((right["gt_events"], right["ev_temporal"], right["ev_spatial"], right["cand_correct"]), (1, 1, 1, 1))
        wrong_person = evaluate_run(g, self._result(100, [self._event(4.5, [600, 100, 650, 200])]))
        self.assertEqual((wrong_person["ev_temporal"], wrong_person["ev_spatial"], wrong_person["cand_correct"]), (1, 0, 0))
        none = evaluate_run(g, self._result(100, []))
        self.assertEqual((none["ev_temporal"], none["fn"], none["tn"]), (0, 10, 90))

    def test_frame_confusion_uses_clip_windows(self):
        g = GTSequence("x", FakeSeq(), {i: [[0, 0, 10, 10]] for i in range(10, 20)})
        r = evaluate_run(g, self._result(100, [self._event(1.5, [0, 0, 10, 10])]))
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"]), (10, 1, 0, 89))


if __name__ == "__main__":
    unittest.main()

"""Causal candidate evidence and externally reviewable model response semantics."""

import json
import tempfile
import unittest
from pathlib import Path

from anomaly_eval.reasoning.evidence import (
    output_schema,
    parse_answer,
    qualify,
    spread,
    validated_views,
)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = []
        for i in range(8):
            p = Path(self.temp.name) / f"{i}.jpg"
            p.write_bytes(b"image-fixture")
            self.paths.append(str(p))
        self.view = {"paths": self.paths[:4], "times_s": [0, 7, 13, 20]}
        self.candidate = {
            "tasks": ["loitering"],
            "timestamp_s": 20,
            "views": {"long": self.view},
        }

    def test_future_reversed_and_repeated_times_are_rejected(self):
        for times in [
            [0, 7, 13, 21],
            [0, 13, 7, 20],
            [0, 0, 13, 20],
            [0, 7, 13, float("nan")],
        ]:
            with self.subTest(times=times), self.assertRaises(ValueError):
                validated_views(
                    {
                        **self.candidate,
                        "views": {"long": {"paths": self.paths[:4], "times_s": times}},
                    }
                )
        self.assertEqual(validated_views(self.candidate)["long"], self.view)

    def test_rounding_tolerance_is_limited_to_writer_precision(self):
        c = {**self.candidate, "observed_until_s": 19.996}
        self.assertEqual(validated_views(c)["long"], self.view)
        with self.assertRaises(ValueError):
            validated_views({**c, "observed_until_s": 19.99})

    def test_unknown_is_preserved_for_missing_history_and_owner(self):
        yes = parse_answer(
            '{"unattended_items":["anomaly",[1,4],"item remains after a person leaves"]}',
            ["unattended_items"],
            4,
        )["unattended_items"]
        self.assertEqual(
            qualify("unattended_items", yes, self.candidate, self.view)["decision"],
            "unknown",
        )
        self.assertTrue(
            qualify("unattended_items", yes, {"owner": "person1"}, self.view)["alert"]
        )
        self.assertEqual(
            qualify("loitering", yes, {}, {"times_s": [0, 1, 2, 3]})["decision"],
            "unknown",
        )

    def test_invalid_json_keys_decisions_and_frame_citations_cannot_alert(self):
        for text in [
            "",
            "{}",
            '{"smoking":[[],[],"bad"]}',
            '{"smoking":["anomaly",[9],"cigarette"]}',
            '{"smoking":["anomaly",[],"cigarette"]}',
            '{"smoking":["yes",[1],"cigarette"]}',
        ]:
            out = parse_answer(text, ["smoking"], 4)["smoking"]
            self.assertEqual(out["decision"], "unknown")
            self.assertFalse(out["alert"])
        out = parse_answer(
            '{"smoking":["normal",[4],"no visible cigarette"]}', ["smoking"], 4
        )
        self.assertEqual(out["smoking"]["frame_refs"], [4])

    def test_sampling_keeps_actual_chronology_and_endpoint(self):
        view = {"paths": self.paths, "times_s": list(range(8))}
        subset = spread(view, 4)
        self.assertEqual(subset["times_s"], [0, 2, 5, 7])
        self.assertEqual(spread(view, 1)["times_s"], [7])
        self.assertEqual(len(set(subset["paths"])), 4)

    def test_native_schema_objects_are_parsed_and_guards_preserve_validity(self):
        text = json.dumps(
            {
                "unattended_items": {
                    "decision": "anomaly",
                    "frame_refs": [1, 4],
                    "evidence": "bag remains after person leaves",
                }
            }
        )
        value = parse_answer(text, ["unattended_items"], 4)["unattended_items"]
        self.assertTrue(value["response_valid"])
        guarded = qualify("unattended_items", value, self.candidate, self.view)
        self.assertTrue(guarded["response_valid"])
        self.assertEqual(guarded["decision"], "unknown")
        schema = output_schema(["unattended_items"], 4)
        self.assertEqual(schema["required"], ["unattended_items"])
        self.assertEqual(
            schema["properties"]["unattended_items"]["properties"]["frame_refs"][
                "items"
            ]["enum"],
            [1, 2, 3, 4],
        )

    def test_schema_compliant_placeholder_evidence_is_still_invalid(self):
        for placeholder in ["[]", "{}", "null", "..."]:
            value = parse_answer(
                json.dumps(
                    {
                        "smoking": {
                            "decision": "unknown",
                            "frame_refs": [],
                            "evidence": placeholder,
                        }
                    }
                ),
                ["smoking"],
                4,
            )["smoking"]
            self.assertFalse(value["response_valid"])
            self.assertFalse(value["alert"])


if __name__ == "__main__":
    unittest.main()

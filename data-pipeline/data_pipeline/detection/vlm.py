"""Send Stage 1 candidates to the VLM stage and record its answers."""

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

# predict-clip answers smoking and rough sleeping from the last view, littering
# from eight views of the action and the duration tasks from ~20 s of history.
VIEW_FOR_TASK = {
    "littering": "short",
    "smoking": "long",
    "rough_sleeping": "long",
    "loitering": "long",
    "unattended_items": "long",
}


class VLMClient(Protocol):
    def classify(self, candidate: dict) -> dict[str, dict]:
        """Return {task: prediction} for the candidate's tasks."""
        ...


class NoVLM:
    """Records candidates without classifying them (no VLM available)."""

    def classify(self, candidate: dict) -> dict[str, dict]:
        return {task: {"decision": "not_run"} for task in candidate["tasks"]}


Runner = Callable[[list[str]], None]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


class PredictClipVLM:
    """Yuchuan's fine-tuned CLIP classifier (python -m anomaly_eval predict-clip).

    Runs once per view set a candidate needs. Each call loads the model, so this
    suits demos and low candidate rates, not a busy camera.

    Args:
        python: Interpreter of an environment with anomaly_eval installed.
        repo_root: Checkout containing anomaly_eval and results/.
        model: "siglip2-b16-224" or "mobileclip2-s2".
        extra_args: More predict-clip options, e.g. ["--study", "<bundle dir>"].
        runner: Runs a command; replaced in tests.
    """

    def __init__(
        self,
        python: str,
        repo_root: Path,
        *,
        model: str = "siglip2-b16-224",
        extra_args: tuple[str, ...] = (),
        runner: Runner = _run,
    ) -> None:
        self.python = python
        self.repo_root = Path(repo_root)
        self.model = model
        self.extra_args = list(extra_args)
        self.runner = runner

    def command(self, view: dict, output: Path) -> list[str]:
        return [
            self.python,
            "-m",
            "anomaly_eval",
            "predict-clip",
            "--model",
            self.model,
            "--images",
            *view["paths"],
            "--timestamps",
            *(f"{t:g}" for t in view["times_s"]),
            "--output",
            str(output),
            *self.extra_args,
        ]

    def classify(self, candidate: dict) -> dict[str, dict]:
        out: dict[str, dict] = {}
        by_view: dict[str, list[str]] = {}
        for task in candidate["tasks"]:
            by_view.setdefault(VIEW_FOR_TASK[task], []).append(task)
        for name, tasks in by_view.items():
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "prediction.json"
                self.runner(self.command(candidate["views"][name], output))
                predictions = json.loads(output.read_text())["predictions"]
            for task in tasks:
                out[task] = {**predictions[task], "view": name}
        return out


class CandidateLog:
    """Append each classified candidate to <out_dir>/<source_id>/candidates.jsonl.

    Args:
        out_dir: Same directory given to Stage1Detector.
        vlm: Classifier for the candidates; NoVLM records them unclassified.
    """

    def __init__(self, out_dir: Path, vlm: VLMClient) -> None:
        self.out_dir = Path(out_dir)
        self.vlm = vlm

    def __call__(self, candidate: dict) -> dict:
        record = {**candidate, "vlm": self.vlm.classify(candidate)}
        path = self.out_dir / candidate["source_id"] / "candidates.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

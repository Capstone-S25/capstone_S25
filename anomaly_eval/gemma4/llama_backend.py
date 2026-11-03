"""Locally owned llama.cpp scoring for the official large Gemma QAT models."""

from __future__ import annotations

import math
import time

from anomaly_eval.gemma4.backends import data_url, images_for
from anomaly_eval.gemma4.protocol import question


class LlamaServerBackend:
    """QAT GGUF answer logits exposed by a locally owned llama.cpp server."""

    images_for = staticmethod(images_for)
    question = staticmethod(question)

    def __init__(self, url, *, batch_size=4, model_name="gemma4"):
        import requests

        self.requests = requests
        self.url = url.rstrip("/")
        self.batch_size = batch_size
        self.model_name = model_name
        self.metadata = {
            "backend": "llama.cpp",
            "quantization": "official_google_QAT_Q4_0",
            "thinking": False,
            "score": "Yes-minus-No log probability after equal +100 label bias; temperature=1, no truncation or penalties; bias cancels in the margin",
            "canonical_microbatch": 1,
        }
        self._tokens = None

    def _one(self, row, policy, label_bias=100, diagnostic=False):
        images = self.images_for(row, policy)
        message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(image)}}
                for image in images
            ]
            + [{"type": "text", "text": self.question(row, policy)}],
        }
        body = {
            "model": self.model_name,
            "messages": [message],
            "max_tokens": 1,
            "temperature": 1,
            "seed": 42,
            "samplers": ["temperature"],
            "top_k": 0,
            "top_p": 1,
            "min_p": 0,
            "repeat_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "logprobs": True,
            "top_logprobs": 2,
            "post_sampling_probs": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": False,
        }
        if self._tokens is None:
            tokens = []
            for word in ["Yes", "No"]:
                response = self.requests.post(
                    self.url + "/tokenize",
                    json={"content": word, "add_special": False},
                    timeout=30,
                )
                response.raise_for_status()
                values = response.json()["tokens"]
                if len(values) != 1:
                    raise ValueError("Expected one-token GGUF verbalizer")
                tokens.append(values[0])
            self._tokens = tokens
            self.metadata.update(yes_token=tokens[0], no_token=tokens[1])
        # The same offset promotes both labels into the returned top probabilities.
        # Their log-probability difference cancels this constant exactly.
        body["logit_bias"] = {str(token): label_bias for token in self._tokens}
        if diagnostic:
            body["logit_bias"] = {}
            body["top_logprobs"] = 20
        for attempt in range(3):
            response = self.requests.post(
                self.url + "/v1/chat/completions", json=body, timeout=(15, 300)
            )
            if response.status_code in {429, 503} and attempt < 2:
                time.sleep(1 + attempt)
                continue
            response.raise_for_status()
            break
        result = response.json()
        choice = result["choices"][0]
        entries = choice["logprobs"]["content"][0]["top_probs"]
        probs = {
            entry["token"]: math.log(entry["prob"])
            for entry in entries
            if entry["prob"] > 0
        }
        if diagnostic:
            return {
                "generated_answer": choice["message"].get("content"),
                "unbiased_top_probabilities": entries,
                "answer_token_mass_lower_bound": sum(
                    entry["prob"]
                    for entry in entries
                    if entry["token"] in {"Yes", "No"}
                ),
                "both_labels_returned": "Yes" in probs and "No" in probs,
            }
        if "Yes" not in probs or "No" not in probs:
            raise ValueError("Server did not expose both answer log probabilities")
        score = float(probs["Yes"] - probs["No"])
        if not math.isfinite(score):
            raise ValueError("Nonfinite GGUF margin")
        return {
            "score": score,
            "answer_token_mass": None,
            "images": len(images),
            "generated_answer": choice["message"].get("content"),
            "completion_tokens": result.get("usage", {}).get("completion_tokens"),
            "score_evidence": {
                "yes_logprob": probs["Yes"],
                "no_logprob": probs["No"],
                "equal_logit_bias": label_bias,
                "post_sampling_probabilities": True,
            },
        }

    def score(self, rows, policy):
        # Matching the native backend, canonical one-case forwards prevent
        # batch-composition-dependent numerical changes in answer margins.
        return [self._one(row, policy) for row in rows]

    def verify_batch(self, rows, policy, tolerance=0.12):
        batch = self.score(rows, policy)
        single = [self._one(row, policy) for row in rows]
        error = max(abs(a["score"] - b["score"]) for a, b in zip(batch, single))
        if error > tolerance:
            raise ValueError("Repeat GGUF score mismatch")
        bias_scores = [self._one(row, policy, label_bias=50) for row in rows]
        bias_error = max(
            abs(a["score"] - b["score"]) for a, b in zip(single, bias_scores)
        )
        if bias_error > 0.002:
            raise ValueError("Equal-label bias did not cancel in the GGUF margin")
        return {
            "examples": len(rows),
            "max_margin_error": error,
            "tolerance": tolerance,
            "equal_bias_50_vs_100_margin_error": bias_error,
            "execution": "Canonical single-example forwards; +50/+100 label-bias cancellation checked on development examples",
            "unbiased_answer_format": [
                self._one(row, policy, diagnostic=True) for row in rows
            ],
        }

    def close(self):
        pass


class ManagedLlamaBackend(LlamaServerBackend):
    """Own only this run's server; set image budgets with verified global flags."""

    def __init__(self, checkpoint, model_name, directory, *, batch_size=4):
        import socket
        from pathlib import Path

        from anomaly_eval.gemma4.protocol import BASELINE, ROOT

        self.checkpoint = Path(checkpoint)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.root = ROOT
        self.process = None
        self.log = None
        self.current_budget = None
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        super().__init__(
            f"http://127.0.0.1:{self.port}",
            batch_size=batch_size,
            model_name=model_name,
        )
        self.metadata.update(
            runtime="b11925",
            kv_cache="f16",
            cache_prompt=False,
            vision_budget="Global --image-max-tokens; server restarts when the policy budget changes",
        )
        self._ensure_policy(BASELINE)
        # Resolve IDs before the experiment identity is written.
        tokens = []
        for word in ["Yes", "No"]:
            response = self.requests.post(
                self.url + "/tokenize",
                json={"content": word, "add_special": False},
                timeout=30,
            )
            response.raise_for_status()
            value = response.json()["tokens"]
            if len(value) != 1:
                raise ValueError("Expected single-token GGUF answers")
            tokens.append(value[0])
        self._tokens = tokens
        self.metadata.update(yes_token=tokens[0], no_token=tokens[1])

    def _stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except __import__("subprocess").TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def _ensure_policy(self, policy):
        if self.current_budget == policy.image_tokens and self.process is not None:
            return
        import subprocess

        from anomaly_eval.datasets.transfer import atomic_json

        self._stop()
        models = [p for p in self.checkpoint.glob("*.gguf") if "mmproj" not in p.name]
        projectors = list(self.checkpoint.glob("*mmproj*.gguf"))
        if len(models) != 1 or len(projectors) != 1:
            raise ValueError("Expected one GGUF decoder and one vision projector")
        command = [
            "bash",
            str(self.root / "scripts/gemma4_llama.sh"),
            "-m",
            str(models[0]),
            "--mmproj",
            str(projectors[0]),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--alias",
            self.model_name,
            "--ctx-size",
            "16384",
            "--parallel",
            "1",
            "--batch-size",
            "4096",
            "--ubatch-size",
            "4096",
            "--flash-attn",
            "on",
            "--n-gpu-layers",
            "99",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--jinja",
            "--no-warmup",
            "--image-max-tokens",
            str(policy.image_tokens),
        ]
        self.log = (self.directory / f"llama_server_{policy.image_tokens}.log").open(
            "a"
        )
        self.process = subprocess.Popen(
            command, stdout=self.log, stderr=subprocess.STDOUT, cwd=self.root
        )
        atomic_json(
            self.directory / "server_launch.json",
            {
                "command": command,
                "pid": self.process.pid,
                "image_tokens": policy.image_tokens,
            },
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("llama.cpp exited; inspect its server log")
            try:
                response = self.requests.get(self.url + "/health", timeout=2)
                if response.ok:
                    self.current_budget = policy.image_tokens
                    return
            except self.requests.RequestException:
                pass
            time.sleep(0.5)
        self._stop()
        raise TimeoutError("Local model server did not become ready")

    def score(self, rows, policy):
        self._ensure_policy(policy)
        return super().score(rows, policy)

    def close(self):
        self._stop()

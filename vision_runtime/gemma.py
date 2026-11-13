"""Optional Gemma review via a configured OpenAI-compatible inference server."""

import base64
import io
import os
import threading
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

from .evidence import (
    VIEW_FOR_TASK,
    headline,
    history_problem,
    output_schema,
    parse_answer,
    parse_summary,
    prompt,
    qualify,
    spread,
    unknown,
    validated_views,
)


class GemmaClient:
    def __init__(self, url, model="gemma4-26b-a4b", timeout=None):
        # GEMMA_TIMEOUT_S: read timeout per request; raise it for a CPU server.
        timeout = float(timeout or os.environ.get("GEMMA_TIMEOUT_S") or 110)
        self.url = url.rstrip("/")
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("Gemma URL must use HTTP or HTTPS")
        self.model, self.timeout = model, timeout
        self.lock = threading.Lock()

    @staticmethod
    def image_url(path):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1536, 1536))
            stream = io.BytesIO()
            image.save(stream, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()

    def classify(self, candidate):
        # The service serializes requests across uploads. CLIP predictions are
        # intentionally not put into Gemma's prompt as a claimed ground truth.
        with self.lock:
            started = time.perf_counter()
            views, output, raw, summaries = validated_views(candidate), {}, [], []
            for name in dict.fromkeys(VIEW_FOR_TASK[t] for t in candidate["tasks"]):
                tasks = [t for t in candidate["tasks"] if VIEW_FOR_TASK[t] == name]
                view = spread(views[name], 8 if name == "short" else 4)
                active = []
                for task in tasks:
                    problem = history_problem(task, view)
                    if problem:
                        output[task] = unknown(problem)
                    else:
                        active.append(task)
                if not active:
                    continue
                paths = list(view["paths"])
                crop = candidate.get("crop_path")
                has_crop = bool(
                    crop
                    and Path(crop).is_file()
                    and candidate.get("candidate_origin") != "scheduled_scene_review"
                )
                if has_crop:
                    paths.append(crop)
                text = prompt(candidate, active, view, has_crop, object_output=True)
                body = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": self.image_url(p)},
                                }
                                for p in paths
                            ]
                            + [{"type": "text", "text": text}],
                        }
                    ],
                    "max_tokens": 280,
                    "temperature": 0,
                    "seed": 42,
                    "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "candidate_evidence",
                            "strict": True,
                            "schema": output_schema(active, len(view["paths"])),
                        },
                    },
                }
                key = os.environ.get("GEMMA_API_KEY")
                headers = {"Authorization": "Bearer " + key} if key else {}
                response = requests.post(
                    self.url + "/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=(5, self.timeout),
                )
                response.raise_for_status()
                answer = response.json()["choices"][0]["message"].get("content") or ""
                parsed = parse_answer(answer, active, len(view["paths"]))
                summary = parse_summary(answer)
                if summary:
                    summaries.append(summary)
                for task in active:
                    output[task] = qualify(task, parsed[task], candidate, view)
                raw.append(
                    {
                        "tasks": active,
                        "text": answer,
                        "frame_times_s": view["times_s"],
                        "crop_used": has_crop,
                    }
                )
            return {
                "predictions": output,
                "headline": headline(output),
                "summary": " ".join(summaries) or None,
                "raw_responses": raw,
                "seconds": time.perf_counter() - started,
                "model": self.model,
            }

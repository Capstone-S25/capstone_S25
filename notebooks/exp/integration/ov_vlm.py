"""OpenVINO for the VLM's MobileCLIP2-S2 image encoder (one compiled model per weight blend alpha the frozen policies
use) and, optionally, its Faster R-CNN. A test harness: patches a ClipStrategy instance, Yuchuan's code untouched.
  python -I ov_vlm.py export BUNDLE OUTDIR          -> OUTDIR/encoder_a<alpha>.xml
  python -I ov_vlm.py compare BUNDLE OUTDIR CANDS ROUTING   torch vs OpenVINO scores and time on the same candidates
"""
import json, sys, time
import numpy as np

def engine(bundle):
    sys.path.insert(0, bundle)
    from anomaly_eval.reasoning.backends import ClipStrategy
    return ClipStrategy(model="mobileclip2-s2", device="cpu")

def alphas(eng):
    a = {float(p["selected"]["selection"]["strategy"].rsplit("a", 1)[1]) for p in eng.policy.values()}
    if any("crop" in p["selected"]["selection"]["strategy"] for p in eng.policy.values()):
        a.add(0.0)
    return sorted(a)

def set_alpha(eng, a):
    import torch
    for name, base in eng.original.items():
        eng.parameters[name].copy_(torch.lerp(base, eng.previous["parameters"][name], a))

class Visual(__import__("torch").nn.Module):
    def __init__(self, backbone):
        super().__init__(); self.b = backbone
    def forward(self, x):
        return self.b.encode_image(x, normalize=True)

def export(bundle, out):
    import torch, openvino as ov
    from pathlib import Path
    eng = engine(bundle); Path(out).mkdir(parents=True, exist_ok=True)
    size = eng.encoder.processor.transforms[0].size if hasattr(eng.encoder.processor, "transforms") else 256
    size = size if isinstance(size, int) else size[0]
    for a in alphas(eng):
        set_alpha(eng, a)
        m = ov.convert_model(Visual(eng.encoder.backbone).eval(), example_input=torch.zeros(1, 3, size, size),
                             input=[(-1, 3, size, size)])
        ov.save_model(m, f"{out}/encoder_a{a:g}.xml")
        print("alpha", a, "->", f"{out}/encoder_a{a:g}.xml", "input", size, flush=True)

def patch(eng, out, threads=None):
    """Route encoder.encode to the OpenVINO model of whichever alpha is currently loaded (identified by a fingerprint
    of one blended parameter)."""
    import torch, openvino as ov
    from anomaly_eval.training.clip import pool_views
    core = ov.Core()
    cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"} | ({"INFERENCE_NUM_THREADS": threads} if threads else {})
    probe = next(iter(eng.original))
    models = {}
    for a in alphas(eng):
        set_alpha(eng, a)
        key = float(eng.parameters[probe].detach().double().sum())
        models[key] = core.compile_model(f"{out}/encoder_a{a:g}.xml", "CPU", cfg)
    def encode(images, lengths):
        key = float(eng.parameters[probe].detach().double().sum())
        m = models[min(models, key=lambda k: abs(k - key))]
        feats = torch.from_numpy(m(images.numpy())[0])
        return pool_views(feats.float(), lengths)
    eng.encoder.encode = encode
    return eng

def patch_detector(eng, out, bundle):
    """Faster R-CNN through OpenVINO, one compiled model per input size (the trace fixes the resize constants),
    cached as OUT/frcnn_<h>x<w>.xml. Same preprocessing and filtering as ClipStrategy._detections."""
    import os, torch, openvino as ov
    from PIL import Image, ImageOps
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.transforms.functional import pil_to_tensor
    core, compiled, holder = ov.Core(), {}, {}
    cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
    class Wrap(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x):
            r = s.m([x])[0]; return r["boxes"], r["labels"], r["scores"]
    def model_for(h, w):
        if (h, w) not in compiled:
            path = f"{out}/frcnn_{h}x{w}.xml"
            if not os.path.exists(path):
                if "torch" not in holder:
                    m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
                    m.load_state_dict(torch.load(f"{bundle}/data/models/detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
                                                 weights_only=True, map_location="cpu"))
                    holder["torch"] = Wrap(m.eval())
                ov.save_model(ov.convert_model(holder["torch"], example_input=(torch.zeros(3, h, w),)), path)
            compiled[(h, w)] = core.compile_model(path, "CPU", cfg)
        return compiled[(h, w)]
    def detections(paths):
        out_ = {}
        for name in paths:
            with Image.open(name) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((1280, 1280))
                w, h = image.size
                tensor = pil_to_tensor(image).float() / 255
            r = model_for(h, w)(tensor.numpy())
            boxes, labels, scores = r[0], r[1], r[2]
            out_[name] = [{"box": [float(v) / s for v, s in zip(box, [w, h, w, h])], "label": int(label), "confidence": float(score)}
                          for box, label, score in zip(boxes, labels, scores) if int(label) in {1, 27, 31, 33} and float(score) >= 0.5]
        return out_
    eng._detections = detections
    return eng


def patch_detector_yolo(eng, weights):
    """Replace Faster R-CNN with YOLO26s (OpenVINO) for the crop boxes: COCO person / backpack / handbag / suitcase
    (YOLO ids 0, 24, 26, 28 = torchvision ids 1, 27, 31, 33), confidence >= 0.5, same 1280 px thumbnail."""
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    yolo, ids = YOLO(weights, task="detect"), {0: 1, 24: 27, 26: 31, 28: 33}
    def detections(paths):
        out_ = {}
        for name in paths:
            with Image.open(name) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((1280, 1280))
                w, h = image.size
                r = yolo.predict(image, imgsz=640, conf=0.5, classes=list(ids), verbose=False, device="cpu")[0]
            out_[name] = [{"box": [float(v) / s for v, s in zip(b, [w, h, w, h])], "label": ids[int(c)], "confidence": float(cf)}
                          for b, c, cf in zip(r.boxes.xyxy.numpy(), r.boxes.cls.numpy(), r.boxes.conf.numpy())]
        return out_
    eng._detections = detections
    return eng


def compare(bundle, out, cands, routing):
    eng = engine(bundle)
    routing = json.loads(routing); cands = [e for e in json.load(open(cands)) if routing.get(e["event_type"])]
    def go():
        res, ts = {}, []
        for e in cands:
            t = time.perf_counter()
            r = eng.classify({"views": e["views"], "observed_until_s": e["timestamp_s"], "tasks": routing[e["event_type"]]})
            ts.append(time.perf_counter() - t); res[e["event_id"]] = r["predictions"]
        return res, ts
    go()  # warm-up
    ref, t_ref = go()
    patch(eng, out)
    if len(sys.argv) > 6:
        patch_detector_yolo(eng, sys.argv[6])
    else:
        patch_detector(eng, out, bundle)
    go()
    ovr, t_ov = go()
    diffs, flips = [], 0
    for k in ref:
        for task, p in ref[k].items():
            if "score" in p:
                diffs.append(abs(p["score"] - ovr[k][task]["score"])); flips += p["decision"] != ovr[k][task]["decision"]
    print(json.dumps({"candidates": len(cands), "torch mean s": round(float(np.mean(t_ref)), 2), "openvino mean s": round(float(np.mean(t_ov)), 2),
                      "score |diff| max": round(max(diffs), 5), "decisions changed": flips, "scored answers": len(diffs)}))

if __name__ == "__main__":
    {"export": lambda: export(*sys.argv[2:4]), "compare": lambda: compare(*sys.argv[2:6])}[sys.argv[1]]()

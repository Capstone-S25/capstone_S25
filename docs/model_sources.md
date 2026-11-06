# Model sources and licenses

Weights are transferred separately from Git. `models/manifest.json`, written
by `tools/import_models.py`, records the source commit, pinned base revision,
file sizes and SHA-256 hashes. The original archive's manifest records the
source paths. The importer verifies copied files against that manifest; verify
the archive checksum sidecar before importing an untrusted copy.

**Attribution:** Apple Machine Learning Research Model is licensed under the
Apple Machine Learning Research Model License Agreement.

The MobileCLIP artifacts are model derivatives: project work fine-tuned the
visual backbone, fitted five task classifiers, selected per-task feature and
weight-interpolation recipes, and calibrated development thresholds. They are
not Apple-endorsed checkpoints. The included Apple agreement limits the base
and derivatives to non-commercial scientific research and academic development.
This application is a research prototype; transferring weights does not grant
commercial deployment rights.

| Component / application location | Source | License reference |
|---|---|---|
| `models/mobileclip2-s2/base/` | [timm/MobileCLIP2-S2-OpenCLIP](https://huggingface.co/timm/MobileCLIP2-S2-OpenCLIP), revision `ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6` | [Original Apple model](https://huggingface.co/apple/MobileCLIP2-S2); copied agreement in [licenses/mobileclip2.txt](licenses/mobileclip2.txt). |
| `selected_heads.pt`, `finetuned_best.pt`, policies and thresholds | Project training/strategy selection, 15–16 October 2026 | Transferred `models/mobileclip2-s2/provenance.json` records training provenance. Upstream model and training-source terms still apply. |
| `models/detectors/yolo26s.pt` and `yolo26s-pose.pt` | [Ultralytics assets](https://github.com/ultralytics/assets/releases/tag/v8.4.0) | [Ultralytics AGPL-3.0 / Enterprise licensing](https://www.ultralytics.com/license), [copied license](licenses/ultralytics.txt). |
| `models/detectors/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` | [Official torchvision checkpoint](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth) | [Copied torchvision license](licenses/torchvision.txt); [pretrained-model notice](https://github.com/pytorch/vision#models) addresses training-data terms separately. |
| `vision_runtime/tracking/weights/*.npz` | Fitted pose action, reach and crop checks from `feat/stage1-integration` commit `a206896` | Project training artifacts. Their research studies/data provenance remain on the source branch rather than being duplicated in this application branch. |
| ByteTrack | Ultralytics integration; no additional learned checkpoint | [Upstream ByteTrack](https://github.com/ifzhang/ByteTrack) and installed integration retain their respective terms. |
| Optional Gemma endpoint | Independently provisioned model server; weights are not included | Apply the actual served model's terms, e.g. [Gemma terms](https://ai.google.dev/gemma/terms). An endpoint alias does not verify its weights or license. |

The Apple, Ultralytics and torchvision license texts were retrieved on
3 November 2026; the package texts correspond to source versions 8.4.199 and
0.23.0 respectively. This document consolidates the known runtime sources and
does not replace upstream terms or independently certify the rights of all
training datasets.

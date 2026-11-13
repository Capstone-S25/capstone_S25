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
| `models/ensemble-20261113/siglip2-b/` | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224), revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | Apache-2.0 per its model card. |
| `models/ensemble-20261113/mobileclip2-b/` | [timm/MobileCLIP2-B-OpenCLIP](https://huggingface.co/timm/MobileCLIP2-B-OpenCLIP), revision `6a3bc3dc48d04504ecfca54f8b26bd78a71ddc4e` | Apple Machine Learning Research Model License ([licenses/mobileclip2.txt](licenses/mobileclip2.txt)); same research-only limits as above. |
| `models/ensemble-20261113/heads.npz`, `ensemble.json` | Project fitting, 11–13 November 2026 ([F1 study](f1_study.md)) | Project artifacts; upstream encoder and training-data terms still apply. |
| `models/ensemble-20261113/smoking_detector.pt` | YOLO26s fine-tuned by the project on the CCTV smoking head boxes | Derived from Ultralytics YOLO26s (AGPL-3.0 / Enterprise) and the dataset's own terms. |
| `gemma/models/*.gguf` (transfer bundle only) | [google/gemma-4-E4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf), revision `4b4a2c1d584be7264f87aac328a1bc739ce81b6c` | Apache-2.0 per its model card; `gemma/gemma.json` records file hashes. |
| `gemma/llama/` (transfer bundle only) | llama.cpp release build 11925, CPU libraries only | MIT; `gemma/llama/LICENSE`. |
| Other Gemma endpoint | Independently provisioned model server | Apply the actual served model's terms. An endpoint alias does not verify its weights or license. |

The Apple, Ultralytics and torchvision license texts were retrieved on
3 November 2026; the package texts correspond to source versions 8.4.199 and
0.23.0 respectively. This document consolidates the known runtime sources and
does not replace upstream terms or independently certify the rights of all
training datasets.

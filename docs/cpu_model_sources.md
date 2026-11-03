# CPU bundle model sources

The archive's SHA-256 manifest identifies the exact copied files. This source
list does not replace upstream license terms or grant a new license to weights.
Copies of the upstream Apple model, Ultralytics and torchvision license texts
are included under `docs/licenses/` (retrieved 3 November 2026). The Apple text
comes from the original model's `LICENSE`; Ultralytics and torchvision texts are
from the installed `v8.4.199` and `v0.23.0` source tags respectively.

**Attribution:** Apple Machine Learning Research Model is licensed under the
Apple Machine Learning Research Model License Agreement.

The trained MobileCLIP artifacts in this bundle are **model derivatives**:
the project fine-tuned visual backbone parameters, fitted five anomaly
classification heads, selected per-task feature/weight-interpolation recipes,
and calibrated development thresholds. These changes are project work, not an
Apple-endorsed checkpoint. The included Apple agreement limits the base and
derivatives to non-commercial scientific research and academic development;
this bundle is a research prototype transfer, not a commercial deployment grant.

| Component | Source | License reference |
|---|---|---|
| MobileCLIP2-S2 OpenCLIP base | [timm/MobileCLIP2-S2-OpenCLIP](https://huggingface.co/timm/MobileCLIP2-S2-OpenCLIP), revision `ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6` | Model card identifies Apple AMLR; see [original Apple model](https://huggingface.co/apple/MobileCLIP2-S2) for model license terms. |
| YOLO26s and YOLO26s-pose | [Ultralytics assets](https://github.com/ultralytics/assets/releases/tag/v8.4.0) | [Ultralytics AGPL-3.0 / Enterprise licensing](https://www.ultralytics.com/license). |
| Faster R-CNN ResNet50 FPN v2 reference weights | [Official torchvision checkpoint](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth) | [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE); [pretrained-model notice](https://github.com/pytorch/vision#models) covers training-data terms separately. |
| Selected classifier heads and backbone updates | Local 15–16 October training / strategy study | Included `results/clip_finetuning_20261015/provenance.json` records training provenance. Upstream model and training-source terms still apply. |
| Pose action MLP | Tracked `notebooks/weights/pose_action_mlp.npz` | Project training artifact; see the repository's pose study and dataset-source registry for provenance. |
| ByteTrack | Included through the Ultralytics package | No additional learned checkpoint. [Upstream ByteTrack](https://github.com/ifzhang/ByteTrack) and the installed integration retain their respective terms. |

CPU wheel installation follows the [official PyTorch previous-version
instructions](https://pytorch.org/get-started/previous-versions/#v280). This
archive does not contain third-party Python package wheels; pip installs their
license metadata with the dependencies.

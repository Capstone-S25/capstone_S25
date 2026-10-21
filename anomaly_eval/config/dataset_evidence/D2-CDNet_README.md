---
language: en
pretty_name: D2-CDNet
tags:
- computer-vision
- video
- foreground-segmentation
- background-subtraction
- cdnet
- surveillance
- webdataset
license: apache-2.0
---

# D²-CDNet Dataset

This Hugging Face dataset repository hosts **archive shards** of the *D²-CDNet* (Diversified Domain CDNet) data package used in our foreground-segmentation project (3-stream model: RGB + static background + BGS + Optical Flow).

- GitHub (code + training + evaluation): https://github.com/tauseefakhund/TriUNet
- Dataset repository (this page): https://huggingface.co/datasets/tauseefakhund/D2-CDNet

## What is inside this repo?

All dataset shards are uploaded as `.tar` archives under:

- `raw_tar/`

Typically, **one `.tar` per CDnet category** (e.g., `raw_tar/badWeather.tar`, `raw_tar/baseline.tar`, `raw_tar/cameraJitter.tar`, ...).

These archives contain the folder structure expected by the training code in the GitHub repository (CDnet14 category/video folders and the precomputed modalities).

## Folder structure (after extracting a shard)

```text
CDnet14/<category>/<video>/
├── input/                      # RGB frames (e.g., in000123.png)
├── backgrounds/                # Static background reference (e.g., in000123.png)
├── bgs_subsense/               # SuBSENSE-based BGS masks
├── FLUX/                       # Optical-flow / motion maps
├── groundtruth/                # Binary GT masks
├── diversified/                # Synthetic diversified frames (if available)
├── diversified_from_original/  # Fallback sampled real frames (if available)
├── div_groundtruth/            # Paired GT for diversified/paired training (if available)
├── div_FLUX/                   # Paired Optical flow for diversified/paired training (if available)
├── ROI.bmp / ROI.jpg           # Spatial ROI mask (CDnet standard)
└── temporalROI.txt             # Temporal ROI file (CDnet standard)

```

Note: Not every video necessarily contains every folder above; availability depends on the category/video and the shard version.

## How to download

Go to **Files and versions** → open `raw_tar/` → download the desired category tar file.

Example: download `raw_tar/badWeather.tar`.

## How to extract (Linux / WSL / macOS)

Extract a downloaded tar into a chosen directory:

```bash
mkdir -p data
tar -xf badWeather.tar -C data
```
Recommended usage
This HF repository provides the data shards only.

Use the GitHub repository for:

Dataloaders and expected paths

Training scripts

Evaluation and cross-dataset testing (e.g., LaSIESTA)

GitHub: https://github.com/tauseefakhund/TriUNet

License & attribution
The apache-2.0 license here applies to this repository’s packaging/metadata (and any files authored in this repo).

Please also acknowledge the original dataset sources (e.g., CDnet14) and follow their terms.
## Citation

If you use **D²-CDNet (Diversified Domain CDNet)**, please cite it as:

```bibtex
@misc{Akhund2026D2CDNet,
  author       = {Tauseef Ud Din Akhund},
  title        = {D\textsuperscript{2}-CDNet (Diversified Domain CDNet)},
  howpublished = {\url{https://huggingface.co/datasets/tauseefakhund/D2-CDNet}},
  url          = {https://huggingface.co/datasets/tauseefakhund/D2-CDNet},
  year         = {2026},
  note         = {Accessed: 2026-01-04}
}

```


Contact
For questions/issues:

Email: tauseefakhund11@gmail.com

---
license: apache-2.0
---
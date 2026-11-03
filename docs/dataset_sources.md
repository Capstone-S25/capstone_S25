# Dataset sources, downloads and licenses

Catalog reviewed on 2026-10-20. The canonical machine-readable registry is [`anomaly_eval/config/dataset_sources.json`](../anomaly_eval/config/dataset_sources.json). It includes author homepages, versions, download manifests, license evidence and limitations.

The later additions include Stanford40, MIVIA IWDD, Street Scene, smoking image transfer data, thermal trajectories, Commons/Flickr photographs, HMDB51, MEVA, AVSS, small-item component footage and the XRF55 RGB public pilot. Acquiring files does not make all of them eligible labeled training examples.

The catalog records **17 source entries** and **4,520 pinned file assets** (**9,662,138,000 bytes** of file payload). These are acquisition artifacts, not independent examples. Archives may contain unused classes or duplicate mirrors; Street Scene entries are selected frames from related videos. The original Drive source and gated Roboflow exports use separate live providers, so their contents are not included in this pinned-file count.

## Consolidated tables

- [Source and license CSV](dataset_sources.csv): one row per source.
- [All pinned download links and hashes](dataset_downloads.csv): one row per file/mirror, with source page and license.
- [Commons/Flickr author and license attribution](commons_attribution.csv): one row per acquired photograph, including artist/credit text and original HTML links, original page, license URL, restrictions and exact byte hash.
- [Downloader usage and recovery](dataset_fetching.md).

## Source inventory

| Source | Recorded assets | License declaration / evidence | Use |
| --- | ---: | --- | --- |
| [ABODA abandoned-object videos](https://github.com/kevinlin311tw/ABODA) | 12 | [not_specified](https://github.com/kevinlin311tw/ABODA) | Unattended-item video evidence; project review required for event timing. |
| [CAVIAR INRIA selected videos and XML](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) | 17 | [CC-BY-SA (version unspecified)](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) | Eight selected sequences; object presence alone is not an unattended state. |
| [Stanford 40 Actions](http://vision.stanford.edu/Datasets/40actions.html) | 5 | [not_specified](http://vision.stanford.edu/Datasets/40actions.html) | 40 action classes, 9,532 images. Smoking and action confusers; acquired primary archives and one independent mirror archive. |
| [MIVIA Illegal Waste Dumping Dataset](https://github.com/MiviaLab/IllegalWasteDumping) | 999 | [not_specified](https://github.com/MiviaLab/IllegalWasteDumping) | Littering/dumping videos and event annotations. 500 listed videos; 498 acquired. Study partitions contain 435 after source-relationship exclusion. |
| [Street Scene (selected frames and track annotations)](https://zenodo.org/records/10870472) | 2464 | [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) | RGB loitering proxy from published tracks and anomaly types. Supervised research re-split of original Test videos; not the official benchmark. |
| [Smoker Detection image classification dataset](https://www.kaggle.com/datasets/sujaykapadnis/smoking) | 1 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | Smoking image transfer set; not independent CCTV evidence. |
| [RS-WACV24 thermal loitering trajectories](https://github.com/johnnynunez/RS-WACV24_Loitering) | 6 | [MIT (repository release)](https://github.com/johnnynunez/RS-WACV24_Loitering/blob/64e6ffdda4dad87b8d13daf69e8b2b9655a6459c/LICENSE) | 19,737 trajectories; 17,878 eligible. Single thermal camera over 31 dates; separate modality study, not RGB CLIP training. |
| [Wikimedia Commons / original Flickr photographs](https://commons.wikimedia.org/) | 259 | [per-file (CC-BY / CC-BY-SA / CC0 / public domain / other)](https://commons.wikimedia.org/) | Apparently sleeping/resting people and controls. Project proxy labels do not establish housing status; most acquired candidates are not training examples. |
| [HMDB51 original action archive](https://serre.lab.brown.edu/) | 1 | [not_specified](https://serre.lab.brown.edu/) | Seven extracted confuser/smoking classes, 906 clips; downloaded archive includes all 51 classes. The fetcher retrieves the archive only. |
| [MEVA abandon-package video subset](https://mevadata.org/) | 13 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | 13 acquired videos of 14 annotation matches. Abandon-package actions are distinct from persistent unattended states. |
| [AVSS 2007 abandoned-baggage pilot (AODsurvey bundle)](http://www-vpu.eps.uam.es/publications/AODsurvey/) | 1 | [not_specified](http://www-vpu.eps.uam.es/publications/AODsurvey/) | Three EASY/MEDIUM/HARD videos plus original annotations. Data extraction only; downloaded source code is never executed. |
| [Author small-item video component](https://github.com/hangsuuuu/video-of-abandoned-object-detection) | 1 | [not_specified](https://github.com/hangsuuuu/video-of-abandoned-object-detection) | Item/owner component investigation; not a full unattended-baggage benchmark. |
| [XRF55 RGB Public (selected scene/subject pilot)](https://www.kaggle.com/datasets/airslab2020/xrf55-rgb-depth-ir) | 740 | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) | 740 selected five-second RGB clips (886,570,077 bytes) from the 43.36 GB release; smoking gesture plus phone/drink/eat controls. Weak action labels, not verified cigarette visibility. |
| [D2-CDNet foreground segmentation (metadata only)](https://huggingface.co/datasets/tauseefakhund/D2-CDNet) | 1 | [Apache-2.0 (mirror declaration; upstream unresolved)](https://huggingface.co/datasets/tauseefakhund/D2-CDNet/blob/main/README.md) | Metadata fetched; optional multi-GB shards are auxiliary masks, not anomaly event labels. |
| [Original shared Drive corpus](https://drive.google.com/drive/folders/1sj9yoyBsmaqMxpnueemZOoPDNrb4p54j) | live provider | [mixed / partly unspecified](https://drive.google.com/drive/folders/1sj9yoyBsmaqMxpnueemZOoPDNrb4p54j) | Initial smoking, littering, loitering, rough-sleeping and unattended-item folders; 19,678 listed file entries in the original receipt. Listing and retrieval remain dynamic in the existing concurrent Drive downloader. |
| [roboflow suspicious](https://universe.roboflow.com/abandoned-object/abandoned-object-detection-9fijk) | live provider | [not_verified (export access required)](https://universe.roboflow.com/abandoned-object/abandoned-object-detection-9fijk) | Object-annotation candidates; unacquired gated exports; no claim that object presence labels unattended state. |
| [roboflow objects](https://universe.roboflow.com/getting/abandoned-w7arq) | live provider | [not_verified (export access required)](https://universe.roboflow.com/getting/abandoned-w7arq) | Object-annotation candidates; unacquired gated exports; no claim that object presence labels unattended state. |

## License scope and attribution

A public URL is not a license grant. Entries marked `not_specified`, `not_verified` or `upstream unresolved` retain that uncertainty. A repository code license or mirror declaration does not establish rights to an underlying video collection. The registry records source statements and evidence; it does not grant new rights.

Per-file Commons terms take precedence over the aggregate row. Preserve the creator, original source page and license link when reusing a photograph; record crop/resize or other modifications in downstream exports. The acquisition manifest describes original downloaded bytes. Review labels and model predictions are project artifacts.

### ABODA abandoned-object videos

No dataset license located in the author repository; public download is not a license grant.

Revision: 08284de80c5a8c5120e95902b5b44f548a691321

### CAVIAR INRIA selected videos and XML

Author page explicitly states CC BY-SA without a version. Acknowledge EC Funded CAVIAR / IST 2001 37540.

### Stanford 40 Actions

Author page requests citation; no explicit dataset license located in the saved page. The Hugging Face mirror does not establish original rights.

Requested/reference citation: Yao et al., Human Action Recognition by Learning Bases of Action Attributes and Parts, ICCV 2011.

Mirrors: https://huggingface.co/datasets/zrchen03/Stanford40_Dataset

### MIVIA Illegal Waste Dumping Dataset

No explicit data license located in the author repository. The Drive host does not determine licensing.

Historical unavailable files: 2; original public links remain recorded in the JSON registry.

### Street Scene (selected frames and track annotations)

Author Zenodo release lists CC BY-SA 4.0.

Requested/reference citation: Ramachandra and Jones, Street Scene: A New Dataset and Evaluation Protocol for Video Anomaly Detection, WACV 2020.

### Smoker Detection image classification dataset

Kaggle API licenseName verified and saved in dataset_evidence/kaggle_terms_20261020.json; uploader declaration.

Revision: datasetVersionNumber=5

### RS-WACV24 thermal loitering trajectories

MIT is the repository license. Separate original sensor-data rights are not documented; this entry covers the published CSV repository files, not arbitrary raw thermal videos.

Revision: 64e6ffdda4dad87b8d13daf69e8b2b9655a6459c

### Wikimedia Commons / original Flickr photographs

Each file has its own original author, source page, license URL, usage terms and restrictions in its manifest and docs/commons_attribution.csv. No blanket license applies. Candidate collection is larger than the reviewed training subset.

### HMDB51 original action archive

Original Brown download URL retained. No explicit current data license verified; underlying movie/Internet sources may retain rights. Do not infer a license from third-party mirrors.

Instructions: Extract the selected classes with scripts/datasets/extract_hmdb_sources.py after acquisition.

### MEVA abandon-package video subset

Official MEVA page states all data are CC BY 4.0. Cite the MEVA dataset paper; source video and annotation provenance retained.

Requested/reference citation: Corona et al., MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection, WACV 2021.

Annotation repository: https://gitlab.kitware.com/meva/meva-data-repo

Annotation revision: 421841a75577b697c314e952e585aecbb1b99e17

Annotation snapshot: dataset_evidence/meva_annotations.json

### AVSS 2007 abandoned-baggage pilot (AODsurvey bundle)

Redistribution bundle contains AVSS footage and separate code. No explicit footage license verified. A code license would not establish video licensing.

### Author small-item video component

No explicit dataset license verified in author release. Three highway/debris clips serve as a component probe. Bundled ABODA aliases are not independent data.

Revision: 966439fcac1103b392c49d4135f128860a39d1f8

### XRF55 RGB Public (selected scene/subject pilot)

The public release README specifies CC BY-NC 4.0. Kaggle says Other (specified in description). Repository code licensing is separate.

Revision: datasetVersionNumber=6

Author repository: https://github.com/airslab2020/XRF55-repo

License snapshot: dataset_evidence/xrf55_public_release_README.md

### D2-CDNet foreground segmentation (metadata only)

The mirror README declares Apache-2.0. The repack contains CDnet-derived material; original CDnet terms are separate and not established by the mirror declaration.

Revision: 298666656aa857471fc6668b03f0225e3c3d9eef

Instructions: anomaly-eval fetch --sources d2_cdnet --d2-shards intermittentObjectMotion

### Original shared Drive corpus

An aggregation, not a single licensed dataset. Preserve component export notices. Some subsets are synthetic, object-only or lack reliable event labels.

Instructions: anomaly-eval fetch-drive --output DATA_ROOT/raw --extract-zip --continue-on-error --anonymous-listing --workers 8 --name-conflicts rename

- [ATMGUARD-Loitering-Tracking v2](https://universe.roboflow.com/s-workspace-1cynf/atmguard-loitering-tracking): CC-BY-4.0 (bundled README.dataset.txt).  Evidence: data/raw/datasets/loitering/Roboflow Dataset (New Workspace)/README.dataset.txt
- [Farjana Hossain Loitering Detection v1](https://universe.roboflow.com/farjana-hossain-d6hl9/loitering-detection-6ir62): CC-BY-4.0 (bundled README.dataset.txt).  Evidence: data/raw/datasets/loitering/Roboflow Dataset (Farjana Hossain)/README.dataset.txt
- [Simuletic synthetic fall/pose sample](https://simuletic.com/datasets): CC-BY-4.0 (bundled Hugging Face README). Synthetic falls/lying are not verified real rough sleeping. Evidence: data/raw/datasets/rough sleeping/hugging_face/README.md
- Littering YOLOv12 export, smoking-person-1 and smoking CCTV COCO exports: Not specified in bundled export notices. Exact upstream project URLs not present in the export README; aggregation URL remains available. Evidence: export notices

### roboflow suspicious

Read the selected project/version license before redistribution. API access credentials are not license evidence.

Instructions: anomaly-eval fetch --sources roboflow_suspicious --roboflow-version roboflow_suspicious=VERSION

### roboflow objects

Read the selected project/version license before redistribution. API access credentials are not license evidence.

Instructions: anomaly-eval fetch --sources roboflow_objects --roboflow-version roboflow_objects=VERSION

## Provenance and updating

SHA-256 recorded from acquired local bytes or historical receipts; not publisher-authenticated unless explicitly noted.

Historical date-specific fetch scripts and scientific receipts are preserved. The unified downloader writes new receipts under `DATA_ROOT/.fetch_state/` and keeps the previous `.receipt.json` / `.download.json` files. XRF55 release terms, Kaggle API license metadata, D2-CDNet mirror terms and original MEVA activity annotations are retained under `anomaly_eval/config/dataset_evidence/`.

After reviewing edits to the catalog/manifests, regenerate these tables with `python scripts/datasets/export_dataset_sources.py`. Use `--update-manifest-hashes` only when intentionally publishing a revised acquisition plan. This does not revise historical experiment splits or results.

# XRF55 RGB Public

This dataset is one of five Kaggle parts containing the processed Azure Kinect data associated with **XRF55**. This part contains **Public RGB clips from all four scenes (27 consenting subjects).** It has 27 ZIP archives totaling approximately 40.38 GiB.

For the project description, paper, hardware setup, action definitions, and other sensing modalities, visit the official [XRF55 dataset homepage](https://aiotgroup.github.io/XRF55/).

## All five Kinect parts

- [XRF55 RGB Public](https://www.kaggle.com/datasets/airslab2020/xrf55-rgb-depth-ir)
- [XRF55 Depth Scene 1](https://www.kaggle.com/datasets/airslab2020/xrf55-depth-scene1)
- [XRF55 Depth Scenes 2-4](https://www.kaggle.com/datasets/airslab2020/xrf55-depth-scenes2-4)
- [XRF55 IR Scene 1 Part 1](https://www.kaggle.com/datasets/airslab2020/xrf55-ir-scene1-part1)
- [XRF55 IR Scene 1 Part 2 and Scenes 2-4](https://www.kaggle.com/datasets/airslab2020/xrf55-ir-scene1-part2-scenes2-4)

## Packaging and directory structure

Each archive contains one modality/scene/subject group and is named `MODALITY__SCENE__SUBJECT.zip`. The ZIPs use ZIP64 with stored entries because the MKV clips are already compressed. Extract all required parts into the same destination to reconstruct the original tree; archive contents do not overlap.

```text
rgb_public|depth|ir/
  sceneN/
    SS/
      SS_AA/
        SS_AA_RR.mkv
```

`SS` is the two-digit subject ID, `AA` is the action ID (`01`-`55`), and `RR` is the repetition (`01`-`20`). For example, `01_06_14.mkv` means subject 01, action 06, repetition 14. Repetitions 01-14 are the official training split and 15-20 are the test split.

## Video processing

- Every MKV contains one 5-second action window sampled at 15 fps (75 frames).
- RGB uses H.265/HEVC.
- Depth and infrared use lossless FFV1 level 3 with 16-bit grayscale samples.
- Rest intervals from the approximately 129-second source recordings were removed.
- The same relative path identifies synchronized modalities whenever public RGB is available.
- Private facial RGB from non-consenting subjects is excluded; their depth and infrared data are included under the release policy.

XRF55 has 55 action classes performed 20 times by 39 subjects across four scenes. Scene 1 has 30 subjects, while scenes 2, 3, and 4 have 3 subjects each.

## Archives in this part

- `rgb_public__scene1__01.zip`
- `rgb_public__scene1__03.zip`
- `rgb_public__scene1__04.zip`
- `rgb_public__scene1__05.zip`
- `rgb_public__scene1__06.zip`
- `rgb_public__scene1__07.zip`
- `rgb_public__scene1__08.zip`
- `rgb_public__scene1__09.zip`
- `rgb_public__scene1__10.zip`
- `rgb_public__scene1__11.zip`
- `rgb_public__scene1__13.zip`
- `rgb_public__scene1__14.zip`
- `rgb_public__scene1__19.zip`
- `rgb_public__scene1__21.zip`
- `rgb_public__scene1__22.zip`
- `rgb_public__scene1__24.zip`
- `rgb_public__scene1__26.zip`
- `rgb_public__scene1__27.zip`
- `rgb_public__scene1__28.zip`
- `rgb_public__scene2__05.zip`
- `rgb_public__scene2__24.zip`
- `rgb_public__scene2__31.zip`
- `rgb_public__scene3__06.zip`
- `rgb_public__scene3__07.zip`
- `rgb_public__scene4__03.zip`
- `rgb_public__scene4__04.zip`
- `rgb_public__scene4__13.zip`

## Citation

```bibtex
@article{wang2024xrf55,
  title = {XRF55: A Radio Frequency Dataset for Human Indoor Action Analysis},
  author = {Wang, Fei and Lv, Yizhe and Zhu, Mengdie and Ding, Han and Han, Jinsong},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume = {8}, issue = {1}, year = {2024}, publisher = {ACM}
}
```

License: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Consult the [official XRF55 homepage](https://aiotgroup.github.io/XRF55/) for current usage information.

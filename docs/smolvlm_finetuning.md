# SmolVLM adaptation for the five target definitions

The 19 October 2026 experiment adapts the pinned
`HuggingFaceTB/SmolVLM-500M-Instruct` model to smoking, littering, loitering,
rough sleeping, and unattended items. It extends the prior CLIP experiments;
the historical binary smoking VadCLIP adapter is unchanged.

## Method and interpretation

The source-separated manifest combines the 16 October strategy manifest and
its historical holdouts: 3,973 training, 1,812 development, 2,025 test, and 806
external examples. All 13,596 distinct image contents are fingerprinted. Old
holdouts have been inspected in earlier studies, so this is retrospective
validation. No test labels enter training, development selection or calibration.

Each example supervises one known binary objective, using its existing
definition and annotation. Other objectives are unlabelled, not negatives.
Inputs use every existing ordered view: one for static images, four for
loitering and unattended-item windows, eight for littering clips. Littering is
an offline clip task here; its label does not assert that every frame shows a
discarding action. Four sparse past images do not establish identity,
continuous dwell time or item ownership reliably.

Images retain aspect ratio through centered gray letterboxing before the
publisher's 512-pixel processor. Image splitting is disabled. The frozen image
encoder and connector produce 64 visual tokens per image, cached in float32.
Both original and adapted models receive identical tokens and prompts.

LoRA changes only the language model's q/k/v/o and gate/up/down projections:
rank 8, alpha 16, dropout 0.05. Vision, connector, embeddings and the vocabulary
output head stay frozen. This is actual model weight adaptation, not a linear
probe, although it does **not** adapt the vision tower. The first response token
(` Yes` or ` No`) has full-vocabulary cross entropy; all input positions are
excluded from the loss. Training uses bfloat16 and AdamW with weight decay 0.01,
5% warmup, linear decay and gradient clipping at 1.0. Each epoch visits every
training example once and adds an equally sized balanced resampled pass.
Sampling balances task/class, then dataset source, then recording/author group.

The primary seed 42 compares learning rates 0.00003 and 0.0001, maximum six
epochs, patience two, microbatch eight and effective batch 32. Selection uses
float32 development mean-source AP within each task and equal weighting of the
five tasks. The best joint adapter is reported separately from per-task routes
chosen between original and best-per-learning-rate adapters. Seed 43 can repeat
the primary winning learning rate for a limited stability check. Held-out
metrics must never choose which seed, route or checkpoint is deployed.

The score is the first-response `Yes` minus `No` logit. This is a
task-conditioned binary detector, not the earlier six-class JSON generation
protocol. Its AP should be compared to the **matched original model** here;
do not attribute the combined prompt/preprocessing change versus old results
entirely to fine-tuning. The sum of the two token probabilities is also saved,
because the binary normalization alone can hide a model preferring some other
text. There is no learned unknown class, calibrated confidence, explanation
training, bounding-box detection or continuous event evaluation in this run.

Raw margin zero, development maximum-F1 and development pooled FPR at most 5%
thresholds are frozen before held-out inference. A development FPR constraint
does not guarantee a held-out or deployment false-alert rate. Source-level
recall and FPR are reported alongside AP. Rough-sleeping photos and provisional
duration/ownership annotations retain all qualifications from the earlier work.

## Reproduce

Use the existing isolated CUDA environment. Additional dependencies are small
and installed separately, without upgrading its PyTorch or Transformers:

```bash
.venv/bin/python -m pip install --target data/tools/smol_training_deps \
  --no-deps -r requirements/smol-training.txt
bash scripts/smolvlm_gpu.sh train --stage all
```

The default output is `results/smolvlm_finetuning_20261019`. It contains the
immutable manifest/protocol, frozen visual cache, original development scores,
per-learning-rate adapter checkpoints, histories, frozen selections and
held-out predictions. Reusing complete phases is allowed; incomplete training
attempts must be preserved under a new name before retrying. There is no
training resume checkpoint for the optimizer, so an interrupted candidate must
be restarted in a separately recorded attempt. No GPU pacing or latency test
is used.

Additional transfer checks reuse the previously screened HMDB (906 clips),
new-author apparent public-sleep (28 photos) and MEVA abandonment-action
(27 windows) cohorts. Their labels and images never enter SmolVLM training or
selection. A separate frozen manifest checks source and pixel overlap against
the main experiment, and records current decoded image hashes. Evaluate the
frozen primary selection with:

```bash
bash scripts/smolvlm_gpu.sh transfer
```

HMDB primary-action controls are non-exhaustive negatives; eight-view clip
inference differs from the single-image smoking training inputs. MEVA action
windows do not establish persistent unattended-item status. These are source
transfer diagnostics, with the same qualifications as the earlier study.

For an independent seed, first read the primary `selection.json` and repeat
only its winning joint learning rate:

```bash
bash scripts/smolvlm_gpu.sh train --stage train --seed 43 --learning-rates 0.0001
bash scripts/smolvlm_gpu.sh train --stage evaluate --seed 43
```

The learning rate above is an example; use the actual primary selection.
For real inputs, use the selected target route:

```bash
bash scripts/smolvlm_gpu.sh predict --task smoking --images /path/frame.jpg
bash scripts/smolvlm_gpu.sh predict --task unattended_items \
  --images /path/t0.jpg /path/t7.jpg /path/t13.jpg /path/t20.jpg \
  --timestamps 0 7 13 20
bash scripts/smolvlm_gpu.sh export
bash scripts/smolvlm_gpu.sh verify
bash scripts/smolvlm_gpu.sh report
```

Static targets use the final image. Duration/ownership targets return unknown
when the required timed context is missing. This input guard does not replace
tracking or prove the temporal definition was satisfied. `target_absent` only
refers to the queried objective, not all five objectives.
Use `--original` for the matched original baseline or `--joint` for the selected
single adapter across all five tasks. The default uses per-target development
routes. These are research inference configurations; inspect the held-out
ranking and threshold-transfer failures before using them for alerts.

## Optional routing that requires development gains in both seeds

After the primary held-out results were inspected and the first independent-seed
development epoch was available, a separate exploratory rule was recorded. It
retains a primary target adapter only when its development source AP and the
second seed's joint-adapter development source AP both exceed the original.
Otherwise it uses the original model and the original development thresholds.
The primary selection and tables remain unchanged. The secondary seed repeats
only the primary winning joint learning rate, so this checks the adaptation
family, not a second run of every per-target learning-rate choice.

```bash
bash scripts/smolvlm_gpu.sh repeated-gains
bash scripts/smolvlm_gpu.sh predict --repeated-gains --task smoking --images /path/frame.jpg
```

The rule reads development predictions only. Because it was proposed after
inspecting primary test results, its benefits remain retrospective. It is a
guard against an unreplicated optimization gain, not independent evidence of
camera generalization or calibrated alert rates. The protocol and resulting
routes are stored separately from the frozen primary selection.

## Primary implementation references

- [SmolVLM model card and publisher fine-tuning tutorial](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
- [PEFT 0.17 LoRA reference](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora)
- [Pinned Transformers 4.57.1 Idefics3 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/idefics3/modeling_idefics3.py)

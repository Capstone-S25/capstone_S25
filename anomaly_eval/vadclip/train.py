"""Train the VadCLIP adapter with frame-level binary supervision (the documented deviation).

Run from the project root:

    python -m anomaly_eval.vadclip.train \
        --cache-dir results/vadclip/features \
        --weights data/models/openai_clip_vitb16.pt \
        --out-dir  results/vadclip/runs/full

Why frame-level (not the paper's video-MIL): every clip in this corpus contains at least one
positive frame, so a clip-level weak label is all-ones and the MIL objective degenerates. The
meaningful unit is the frame -- exactly what VadCLIP's headline frame average precision
measures -- so each representative frame is supervised with a soft target equal to the positive
fraction of its extraction block (exact 0/1 for sessions <= visual_length frames).

Both detection branches are trained:
    * visual   : BCE-with-logits on ``logits1`` (the per-frame anomaly classifier),
    * alignment: soft-target cross-entropy over the author-scaled cosine logits (/0.07),
      computed with log_softmax for numerical stability; no probability clamping is needed.

Ablation knobs: --visual-length, --attn-window, --temporal-layers, --no-graph. The best adapter
(state dict + config + valid metrics) is written to ``--out-dir``. No image content is displayed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from anomaly_eval.data import file_sha256
from anomaly_eval.metrics import frame_metrics
from anomaly_eval.paths import resolve_arguments, resolve_path
from anomaly_eval.runtime import add_device_argument, resolve_device

from .clip_b16 import load_clip_b16
from .data import PROMPTS, VadClipDataset, build_clip, collate, expand_scores
from .model import CLIPVAD


def load_clips(cache_dir: Path, visual_length=256):
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("schema") != 2 or not all(
        manifest.get(split) for split in ("train", "dev", "test")
    ):
        raise ValueError(
            "Re-extract features: schema 2 with separate train/dev/test groups is required"
        )
    partitions = [manifest[split] for split in ("train", "dev", "test")]
    flat = [key for keys in partitions for key in keys]
    if len(flat) != len(set(flat)) or set(flat) != set(manifest["sessions"]):
        raise ValueError("Feature cache has overlapping or missing split groups")
    items = {}
    for key, meta in manifest["sessions"].items():
        path = (cache_dir / meta["file"]).resolve()
        if (
            not path.is_relative_to(cache_dir.resolve())
            or meta["split"] not in ("train", "dev", "test")
            or key not in manifest[meta["split"]]
        ):
            raise ValueError("Invalid cache path or source split")
        if file_sha256(path) != meta["sha256"]:
            raise ValueError(f"Feature cache content changed: {key}")
        with np.load(path, allow_pickle=False) as d:
            feat, labels = d["feat"], d["labels"]
        if (
            not np.isfinite(feat).all()
            or feat.ndim != 2
            or len(labels) != meta.get("n_frames", len(labels))
        ):
            raise ValueError("Invalid feature tensor or original frame alignment")
        cf, cl, length = build_clip(feat, labels, visual_length)
        items[key] = {
            "feat": cf,
            "label": cl,
            "length": length,
            "raw_label": labels,
            "raw_feat": feat,
            "split": meta["split"],
            "key": key,
        }
    return items, manifest


def snapshot_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def alignment_loss(logits, labels):
    """Stable soft-label CE using the author's alignment temperature."""
    log_prob = F.log_softmax(logits, dim=-1)
    return -((1 - labels) * log_prob[:, :, 0] + labels * log_prob[:, :, 1])


def frame_scores(model, feats, lengths, device):
    """Return (p_visual [B,T], p_align [B,T]) for a batch of clips."""
    _, logits1, logits2, _ = model(feats.to(device), None, PROMPTS, lengths.to(device))
    p_vis = torch.sigmoid(logits1.squeeze(-1))  # [B, T]
    p_align = F.softmax(logits2.double(), dim=-1)[:, :, 1]  # [B, T] anomaly class prob
    return p_vis.cpu(), p_align.cpu()


def evaluate_split(model, items: list[dict], device):
    """Aggregate frame metrics over a list of clips for each scoring variant."""
    all_true, all_vis, all_al = [], [], []
    model.eval()
    with torch.no_grad():
        for it in items:
            feats = torch.from_numpy(it["feat"]).unsqueeze(0)  # [1, L, W]
            lengths = torch.tensor([it["length"]])
            p_vis, p_al = frame_scores(model, feats, lengths, device)
            L = it["length"]
            all_true.append(it["raw_label"])
            all_vis.append(expand_scores(p_vis[0, :L].numpy(), len(it["raw_label"])))
            all_al.append(expand_scores(p_al[0, :L].numpy(), len(it["raw_label"])))
    y = np.concatenate(all_true)
    vis = np.concatenate(all_vis)
    al = np.concatenate(all_al)
    return {
        "visual": frame_metrics(y, vis),
        "align": frame_metrics(y, al),
        "dual_mean": frame_metrics(y, 0.5 * (vis + al)),
        "dual_max": frame_metrics(y, np.maximum(vis, al)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--visual-length", type=int, default=256)
    ap.add_argument("--attn-window", type=int, default=8)
    ap.add_argument("--temporal-layers", type=int, default=2)
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="ablation: temporal-only (drop GCN branches)",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--threads", type=int, default=4)
    add_device_argument(ap)
    ap.add_argument(
        "--patience",
        type=int,
        default=50,
        help="early-stop patience on valid dual AUPR",
    )
    args = resolve_arguments(ap.parse_args(argv))
    for field in ("cache_dir", "weights", "out_dir"):
        setattr(args, field, str(resolve_path(getattr(args, field))))

    if (
        min(
            args.epochs,
            args.batch_size,
            args.patience,
            args.threads,
            args.visual_length,
            args.attn_window,
            args.temporal_layers,
        )
        < 1
    ):
        ap.error(
            "Epochs, batch size, patience, threads and architecture dimensions must be positive"
        )
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = resolve_device(args.device)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading frozen ViT-B/16 from {args.weights}", flush=True)
    clip_model = load_clip_b16(args.weights, device).eval()
    model = CLIPVAD(
        num_class=2,
        clip_model=clip_model,
        device=device,
        visual_length=args.visual_length,
        attn_window=args.attn_window,
        visual_layers=args.temporal_layers,
        use_graph=not args.no_graph,
    ).to(device)

    items_by_key, manifest = load_clips(Path(args.cache_dir), args.visual_length)
    if manifest["weights_sha256"] != file_sha256(Path(args.weights)):
        raise ValueError("Cached feature encoder and training weights differ")
    train_items = [
        items_by_key[k]
        for k in sorted(items_by_key)
        if items_by_key[k]["split"] == "train"
    ]
    valid_items = [
        items_by_key[k]
        for k in sorted(items_by_key)
        if items_by_key[k]["split"] == "dev"
    ]
    n_train_frames = sum(len(it["raw_label"]) for it in train_items)
    n_valid_frames = sum(len(it["raw_label"]) for it in valid_items)
    print(
        f"[train] {len(train_items)} train clips ({n_train_frames} frames) | "
        f"{len(valid_items)} valid clips ({n_valid_frames} frames)",
        flush=True,
    )

    loader = DataLoader(
        VadClipDataset(train_items),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(
        f"[train] trainable params: {n_params / 1e6:.2f}M | config: L={args.visual_length} "
        f"win={args.attn_window} layers={args.temporal_layers} graph={not args.no_graph}",
        flush=True,
    )

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    best_aupr, best_state, bad_epochs, best_epoch = float("-inf"), None, 0, None
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for feats, labels, lengths in loader:
            feats, labels, lengths = (
                feats.to(device),
                labels.to(device),
                lengths.to(device),
            )
            opt.zero_grad()
            _, logits1, logits2, _ = model(feats, None, PROMPTS, lengths)
            L = feats.shape[1]
            mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(
                1
            )  # [B, L] valid frames

            vis_loss = F.binary_cross_entropy_with_logits(
                logits1.squeeze(-1), labels, reduction="none"
            )
            al_loss = alignment_loss(logits2, labels)
            loss = (vis_loss + al_loss)[mask].mean()
            loss.backward()
            opt.step()
            run_loss += loss.item()

        val = evaluate_split(model, valid_items, device)
        dual_aupr = val["dual_mean"]["aupr"]
        if dual_aupr is None:
            raise ValueError("Development set needs both positive and negative frames")
        history.append(
            {
                "epoch": epoch,
                "loss": run_loss / max(1, len(loader)),
                **{k: v["aupr"] for k, v in val.items()},
            }
        )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[train] ep {epoch:3d} loss={run_loss / max(1, len(loader)):.4f} | "
                f"val AUPR vis={val['visual']['aupr']:.3f} al={val['align']['aupr']:.3f} "
                f"dual={dual_aupr:.3f}",
                flush=True,
            )

        if dual_aupr > best_aupr + 1e-4:
            best_aupr, bad_epochs = dual_aupr, 0
            best_epoch = epoch
            best_state = snapshot_state(model)
            torch.save(
                {
                    "state": best_state,
                    "config": vars(args),
                    "best_epoch": epoch,
                    "cache_fingerprint": manifest["fingerprint"],
                    "weights_sha256": manifest["weights_sha256"],
                },
                out_dir / "checkpoint.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"[train] early stop at epoch {epoch} (no valid-AUPR gain for {args.patience})",
                    flush=True,
                )
                break

    # final evaluation with the best checkpoint
    model.load_state_dict(best_state)
    final = evaluate_split(model, valid_items, device)
    # Freeze operating points on development frames. Test evaluation never searches thresholds.
    thresholds = {name: metric["threshold"] for name, metric in final.items()}
    checkpoint = {
        "state": best_state,
        "config": vars(args),
        "best_epoch": best_epoch,
        "cache_fingerprint": manifest["fingerprint"],
        "weights_sha256": manifest["weights_sha256"],
        "thresholds": thresholds,
        "selected_branch": "dual_mean",
        "protocol": "offline_supervised_smoking",
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")
    result = {
        "config": vars(args),
        "n_train_clips": len(train_items),
        "n_valid_clips": len(valid_items),
        "n_train_frames": n_train_frames,
        "n_valid_frames": n_valid_frames,
        "trainable_params_M": round(n_params / 1e6, 3),
        "best_epoch": best_epoch,
        "best_dual_aupr": best_aupr,
        "valid_metrics": final,
        "history_tail": history[-20:],
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"[train] done in {time.time() - t0:.1f}s | best dual AUPR={best_aupr:.4f} -> {out_dir}",
        flush=True,
    )

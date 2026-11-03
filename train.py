"""Train the grader of the paper: NaFlexViT-B/16 (SigLIP 2 weights) with its attention-pooling head and a CORAL ordinal
head (or a 5-class cross-entropy head), all weights trained, in two stages:

  stage 1   12 epochs at 1,024 px, backbone learning rate 1e-5, one epoch of warmup
  stage 2   2 epochs at 1,536 px from the best stage-1 checkpoint, learning rate 3e-6, a quarter epoch of warmup
Both: AdamW (weight decay 0.05 on backbone weights, 0 on norms and biases; heads: learning rate 1e-3, decay 0.01),
cosine schedule, gradient clipping at 1.0, batch 32, bfloat16 autocast, gradient checkpointing. Each epoch uses every
training image with grade > 0 and as many grade-0 images, drawn anew for each epoch. The kept checkpoint of a stage has
the lowest loss on the full validation split. Augmentation (on the GPU): rotation 0-360 degrees, flips, scale
0.9-1.1, brightness, contrast and hue jitter.
Loss: CORAL (summed binary cross-entropy of the four thresholds) or cross-entropy, plus 0.1 x a VICReg-style spread
term on a 32-d projection of the pooled feature (it acts on the whole batch, so the batch size is part of the method).
After training, evaluate.py fits the grade thresholds on the validation split and writes them into the checkpoint.

    uv run python train.py --seed 0                 # CORAL; then evaluate
    uv run python train.py --seed 0 --head ce       # cross-entropy head
    uv run python train.py --seed 0 --stage 2       # stage 2 only (from checkpoints/<head>_seed<s>_stage1.pt)
    uv run python train.py --seed 0 --limit 256     # smoke test: 256 images, one epoch per stage, outputs/smoke/

Output: checkpoints/<head>_seed<s>.safetensors (final), checkpoints/<head>_seed<s>_stage1.pt,
        <out>/train/<head>_seed<s>/{config,log}_stage{1,2}.{json,csv}
Training data: the cached crops of prepare_data.py. Time: about 2 h (stage 1) + 1 h (stage 2) on one RTX PRO 6000.
"""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drxai import data
from drxai.model import BACKBONE, DEV, Grader, save
from evaluate import predict, run_one

STAGES = {1: dict(side=1024, epochs=12, lr=1e-5, warmup=1.0), 2: dict(side=1536, epochs=2, lr=3e-6, warmup=0.25)}
HEAD_LR, BATCH, SPREAD_WEIGHT = 1e-3, 32, 0.1


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def augment(x):
    """random rotation, flips and scale (one affine resampling), then brightness, contrast and hue (YIQ rotation);
    x: float [B, 3, H, W] in [0, 1] on the GPU"""
    B, dev = x.shape[0], x.device
    th = torch.rand(B, device=dev) * 2 * torch.pi
    s = 0.9 + 0.2 * torch.rand(B, device=dev)
    fx = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(th) / s, torch.sin(th) / s
    A = torch.stack([torch.stack([cos * fx, -sin * fy, torch.zeros_like(s)], 1),
                     torch.stack([sin * fx, cos * fy, torch.zeros_like(s)], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(A, x.shape, align_corners=False), mode="bilinear", padding_mode="zeros",
                      align_corners=False)
    b = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=dev)
    c = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=dev)
    mean = x.mean((1, 2, 3), keepdim=True)
    x = ((x - mean) * c + mean) * b
    h = (torch.rand(B, device=dev) - 0.5) * 0.1 * 2 * torch.pi
    T = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
    R = torch.zeros(B, 3, 3, device=dev)
    R[:, 0, 0] = 1
    R[:, 1, 1], R[:, 1, 2], R[:, 2, 1], R[:, 2, 2] = torch.cos(h), -torch.sin(h), torch.sin(h), torch.cos(h)
    x = torch.einsum("bij,bjhw->bihw", torch.linalg.inv(T) @ R @ T, x)
    return x.clamp(0, 1)


def spread(z):
    """VICReg-style anti-collapse term on a batch z [B, d], about [0, 1]: a hinge that keeps the spread of every
    dimension at least the mean spread, plus the mean squared correlation between dimensions"""
    std = z.std(0) + 1e-4
    var = F.relu(1 - std / std.mean().detach()).mean()
    zc = (z - z.mean(0)) / std
    corr = zc.T @ zc / (len(z) - 1)
    off = corr - torch.diag(torch.diag(corr))
    return var + off.pow(2).sum() / (z.shape[1] * (z.shape[1] - 1))


def grade_loss(logits, y):
    """per image: summed threshold BCE (CORAL, 4 logits) or cross-entropy (5 logits)"""
    if logits.shape[1] == 5:
        return F.cross_entropy(logits.float(), y, reduction="none")
    targets = (y[:, None] >= torch.arange(1, 5, device=y.device)).float()
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(1)


def train_stage(stage, head, seed, rows, init, out_dir, limit):
    cfg = STAGES[stage]
    side, epochs = cfg["side"], 1 if limit else cfg["epochs"]
    seed_all(seed)
    keys = np.array([r["key"] for r in rows])
    y = np.array([r["grade"] for r in rows])
    split = np.array([r["split"] for r in rows])
    tr, va = np.where(split == "train")[0], np.where(split == "val")[0]
    if limit:
        tr, va = tr[:: len(tr) // limit][:limit], va[:: len(va) // max(1, limit // 4)]
    pos, neg = tr[y[tr] > 0], tr[y[tr] == 0]
    steps_per_epoch = 2 * len(pos) // BATCH

    model = Grader(head=head, pretrained=True, spread=True).to(DEV)
    if init is not None:
        model.load_state_dict(torch.load(init, map_location=DEV), strict=True)
    model.backbone.set_grad_checkpointing(True)
    bb = list(model.backbone.parameters())
    heads = list(model.head.parameters()) + list(model.spread_proj.parameters())
    opt = torch.optim.AdamW([dict(params=[p for p in bb if p.ndim > 1], lr=cfg["lr"], weight_decay=0.05),
                             dict(params=[p for p in bb if p.ndim <= 1], lr=cfg["lr"], weight_decay=0.0),
                             dict(params=heads, lr=HEAD_LR, weight_decay=1e-2)])
    total, warm = epochs * steps_per_epoch, max(1, int(cfg["warmup"] * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * min((s - warm) / max(1, total - warm), 1))))
    rng = np.random.default_rng(seed)
    y_gpu = torch.as_tensor(y, device=DEV)
    print(f"stage {stage} ({side} px), {head} seed {seed}: {len(pos)} grade > 0 + {len(pos)} of {len(neg)} grade-0 "
          f"images per epoch, {steps_per_epoch} steps per epoch, {len(va)} validation images", flush=True)

    best, best_state, log, t0 = np.inf, None, [], time.time()
    for ep in range(epochs):
        model.train()
        ep_rows = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
        rng.shuffle(ep_rows)
        tl = tp = nb = 0
        for x, idx in data.crop_loader(list(keys[ep_rows]), side, BATCH, workers=24):
            yb = y_gpu[torch.as_tensor(ep_rows[idx.numpy()], device=DEV)]
            with torch.autocast("cuda", torch.bfloat16):
                o = model(Grader.norm(augment(Grader.prep(x))))
            pen = spread(o["spread"].float())
            loss = grade_loss(o["main"], yb).mean() + SPREAD_WEIGHT * pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tl, tp, nb = tl + loss.item(), tp + pen.item(), nb + 1
        v = float(grade_loss(torch.from_numpy(predict(model, list(keys[va]), side, 32 if side <= 1024 else 8)),
                             torch.as_tensor(y[va])).mean())
        log.append(dict(epoch=ep + 1, train_loss=tl / nb, spread=tp / nb, val_loss=v, minutes=(time.time() - t0) / 60))
        print(f"  epoch {ep + 1}: train {tl / nb:.4f} (spread {tp / nb:.4f}) val {v:.4f} [{log[-1]['minutes']:.1f} min, "
              f"peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB]", flush=True)
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"log_stage{stage}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    config = dict(stage=stage, head=head, seed=seed, side=side, epochs=epochs, lr=cfg["lr"], head_lr=HEAD_LR,
                  warmup_epochs=cfg["warmup"], batch=BATCH, spread_weight=SPREAD_WEIGHT, limit=limit,
                  best_val_loss=best, best_epoch=int(np.argmin([r["val_loss"] for r in log]) + 1))
    json.dump(config, open(out_dir / f"config_stage{stage}.json", "w"), indent=2)
    model.load_state_dict(best_state)
    return model, config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", choices=["coral", "ce"], default="coral")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=None, help="run one stage only (default: both)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: this many training images, one epoch per stage")
    ap.add_argument("--out", default=None, help="output folder (default: outputs, or outputs/smoke with --limit)")
    args = ap.parse_args()
    out = Path(args.out or ("outputs/smoke" if args.limit else "outputs"))
    ck_dir = out / "checkpoints" if args.limit else data.REPO / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.head}_seed{args.seed}"
    rows = [r for r in data.grading_rows() if r["split"] in ("train", "val")]
    missing = [r["key"] for r in rows if not (data.CROPS / f"{r['key']}.jpg").exists()]
    if missing and not args.limit:
        raise SystemExit(f"{len(missing)} training crops are missing (e.g. {missing[0]}): run  uv run python prepare_data.py")
    if args.limit:
        rows = [r for r in rows if (data.CROPS / f"{r['key']}.jpg").exists()]
    stage1 = ck_dir / f"{name}_stage1.pt"
    configs = []
    if args.stage in (None, 1):
        model, cfg = train_stage(1, args.head, args.seed, rows, None, out / "train" / name, args.limit)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, stage1)
        configs.append(cfg)
        del model
        torch.cuda.empty_cache()
    if args.stage in (None, 2):
        if not stage1.exists():
            raise SystemExit(f"{stage1} is missing: run stage 1 first")
        model, cfg = train_stage(2, args.head, args.seed, rows, stage1, out / "train" / name, args.limit)
        if not configs and (out / "train" / name / "config_stage1.json").exists():
            configs.append(json.load(open(out / "train" / name / "config_stage1.json")))
        configs.append(cfg)
        final = ck_dir / f"{name}.safetensors"
        save(model, final, head=args.head, seed=args.seed, backbone=BACKBONE, train=configs)
        print(f"saved {final}", flush=True)
        del model
        torch.cuda.empty_cache()
        run_one(final, out, limit=args.limit, write_thresholds=True)


if __name__ == "__main__":
    main()

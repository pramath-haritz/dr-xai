"""Lesion localisation without pixel labels: average precision of each attribution map against the expert lesion masks
of MAPLES-DR and the DDR test set (Table 1, lesion AP; Table 2, AP with --head ce).

    uv run python lesion_ap.py run                          # CORAL, seeds 0-2, MAPLES-DR and DDR (reuses cached maps)
    uv run python lesion_ap.py run --head ce                # cross-entropy model, MAPLES-DR
    uv run python lesion_ap.py run --seeds 0 --limit 2      # smoke test, outputs/smoke/
    uv run python lesion_ap.py report                       # Table 1 (lesion AP) from outputs/lesion_ap/
    uv run python lesion_ap.py report --head ce             # Table 2 (AP of CORAL against CE)
    uv run python lesion_ap.py report --from paper_results  # the paper's numbers, no GPU needed

Each 96 x 96 map is upsampled (bilinear) to the 1,536-px crop of the image and scored, inside the fundus, against the
union of the expert masks (any lesion) and each lesion type (MA, HE, EX, SE):
  ap       average precision (exact: all fundus pixels sorted by the map)
  auc      pixel ROC AUC
  rec@5    fraction of lesions (8-connected components) whose maximum is in the top 5% of the fundus pixels
Means over images (with a lesion of that type) and seeds, 95% intervals of a hierarchical bootstrap; pull against each
other map: paired bootstrap, two-sided. The supervised segmentation models of Table 1 (HRDecoder, nnU-Net) are not
part of this repository; their numbers are those of the paper.
Output of run: <out>/lesion_ap/<head>_seed<s>.json ({dataset: {map: {image: {target: {ap, auc, rec@5}}}}}).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drxai import data, maps, stats
from drxai.model import DEV, load

AP_MAPS = {"coral": ("pull", "attention", "contribution", "attention_term", "rollout", "gradcam", "attnlrp", "occlusion"),
           "ce": ("attention", "pull")}
DATASETS = {"coral": ("maples", "ddr"), "ce": ("maples",)}
TARGETS = ("ANY", "MA", "HE", "EX", "SE")
LABEL = {"pull": "Pull", "attention": "Attention weights", "contribution": "Contribution", "attention_term": "Attention term",
         "rollout": "Rollout", "gradcam": "Grad-CAM", "attnlrp": "AttnLRP", "occlusion": "Occlusion"}


def per_image(score, t):
    """{target: {ap, auc, rec@5}} of one score map [H, W] (torch, on the GPU) against the targets t of one image"""
    fm = torch.from_numpy(t["fm"]).to(DEV)
    s = score[fm].float()
    order = torch.argsort(s, descending=True)
    n = len(s)
    thr5 = torch.topk(s, max(1, int(n * 5 / 100))).values[-1]
    out = {}
    for target, m in t["masks"].items():
        y = torch.from_numpy(m).to(DEV)[fm]
        P = int(y.sum())
        if P == 0:
            continue
        ys = y[order].float()
        prec = torch.cumsum(ys, 0) / torch.arange(1, n + 1, device=DEV)
        ranks = torch.empty(n, device=DEV)
        ranks[order] = torch.arange(n, 0, -1, device=DEV, dtype=torch.float32)
        lab, nl = t["labels"][target]
        peak = torch.full((nl + 1,), -float("inf"), device=DEV).scatter_reduce(0, torch.from_numpy(lab).to(DEV)[fm], s, "amax")[1:]
        out[target] = {"ap": float((prec * ys).sum() / P), "auc": float((ranks[y].sum() - P * (P + 1) / 2) / (P * (n - P))),
                       "rec@5": float((peak >= thr5).float().mean())}
    return out


def score_maps(mp, t):
    """per-image metrics of every map (dict name -> [GRID, GRID]) of one image"""
    res = {}
    for m, a in mp.items():
        up = F.interpolate(torch.from_numpy(np.asarray(a, np.float32)).to(DEV)[None, None], size=t["fm"].shape,
                           mode="bilinear", align_corners=False)[0, 0]
        res[m] = per_image(up, t)
    return res


# ----------------------------------------------------------------------------- run
def run(head, seed, out, limit, occlusion_batch):
    model, _ = load(data.REPO / "checkpoints" / f"{head}_seed{seed}.safetensors")
    res = {}
    for ds in DATASETS[head]:
        rows = data.lesion_rows(ds)[:limit or None]
        mp = maps.get(model, head, seed, ds, rows, maps.names_of(head), out, batch=occlusion_batch)
        t0 = time.time()
        for i, (name, sub) in enumerate(rows):
            t = data.load_lesion_targets(ds, name, sub)
            for m, r in score_maps({m: mp[m][name] for m in AP_MAPS[head]}, t).items():
                res.setdefault(ds, {}).setdefault(m, {})[name] = r
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                print(f"  {head} seed {seed} {ds}: {i + 1}/{len(rows)} scored [{(time.time() - t0) / 60:.1f} min]", flush=True)
    p = Path(out) / "lesion_ap" / f"{head}_seed{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p, "w"))
    print(f"saved {p} (peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB)", flush=True)


# ----------------------------------------------------------------------------- report
def load_runs(src, head):
    out = {}
    for s in (0, 1, 2):
        p = Path(src) / "lesion_ap" / f"{head}_seed{s}.json"
        if p.exists():
            out[s] = json.load(open(p))
    return out


def values(runs, ds, m, target, metric, names=None):
    """(images, V [seeds, images]): the images with the value in every seed (sorted, unless names gives the order)"""
    per = [runs[s][ds][m] for s in sorted(runs)]
    have = set.intersection(*[{n for n, r in p.items() if target in r} for p in per])
    names = sorted(have) if names is None else [n for n in names if n in have]
    return names, np.array([[p[n][target][metric] for n in names] for p in per])


def table1(src):
    runs = load_runs(src, "coral")
    if not runs:
        raise SystemExit(f"no CORAL results in {src}/lesion_ap: run  uv run python lesion_ap.py run")
    # bootstrap draws in the order of the paper: datasets, maps, targets, metrics, then the paired differences
    rng = np.random.default_rng(13)
    res = {}
    for ds in DATASETS["coral"]:
        for m in AP_MAPS["coral"]:
            for t in TARGETS:
                for k in ("ap", "auc", "rec@5"):
                    names, V = values(runs, ds, m, t, k)
                    if not names:
                        continue
                    lo, hi = stats.ci(stats.boot_mean(V, rng, 2000))
                    res.setdefault(ds, {}).setdefault(m, {})[f"{k}_{t}"] = dict(mean=float(V.mean()), lo=lo, hi=hi)
        for m in AP_MAPS["coral"][1:]:
            names, A = values(runs, ds, "pull", "ANY", "ap")
            names, B = values(runs, ds, m, "ANY", "ap", names)
            if not names:
                continue
            A = values(runs, ds, "pull", "ANY", "ap", names)[1]
            d = stats.boot_diff(A, B, rng, 5000)
            res[ds][f"pull_vs_{m}"] = dict(delta=float(A.mean() - B.mean()), ci=stats.ci(d), p=stats.p_two_sided(d),
                                           frac=float(np.mean(A.mean(0) > B.mean(0))))
    print(f"# Table 1, lesion AP (any lesion; {src}; seeds {sorted(runs)})\n")
    print("Mean over images with a lesion and over seeds [95% CI]. Supervised models (paper): HRDecoder 0.396 (MAPLES-DR), "
          "0.639 (DDR); nnU-Net 0.384, 0.582.\n")
    print("| Map | AP MAPLES-DR | AP DDR | AUC MAPLES-DR | AUC DDR |\n|---|---|---|---|---|")
    c = lambda ds, m, k: (f"{res[ds][m][k]['mean']:.3f} [{res[ds][m][k]['lo']:.3f}, {res[ds][m][k]['hi']:.3f}]"
                          if k in res[ds][m] else "–")
    for m in AP_MAPS["coral"]:
        print(f"| {LABEL[m]} | {c('maples', m, 'ap_ANY')} | {c('ddr', m, 'ap_ANY')} | {c('maples', m, 'auc_ANY')} | "
              f"{c('ddr', m, 'auc_ANY')} |")
    for ds, title in (("maples", "MAPLES-DR"), ("ddr", "DDR")):
        print(f"\n## {title}: per lesion type, and the pull against each map\n")
        print("| Map | AP MA | AP HE | AP EX | AP SE | recall@5% (any) | pull − map, AP [95% CI] | p |\n|---|---|---|---|---|---|---|---|")
        for m in AP_MAPS["coral"]:
            r = res[ds][m]
            pv = res[ds].get(f"pull_vs_{m}")
            diff = f"{pv['delta']:+.3f} [{pv['ci'][0]:+.3f}, {pv['ci'][1]:+.3f}] | {pv['p']:.4f}" if pv else "– | –"
            cell = lambda k: f"{r[k]['mean']:.3f}" if k in r else "–"
            print(f"| {LABEL[m]} | " + " | ".join(cell(f"ap_{t}") for t in TARGETS[1:]) + f" | {cell('rec@5_ANY')} | {diff} |")


def table2(src):
    rc, re = load_runs(src, "coral"), load_runs(src, "ce")
    if not rc or not re:
        raise SystemExit(f"Table 2 needs the CORAL and the CE results in {src}/lesion_ap")
    order = [n for n, _ in data.lesion_rows("maples")]                 # the images in the order of splits/lesion.csv
    rng = np.random.default_rng(14)
    print(f"# Table 2, lesion AP (MAPLES-DR, any lesion; {src})\n\n| Map | AP CORAL | AP CE | Δ [95% CI] | p |\n|---|---|---|---|---|")
    for m in AP_MAPS["ce"]:
        names, A = values(rc, "maples", m, "ANY", "ap", order)
        names, B = values(re, "maples", m, "ANY", "ap", names)
        A = values(rc, "maples", m, "ANY", "ap", names)[1]
        d = stats.boot_diff(A, B, rng, 5000)
        lo, hi = stats.ci(d)
        print(f"| {LABEL[m]} | {A.mean():.3f} | {B.mean():.3f} | {A.mean() - B.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | "
              f"{stats.p_two_sided(d):.4f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["run", "report"])
    ap.add_argument("--head", choices=["coral", "ce"], default="coral")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--from", dest="src", default="outputs", help="report: folder with the lesion_ap/ results")
    ap.add_argument("--out", default=None, help="run: output folder (default: outputs, or outputs/smoke with --limit)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: the first images of each dataset only")
    ap.add_argument("--occlusion-batch", type=int, default=32, help="occlusion batch (lower it on a small GPU)")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    data.set_data_root(args.data_root)
    if args.part == "report":
        return table2(args.src) if args.head == "ce" else table1(args.src)
    out = Path(args.out or ("outputs/smoke" if args.limit else "outputs"))
    for s in args.seeds:
        run(args.head, s, out, args.limit, args.occlusion_batch)


if __name__ == "__main__":
    main()

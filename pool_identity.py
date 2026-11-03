"""The decomposition at the pool (Section 2 of the paper): completeness (Eq. 2), the identity (Eq. 4) and whether each map
predicts the exact effect of removing patches from the pool (Fig. 2); and Fig. 1.

    uv run python pool_identity.py run                          # CORAL, seeds 0-2, MAPLES-DR (about 10 min per seed)
    uv run python pool_identity.py run --seeds 0 --limit 2      # smoke test, outputs/smoke/
    uv run python pool_identity.py report                       # numbers and figures from outputs/pool_identity/
    uv run python pool_identity.py report --from paper_results  # the paper's numbers and figures, no GPU needed

For each image and seed: the maps of the attention-pooling head (drxai.attribution), the largest violation of
completeness and of the identity, the head evidences E_h, and the alignment q. Removal from the pool: the 128 random
groups of 2% of the fundus patches of the causal tests (the same draws) are removed from the pool in all heads (attention
logits -inf, backbone fixed); Pearson r over the groups between the exact change of S and each map summed over the group
(pull, contribution, attention weights, attention term) and the change under Value Zeroing (the values of the group set
to zero, the attention held fixed, read out through the head); and r between Value Zeroing and the summed contribution
(the paper's r = 0.996: Value Zeroing follows the contribution, not the removal).
Fig. 1 needs one more run on its image (--example, seed 1): the grade score as the top-ranked patches of the pull and
of the contribution are deleted one by one (token dropping), and the union of the image's MAPLES-DR expert masks,
saved with the record (fig1_lesions.png). Its panels draw the image itself from the original Messidor (data/messidor):
the Messidor terms do not allow the image to be redistributed.
Output of run: <out>/pool_identity/coral_seed<s>.json, <out>/pool_identity/fig1_example.json, fig1_lesions.png;
report: <out>/figures/fig1.pdf, fig2.pdf (default outputs/figures).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from drxai import attribution, data, figures
from drxai.model import GRID, input_tensor, load
from drxai.removal import TokenDropper, draws, fundus_patches, grade, pearson, ranking

EXAMPLE, EXAMPLE_SEED, FIG1_STEPS = "20060412_59100_0200_PP", 1, 40
SUMMED = ("pull", "contribution", "attention", "attention_term")


def summed_maps(d):
    """the four maps of the head over all patches, float64 [N]"""
    a, E = d["a"], d["E"]
    return dict(pull=d["p"].sum(0), contribution=d["c"].sum(0), attention=a.mean(0), attention_term=(a * E[:, None]).sum(0))


def group_sums(m, groups):
    M = torch.zeros(len(groups), len(m), dtype=torch.float64, device=m.device)
    for j, g in enumerate(groups):
        M[j, torch.from_numpy(np.asarray(g, np.int64)).to(m.device)] = 1
    return (M @ m).cpu().numpy()


@torch.no_grad()
def run(seed, out, limit, batch, example):
    model, meta = load(data.REPO / "checkpoints" / f"coral_seed{seed}.safetensors")
    rows = data.lesion_rows("maples")[:limit or None]
    images, t0 = [], time.time()
    for i, (name, sub) in enumerate(rows):
        it = data.load_lesion_image("maples", name, sub)
        fund = fundus_patches(it["fundus"])
        rnd = draws(seed, i, fund)                                     # the draws of the causal tests
        groups = rnd["groups"]
        d = attribution.decompose(model, input_tensor(it["image"]))
        m32 = attribution.maps(d, GRID)
        dS, dS_vz = attribution.pool_removal(model, d, groups)
        sums = {k: group_sums(v, groups) for k, v in summed_maps(d).items()}
        r = {k: pearson(v, dS) for k, v in sums.items()} | dict(value_zeroing=pearson(dS_vz, dS),
                                                                value_zeroing_vs_contribution=pearson(dS_vz, sums["contribution"]))
        images.append(dict(name=name, S=d["S"], S_empty=d["S_empty"], q=attribution.alignment(m32, fund),
                           E=d["E"].tolist(), checks=attribution.checks(d), r=r,
                           groups=dict(dS=dS.tolist(), value_zeroing=dS_vz.tolist(), **{k: v.tolist() for k, v in sums.items()})))
        if name == example and seed == EXAMPLE_SEED:
            fig1_example(model, meta, it, fund, rnd["tie"], m32, seed, out, batch)
        if i % 20 == 0 or limit:
            print(f"  seed {seed}: {i + 1}/{len(rows)} [{(time.time() - t0) / 60:.1f} min] r " +
                  ", ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
    p = Path(out) / "pool_identity" / f"coral_seed{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(seed=seed, images=images), open(p, "w"))
    print(f"saved {p} (peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB)", flush=True)


def fig1_example(model, meta, it, fund, tie, m32, seed, out, batch):
    """the record of Fig. 1: the maps of the image, and S after deleting the top k = 0..FIG1_STEPS patches of the pull
    and of the contribution (token dropping, ranking and tie order of the causal tests)"""
    dropper = TokenDropper(model, it["image"])
    curves = {}
    for m in ("pull", "contribution"):
        o = ranking(m32[m].reshape(-1), fund, tie)
        curves[m] = dropper.scores([o[:k] for k in range(FIG1_STEPS + 1)], batch).tolist()
    T = meta["thresholds"]
    ex = dict(name=it["name"], seed=seed, grade=int(grade(curves["pull"][0], T)), thresholds=T, n_fundus=int(len(fund)),
              fund=fund.tolist(), deletion=curves, **{m: m32[m].tolist() for m in ("attention", "contribution", "attention_term")})
    p = Path(out) / "pool_identity" / "fig1_example.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(ex, open(p, "w"))
    t = data.load_lesion_targets("maples", it["name"], dict(data.lesion_rows("maples"))[it["name"]])
    Image.fromarray(t["masks"]["ANY"]).save(p.parent / "fig1_lesions.png", optimize=True)
    print(f"  saved {p}, fig1_lesions.png", flush=True)


def fig1_image(src, name):
    """the 1,536-px crop of the Fig. 1 image (from Messidor in data/) and its lesion union (saved with the record, else
    from the MAPLES-DR masks in data/); None if the image or the masks are missing"""
    path = data.find_stem("messidor", name)
    if path is None:
        return None
    p = Path(src) / "pool_identity" / "fig1_lesions.png"
    if p.exists():
        return data.score_crop(path)[0], np.asarray(Image.open(p)) > 0
    if data.find_dir("maples-dr", "train/Microaneurysms") is None:
        return None
    t = data.load_lesion_targets("maples", name, dict(data.lesion_rows("maples"))[name])
    return t["image"], t["masks"]["ANY"]


# ----------------------------------------------------------------------------- report
def median_ci(pools, k, n_boot=2000):
    """median of r over images and seeds and its 95% interval (hierarchical bootstrap: seeds and images resampled)"""
    names = sorted(set.intersection(*[{im["name"] for im in ims} for ims in pools.values()]))
    X = np.array([[{im["name"]: im["r"][k] for im in ims}[n] for n in names] for _, ims in sorted(pools.items())], float)
    rng = np.random.default_rng(13)
    b = [np.nanmedian(X[rng.choice(len(X), len(X))][:, rng.choice(len(names), len(names))]) for _ in range(n_boot)]
    return float(np.nanmedian(X)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def report(src, fig_dir):
    pools = {}
    for s in (0, 1, 2):
        p = Path(src) / "pool_identity" / f"coral_seed{s}.json"
        if p.exists():
            pools[s] = json.load(open(p))["images"]
    if not pools:
        raise SystemExit(f"no results in {src}/pool_identity: run  uv run python pool_identity.py run")
    every = [im for ims in pools.values() for im in ims]
    r = lambda k, sel=lambda im: True: np.array([im["r"][k] for im in every if sel(im)])
    print(f"# Removal from the pool (MAPLES-DR; {src}; seeds {sorted(pools)}; {len(every)} image-seed pairs)\n")
    print("Pearson r over the 128 random groups between the exact change of S and the summed map (last row: between Value "
          "Zeroing and the contribution): median over images and seeds [95% CI], interquartile range (Fig. 2b), and the "
          "median of each seed.\n")
    print("| Map | median r [95% CI] | IQR | " + " | ".join(f"seed {s}" for s in pools) + " |\n|---|---|---|" + "---|" * len(pools))
    labels = dict(pull="Pull", contribution="Contribution", value_zeroing="Value Zeroing", attention="Attention weights",
                  attention_term="Attention term", value_zeroing_vs_contribution="Value Zeroing vs contribution")
    for k, lab in labels.items():
        m, lo, hi = median_ci(pools, k)
        print(f"| {lab} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {np.nanpercentile(r(k), 25):.3f} to {np.nanpercentile(r(k), 75):.3f} | "
              + " | ".join(f"{np.nanmedian([im['r'][k] for im in ims]):.3f}" for ims in pools.values()) + " |")
    neg, pos = r("contribution", lambda im: im["q"] < 0), r("contribution", lambda im: im["q"] >= 0)
    print(f"\nContribution by the sign of the alignment q: median r {np.nanmedian(neg):.2f} for q < 0 ({len(neg)} pairs), "
          f"{np.nanmedian(pos):.2f} for q >= 0 ({len(pos)} pairs)")
    mixed = {s: np.mean([(np.array(im["E"]) > 0).any() and (np.array(im["E"]) < 0).any() for im in ims])
             for s, ims in pools.items()}
    print("Head evidences E_h of mixed signs: " + ", ".join(f"{v:.0%} (seed {s})" for s, v in mixed.items()) + " of the images")
    print(f"Images with q < 0: " + ", ".join(f"{np.mean([im['q'] < 0 for im in ims]):.0%} (seed {s})" for s, ims in pools.items()))
    chk = {k: max(im["checks"][k] for im in every if k in im.get("checks", {})) for k in ("completeness", "identity")
           if any(k in im.get("checks", {}) for im in every)}
    print("Largest violation over all images and seeds: " + ", ".join(f"{k} {v:.1e}" for k, v in chk.items()))

    fig_dir.mkdir(parents=True, exist_ok=True)
    if 1 in pools:
        rc = np.array([im["r"]["contribution"] for im in pools[1]])
        pick = pools[1][int(np.argmin(np.abs(rc - np.median(rc))))]                # seed-1 image with the median r of c
        if "groups" in pick:
            rows = [(r"Pull $p_t$", figures.PULL, r("pull")),
                    (r"Contribution $c_t$, $q\geq0$", figures.CONTRIB, pos), (r"Contribution $c_t$, $q<0$", figures.CONTRIB, neg),
                    ("Value Zeroing", figures.GREY, r("value_zeroing")), (r"Attention weights $\bar a_t$", figures.GREY, r("attention"))]
            figures.fig2(pick["groups"], rows, fig_dir / "fig2.pdf")
            print(f"\nFig. 2: {fig_dir / 'fig2.pdf'} (panel a: {pick['name']}, seed 1)")
    p = Path(src) / "pool_identity" / "fig1_example.json"
    if p.exists():
        ex = json.load(open(p))
        im = fig1_image(src, ex["name"])
        if im is None:
            print(f"Fig. 1 needs the Messidor image {ex['name']} in data/messidor, and the MAPLES-DR masks "
                  f"where {p.parent / 'fig1_lesions.png'} is missing (see README); skipped")
            return
        share = figures.fig1(ex, *im, fig_dir / "fig1.pdf")
        print(f"Fig. 1: {fig_dir / 'fig1.pdf'} ({ex['name']}, seed {ex['seed']}, grade {ex['grade']}; the top 6 patches hold "
              f"{share:.0%} of the attention)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["run", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--from", dest="src", default="outputs", help="report: folder with the pool_identity/ results")
    ap.add_argument("--out", default=None, help="output folder (default: outputs, or outputs/smoke with --limit)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: the first images only")
    ap.add_argument("--example", default=EXAMPLE, help="the MAPLES-DR image of Fig. 1 (seed 1)")
    ap.add_argument("--batch", type=int, default=24, help="token-dropping batch of Fig. 1 (c)")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    data.set_data_root(args.data_root)
    if args.part == "report":
        return report(args.src, Path(args.out or "outputs") / "figures")
    out = Path(args.out or ("outputs/smoke" if args.limit else "outputs"))
    for s in args.seeds:
        run(s, out, args.limit, args.batch, args.example)


if __name__ == "__main__":
    main()

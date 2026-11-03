"""Causal tests of the attribution maps by token dropping on MAPLES-DR: Table 1 (N, Su, FC), Table 2 (--head ce) and the
cost columns of Table 1 (profile).

    uv run python causal_tests.py run                          # CORAL, seeds 0-2 (about 6 h per seed on one GPU, plus
                                                               # the occlusion maps, about 5 h per seed, cached)
    uv run python causal_tests.py run --head ce                # cross-entropy model (Table 2)
    uv run python causal_tests.py run --seeds 0 --limit 2      # smoke test: two images, outputs/smoke/
    uv run python causal_tests.py report                       # Table 1 from outputs/causal_tests/
    uv run python causal_tests.py report --head ce             # Table 2 (needs the CORAL and the CE results)
    uv run python causal_tests.py report --from paper_results  # the paper's numbers, no GPU needed
    uv run python causal_tests.py profile                      # time and FLOPs of each map per image

Maps (drxai.maps): attention weights, contribution and pull (ours), attention rollout, Grad-CAM, AttnLRP and occlusion
(CORAL only); controls: expert lesions and random orders. Tests (drxai.removal), per image and seed:
  N   necessity: the smallest fraction of the fundus patches, removed in the order of the map, that lowers the
      predicted grade g (1.0 if none); Su sufficiency: the smallest fraction that, inserted into the empty fundus,
      restores g; both on the grid FRACTIONS (0.25% to 50%) and for images that all three seeds grade >= 1
  FC  faithfulness correlation over 128 random groups of 2% of the fundus patches
Table 1 gives the geometric means of N and Su (in % of the fundus patches) and the mean FC over images and seeds, with
95% intervals of a hierarchical bootstrap (images and seeds, 5,000 resamples for tests, 2,000 for intervals); each of our
maps is compared with the four baselines on N, Su and FC, two-sided, Holm-corrected over its 12 comparisons.
Output of run: <out>/causal_tests/<head>_seed<s>.json (per image: grade, scores, and for each map the deletion and
insertion curves, necessity, sufficiency and FC; the alignment q of the image).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from drxai import attribution, data, maps, stats
from drxai.model import GRID, load
from drxai.removal import FRACTIONS, TokenDropper, causal_record, draws, fundus_patches, lesion_fraction, spearman

TEST_MAPS = {"coral": ("attention", "contribution", "pull", "rollout", "gradcam", "attnlrp", "occlusion"),
             "ce": ("attention", "contribution", "pull", "rollout", "gradcam")}
OURS, BASELINES, KEYS = ("attention", "contribution", "pull"), ("rollout", "gradcam", "attnlrp", "occlusion"), ("N", "Su", "FC")
LABEL = {"attention": "Attention weights", "contribution": "Contribution", "pull": "Pull", "rollout": "Rollout",
         "gradcam": "Grad-CAM", "attnlrp": "AttnLRP", "occlusion": "Occlusion", "lesion": "Expert lesion masks",
         "random": "Random order"}
N_TEST, N_CI = 5000, 2000


# ----------------------------------------------------------------------------- run
def run(head, seed, out, limit, batch, occlusion_batch):
    model, meta = load(data.REPO / "checkpoints" / f"{head}_seed{seed}.safetensors")
    T = np.array(meta["thresholds"])
    rows = data.lesion_rows("maples")[:limit or None]
    mp = maps.get(model, head, seed, "maples", rows, maps.names_of(head), out, batch=occlusion_batch)
    images, t0 = [], time.time()
    for i, (name, sub) in enumerate(rows):
        it = data.load_lesion_image("maples", name, sub)
        fund = fundus_patches(it["fundus"])
        flat = {m: mp[m][name].reshape(-1).astype(np.float64) for m in TEST_MAPS[head]}
        rec = causal_record(TokenDropper(model, it["image"]), flat, fund, draws(seed, i, fund), T,
                            lesions=lesion_fraction(it["lesions"]), batch=batch)
        q = attribution.alignment({k: mp[k][name] for k in ("attention", "attention_term")}, fund)
        images.append(dict(name=name, q=q, **rec))
        if i % 10 == 0 or limit:
            print(f"  {head} seed {seed}: {i + 1}/{len(rows)} [{(time.time() - t0) / 60:.1f} min] grade {rec['grade']}, "
                  f"FC " + ", ".join(f"{m} {rec['maps'][m]['fc']:.2f}" for m in TEST_MAPS[head]), flush=True)
    p = Path(out) / "causal_tests" / f"{head}_seed{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(head=head, seed=seed, thresholds=T.tolist(), score_sd=meta["score_sd"], fractions=FRACTIONS,
                   maps=list(TEST_MAPS[head]), images=images), open(p, "w"))
    print(f"saved {p} (peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB)", flush=True)


# ----------------------------------------------------------------------------- report
def load_runs(src, head):
    runs = {}
    for s in (0, 1, 2):
        p = Path(src) / "causal_tests" / f"{head}_seed{s}.json"
        if p.exists():
            runs[s] = json.load(open(p))
    return runs


def measures(run, m):
    """{image: per-image measures of map m}: N, Su (log10 fractions; images with a predicted grade >= 1), FC, the
    grade fall at each fraction, N and Su for each grade level, and the deletion and insertion areas (in SD of S)"""
    out, T, sd = {}, np.array(run["thresholds"]), run["score_sd"]
    for im in run["images"]:
        r = im["maps"].get(m)
        if r is None or "deletion" not in r:
            continue
        g = im["grade"]
        x = dict(FC=r["fc"], deletion_area=float(np.mean(np.array(r["deletion"]) - im["S0"]) / sd),
                 insertion_area=float(np.mean(np.array(r["insertion"]) - im["S_empty"]) / sd))
        if g >= 1:
            x["N"] = float(np.mean(np.log10(r["necessity"][str(g)])))
            x["fall"] = ((np.array(r["deletion"])[:, None] > T).sum(1) < g).astype(float)
            if str(g) in r["sufficiency"]:
                x["Su"] = float(np.mean(np.log10(r["sufficiency"][str(g)])))
            x["Nj"] = {int(j): float(np.mean(np.log10(v))) for j, v in r["necessity"].items()}
            x["Suj"] = {int(j): float(np.mean(np.log10(v))) for j, v in r["sufficiency"].items()}
        out[im["name"]] = x
    return out


def mat(runs, m, key, seeds=None):
    """(image names, V [seeds, images]) of one measure; the images that have it (finite) in every seed"""
    per = {s: measures(r, m) for s, r in runs.items() if seeds is None or s in seeds}
    per = {s: p for s, p in per.items() if p}
    if not per:
        return [], np.zeros((0, 0))
    names = sorted(set.intersection(*[{n for n, v in p.items() if key in v and np.all(np.isfinite(v[key]))}
                                      for p in per.values()]))
    return names, np.array([[per[s][n][key] for n in names] for s in sorted(per)])


def has(run, m):
    return any(m in im["maps"] for im in run["images"])


def paired(runs, a, b, key, rng):
    """map a against map b on one measure: images paired, seeds resampled; lower N, Su and higher FC are better"""
    seeds = [s for s in runs if has(runs[s], a) and has(runs[s], b)]
    na, Va = mat(runs, a, key, seeds)
    nb, Vb = mat(runs, b, key, seeds)
    common = sorted(set(na) & set(nb))
    Va, Vb = Va[:, [na.index(n) for n in common]], Vb[:, [nb.index(n) for n in common]]
    d = stats.boot_diff(Va, Vb, rng, N_TEST)
    lo, hi = stats.ci(d)
    better = (Va.mean(0) < Vb.mean(0)) if key != "FC" else (Va.mean(0) > Vb.mean(0))
    return dict(n=len(common), a=float(Va.mean()), b=float(Vb.mean()), delta=float(Va.mean() - Vb.mean()), lo=lo,
                hi=hi, p=stats.p_two_sided(d), frac_a_better=float(better.mean()))


def describe(runs, m, key, rng):
    _, V = mat(runs, m, key)
    if V.size == 0:
        return None
    lo, hi = stats.ci(stats.boot_mean(V, rng, N_CI))
    return dict(mean=float(V.mean()), lo=lo, hi=hi, n=V.shape[1], seeds=V.shape[0])


def is_better(t, key):
    return t["p_holm"] < 0.05 and (t["delta"] < 0 if key != "FC" else t["delta"] > 0)


def pct(v):
    """log10 fraction -> % of the fundus patches"""
    return 100 * 10 ** v


def table1(src):
    runs = load_runs(src, "coral")
    if not runs:
        raise SystemExit(f"no CORAL results in {src}/causal_tests: run  uv run python causal_tests.py run")
    # bootstrap draws, in the order of the paper: one generator for the tests of the contribution and the pull and for
    # the intervals, another for the tests of the attention weights
    rng = np.random.default_rng(0)
    fam = {m: stats.holm({f"{k} vs {b}": paired(runs, m, b, k, rng) for b in BASELINES for k in KEYS})
           for m in ("contribution", "pull")}
    ours = {f"{k}: pull vs {o}": paired(runs, "pull", o, k, rng) for o in ("contribution", "attention") for k in KEYS}
    shown = ("contribution", "pull", "rollout", "gradcam", "attnlrp", "occlusion", "attention", "lesion", "random")
    desc = {m: {k: describe(runs, m, k, rng) for k in ("N", "Su", "FC", "deletion_area", "insertion_area")} for m in shown}
    rng14 = np.random.default_rng(14)
    fam["attention"] = stats.holm({f"{k} vs {b}": paired(runs, "attention", b, k, rng14) for b in BASELINES for k in KEYS})

    seeds = sorted(runs)
    n_imgs = desc["pull"]["N"]["n"]
    print(f"# Table 1, causal tests (token dropping, MAPLES-DR; {src}; seeds {seeds})\n")
    print(f"N, Su: geometric mean over images and seeds, % of the fundus patches (lower is better; {n_imgs} images that "
          f"all seeds grade >= 1). FC: mean (higher is better). [95% CI]; †: better than all four baselines "
          f"(Holm, 12 comparisons per map).\n")
    print("| Map | N (%) | Su (%) | FC | grade falls at 1% |")
    print("|---|---|---|---|---|")
    fall = {}
    for m in ("attention", "contribution", "pull") + BASELINES + ("lesion", "random"):
        d = desc[m]
        _, Fm = mat(runs, m, "fall")
        fall[m] = float(Fm.mean((0, 1))[FRACTIONS.index(0.01)]) if Fm.size else float("nan")
        dag = {k: m in OURS and all(is_better(fam[m][f"{k} vs {b}"], k) for b in BASELINES) for k in KEYS}
        f = lambda k, v: f"{pct(v):.3g}" if k != "FC" else f"{v:.3f}"
        cell = lambda k: (f"{f(k, d[k]['mean'])}{'†' if dag[k] else ''} [{f(k, d[k]['lo'])}, {f(k, d[k]['hi'])}]"
                          if d[k] else "–")
        print(f"| {LABEL[m]} | {cell('N')} | {cell('Su')} | {cell('FC')} | {fall[m]:.0%} |")

    print("\n## Our maps against the four baselines (two-sided, Holm over the 12 comparisons of each map)\n")
    print("| Map | Comparison | n | Δ (log10 for N, Su) [95% CI] | p | p (Holm) |")
    print("|---|---|---|---|---|---|")
    for m in OURS:
        for h, t in fam[m].items():
            print(f"| {LABEL[m]} | {h} | {t['n']} | {t['delta']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}] | {t['p']:.4f} | "
                  f"{t['p_holm']:.4f}{' ✓' if is_better(t, h.split()[0]) else ''} |")
    print("\n## Numbers in the text\n")
    for m in OURS:
        wins = [t for h, t in fam[m].items() if is_better(t, h.split()[0])]
        sig = [t for t in fam[m].values() if t["p_holm"] < 0.05]
        print(f"- {LABEL[m]}: better than the baselines in {len(wins)} of 12 comparisons; {len(sig)} of 12 significant, "
              f"largest p_Holm among them {max([t['p_holm'] for t in sig], default=float('nan')):.3f}")
    print(f"- deleting the top 1% lowers the grade in {fall['attention']:.0%} of the images (attention weights), "
          f"{fall['pull']:.0%} (pull)")
    for o in ("attention", "contribution"):
        t = ours[f"FC: pull vs {o}"]
        print(f"- FC of the pull {t['a']:.3f}, above the {LABEL[o].lower()} by {t['delta']:.3f} [{t['lo']:.3f}, {t['hi']:.3f}]")
    lv = lambda m, key, j: pct(np.mean([x[key][j] for r in runs.values() for x in measures(r, m).values()
                                         if key in x and j in x[key]]))
    print("- sufficiency at grade levels 3 and 4: contribution {:.0f}% and {:.0f}%, pull {:.0f}% and {:.0f}%".format(
        lv("contribution", "Suj", 3), lv("contribution", "Suj", 4), lv("pull", "Suj", 3), lv("pull", "Suj", 4)))
    q, fc_c, fc_p = (np.array([im["q"] if k == "q" else im["maps"][k]["fc"] for r in runs.values() for im in r["images"]])
                     for k in ("q", "contribution", "pull"))
    ok = np.isfinite(q) & np.isfinite(fc_c) & np.isfinite(fc_p)
    print(f"- FC against the alignment q (Spearman over images and seeds): contribution {spearman(q[ok], fc_c[ok]):.3f}, "
          f"pull {spearman(q[ok], fc_p[ok]):.3f}")
    print(f"- deletion / insertion areas (mean change of S over the fractions, in SD of S): " + ", ".join(
        f"{LABEL[m].lower()} {desc[m]['deletion_area']['mean']:.2f} / {desc[m]['insertion_area']['mean']:.2f}"
        for m in ("attention", "contribution", "pull")))


def table2(src):
    rc, re = load_runs(src, "coral"), load_runs(src, "ce")
    if not rc or not re:
        raise SystemExit(f"Table 2 needs the CORAL and the CE results in {src}/causal_tests")

    def relative(runs, m, key):
        """N and Su relative to the random order of the same image (log10 ratio); FC as it is"""
        names, V = mat(runs, m, key)
        if key == "FC":
            return names, V
        nr, Vr = mat(runs, "random", key)
        common = sorted(set(names) & set(nr))
        return common, V[:, [names.index(n) for n in common]] - Vr[:, [nr.index(n) for n in common]]

    def compare(m, key, rng):
        na, Va = relative(rc, m, key)
        nb, Vb = relative(re, m, key)
        common = sorted(set(na) & set(nb))
        Va, Vb = Va[:, [na.index(n) for n in common]], Vb[:, [nb.index(n) for n in common]]
        d = stats.boot_diff(Va, Vb, rng, N_TEST)
        lo, hi = stats.ci(d)
        return dict(n=len(common), coral=float(Va.mean()), ce=float(Vb.mean()), delta=float(Va.mean() - Vb.mean()),
                    lo=lo, hi=hi, p=stats.p_two_sided(d))

    rng = np.random.default_rng(14)
    tests = stats.holm({(m, k): compare(m, k, rng) for m in ("pull", "attention") for k in KEYS})
    print(f"# Table 2, CORAL against cross-entropy (CE) head, causal tests (MAPLES-DR; {src})\n")
    print("Means over images and seeds (N, Su: geometric, % of the fundus patches). †: significant difference between "
          "the heads (N and Su relative to the random order of the image; bootstrap, Holm over the 6 tests).\n")
    print("| Map | Head | N (%) | Su (%) | FC |")
    print("|---|---|---|---|---|")
    for m in ("attention", "pull"):
        for head, runs in (("CORAL", rc), ("CE", re)):
            cells = []
            for k in KEYS:
                v = mat(runs, m, k)[1].mean()
                dag = "†" if head == "CORAL" and tests[(m, k)]["p_holm"] < 0.05 else ""
                cells.append(f"{pct(v):.3g}{dag}" if k != "FC" else f"{v:.3f}{dag}")
            print(f"| {LABEL[m]} | {head} | " + " | ".join(cells) + " |")
    print("\n| Test (CORAL − CE) | n | Δ [95% CI] | p | p (Holm) |\n|---|---|---|---|---|")
    for (m, k), t in tests.items():
        print(f"| {LABEL[m]}, {k}{' (vs random)' if k != 'FC' else ''} | {t['n']} | {t['delta']:+.3f} "
              f"[{t['lo']:+.3f}, {t['hi']:+.3f}] | {t['p']:.4f} | {t['p_holm']:.4f} |")
    print(f"\nNumbers in the text: the pull is as causal for either head (p_Holm >= "
          f"{min(t['p_holm'] for (m, _), t in tests.items() if m == 'pull'):.3f})")


# ----------------------------------------------------------------------------- profile
def profile(seed, n_images):
    """FLOPs and time per image of each map (FLOPs counted on the first image; time: mean over n_images after one
    warm-up run; occlusion: one image, counted and timed in the same run)"""
    from torch.utils.flop_counter import FlopCounterMode
    from drxai import baselines
    from drxai.model import input_tensor
    model, _ = load(data.REPO / "checkpoints" / f"coral_seed{seed}.safetensors")
    model.requires_grad_(False)
    items = [data.load_lesion_image("maples", *r) for r in data.lesion_rows("maples")[:n_images]]
    fns = {"forward pass (attention weights)": lambda x, it: attribution.pool_tokens(model, x),
           "contribution, pull and attention term": lambda x, it: attribution.maps(attribution.decompose(model, x), GRID),
           "attention rollout": lambda x, it: baselines.rollout(model, x),
           "Grad-CAM": lambda x, it: baselines.gradcam(model, x),
           "AttnLRP": lambda x, it: baselines.attnlrp(model, x),
           "occlusion": lambda x, it: baselines.occlusion(model, x, it["fundus"])}

    def timed(fn, it):
        x = input_tensor(it["image"])
        torch.cuda.synchronize()
        t0 = time.time()
        fn(x, it)
        torch.cuda.synchronize()
        return time.time() - t0

    print(f"# Cost per image at 1,536 px ({torch.cuda.get_device_name()})\n\n| Map | TFLOPs | Time (s) |\n|---|---|---|")
    for name, fn in fns.items():
        with FlopCounterMode(display=False) as fc:
            t = timed(fn, items[0])
        if name != "occlusion":
            t = float(np.mean([timed(fn, it) for it in items]))
        print(f"| {name} | {fc.get_total_flops() / 1e12:,.1f} | {t:.2f} |", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["run", "report", "profile"])
    ap.add_argument("--head", choices=["coral", "ce"], default="coral")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--from", dest="src", default="outputs", help="report: folder with the causal_tests/ results")
    ap.add_argument("--out", default=None, help="run: output folder (default: outputs, or outputs/smoke with --limit)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: the first images only")
    ap.add_argument("--batch", type=int, default=24, help="token-dropping batch (lower it on a small GPU)")
    ap.add_argument("--occlusion-batch", type=int, default=32, help="occlusion batch (lower it on a small GPU)")
    ap.add_argument("--images", type=int, default=3, help="profile: images to time")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    data.set_data_root(args.data_root)
    if args.part == "report":
        return table2(args.src) if args.head == "ce" else table1(args.src)
    if args.part == "profile":
        return profile(args.seeds[0], args.images)
    out = Path(args.out or ("outputs/smoke" if args.limit else "outputs"))
    for s in args.seeds:
        run(args.head, s, out, args.limit, args.batch, args.occlusion_batch)


if __name__ == "__main__":
    main()

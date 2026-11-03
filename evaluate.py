"""Grade thresholds and grading performance of the trained models.

Training samples grade 0 and grade > 0 equally, so the thresholds of the head are recalibrated on the validation split:
for each k = 1..4, a logistic regression of [grade >= k] on the k-th CORAL logit (cross-entropy head: on the expected
grade). Its decision boundary, written on the grade-score scale, is the grade threshold T_k; the predicted grade is the
number of thresholds that S exceeds. The thresholds and the validation SD of S are stored in the checkpoint.

    uv run python evaluate.py run                              # every checkpoint in checkpoints/ (about 10 min each)
    uv run python evaluate.py run --checkpoint checkpoints/coral_seed0.safetensors
    uv run python evaluate.py report                           # grading table from outputs/evaluate/
    uv run python evaluate.py report --from paper_results      # the paper's numbers, no GPU needed

run: predicts the validation and test splits and Messidor-2 at 1,536 px, writes <out>/evaluate/<head>_seed<s>.csv
(per image: the head outputs) and the thresholds (into the checkpoint if it has none; a released checkpoint keeps its
thresholds unless --overwrite-thresholds). Needs the crops of prepare_data.py for those splits.
report: quadratic-weighted kappa, accuracy and AUCs, mean +- SD over seeds (Section 3, "Model and training", and the
cross-entropy comparison). The grades are not in the CSVs: they are joined by key from the label files of the four
grading datasets (see prepare_data.py --check), so the report needs those files but no images.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from drxai import data
from drxai.model import SIDE, Grader, load, update_meta

SPLITS = ("val", "test", "messidor2")
HEADS = ("coral", "ce")


@torch.no_grad()
def predict(model, keys, side, batch):
    """head outputs (CORAL: 4 logits; cross-entropy: 5 logits) for the crops of keys, float32 [n, k]"""
    model.eval()
    out = None
    for x, idx in data.crop_loader(keys, side, batch):
        with torch.autocast("cuda", torch.bfloat16):
            o = model(Grader.norm(Grader.prep(x)))["main"].float().cpu().numpy()
        out = np.zeros((len(keys), o.shape[1]), np.float32) if out is None else out
        out[idx.numpy()] = o
    return out


def expected_grade(L):
    p = np.exp(L - L.max(1, keepdims=True))
    return (p / p.sum(1, keepdims=True)) @ np.arange(L.shape[1])


def scores(L):
    """[n, 4] per-threshold scores: the four CORAL logits, or the expected grade (repeated) of the CE head"""
    L = np.asarray(L, np.float64)
    return L if L.shape[1] == 4 else expected_grade(L)[:, None].repeat(4, 1)


def fit_thresholds(L_val, y_val):
    """thresholds on the per-threshold scores (logistic regression on the validation split)"""
    Z = scores(L_val)
    T = []
    for k in range(4):
        lr = LogisticRegression(C=1e6).fit(Z[:, k:k + 1], (y_val > k).astype(int))
        T.append(float(-lr.intercept_[0] / lr.coef_[0, 0]))
    return np.array(T)


def thresholds_on_S(L_val, y_val, bias=None):
    """(T_1..T_4 on the grade-score scale, validation SD of S); CORAL: logit_k = S + b_k"""
    T = fit_thresholds(L_val, y_val)
    if L_val.shape[1] == 4:
        return T - bias, float(np.std(np.asarray(L_val, np.float64)[:, 0]))
    return T, float(np.std(expected_grade(np.asarray(L_val, np.float64))))


# ----------------------------------------------------------------------------- run
def write_csv(path, rows, L):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "dataset", "split"] + [f"out{k}" for k in range(L.shape[1])])
        for r, l in zip(rows, L):
            w.writerow([r["key"], r["dataset"], r["split"]] + [f"{v:.9g}" for v in l])


def read_csv(path, grade):
    """the outputs of run, with each image's grade from grade {key: grade}"""
    rows = list(csv.DictReader(open(path)))
    cols = [c for c in rows[0] if c.startswith("out")]
    L = np.array([[r[c] for c in cols] for r in rows], np.float32)
    return dict(split=np.array([r["split"] for r in rows]), dataset=np.array([r["dataset"] for r in rows]),
                grade=np.array([grade[r["key"]] for r in rows]), L=L)


def run_one(path, out, limit=0, write_thresholds=False, batch=8):
    path = Path(path)
    model, meta = load(path, need_thresholds=False)
    rows = [r for r in data.grading_rows() if r["split"] in SPLITS]
    if limit:                                                       # evenly spaced images of each split
        rows = [r for r in rows if (data.CROPS / f"{r['key']}.jpg").exists()]
        per = {s: [q for q in rows if q["split"] == s] for s in SPLITS}
        rows = [r for s in SPLITS for r in per[s][:: max(1, len(per[s]) // limit)][:limit]]
    missing = [r["key"] for r in rows if not (data.CROPS / f"{r['key']}.jpg").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} crops are missing (e.g. {missing[0]}): run  uv run python prepare_data.py")
    print(f"evaluate {path.name}: {len(rows)} images at {SIDE} px", flush=True)
    L = predict(model, [r["key"] for r in rows], SIDE, batch)
    write_csv(Path(out) / "evaluate" / f"{path.stem}.csv", rows, L)
    print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
    va = np.array([r["split"] == "val" for r in rows])
    y_val = np.array([r["grade"] for r in rows])[va]
    if any(len(set((y_val > k).tolist())) < 2 for k in range(4)):
        print("  too few validation images of each grade to fit the thresholds (smoke run): none fitted", flush=True)
        return
    bias = model.head.bias.detach().double().cpu().numpy() if meta.get("head", "coral") == "coral" else None
    T, sd = thresholds_on_S(L[va], y_val, bias)
    print(f"  thresholds on S: {np.round(T, 4).tolist()}, validation SD of S: {sd:.4f}", flush=True)
    if "thresholds" in meta and not write_thresholds:
        print(f"  stored thresholds: {np.round(meta['thresholds'], 4).tolist()} (kept; --overwrite-thresholds replaces them)")
        return
    if limit and not write_thresholds:
        return
    update_meta(path, thresholds=T.tolist(), score_sd=sd)
    print(f"  thresholds written to {path}", flush=True)


# ----------------------------------------------------------------------------- report
def metrics(d, T_fit_rows, mask):
    """QWK, accuracy and AUCs on the rows in mask, with thresholds fitted on the validation rows"""
    Z = scores(d["L"])
    T = fit_thresholds(d["L"][T_fit_rows], d["grade"][T_fit_rows])
    y, z = d["grade"][mask], Z[mask]
    pred = (z > T).sum(1)
    auc = lambda k: float(roc_auc_score(y >= k, z[:, k - 1])) if 0 < (y >= k).sum() < len(y) else float("nan")
    return dict(qwk=float(cohen_kappa_score(y, pred, weights="quadratic")), acc=float((pred == y).mean()),
                auc_any=auc(1), auc_referable=auc(2), auc_vt=auc(3), n=int(mask.sum()))


def report(src):
    tests = {"in-distribution test (EyePACS + BRSET)": lambda d: d["split"] == "test",
             "EyePACS test": lambda d: (d["split"] == "test") & (d["dataset"] == "eyepacs"),
             "BRSET test": lambda d: (d["split"] == "test") & (d["dataset"] == "brset"),
             "Messidor-2 (out-of-distribution)": lambda d: d["split"] == "messidor2"}
    res, grade = {}, None
    for head in HEADS:
        for s in (0, 1, 2):
            p = Path(src) / "evaluate" / f"{head}_seed{s}.csv"
            if p.exists():
                grade = grade or {r["key"]: r["grade"] for r in data.grading_rows()}
                d = read_csv(p, grade)
                y_val = d["grade"][d["split"] == "val"]
                if any(len(set((y_val > k).tolist())) < 2 for k in range(4)):
                    print(f"{p}: too few validation images of each grade to fit the thresholds (smoke run?); skipped\n")
                    continue
                for t, f in tests.items():
                    res.setdefault(head, {}).setdefault(t, []).append(metrics(d, d["split"] == "val", f(d)))
    if not res:
        raise SystemExit(f"no results in {src}/evaluate: run  uv run python evaluate.py run")
    print(f"# Grading ({src}): mean ± SD over seeds; thresholds recalibrated on the validation split\n")
    print("| Head | Test set | Seeds | n | QWK | Accuracy | AUC any DR | AUC referable | AUC vision-threatening |")
    print("|---|---|---|---|---|---|---|---|---|")
    ms = lambda v, k: f"{np.mean([x[k] for x in v]):.3f} ± {np.std([x[k] for x in v]):.3f}"
    for head, per in res.items():
        for t, v in per.items():
            print(f"| {head.upper()} | {t} | {len(v)} | {v[0]['n']} | {ms(v, 'qwk')} | {ms(v, 'acc')} | {ms(v, 'auc_any')} | "
                  f"{ms(v, 'auc_referable')} | {ms(v, 'auc_vt')} |")
    q = lambda head, t: np.mean([x["qwk"] for x in res[head][t]]) if head in res else float("nan")
    print(f"\nNumbers in the text: CORAL on Messidor-2, QWK {ms(res['coral']['Messidor-2 (out-of-distribution)'], 'qwk')}, "
          f"referable-DR AUC {ms(res['coral']['Messidor-2 (out-of-distribution)'], 'auc_referable')}" if "coral" in res else "")
    if "ce" in res and "coral" in res:
        print(f"Numbers in the text: CE vs CORAL, QWK {q('ce', 'in-distribution test (EyePACS + BRSET)'):.3f} vs "
              f"{q('coral', 'in-distribution test (EyePACS + BRSET)'):.3f} in distribution and "
              f"{q('ce', 'Messidor-2 (out-of-distribution)'):.3f} vs {q('coral', 'Messidor-2 (out-of-distribution)'):.3f} on Messidor-2")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["run", "report"])
    ap.add_argument("--checkpoint", default=None, help="one checkpoint (default: every checkpoint in checkpoints/)")
    ap.add_argument("--from", dest="src", default="outputs", help="report: folder with the evaluate/ results")
    ap.add_argument("--out", default=None, help="run: output folder (default: outputs, or outputs/smoke with --limit)")
    ap.add_argument("--overwrite-thresholds", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: this many images of each split")
    args = ap.parse_args()
    if args.part == "report":
        return report(args.src)
    out = Path(args.out or ("outputs/smoke" if args.limit else "outputs"))
    paths = [Path(args.checkpoint)] if args.checkpoint else [data.REPO / "checkpoints" / f"{h}_seed{s}.safetensors"
                                                              for h in HEADS for s in (0, 1, 2)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("no checkpoint found: run  uv run python download_weights.py  (or train.py)")
    for p in paths:
        run_one(p, out, args.limit, args.overwrite_thresholds, args.batch)


if __name__ == "__main__":
    main()

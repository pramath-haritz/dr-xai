"""All attribution maps of one model on one lesion test set (96 x 96 per image), computed once and cached in
<out>/maps/<head>_seed<s>/<dataset>_<map>.npz, so that causal_tests.py and lesion_ap.py share them.

Maps of the attention-pooling head (ours, drxai.attribution): attention, contribution, pull, attention_term.
Baselines (drxai.baselines): rollout and gradcam for both heads; attnlrp and occlusion for the CORAL model (the maps
of Table 1). Occlusion takes about 4 minutes per image (1,800 forward passes); the other maps take seconds.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from drxai import attribution, baselines, data
from drxai.model import GRID, input_tensor

HEAD_MAPS = ("attention", "contribution", "pull", "attention_term")
BASELINES = {"coral": ("rollout", "gradcam", "attnlrp", "occlusion"), "ce": ("rollout", "gradcam")}


def names_of(head):
    return HEAD_MAPS + BASELINES[head]


def _path(out, head, seed, dataset, method):
    return out / "maps" / f"{head}_seed{seed}" / f"{dataset}_{method}.npz"


def _load(path):
    if not path.exists():
        return {}
    z = np.load(path)
    return {n: z["maps"][i] for i, n in enumerate(z["names"])}


def _compute(model, item, methods, batch):
    x = input_tensor(item["image"])
    out = {}
    if any(m in HEAD_MAPS for m in methods):
        out.update(attribution.maps(attribution.decompose(model, x), GRID))
    if "rollout" in methods:
        out["rollout"] = baselines.rollout(model, x)
    if "gradcam" in methods:
        out["gradcam"] = baselines.gradcam(model, x)
    if "attnlrp" in methods:
        out["attnlrp"] = baselines.attnlrp(model, x)
    if "occlusion" in methods:
        out["occlusion"] = baselines.occlusion(model, x, item["fundus"], batch=batch)
    torch.cuda.empty_cache()
    return {m: np.asarray(out[m], np.float32).reshape(GRID, GRID) for m in methods}


def get(model, head, seed, dataset, rows, methods, out, batch=32):
    """{map: {image name: [GRID, GRID] float32}} for the lesion images rows [(name, subset)]; maps that are not
    cached yet are computed (and cached, unless rows is a subset of the test set, as in a --limit smoke run)"""
    full = len(rows) == len(data.lesion_rows(dataset))
    res = {m: _load(_path(out, head, seed, dataset, m)) for m in methods}
    todo = {m for m in methods if any(n not in res[m] for n, _ in rows)}
    if not todo:
        return res
    model.requires_grad_(False)
    fast, slow = [m for m in methods if m in todo and m != "occlusion"], ["occlusion"] if "occlusion" in todo else []
    for group in (fast, slow):
        if not group:
            continue
        print(f"  maps {head} seed {seed} {dataset}: computing {', '.join(group)} for {len(rows)} images", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(4) as ex:
            items = ex.map(lambda r: data.load_lesion_image(dataset, *r), rows)
            for i, it in enumerate(items):
                for m, v in _compute(model, it, group, batch).items():
                    res[m][it["name"]] = v
                if (i + 1) % (1 if group is slow else 20) == 0 or i + 1 == len(rows):
                    print(f"    {i + 1}/{len(rows)} [{(time.time() - t0) / 60:.1f} min]", flush=True)
        if full:
            for m in group:
                p = _path(out, head, seed, dataset, m)
                p.parent.mkdir(parents=True, exist_ok=True)
                names = [n for n, _ in rows]
                np.savez_compressed(p, names=np.array(names), maps=np.stack([res[m][n] for n in names]))
    return res

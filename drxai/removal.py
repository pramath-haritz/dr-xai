"""Causal tests by token dropping (Section 3 of the paper).

A patch is removed by dropping its token from the sequence after the patch and position embedding; NaFlexViT runs on
any set of tokens and the attention pool renormalises over the rest. Only fundus patches are ranked or removed; the
tokens outside the field of view are always kept, so "all fundus patches removed" is the empty fundus.

For one image and one map (patches ranked by the map, largest first; ties broken by a fixed random order):
  deletion     S after removing the top f of the fundus patches, f in FRACTIONS
  insertion    S after removing all fundus patches except the top f
  necessity    for each grade level j <= g (g: predicted grade): the smallest f whose deletion gives a grade < j
               (1.0 if none does)
  sufficiency  for each level j <= g that the empty fundus does not reach: the smallest f whose insertion gives a
               grade >= j (1.0 if none does)
  FC           faithfulness correlation (Bhatt et al. 2020): Pearson r over 128 random groups of 2% of the fundus
               patches (the same groups for all maps) between the map summed over the group and the drop of S when
               the group is removed
Controls: expert lesions (fraction of lesion pixels in the patch) and random orders (3 orders; for FC a random map).
The random draws come from one generator per image, seeded with (2000 + model seed, image index).
"""
import numpy as np
import torch
from PIL import Image

from drxai.model import GRID, input_tensor

FRACTIONS = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
N_GROUPS, GROUP_FRACTION, N_RANDOM_ORDERS = 128, 0.02, 3


def fundus_patches(fundus):
    """flat indices of the patches of the GRID x GRID grid that are at least half inside the fundus mask"""
    f = Image.fromarray(fundus.astype(np.uint8) * 255).resize((GRID, GRID), Image.BOX)
    return np.where(np.asarray(f).reshape(-1) > 127)[0]


def lesion_fraction(lesions):
    """fraction of expert lesion pixels in each patch, flat [GRID * GRID] (the lesion control)"""
    f = Image.fromarray(lesions.astype(np.uint8) * 255).resize((GRID, GRID), Image.BOX)
    return np.asarray(f).reshape(-1) / 255.0


def draws(seed, index, fund):
    """the per-image random draws, in this order: tie order, random orders, FC groups, random map"""
    rng = np.random.default_rng([2000 + seed, index])
    tie = rng.random(GRID * GRID)
    orders = [rng.permutation(fund) for _ in range(N_RANDOM_ORDERS)]
    groups = [rng.choice(fund, max(1, int(round(GROUP_FRACTION * len(fund)))), replace=False) for _ in range(N_GROUPS)]
    return dict(tie=tie, orders=orders, groups=groups, random_map=rng.standard_normal(GRID * GRID))


def ranking(m, fund, tie):
    """fundus patches sorted by the map m (flat), largest first; ties broken by the tie order"""
    return fund[np.lexsort((tie[fund], -m[fund]))]


class TokenDropper:
    """grade score S of one image with sets of patch tokens removed"""

    def __init__(self, model, image):
        self.model = model
        x = input_tensor(image)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            self.tokens = model.backbone._forward_embeds(x, None, None, None)["patches"][0]          # [N, C]

    @torch.no_grad()
    def scores(self, removed, batch=24):
        """removed: list of index arrays -> S [len(removed)] (float64); sets of the same size share a batch"""
        bb, out = self.model.backbone, np.zeros(len(removed))
        sizes = {}
        for i, r in enumerate(removed):
            sizes.setdefault(len(r), []).append(i)
        everything = np.arange(GRID * GRID)
        for ids in sizes.values():
            for c in range(0, len(ids), batch):
                chunk = ids[c:c + batch]
                keep = np.stack([np.setdiff1d(everything, removed[i]) for i in chunk])
                with torch.autocast("cuda", torch.bfloat16):
                    t = self.tokens[torch.from_numpy(keep).cuda()]
                    for blk in bb.blocks:
                        t = blk(t)
                    f = bb.forward_head(bb.norm(t)).float()
                out[chunk] = self.model.head.S(f).double().cpu().numpy()
        return out


def grade(S, T):
    return (np.asarray(S)[..., None] > np.asarray(T)).sum(-1)


def first_fraction(ok):
    """the smallest f in FRACTIONS where ok is true; 1.0 if none"""
    i = int(np.argmax(ok))
    return float(FRACTIONS[i]) if ok[i] else 1.0


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rank = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(rank(a), rank(b))[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def causal_record(dropper, maps, fund, rnd, T, lesions=None, batch=24):
    """the causal tests of all maps (dict name -> flat map) on one image; lesions: flat lesion fraction (controls)"""
    orders = {m: [ranking(mp, fund, rnd["tie"])] for m, mp in maps.items()}
    fc_maps = dict(maps)
    if lesions is not None:
        if lesions[fund].max() > 0:
            orders["lesion"] = [ranking(lesions, fund, rnd["tie"])]
        orders["random"] = rnd["orders"]
        fc_maps["lesion"], fc_maps["random"] = lesions, rnd["random_map"]
    ns = [max(1, int(round(f * len(fund)))) for f in FRACTIONS]
    sets = [np.array([], np.int64), fund]
    for ords in orders.values():
        for o in ords:
            sets += [o[:n] for n in ns] + [o[n:] for n in ns]
    sets += rnd["groups"]
    S = dropper.scores(sets, batch)
    S0, S_empty = S[0], S[1]
    g, g_empty = int(grade(S0, T)), int(grade(S_empty, T))
    rec = dict(S0=float(S0), S_empty=float(S_empty), grade=g, grade_empty=g_empty, n_fundus=int(len(fund)), maps={})
    p = 2
    for m, ords in orders.items():
        dels, inss, nec, suf = [], [], {}, {}
        for _ in ords:
            dele, ins = S[p:p + len(ns)], S[p + len(ns):p + 2 * len(ns)]
            p += 2 * len(ns)
            dels.append(dele)
            inss.append(ins)
            for j in range(1, g + 1):
                nec.setdefault(str(j), []).append(first_fraction(grade(dele, T) < j))
                if g_empty < j:
                    suf.setdefault(str(j), []).append(first_fraction(grade(ins, T) >= j))
        rec["maps"][m] = dict(deletion=np.mean(dels, 0).tolist(), insertion=np.mean(inss, 0).tolist(),
                              necessity=nec, sufficiency=suf)
    dS = S0 - S[p:p + len(rnd["groups"])]
    for m, mp in fc_maps.items():
        a = np.array([mp[s].sum() for s in rnd["groups"]])
        rec["maps"].setdefault(m, {}).update(fc=pearson(a, dS), fc_spearman=spearman(a, dS))
    return rec

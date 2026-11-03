"""Statistics of the paper: hierarchical bootstrap over test images and seeds, paired comparisons and Holm correction.

A measure is a matrix V [seeds, images]. One bootstrap draw resamples the images (multinomial weights W) and the seeds
(weights C) with replacement and returns the weighted mean ((C @ V) * W).sum() / V.size. A paired comparison of two
maps uses the same image weights for both and resamples the seeds of each separately; the two-sided p-value is
twice the smaller tail of the bootstrap difference at 0.

The bootstrap draws come from generators with fixed seeds; each report makes its draws in a fixed order, so that it
prints the paper's intervals and p-values exactly.
"""
import numpy as np


def boot_mean(V, rng, n_boot):
    """bootstrap draws of the mean of V [seeds, images]"""
    k, n = V.shape
    W = rng.multinomial(n, np.ones(n) / n, size=n_boot).astype(float)
    C = rng.multinomial(k, np.ones(k) / k, size=n_boot).astype(float)
    return ((C @ V) * W).sum(1) / V.size


def boot_diff(Va, Vb, rng, n_boot):
    """bootstrap draws of mean(Va) - mean(Vb), images paired (the same columns), seeds resampled within each"""
    n, ka, kb = Va.shape[1], Va.shape[0], Vb.shape[0]
    W = rng.multinomial(n, np.ones(n) / n, size=n_boot).astype(float)
    Ca = rng.multinomial(ka, np.ones(ka) / ka, size=n_boot).astype(float)
    Cb = rng.multinomial(kb, np.ones(kb) / kb, size=n_boot).astype(float)
    return ((Ca @ Va) * W).sum(1) / Va.size - ((Cb @ Vb) * W).sum(1) / Vb.size


def ci(draws):
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def p_two_sided(draws):
    return float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))


def holm(tests):
    """adds p_holm to each test (dicts with a p-value 'p'), over all tests of the family"""
    order, running = sorted(tests, key=lambda h: tests[h]["p"]), 0.0
    for i, h in enumerate(order):
        running = max(running, min(1.0, tests[h]["p"] * (len(order) - i)))
        tests[h]["p_holm"] = running
    return tests

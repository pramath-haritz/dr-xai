"""Fig. 1 and Fig. 2 of the paper (drawn by pool_identity.py report). All text is 9 pt; Fig. 1 spans two columns (7.0 in),
Fig. 2 one column (3.39 in).

Fig. 1  the identity of Eq. (4) on the patches the head attends to, for one MAPLES-DR image (seed 1):
        (a) the image, the expert lesion outlines and the K patches with the highest mean attention;
        (b) for each of them, a thumbnail and the terms of the identity: pull term (bar), attention term (bar) and their
            sum, the contribution (marker);
        (c) the grade score as the top-ranked patches of the pull and of the contribution are deleted (token dropping).
Fig. 2  the pull predicts removal from the pool: (a) exact change of S against the summed maps for the 128 random groups
        of one image (seed 1, the image whose contribution r is closest to the seed-1 median); (b) Pearson r with the
        exact change for every image and seed, the contribution split by the sign of the alignment q.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

COLW, TEXTW = 3.39, 7.0
PULL, CONTRIB, TERM, GREY = "#2a78d6", "#eb6834", "#8a8984", "#8a8984"
INK, INK2, HAIR, MARK = "#0b0b0b", "#52514e", "#c3c2b7", "#ffe14d"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2, "legend.frameon": False,
                     "pdf.fonttype": 42, "savefig.dpi": 300})


def _clean(a):
    a.set_xticks([]), a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)


def _display(image, lesions, side):
    """the 1,536-px crop and the union of its lesion masks, resized for display"""
    img = np.asarray(Image.fromarray(image).resize((side, side), Image.LANCZOS))
    les = np.asarray(Image.fromarray(lesions.astype(np.uint8)).resize((side, side), Image.NEAREST)) > 0
    return img, les


def fig1(ex, image, lesions, path, k=6):
    """ex: the Fig. 1 record of pool_identity.py; image, lesions: its 1,536-px scoring crop (uint8) and the union of the
    expert masks in that frame (bool), as saved by pool_identity.fig1_image"""
    G = len(ex["attention"])
    c, s, a = (np.array(ex[m], np.float64) for m in ("contribution", "attention_term", "attention"))
    pp = c - s                                                        # sum_h (1 - a) p, exact by the identity
    fund = np.zeros(G * G, bool)
    fund[ex["fund"]] = True
    fund = fund.reshape(G, G)
    top = np.argsort(-np.where(fund, a, -1).ravel())[:k]
    rc = [divmod(int(j), G) for j in top]
    share = a.ravel()[top].sum() / a[fund].sum()
    img_hi, les_hi = _display(image, lesions, 1536)
    img, les = _display(image, lesions, 768)
    les = les.astype(float)

    fig = plt.figure(figsize=(TEXTW, 1.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.15, 1.05], wspace=0.30, left=0.0, right=0.985, top=0.88, bottom=0.2)
    A = fig.add_subplot(gs[0, 0])
    gb = gs[0, 1].subgridspec(2, k, height_ratios=[0.62, 1.0], hspace=0.08, wspace=0.12)
    B = fig.add_subplot(gb[1, :])
    thumbs = [fig.add_subplot(gb[0, j]) for j in range(k)]
    C = fig.add_subplot(gs[0, 2])

    # (a) where the head looks
    A.imshow(img, interpolation="lanczos")
    A.contour(les, [0.5], colors="white", linewidths=0.35)
    px = 768 // G
    placed = []
    for n_, (r_, c_) in enumerate(rc, 1):
        A.add_patch(Rectangle((c_ * px - 0.5 - 3, r_ * px - 0.5 - 3), px + 6, px + 6, fill=False, ec=MARK, lw=0.9))
        left = any(abs(r_ - r2) <= 2 and abs(c_ - c2) <= 3 for r2, c2 in placed)       # neighbour labelled: go left
        A.text(c_ * px - 8 if left else c_ * px + px + 6, r_ * px + px / 2, str(n_), color=MARK, fontsize=9,
               fontweight="bold", ha="right" if left else "left", va="center")
        placed.append((r_, c_))
    _clean(A)
    A.set_title("(a)", loc="center", fontweight="bold", pad=3)
    A.text(0.5, -0.05, f"grade {ex['grade']}; patches 1–{k} hold\n{share:.0%} of the attention", transform=A.transAxes,
           ha="center", va="top", color=INK2, linespacing=1.15)

    # (b) the identity on each patch
    x = np.arange(k)
    vp, vs, vc = (np.array([v[r_, c_] for r_, c_ in rc]) for v in (pp, s, c))
    bw = 0.34
    B.bar(x - bw / 2, vp, bw, color=PULL, zorder=2)
    B.bar(x + bw / 2, vs, bw, color=TERM, zorder=2)
    B.plot(x, vc, ls="", marker="D", ms=4.6, mfc=CONTRIB, mec="white", mew=0.6, zorder=4)
    B.axhline(0, color=INK2, lw=0.6, zorder=1)
    B.set_xlim(-0.5, k - 0.5)
    B.set_xticks([])
    B.spines["bottom"].set_visible(False)
    B.set_ylabel("effect on $S$", labelpad=1)
    B.locator_params(axis="y", nbins=4)
    hd = [Patch(color=PULL), Patch(color=TERM), Line2D([], [], ls="", marker="D", ms=4.6, mfc=CONTRIB, mec="white")]
    B.legend(hd, ["pull term  +", "attention term  =", "contribution $c_t$"], loc="upper center",
             bbox_to_anchor=(0.5, -0.04), ncol=3, borderaxespad=0.0, handlelength=0.8, handletextpad=0.3,
             columnspacing=0.6, borderpad=0.0)
    for j, (r_, c_) in enumerate(rc):
        y0, x0 = (r_ - 1) * 16, (c_ - 1) * 16
        thumbs[j].imshow(img_hi[y0:y0 + 48, x0:x0 + 48], interpolation="lanczos")
        thumbs[j].contour(les_hi[y0:y0 + 48, x0:x0 + 48].astype(float), [0.5], colors="white", linewidths=0.5)
        thumbs[j].add_patch(Rectangle((15.5, 15.5), 16, 16, fill=False, ec=MARK, lw=0.8))
        thumbs[j].text(2.5, 2.5, str(j + 1), color=MARK, fontsize=9, fontweight="bold", ha="left", va="top")
        _clean(thumbs[j])
    fig.canvas.draw()
    fig.text(0.5 * (thumbs[0].get_position().x0 + thumbs[-1].get_position().x1), thumbs[0].get_position().y1 + 0.02,
             "(b)", ha="center", va="bottom", fontweight="bold")

    # (c) the grade score as the top-ranked patches are deleted one by one (token dropping)
    kk = np.arange(len(ex["deletion"]["pull"]))
    T = np.array(ex["thresholds"])
    yp, yc = np.array(ex["deletion"]["pull"]), np.array(ex["deletion"]["contribution"])
    g = max(1, ex["grade"])
    lower, upper = T[g - 1], (T[g] if g < 4 else T[g - 1] + 1)        # the band of the predicted grade
    C.axhspan(lower, upper, color="#eeede9", lw=0, zorder=0)
    C.axhline(lower, color=INK2, lw=0.6, zorder=1)
    C.plot(kk, yc, color=CONTRIB, lw=1.5, zorder=3)
    C.plot(kk, yp, color=PULL, lw=1.5, zorder=3)
    C.plot([0], [yp[0]], marker="o", ms=3.5, color=INK, zorder=4)
    C.set_xlim(-1, kk[-1])
    C.set_xlabel(f"No. of patches deleted (of {ex['n_fundus']:,})", labelpad=1)
    C.set_ylabel("grade score $S$", labelpad=1)
    lo, hi = min(yp.min(), yc.min(), lower), max(yp.max(), yc.max(), upper if upper < yp.max() + 1 else yp.max())
    C.set_ylim(lo - 0.12 * (hi - lo), hi + 0.32 * (hi - lo))
    C.text(0.97, lower + 0.04 * (hi - lo), f"grade {g}", transform=C.get_yaxis_transform(), ha="right", va="bottom",
           color=INK2)
    C.text(0.97, lower - 0.04 * (hi - lo), f"grade {g - 1}", transform=C.get_yaxis_transform(), ha="right", va="top",
           color=INK2)
    C.text(kk[-1] * 0.30, yc[int(kk[-1] * 0.30)] + 0.06 * (hi - lo), "contribution $c_t$", ha="center", va="bottom", color=INK2)
    C.text(kk[-1] * 0.55, yp[int(kk[-1] * 0.55)] + 0.06 * (hi - lo), "pull $p_t$", ha="center", va="bottom", color=INK2)
    C.set_title("(c)", loc="center", fontweight="bold", pad=3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return share


def fig2(ex, rows, path):
    """ex: dict(dS, pull, contribution, attention_term) of the example image; rows: [(label, colour, r values)]"""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COLW, 1.55), gridspec_kw=dict(width_ratios=[1.16, 1], wspace=0.30))
    for a_ in (ax, bx):
        a_.tick_params(length=2.5, width=0.6, pad=2, color=INK2)
        for sp in a_.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color(INK2)

    # (a) parity plot: equal limits on both axes, so the identity is a 45-degree line
    dS, pull, contrib = (np.array(ex[k]) for k in ("dS", "pull", "contribution"))
    lo, hi = min(dS.min(), pull.min(), contrib.min()), max(dS.max(), pull.max(), contrib.max())
    lim = (lo - 0.06 * (hi - lo), hi + 0.30 * (hi - lo))
    ax.plot(lim, lim, color=HAIR, lw=0.9, ls=(0, (3, 2)), zorder=1)
    gap = contrib - dS                                                # equals the summed attention term
    ax.vlines(dS, contrib, dS, color=HAIR, lw=0.5, zorder=1.5)
    j = int(np.argmin(gap))
    ax.annotate(r"$\sum a_{h,t}E_h$", xy=(dS[j], 0.5 * (dS[j] + contrib[j])), xytext=(6, -2), textcoords="offset points",
                ha="left", va="center", color=INK2)
    ax.scatter(dS, contrib, s=9, marker="s", facecolor=CONTRIB, edgecolor="white", linewidth=0.3, alpha=0.9, zorder=2)
    ax.scatter(dS, pull, s=10, marker="o", facecolor=PULL, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    hd = [Line2D([], [], ls="", marker="o", ms=3.4, color=PULL), Line2D([], [], ls="", marker="s", ms=3.0, color=CONTRIB)]
    ax.legend(hd, ["pull $p_t$", "contribution $c_t$"], loc="upper left", borderaxespad=0.0, handletextpad=0.1,
              borderpad=0.2, handlelength=0.7, labelspacing=0.15)
    ticks = [0.0, 0.05, 0.1]
    ax.set_xticks([t for t in ticks if lim[0] <= t <= lim[1]])
    ax.set_yticks([t for t in ticks if lim[0] <= t <= lim[1]])
    ax.set_xlabel(r"exact $\Delta S$", labelpad=1.5)
    ax.set_ylabel("summed map", labelpad=1.5)
    ax.set_title("(a)", loc="center", fontweight="bold", pad=4)

    # (b) raw points with the quartile bar and the median dot above them
    rng = np.random.default_rng(0)
    for y, (lab, col, r) in enumerate(rows):
        r = np.asarray(r)[np.isfinite(r)]
        bx.text(-1.0, y - 0.11, lab, ha="left", va="bottom", color=INK2)
        if not len(r):
            continue
        bx.scatter(r, y + 0.13 + rng.uniform(-0.12, 0.12, len(r)), s=2.0, color=col, alpha=0.4, linewidth=0,
                   rasterized=True, zorder=2)
        q1, med, q3 = np.percentile(r, [25, 50, 75])
        halo = [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]
        bx.plot([q1, q3], [y + 0.13, y + 0.13], color=INK, lw=1.3, solid_capstyle="butt", zorder=3, path_effects=halo)
        bx.plot([med], [y + 0.13], marker="o", ms=3.2, mfc="white", mec=INK, mew=0.9, zorder=4, clip_on=False)
    bx.set_ylim(len(rows) - 0.66, -0.58)
    bx.set_yticks([])
    bx.spines["left"].set_visible(False)
    bx.set_xlim(-1.0, 1.04)
    bx.set_xticks([-1, 0, 1])
    bx.axvline(0, color=HAIR, lw=0.7, zorder=1)
    bx.set_xlabel(r"$r$ with exact $\Delta S$", labelpad=1.5)
    bx.set_title("(b)", loc="center", fontweight="bold", pad=4)
    fig.canvas.draw()                                                 # align (b) with the square panel (a)
    pa, pb = ax.get_position(), bx.get_position()
    bx.set_position([pb.x0, pa.y0, pb.width, pa.height])
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

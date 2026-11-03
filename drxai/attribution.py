"""Closed-form attribution through the attention-pooling (MAP) head: Section 2 of the paper.

For each head h and patch t of the MAP head:
  a_{h,t}   attention weight (softmax of the learned query against the patch keys; sums to 1 over t)
  v_{h,t}   value of the patch
  z_h       = sum_t a_{h,t} v_{h,t}, the head vector;  z = b_O + sum_h W_O^h z_h, the pooled vector
  S         = phi(z), the grade score (LayerNorm, residual MLP and the linear score of the head)
Eq. (1)   g = integral over alpha of grad phi on the straight path from the empty pool b_O to z (64 midpoints), so that
          S - S_empty = g . (z - b_O) exactly; r_h = W_O^h' g reads a head vector along the score direction
Eq. (2)   patch value e_{h,t} = r_h . v_{h,t}; head evidence E_h = sum_t a_{h,t} e_{h,t} = r_h . z_h;
          contribution c_{h,t} = a_{h,t} e_{h,t}; completeness: S = S_empty + sum_{h,t} c_{h,t}
Eq. (3)   pull p_{h,t} = r_h . (z_h - z'_h) = a_{h,t} / (1 - a_{h,t}) (e_{h,t} - E_h), the effect of removing patch t
          from the pool (z'_h: head vector with t removed and the softmax renormalised over the other patches)
Eq. (4)   identity c_{h,t} = (1 - a_{h,t}) p_{h,t} + a_{h,t} E_h; the pull term (1 - a) p sums to 0 in each head
Maps (96 x 96, one value per patch): attention weights (mean over heads), contribution, pull and attention term
(sum_h a_{h,t} E_h), all summed over heads. Alignment q: Pearson correlation over the fundus patches between the
attention term and the attention weights.
The head is evaluated in float64 on the tokens of a bfloat16 forward pass of the backbone.
"""
import numpy as np
import torch
import torch.nn.functional as F

STEPS = 64


@torch.no_grad()
def pool_tokens(model, x):
    """the tokens that enter the MAP head, [N, C] (bfloat16 forward of the whole model)"""
    store = {}
    hook = model.backbone.attn_pool.register_forward_pre_hook(lambda m, a: store.__setitem__("x", a[0].detach()))
    with torch.autocast("cuda", torch.bfloat16):
        model(x)
    hook.remove()
    return store["x"][0]


def phi(model, z):
    """grade score S of pooled vectors z [..., C] in float64: phi(z) = score(z + MLP(LN(z)))"""
    ap = model.backbone.attn_pool
    n, m = ap.norm, ap.mlp
    y = F.layer_norm(z, (z.shape[-1],), n.weight.double(), n.bias.double(), n.eps)
    y = F.linear(m.act(F.linear(y, m.fc1.weight.double(), m.fc1.bias.double())), m.fc2.weight.double(), m.fc2.bias.double())
    return model.head.S(z + y)


@torch.no_grad()
def head_state(model, tokens):
    """attention a [H, N], values v [N, H, hd], head vectors zh [H, hd] and the pooled vector z [C], in float64"""
    ap = model.backbone.attn_pool
    H, hd = ap.num_heads, ap.head_dim
    lin = lambda mod, t: F.linear(t, mod.weight.double(), mod.bias.double())
    q = lin(ap.q, ap.latent.double()).reshape(1, H, hd)
    kv = lin(ap.kv, tokens.double()).reshape(-1, 2, H, hd)
    k, v = kv[:, 0], kv[:, 1]
    a = (torch.einsum("hd,nhd->hn", q[0], k) * ap.scale).softmax(-1)
    zh = torch.einsum("hn,nhd->hd", a, v)
    Wp, bp = ap.proj.weight.double(), ap.proj.bias.double()
    return dict(a=a, v=v, zh=zh, Wp=Wp, bp=bp, z=Wp @ zh.reshape(-1) + bp, H=H, hd=hd)


def readout(model, st):
    """r [H, hd]: Eq. (1), the integrated gradient of phi from the empty pool b_O to z, read through W_O^h"""
    z, bp = st["z"], st["bp"]
    alpha = (torch.arange(1, STEPS + 1, device=z.device, dtype=torch.float64) - 0.5) / STEPS
    with torch.enable_grad():
        zz = (bp[None] + alpha[:, None] * (z - bp)[None]).requires_grad_(True)
        g = torch.autograd.grad(phi(model, zz).sum(), zz)[0].mean(0)
    return (st["Wp"].T @ g).reshape(st["H"], st["hd"])


@torch.no_grad()
def decompose(model, x):
    """all head quantities of one image x [1, 3, side, side]:
    a, c, p [H, N]; E [H]; S, S_empty (float); st (head state, for pool removal); float64 tensors on the GPU"""
    st = head_state(model, pool_tokens(model, x))
    r = readout(model, st)
    a, v, zh = st["a"], st["v"], st["zh"]
    c = a * torch.einsum("nhd,hd->hn", v, r)                                 # Eq. (2)
    E = (zh * r).sum(-1)
    p = a * torch.einsum("nhd,hd->hn", v - zh[None], r) / (1 - a)            # Eq. (3)
    return dict(a=a, c=c, p=p, E=E, S=float(phi(model, st["z"])), S_empty=float(phi(model, st["bp"])), st=st)


def maps(d, grid):
    """the four maps of the head as float32 [grid, grid] arrays"""
    g = lambda t: np.asarray(t, np.float32).reshape(grid, grid)
    return dict(attention=d["a"].float().cpu().numpy().mean(0).reshape(grid, grid),
                contribution=g(d["c"].cpu().numpy().sum(0)), pull=g(d["p"].sum(0).cpu().numpy()),
                attention_term=g((d["a"] * d["E"][:, None]).sum(0).cpu().numpy()))


def checks(d):
    """the largest violations of completeness (Eq. 2) and of the identity (Eq. 4) for one image"""
    a, c, p, E = d["a"], d["c"], d["p"], d["E"]
    return dict(completeness=float(abs(d["S_empty"] + float(c.sum()) - d["S"])),
                identity=float((c - (1 - a) * p - a * E[:, None]).abs().max()),
                pull_term_sum=float(((1 - a) * p).sum(1).abs().max()))


def alignment(m, fund):
    """q: Pearson correlation over the fundus patches (flat indices) between the attention term and the attention
    weights of the maps m (as returned by maps)"""
    s, a = m["attention_term"].reshape(-1)[fund], m["attention"].reshape(-1)[fund]
    return float(np.corrcoef(s, a)[0, 1]) if s.std() > 0 and a.std() > 0 else float("nan")


@torch.no_grad()
def pool_removal(model, d, sets):
    """exact change of S when each set of patches (flat indices) is removed from the pool in all heads (attention
    logits -inf, softmax renormalised over the rest; backbone fixed), and the change under Value Zeroing (values of the
    set zeroed, attention fixed). -> (dS, dS_value_zeroing) float64 numpy [len(sets)]"""
    st = d["st"]
    a, v, zh, Wp, bp = st["a"], st["v"], st["zh"], st["Wp"], st["bp"]
    M = torch.zeros(len(sets), a.shape[1], dtype=torch.float64, device=a.device)
    for j, s in enumerate(sets):
        M[j, torch.from_numpy(np.asarray(s, np.int64)).to(a.device)] = 1
    removed = torch.einsum("sn,hn,nhd->shd", M, a, v)                         # sum over the set of a v, per head
    mass = M @ a.T                                                            # attention mass of the set, per head
    pooled = lambda zz: zz.reshape(len(sets), -1) @ Wp.T + bp
    S = phi(model, st["z"])
    dS = S - phi(model, pooled((zh[None] - removed) / (1 - mass)[..., None]))
    dS_vz = S - phi(model, pooled(zh[None] - removed))
    return dS.cpu().numpy(), dS_vz.cpu().numpy()

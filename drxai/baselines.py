"""Baseline attribution maps, one value per patch of the 96 x 96 grid (the settings of the paper):

  rollout    attention rollout from the pool query (Abnar & Zuidema 2020): r = a_pool, then for each block from the
             last to the first r <- 0.5 r + 0.5 r A_l (A_l: attention of block l, mean over heads)
  gradcam    Grad-CAM (Selvaraju et al. 2017) on the tokens after block 8: ReLU(sum_c mean_t(dS/dA_tc) A_tc)
  attnlrp    AttnLRP (Achtibat et al. 2024) with the LXT recipe for vision transformers: CP-LRP inside the attention
             (no relevance through queries and keys), generalized gamma rule on every linear layer (gamma 0.1) and on
             the patch embedding (gamma 0.1), identity rule on LayerNorm and GELU; relevance x . dS/dx summed over
             each patch. The configuration was selected on IDRiD (best mean lesion AP over 30 configurations).
  occlusion  2 x 2-patch windows (32 px, no overlap, windows with >= 50% fundus) replaced by a masked Gaussian blur of
             the image (sigma 16 px); value S - S(window occluded)

The rollout, Grad-CAM and AttnLRP maps use a float32 forward pass written out from the timm model (same math).
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from drxai.model import GRID, PATCH, SIDE

GRADCAM_BLOCK = 8
LRP_GAMMA, LRP_EMBED_GAMMA = 0.1, 0.1
OCCLUSION_WINDOW = 2
BLUR_SIGMA = 16


# ----------------------------------------------------------------------------- written-out forward pass
def layer_norm(mod, x, lrp=False):
    """LayerNorm; lrp: the normaliser is a constant (identity rule)"""
    mu = x.mean(-1, keepdim=True)
    sd = (x.var(-1, keepdim=True, unbiased=False) + mod.eps).sqrt()
    return (x - mu) / (sd.detach() if lrp else sd) * mod.weight + mod.bias


def gelu(x, lrp=False):
    """tanh GELU; lrp: the gate y/x is a constant (identity rule)"""
    y = F.gelu(x, approximate="tanh")
    if not lrp:
        return y
    big = x.abs() > 1e-6
    ratio = torch.where(big, y / torch.where(big, x, torch.ones_like(x)), torch.full_like(x, 0.5))
    return x * ratio.detach()


def forward_S(model, x, capture=None, keep_block=None):
    """grade score S of x [1, 3, side, side] (float32). capture: list that receives (q, k, scale) of every block and of
    the pool; keep_block: index of the block whose output tokens keep their gradient. -> (S, kept tokens)"""
    bb = model.backbone
    t = bb._forward_embeds(x, None, None, None)["patches"]
    kept = None
    for i, blk in enumerate(bb.blocks):
        a = blk.attn
        h = layer_norm(blk.norm1, t)
        B, N, C = h.shape
        q, k, v = F.linear(h, a.qkv.weight, a.qkv.bias).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        if capture is not None:
            capture.append((q.detach(), k.detach(), a.scale))
        o = F.scaled_dot_product_attention(q, k, v)
        t = t + F.linear(o.transpose(1, 2).reshape(B, N, C), a.proj.weight, a.proj.bias)
        m = blk.mlp
        t = t + F.linear(gelu(F.linear(layer_norm(blk.norm2, t), m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
        if i == keep_block:
            t.retain_grad()
            kept = t
    t = layer_norm(bb.norm, t)
    ap = bb.attn_pool
    B, N, C = t.shape
    q = F.linear(ap.latent, ap.q.weight, ap.q.bias).reshape(1, ap.latent_len, ap.num_heads, ap.head_dim).transpose(1, 2)
    k, v = F.linear(t, ap.kv.weight, ap.kv.bias).reshape(B, N, 2, ap.num_heads, ap.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    if capture is not None:
        capture.append((q.detach(), k.detach(), ap.scale))
    o = F.scaled_dot_product_attention(q, k, v)
    z = F.linear(o.transpose(1, 2).reshape(B, 1, C), ap.proj.weight, ap.proj.bias)
    m = ap.mlp
    z = z + F.linear(gelu(F.linear(layer_norm(ap.norm, z), m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
    return model.head.S(z[:, 0]), kept


@torch.no_grad()
def rollout(model, x):
    cap = []
    forward_S(model, x, capture=cap)
    *blocks, (q, k, sc) = cap
    r = (torch.einsum("bhqd,bhnd->bhqn", q, k) * sc).softmax(-1)[0, :, 0].mean(0)            # pool attention [N]
    for q, k, sc in reversed(blocks):
        A = torch.zeros(q.shape[2], q.shape[2], device=q.device)
        for h in range(q.shape[1]):
            A += (q[0, h] @ k[0, h].T * sc).softmax(-1)
        r = 0.5 * r + 0.5 * (r @ (A / q.shape[1]))
    return r.cpu().numpy().reshape(GRID, GRID)


def gradcam(model, x, block=GRADCAM_BLOCK):
    with torch.enable_grad():
        S, act = forward_S(model, x.clone().requires_grad_(True), keep_block=block - 1)
        S.sum().backward()
    with torch.no_grad():
        return F.relu((act.grad[0].mean(0, keepdim=True) * act[0]).sum(-1)).cpu().numpy().reshape(GRID, GRID)


# ----------------------------------------------------------------------------- AttnLRP
def _stab(z, eps=1e-6):
    return z + eps * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))


class GammaLinear(torch.autograd.Function):
    """y = x W' + b; backward: the generalized gamma rule, written in the gradient x input form
    (R_out = grad x output, R_in from the rule, grad_in = R_in / input)"""

    @staticmethod
    def forward(ctx, x, W, b, gamma):
        y = F.linear(x, W, b)
        ctx.save_for_backward(x, W, b, y)
        ctx.gamma = gamma
        return y.clone()                                                  # not a view (the embedding adds in place)

    @staticmethod
    def backward(ctx, g):
        x, W, b, y = ctx.saved_tensors
        gm = ctx.gamma
        R = g * y
        Wp, Wn = W + gm * W.clamp(min=0), W + gm * W.clamp(max=0)
        bp = b + gm * b.clamp(min=0) if b is not None else None
        bn = b + gm * b.clamp(max=0) if b is not None else None
        xp, xn = x.clamp(min=0), x.clamp(max=0)
        z_pos = F.linear(xp, Wp, bp) + F.linear(xn, Wn)                    # denominators of the positive-output branch
        z_neg = F.linear(xp, Wn, bn) + F.linear(xn, Wp)                    # and of the negative-output branch
        s_pos = (y > 0) * R / _stab(z_pos)
        s_neg = (y < 0) * R / _stab(z_neg)
        R_in = xp * (s_pos @ Wp + s_neg @ Wn) + xn * (s_pos @ Wn + s_neg @ Wp)
        return R_in / _stab(x), None, None, None


def attnlrp(model, x, gamma=LRP_GAMMA, embed_gamma=LRP_EMBED_GAMMA):
    """signed relevance of the grade score per patch [GRID, GRID] (CORAL head: its score is a linear layer)"""
    lin = lambda m, t: GammaLinear.apply(t, m.weight, m.bias, gamma)
    mlp = lambda m, t: lin(m.fc2, gelu(lin(m.fc1, t), lrp=True))
    bb = model.backbone
    proj = bb.embeds.proj
    proj.forward = lambda t: GammaLinear.apply(t, proj.weight, proj.bias, embed_gamma)
    try:
        xl = x.clone().requires_grad_(True)
        with torch.enable_grad():
            t = bb._forward_embeds(xl, None, None, None)["patches"]
            for blk in bb.blocks:
                a = blk.attn
                h = layer_norm(blk.norm1, t, lrp=True)
                B, N, C = h.shape
                q, k, v = lin(a.qkv, h).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
                q, k = q.detach(), k.detach()                                 # CP-LRP
                o = F.scaled_dot_product_attention(q, k, v)
                t = t + lin(a.proj, o.transpose(1, 2).reshape(B, N, C))
                t = t + mlp(blk.mlp, layer_norm(blk.norm2, t, lrp=True))
            t = layer_norm(bb.norm, t, lrp=True)
            ap = bb.attn_pool
            B, N, C = t.shape
            q = F.linear(ap.latent, ap.q.weight, ap.q.bias).reshape(1, ap.latent_len, ap.num_heads, ap.head_dim).transpose(1, 2)
            k, v = lin(ap.kv, t).reshape(B, N, 2, ap.num_heads, ap.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            k = k.detach()
            o = F.scaled_dot_product_attention(q, k, v)
            z = lin(ap.proj, o.transpose(1, 2).reshape(B, 1, C))
            z = z + mlp(ap.mlp, layer_norm(ap.norm, z, lrp=True))
            S = lin(model.head.score, z[:, 0])[:, 0]
            S.sum().backward()
    finally:
        del proj.forward                                                  # back to the class method
    R = (xl * xl.grad)[0].sum(0)
    return (F.avg_pool2d(R[None, None], PATCH)[0, 0] * PATCH ** 2).detach().cpu().numpy()


# ----------------------------------------------------------------------------- occlusion
def blur(x, fm, sigma=BLUR_SIGMA):
    """masked (normalised) Gaussian blur of x [B, C, H, W] inside the fundus mask fm [1, 1, H, W]"""
    k = int(4 * sigma) * 2 + 1
    t = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = g / g.sum()

    def sep(z):
        c = z.shape[1]
        z = F.conv2d(F.pad(z, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
        return F.conv2d(F.pad(z, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)
    return sep(x * fm) / sep(fm).clamp_min(1e-3)


@torch.no_grad()
def _occluded_scores(model, x, xb, keep, batch):
    """S of x with the patches where keep [n, GRID, GRID] is 0 replaced by xb (bfloat16 forward and head)"""
    out = []
    for i in range(0, len(keep), batch):
        m = F.interpolate(keep[i:i + batch, None].float(), scale_factor=PATCH, mode="nearest")
        with torch.autocast("cuda", torch.bfloat16):
            s = model.head.S(model(x * m + xb * (1 - m))["f"].float())
        out.append(s.double().cpu())
    return torch.cat(out).numpy()


def occlusion(model, x, fundus, batch=32, w=OCCLUSION_WINDOW):
    """fundus: the bool fundus mask of the image (any resolution)"""
    fimg = Image.fromarray(fundus.astype(np.uint8) * 255)
    xb = blur(x, torch.from_numpy(np.asarray(fimg.resize((SIDE, SIDE), Image.NEAREST)) > 127).float().cuda()[None, None])
    ftok = torch.from_numpy(np.asarray(fimg.resize((GRID, GRID), Image.BOX)) / 255.0).float().cuda()
    S0 = float(_occluded_scores(model, x, xb, torch.ones(1, GRID, GRID, device="cuda"), batch)[0])
    wins = torch.nonzero(F.avg_pool2d(ftok[None, None], w)[0, 0] >= 0.5)
    keep = torch.ones(len(wins), GRID, GRID, device="cuda")
    for j, (a, b) in enumerate(wins.tolist()):
        keep[j, a * w:(a + 1) * w, b * w:(b + 1) * w] = 0
    d = S0 - _occluded_scores(model, x, xb, keep, batch)
    occ = np.full((GRID // w, GRID // w), np.nan, np.float32)
    occ[wins[:, 0].cpu().numpy(), wins[:, 1].cpu().numpy()] = d
    occ = np.nan_to_num(occ, nan=float(np.nanmin(occ)))                     # outside the fundus: the lowest value
    return np.kron(occ, np.ones((w, w), np.float32))

"""The grader: NaFlexViT-B/16 (SigLIP 2 weights) with its multi-head attention-pooling (MAP) head, then an ordinal CORAL
head (or a 5-class cross-entropy head) on the pooled 768-d feature. Both heads give one grade score S per image; the
predicted grade is the number of grade thresholds T_1 < ... < T_4 (recalibrated on the validation split by
evaluate.py) that S exceeds.

Checkpoints are safetensors files; their metadata holds the head, the seed, the training config and the thresholds.
"""
import json
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file

DEV = "cuda"
BACKBONE = "naflexvit_base_patch16_siglip.v2_webli"
SIDE = 1536                   # input side of the experiments: 96 x 96 patches of 16 px
PATCH = 16
GRID = SIDE // PATCH
D_SPREAD = 32                 # width of the projection that the spread term of the training loss acts on


class Coral(nn.Module):
    """CORAL ordinal head: one shared score plus four ordered biases, logit_k = S + b_k (k = 1..4)"""

    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)
        self.bias = nn.Parameter(torch.tensor([1.5, 0.5, -0.5, -1.5]))

    def forward(self, f):
        return self.score(f) + self.bias

    def S(self, f):
        """grade score, in the dtype of f"""
        return F.linear(f, self.score.weight.to(f.dtype), self.score.bias.to(f.dtype))[..., 0]


class CEHead(nn.Module):
    """5-class softmax head (cross-entropy); grade score S = expected grade sum_k k p_k"""

    def __init__(self, d, k=5):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, f):
        return self.fc(f)

    def S(self, f):
        p = F.linear(f, self.fc.weight.to(f.dtype), self.fc.bias.to(f.dtype)).softmax(-1)
        return p @ torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)


class Grader(nn.Module):
    def __init__(self, head="coral", pretrained=False, spread=False):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.head = CEHead(d) if head == "ce" else Coral(d)
        self.spread_proj = nn.Linear(d, D_SPREAD) if spread else None      # training only
        bb = self.backbone
        assert isinstance(bb.norm_pre, nn.Identity) and isinstance(bb.fc_norm, nn.Identity) and bb.num_prefix_tokens == 0
        assert bb.attn_pool.latent_len == 1 and bb.attn_pool.pos_embed is None

    @staticmethod
    def prep(x_uint8):
        """uint8 [B, 3, H, W] -> float in [0, 1] on the GPU (augmentation, if any, comes next)"""
        return x_uint8.to(DEV, non_blocking=True).float() / 255

    @staticmethod
    def norm(x):
        return x * 2 - 1                                                    # SigLIP 2: (-1, 1)

    def forward(self, x):
        f = self.backbone(x).float()
        out = dict(f=f, main=self.head(f))
        if self.spread_proj is not None:
            out["spread"] = self.spread_proj(f)
        return out


def input_tensor(image, side=SIDE):
    """uint8 [h, w, 3] image (a fundus crop) -> model input [1, 3, side, side] in (-1, 1), float32 on the GPU"""
    x = torch.from_numpy(np.array(Image.fromarray(image).resize((side, side), Image.BILINEAR))).permute(2, 0, 1)[None]
    return Grader.norm(Grader.prep(x))


def grade(S, T):
    """predicted grade: the number of thresholds that the grade score exceeds"""
    return (np.asarray(S)[..., None] > np.asarray(T)).sum(-1)


# ----------------------------------------------------------------------------- checkpoints
def save(model, path, **meta):
    """weights (without the training-only spread projection) and metadata; values that are not strings are JSON"""
    sd = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items() if not k.startswith("spread_proj.")}
    save_file(sd, str(path), metadata={k: v if isinstance(v, str) else json.dumps(v) for k, v in meta.items()})


def update_meta(path, **meta):
    """rewrite a checkpoint with these metadata entries added or replaced (via a temporary file)"""
    meta = {**read_meta(path), **meta}
    tmp = f"{path}.tmp"
    save_file({k: v.clone() for k, v in load_file(str(path)).items()}, tmp,
              metadata={k: v if isinstance(v, str) else json.dumps(v) for k, v in meta.items()})
    os.replace(tmp, path)


def read_meta(path):
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    out = {}
    for k, v in meta.items():
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def load(path, need_thresholds=True):
    """(model in eval mode on the GPU, metadata); metadata['thresholds'] are T_1..T_4 on the grade-score scale"""
    meta = read_meta(path)
    model = Grader(head=meta.get("head", "coral"))
    model.load_state_dict(load_file(str(path)), strict=True)
    if need_thresholds and "thresholds" not in meta:
        raise SystemExit(f"{path} has no grade thresholds yet: run  uv run python evaluate.py run --checkpoint {path}")
    return model.to(DEV).eval(), meta

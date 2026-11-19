"""The twin network.

One shared tower scores a photo on every attribute; a comparison is the
*difference* of two passes through the same tower:

    P(A beats B on attribute k) = sigmoid( scale_k * (s_k(A) - s_k(B)) )

Because both photos go through identical weights and the verdict is a function
of the score difference, the comparison is antisymmetric by construction
(swap the photos and the answer flips) and every verdict decomposes into
per-attribute gaps that can be explained.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import G_DIM, L_DIM
from .signals import VEC_KEYS

# name, group, where its labels come from
HEADS = [
    # overall
    ("aesthetic",        "overall"),      # AVA votes, AADB score
    ("technical",        "overall"),      # KonIQ MOS, SPAQ MOS, synthetic pairs
    # composition
    ("balance",          "composition"),  # AADB BalancingElements
    ("thirds",           "composition"),  # AADB RuleOfThirds
    ("subject_emphasis", "composition"),  # AADB Object
    ("framing",          "composition"),  # synthetic careless-crop pairs
    ("level",            "composition"),  # synthetic tilt pairs
    ("symmetry",         "style"),        # AADB Symmetry
    ("repetition",       "style"),        # AADB Repetition
    # light & tone
    ("lighting",         "light"),        # AADB Light
    ("exposure",         "light"),        # SPAQ Brightness + synthetic
    ("contrast",         "light"),        # SPAQ Contrast + synthetic
    # colour
    ("color_harmony",    "color"),        # AADB ColorHarmony
    ("vivid_color",      "color"),        # AADB VividColor, SPAQ Colorfulness
    ("color_accuracy",   "color"),        # synthetic WB / saturation pairs
    # focus & depth
    ("depth_of_field",   "focus"),        # AADB DoF
    ("sharpness",        "focus"),        # SPAQ Sharpness + synthetic
    ("motion_blur",      "focus"),        # AADB MotionBlur (higher = less blur)
    # cleanliness / content
    ("noise",            "clean"),        # SPAQ Noisiness + synthetic (higher = cleaner)
    ("content",          "subject"),      # AADB Content
]
HEAD_NAMES = [h[0] for h in HEADS]
HEAD_GROUP = dict(HEADS)
K = len(HEADS)
HIDX = {n: i for i, n in enumerate(HEAD_NAMES)}

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "twin_judge.pt")


class Tower(nn.Module):
    """Scores ONE photo on all K attributes from (global embed, local embeds, measured signals).

    g_dim / l_dim follow the backbone (768 / 512 for the desktop CLIP pair, 512 / 512 for a
    single phone-sized encoder).  h_keys picks which measured signals the tower may use;
    an empty list makes it a pure image-encoder model, which is simpler to port.
    """

    def __init__(self, g_dim: int = G_DIM, l_dim: int = L_DIM, h_keys: list | None = None, width: int = 640, p: float = 0.25):
        super().__init__()
        h_keys = list(VEC_KEYS) if h_keys is None else list(h_keys)
        n_h = len(VEC_KEYS)
        self.g_dim, self.l_dim = g_dim, l_dim
        self.register_buffer("h_mean", torch.zeros(n_h))
        self.register_buffer("h_std", torch.ones(n_h))
        use = torch.zeros(n_h)
        use[[VEC_KEYS.index(k) for k in h_keys]] = 1.0
        self.register_buffer("h_use", use)
        self.g_in = nn.Sequential(nn.Linear(g_dim, width), nn.GELU(), nn.Dropout(p))
        self.l_in = nn.Sequential(nn.Linear(2 * l_dim, 256), nn.GELU(), nn.Dropout(p))
        self.h_in = nn.Sequential(nn.Linear(n_h, 64), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Linear(width + 256 + 64 + 1, width), nn.GELU(), nn.Dropout(p),
            nn.Linear(width, 384), nn.GELU(),
        )
        self.out = nn.Linear(384, K)

    def forward(self, g, l, h, has_l):
        """g (N,g_dim)  l (N,4,l_dim)  h (N,n_h)  has_l (N,) in {0,1}  ->  (N,K)"""
        g = F.normalize(g.float(), dim=-1) * (self.g_dim ** 0.5)
        l = F.normalize(l.float(), dim=-1) * (self.l_dim ** 0.5)
        lf = torch.cat([l.mean(1), l.std(1)], dim=-1)
        hz = ((h.float() - self.h_mean) / self.h_std).clamp(-6, 6) * self.h_use
        m = has_l.float().unsqueeze(-1)
        x = torch.cat([self.g_in(g), self.l_in(lf) * m, self.h_in(hz), m], dim=-1)
        return self.out(self.trunk(x))


class TwinJudge(nn.Module):
    def __init__(self, g_dim: int = G_DIM, l_dim: int = L_DIM, h_keys: list | None = None):
        super().__init__()
        self.tower = Tower(g_dim, l_dim, h_keys)
        self.log_scale = nn.Parameter(torch.zeros(K))          # per-attribute comparison sharpness
        # calibration tables filled in after training
        self.register_buffer("q_table", torch.zeros(K, 101))   # score quantiles over the reference pool
        self.register_buffer("ref_mean", torch.zeros(K))
        self.register_buffer("ref_std", torch.ones(K))

    def score(self, g, l, h, has_l):
        return self.tower(g, l, h, has_l)

    def compare_logits(self, sa, sb):
        return (sa - sb) * self.log_scale.exp()

    def forward(self, a, b):
        """a, b: tuples (g, l, h, has_l).  Returns scores for both and P(a beats b) logits per attribute."""
        sa, sb = self.score(*a), self.score(*b)
        return sa, sb, self.compare_logits(sa, sb)

    # --- calibration helpers (numpy, CPU) ---
    def percentile(self, scores: np.ndarray) -> np.ndarray:
        """Raw scores (...,K) -> 0..100 standing among the reference pool of real photos."""
        q = self.q_table.cpu().numpy()
        out = np.empty_like(scores, dtype=np.float32)
        for k in range(K):
            out[..., k] = np.interp(scores[..., k], q[k], np.arange(101.0))
        return out

    def zscore(self, scores: np.ndarray) -> np.ndarray:
        return (scores - self.ref_mean.cpu().numpy()) / self.ref_std.cpu().numpy()


def load_judge(path: str = CKPT, device: str = "cpu") -> tuple[TwinJudge, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("meta", {}).get("arch", {})          # absent in checkpoints of the original desktop model
    m = TwinJudge(arch.get("g_dim", G_DIM), arch.get("l_dim", L_DIM), arch.get("h_keys"))
    m.load_state_dict(ck["state"], strict=False)       # older checkpoints have no h_use buffer (= use all)
    return m.to(device).eval(), ck.get("meta", {})

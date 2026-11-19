"""Controlled single-fault degradations with KNOWN ground truth.

The old pipeline stacked 4 random effects on every image and then compared two
degraded copies using a hand-written formula as the "label".  Here each variant
has exactly one fault, of known type and severity, and is always paired with
its clean original - so "original beats variant, because of <attribute>" is
true by construction.  These pairs teach the attribute heads that human-rated
datasets do not cover (exposure direction, white balance, level horizon, framing).
"""
from __future__ import annotations

import math
import cv2
import numpy as np

# effect -> attribute head it supervises
EFFECT_HEAD = {
    "underexpose": "exposure", "overexpose": "exposure", "crushed_shadows": "exposure",
    "flat_contrast": "contrast", "harsh_contrast": "contrast",
    "wb_warm": "color_accuracy", "wb_cool": "color_accuracy", "wb_tint": "color_accuracy",
    "desaturate": "color_accuracy", "oversaturate": "color_accuracy",
    "noise": "noise",
    "defocus": "sharpness", "motion_blur": "sharpness",
    "jpeg": None,
    "tilt": "level",
    "bad_crop": "framing",
}
EFFECTS = list(EFFECT_HEAD)
# which overall score the pair also supervises
AESTHETIC_EFFECTS = {"tilt", "bad_crop"}


def _lin(u8):
    return np.power(u8.astype(np.float32) / 255.0, 2.2)


def _srgb(lin):
    return (np.power(np.clip(lin, 0, 1), 1 / 2.2) * 255.0 + 0.5).astype(np.uint8)


def underexpose(rgb, s, rng):
    return _srgb(_lin(rgb) * 2.0 ** -(0.8 + 2.2 * s))


def overexpose(rgb, s, rng):
    return _srgb(_lin(rgb) * 2.0 ** (0.6 + 1.6 * s))


def crushed_shadows(rgb, s, rng):
    f = rgb.astype(np.float32) / 255.0
    cut = 0.08 + 0.22 * s
    return (np.clip((f - cut) / (1 - cut), 0, 1) * 255).astype(np.uint8)


def flat_contrast(rgb, s, rng):
    f = rgb.astype(np.float32) / 255.0
    k = 0.75 - 0.5 * s
    veil = 0.5 + 0.2 * s
    return (np.clip((f - 0.5) * k + veil, 0, 1) * 255).astype(np.uint8)


def harsh_contrast(rgb, s, rng):
    f = rgb.astype(np.float32) / 255.0
    k = 1.5 + 1.8 * s
    return (np.clip((f - 0.5) * k + 0.5, 0, 1) * 255).astype(np.uint8)


def _gains(rgb, gr, gg, gb):
    lin = _lin(rgb) * np.array([gr, gg, gb], np.float32)
    return _srgb(lin)


def wb_warm(rgb, s, rng):
    k = 1.18 + 0.5 * s
    return _gains(rgb, k, 1.0, 1 / k)


def wb_cool(rgb, s, rng):
    k = 1.18 + 0.5 * s
    return _gains(rgb, 1 / k, 1.0, k)


def wb_tint(rgb, s, rng):
    k = 1.15 + 0.4 * s
    return _gains(rgb, 1.0, k, 1.0) if rng.random() < 0.5 else _gains(rgb, k ** 0.5, 1 / k, k ** 0.5)


def _sat(rgb, k):
    f = rgb.astype(np.float32)
    g = f @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.clip(g[..., None] + (f - g[..., None]) * k, 0, 255).astype(np.uint8)


def desaturate(rgb, s, rng):
    return _sat(rgb, 0.6 - 0.5 * s)


def oversaturate(rgb, s, rng):
    return _sat(rgb, 1.7 + 1.6 * s)


def noise(rgb, s, rng):
    lin = _lin(rgb)
    sig = 0.008 + 0.05 * s
    shot = rng.normal(0, 1, lin.shape).astype(np.float32) * np.sqrt(np.maximum(lin, 0) + 0.01) * sig * 2.2
    chroma = cv2.GaussianBlur(rng.normal(0, 1, lin.shape).astype(np.float32), (0, 0), 1.2) * sig * 0.8
    return _srgb(lin + shot + chroma)


def defocus(rgb, s, rng):
    r = 1.2 + 5.0 * s * max(rgb.shape[:2]) / 1024.0
    k = int(2 * math.ceil(r) + 1)
    yy, xx = np.mgrid[-(k // 2):k // 2 + 1, -(k // 2):k // 2 + 1]
    disk = (np.hypot(xx, yy) <= r).astype(np.float32)
    disk /= disk.sum()
    return _srgb(cv2.filter2D(_lin(rgb), -1, disk))


def motion_blur(rgb, s, rng):
    L = int(5 + 28 * s * max(rgb.shape[:2]) / 1024.0) | 1
    ker = np.zeros((L, L), np.float32)
    ker[L // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), rng.uniform(0, 180), 1.0)
    ker = cv2.warpAffine(ker, M, (L, L))
    ker /= ker.sum() + 1e-9
    return _srgb(cv2.filter2D(_lin(rgb), -1, ker))


def jpeg(rgb, s, rng):
    q = int(28 - 22 * s)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def rotate_crop(rgb, deg):
    """Rotate and crop to the largest inscribed axis-aligned rectangle (no borders)."""
    h, w = rgb.shape[:2]
    a = math.radians(abs(deg))
    sa, ca = math.sin(a), math.cos(a)
    side_long, side_short = max(w, h), min(w, h)
    if side_short <= 2.0 * sa * ca * side_long or abs(sa - ca) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sa, x / ca) if w >= h else (x / ca, x / sa)
    else:
        c2 = ca * ca - sa * sa
        wr, hr = (w * ca - h * sa) / c2, (h * ca - w * sa) / c2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    rot = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    x0, y0 = int((w - wr) / 2) + 1, int((h - hr) / 2) + 1
    return rot[y0:h - y0, x0:w - x0]


def tilt(rgb, s, rng):
    return rotate_crop(rgb, (2.5 + 7.5 * s) * (1 if rng.random() < 0.5 else -1))


def bad_crop(rgb, s, rng):
    """Careless framing: keep 40-65% of the frame, shoved into a corner/edge."""
    h, w = rgb.shape[:2]
    keep = 0.8 - 0.18 * s
    ch, cw = int(h * keep), int(w * keep)
    y0 = 0 if rng.random() < 0.5 else h - ch
    x0 = 0 if rng.random() < 0.5 else w - cw
    return rgb[y0:y0 + ch, x0:x0 + cw]


_FN = {k: globals()[k] for k in EFFECTS}


def degrade(rgb: np.ndarray, effect: str, severity: float, rng: np.random.Generator) -> np.ndarray:
    return np.ascontiguousarray(_FN[effect](rgb, float(severity), rng))


def random_variants(rgb: np.ndarray, n: int, rng: np.random.Generator):
    """n distinct single-fault variants: [(effect, severity, image)]."""
    out = []
    for eff in rng.choice(EFFECTS, size=min(n, len(EFFECTS)), replace=False):
        s = float(rng.uniform(0.15, 1.0))
        out.append((str(eff), s, degrade(rgb, str(eff), s, rng)))
    return out

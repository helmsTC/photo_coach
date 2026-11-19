"""Measured (non-learned) photographic signals.

Everything here is a direct measurement on pixels: exposure, clipping, contrast,
colour, sharpness/focus distribution, noise, horizon tilt and visual-weight
placement.  The numbers are used three ways:

  * as extra inputs to the technical heads of the twin network,
  * as hard evidence quoted in the coaching text ("7.9% of the frame is blown out"),
  * as overlays in the UI (clipping zebras, focus map, subject placement).

All functions take an RGB uint8 array of the *standard view* (long side <= 1024).
"""
from __future__ import annotations

import math
import cv2
import numpy as np

STD_LONG_SIDE = 1024
TILE_GRID = 12  # focus / noise maps are TILE_GRID x TILE_GRID

# Fixed-order scalar vector fed to the network.
VEC_KEYS = [
    "luma_mean", "luma_median", "luma_p01", "luma_p05", "luma_p95", "luma_p99",
    "clip_hi", "clip_lo", "near_hi", "near_lo",
    "contrast_rms", "dyn_range", "local_contrast",
    "sat_mean", "colorfulness", "cast_a", "cast_b", "neutral_cast_a", "neutral_cast_b",
    "sharp_top", "sharp_median", "edge_crispness", "focus_spread", "sharp_area",
    "noise_luma", "noise_chroma",
    "log_long_side", "aspect",
]


def to_std(rgb: np.ndarray, long_side: int = STD_LONG_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        rgb = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return rgb


def _luma(rgbf: np.ndarray) -> np.ndarray:
    return rgbf[..., 0] * 0.2126 + rgbf[..., 1] * 0.7152 + rgbf[..., 2] * 0.0722


def _tiles(a: np.ndarray, grid: int = TILE_GRID):
    h, w = a.shape[:2]
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    for i in range(grid):
        for j in range(grid):
            yield i, j, a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]


def tone(rgb: np.ndarray) -> dict:
    f = rgb.astype(np.float32) / 255.0
    y = _luma(f)
    p = np.percentile(y, [1, 5, 50, 95, 99])
    mx = rgb.max(axis=2)
    blur = cv2.GaussianBlur(y, (0, 0), max(2.0, 0.02 * max(y.shape)))
    return {
        "luma_mean": float(y.mean()),
        "luma_median": float(p[2]),
        "luma_p01": float(p[0]), "luma_p05": float(p[1]),
        "luma_p95": float(p[3]), "luma_p99": float(p[4]),
        # blown: any channel pinned at the top; crushed: luma at the floor
        "clip_hi": float((mx >= 252).mean()),
        "clip_lo": float((y <= 0.016).mean()),
        "near_hi": float((y >= 0.92).mean()),
        "near_lo": float((y <= 0.06).mean()),
        "contrast_rms": float(y.std()),
        "dyn_range": float(p[4] - p[0]),
        "local_contrast": float(np.abs(y - blur).mean()),
    }


def color(rgb: np.ndarray) -> dict:
    f = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)  # L 0..100, a/b ~ -127..127
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    a, bb, L = lab[..., 1], lab[..., 2], lab[..., 0]
    chroma = np.hypot(a, bb)
    # "should-be-neutral" pixels: low chroma mid-tones.  Their mean a/b is the cast.
    neutral = (chroma < np.percentile(chroma, 25)) & (L > 20) & (L < 92)
    if neutral.sum() < 50:
        neutral = np.ones_like(L, bool)
    return {
        "sat_mean": float(hsv[..., 1].mean()),
        "colorfulness": float(colorfulness),
        "cast_a": float(a.mean() / 50.0),            # + magenta / - green
        "cast_b": float(bb.mean() / 50.0),           # + warm(yellow) / - cool(blue)
        "neutral_cast_a": float(a[neutral].mean() / 50.0),
        "neutral_cast_b": float(bb[neutral].mean() / 50.0),
    }


def palette(rgb: np.ndarray, k: int = 6) -> list:
    """Dominant colours as [(hex, share)] via k-means in Lab on a thumbnail."""
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(lab, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k).astype(float)
    centers_rgb = cv2.cvtColor(centers.reshape(1, k, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(k, 3)
    order = np.argsort(-counts)
    return [("#%02x%02x%02x" % tuple(int(v) for v in centers_rgb[i]), float(counts[i] / counts.sum())) for i in order]


def focus(rgb: np.ndarray) -> dict:
    """Where is the image sharp, and how sharp?

    Two complementary per-tile measures:
      detail    - strength of the strongest fine-scale edges (after light denoise)
      crispness - fine-scale / coarse-scale edge strength: ~1 for razor edges,
                  falls toward 0.3 as edges get wider (defocus, motion, softness)
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    def grad(sig):
        g = cv2.GaussianBlur(gray, (0, 0), sig)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return np.hypot(gx, gy) / 8.0

    g1, g2 = grad(0.8), grad(2.4)
    n = TILE_GRID
    detail = np.zeros((n, n), np.float32)
    crisp = np.zeros((n, n), np.float32)
    for (i, j, t1), (_, _, t2) in zip(_tiles(g1), _tiles(g2)):
        a = float(np.percentile(t1, 97))
        b = float(np.percentile(t2, 97))
        detail[i, j] = a
        crisp[i, j] = a / (b + 1e-3) if b > 0.004 else 0.0
    flat = np.sort(detail.ravel())
    top = float(flat[int(0.9 * flat.size)])
    med = float(np.median(flat))
    low = float(flat[int(0.2 * flat.size)])
    sharp_mask = detail >= max(0.5 * top, 1e-4)
    top_tiles = detail >= flat[int(0.8 * flat.size)]
    crispness = float(crisp[top_tiles].mean()) if top_tiles.any() else 0.0
    ys, xs = np.mgrid[0:n, 0:n]
    wts = detail * sharp_mask
    tot = float(wts.sum()) + 1e-9
    return {
        "sharp_top": top,
        "sharp_median": med,
        "edge_crispness": crispness,
        "focus_spread": float(math.log((top + 1e-3) / (low + 1e-3))),
        "sharp_area": float(sharp_mask.mean()),
        "sharp_cx": float((wts * (xs + 0.5)).sum() / tot / n),
        "sharp_cy": float((wts * (ys + 0.5)).sum() / tot / n),
        "_detail_map": detail,
        "_crisp_map": crisp,
    }


_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def noise(rgb: np.ndarray) -> dict:
    """Noise sigma estimated in the flattest tiles (Immerkaer), luma and chroma."""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    out = {}
    for name, ch in (("noise_luma", ycc[..., 0]), ("noise_chroma", 0.5 * (ycc[..., 1] + ycc[..., 2]))):
        resp = np.abs(cv2.filter2D(ch, cv2.CV_32F, _IMMERKAER))
        sig = [math.sqrt(math.pi / 2.0) / 6.0 * float(t.mean()) for _, _, t in _tiles(resp) if t.size]
        out[name] = float(np.percentile(sig, 10)) if sig else 0.0
    return out


def horizon(rgb: np.ndarray) -> dict:
    """Angle of the horizon / main waterline in degrees (positive = clockwise tilt).

    A candidate must be a long near-horizontal edge that actually separates two
    different regions (sky|land, land|water): texture such as ripples or rock
    strata produces long lines too, but with the same colour on both sides.
    """
    h0, w0 = rgb.shape[:2]
    s = 512.0 / max(h0, w0)
    small = cv2.resize(rgb, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    h, w = small.shape[:2]
    none = {"tilt_deg": 0.0, "tilt_conf": 0.0, "_horizon_line": None}
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (0, 0), 1.5)
    edges = cv2.Canny(gray, 30, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=40, minLineLength=int(0.25 * w), maxLineGap=int(0.04 * w))
    if lines is None:
        return none
    lab = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32), (0, 0), 2.0)
    band = max(4, int(0.035 * h))
    best = None
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(a) > 10:
            continue
        L = math.hypot(x2 - x1, y2 - y1)
        xs = np.linspace(x1, x2, 40)
        ys = np.linspace(y1, y2, 40)
        up = np.clip(ys - band, 0, h - 1).astype(int)
        dn = np.clip(ys + band, 0, h - 1).astype(int)
        xi = np.clip(xs, 0, w - 1).astype(int)
        if ys.min() - band < 0 or ys.max() + band > h - 1:
            continue
        diff = np.linalg.norm(lab[up, xi] - lab[dn, xi], axis=1)
        dE = float(np.median(diff)) * (100.0 / 255.0)     # 8-bit Lab -> ~CIE units
        if dE < 9:
            continue
        score = (L / w) * min(dE, 40)
        if best is None or score > best[0]:
            best = (score, a, L / w, dE, (float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)))
    if best is None:
        return none
    _, a, frac, dE, seg = best
    conf = min(1.0, frac / 0.55) * min(1.0, dE / 18.0)
    return {"tilt_deg": float(a), "tilt_conf": float(conf), "_horizon_line": seg}


def saliency(rgb: np.ndarray, detail_map: np.ndarray | None = None) -> dict:
    """Visual-weight map: spectral-residual saliency blended with colour contrast
    and the focus map.  Used for subject placement / balance, not for scoring."""
    small = cv2.resize(rgb, (96, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    F = np.fft.fft2(gray)
    la = np.log(np.abs(F) + 1e-8)
    sr = la - cv2.blur(la, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(sr + 1j * np.angle(F)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 4)
    sal /= sal.max() + 1e-9
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    cc = np.linalg.norm(cv2.GaussianBlur(lab, (0, 0), 2) - lab.reshape(-1, 3).mean(0), axis=2)
    cc /= cc.max() + 1e-9
    m = 0.5 * sal + 0.3 * cc
    if detail_map is not None:
        d = cv2.resize(detail_map, (96, 96), interpolation=cv2.INTER_CUBIC)
        m = m + 0.35 * np.clip(d / (d.max() + 1e-9), 0, 1)
    m = np.clip(m - np.percentile(m, 60), 0, None)
    m /= m.sum() + 1e-9
    ys, xs = np.mgrid[0:96, 0:96]
    cx, cy = float((m * (xs + 0.5)).sum() / 96), float((m * (ys + 0.5)).sum() / 96)
    pk = np.unravel_index(np.argmax(cv2.GaussianBlur(m, (0, 0), 5)), m.shape)
    px, py = (pk[1] + 0.5) / 96, (pk[0] + 0.5) / 96
    thirds = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
    d_third = min(math.hypot(px - tx, py - ty) for tx, ty in thirds)
    border = np.ones((96, 96), bool); border[8:-8, 8:-8] = False
    return {
        "weight_cx": cx, "weight_cy": cy,
        "subject_x": float(px), "subject_y": float(py),
        "thirds_dist": float(d_third),
        "center_dist": float(math.hypot(px - 0.5, py - 0.5)),
        "lr_balance": float(m[:, 48:].sum() - m[:, :48].sum()),
        "edge_weight": float(m[border].sum()),
        "_saliency_map": m / (m.max() + 1e-9),
    }


def histogram(rgb: np.ndarray, bins: int = 64) -> dict:
    out = {}
    for i, c in enumerate("rgb"):
        hst = np.bincount((rgb[..., i].ravel().astype(np.int32) * bins) >> 8, minlength=bins).astype(float)
        out[c] = (hst / hst.sum()).round(5).tolist()
    y = (_luma(rgb.astype(np.float32)) * bins / 256.0).astype(np.int32).clip(0, bins - 1)
    hst = np.bincount(y.ravel(), minlength=bins).astype(float)
    out["l"] = (hst / hst.sum()).round(5).tolist()
    return out


def measure(rgb: np.ndarray, full: bool = True) -> dict:
    """All signals for one image. `full=False` skips display-only extras."""
    rgb = to_std(rgb)
    h, w = rgb.shape[:2]
    out = {}
    out.update(tone(rgb))
    out.update(color(rgb))
    out.update(focus(rgb))
    out.update(noise(rgb))
    out["log_long_side"] = math.log(max(h, w) / 1024.0 + 1e-6)
    out["aspect"] = w / h
    if full:
        out.update(horizon(rgb))
        out.update(saliency(rgb, out["_detail_map"]))
        out["palette"] = palette(rgb)
        out["histogram"] = histogram(rgb)
    return out


def vector(m: dict) -> np.ndarray:
    return np.array([m[k] for k in VEC_KEYS], np.float32)

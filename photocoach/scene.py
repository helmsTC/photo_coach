"""Pair-specific analysis: is this the same scene, how do the framings differ,
and what edit would move one photo's tone/colour toward the other's?
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from PIL import Image, ExifTags


# ----------------------------------------------------------------------------
# geometric alignment
# ----------------------------------------------------------------------------
def _gray(rgb, long_side=900):
    h, w = rgb.shape[:2]
    s = min(1.0, long_side / max(h, w))
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if s < 1:
        g = cv2.resize(g, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    # CLAHE makes matching robust to very different exposure / contrast
    return cv2.createCLAHE(2.0, (8, 8)).apply(g), s


def align(rgb_a: np.ndarray, rgb_b: np.ndarray) -> dict:
    """Find a homography mapping A pixels -> B pixels (normalised 0..1 coords)."""
    ga, sa = _gray(rgb_a)
    gb, sb = _gray(rgb_b)
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    res = {"aligned": False, "inliers": 0, "matches": 0}
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return res
    da = np.sqrt(da / (da.sum(1, keepdims=True) + 1e-7))   # RootSIFT
    db = np.sqrt(db / (db.sum(1, keepdims=True) + 1e-7))
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
    res["matches"] = len(good)
    if len(good) < 14:
        return res
    ha, wa = ga.shape; hb, wb = gb.shape
    pa = np.float32([[ka[m.queryIdx].pt[0] / wa, ka[m.queryIdx].pt[1] / ha] for m in good])
    pb = np.float32([[kb[m.trainIdx].pt[0] / wb, kb[m.trainIdx].pt[1] / hb] for m in good])
    # work in pixel-proportional coords so the RANSAC threshold is isotropic
    A = pa * [wa, ha]; B = pb * [wb, hb]
    H, mask = cv2.findHomography(A, B, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
    if H is None:
        return res
    inl = int(mask.sum())
    res["inliers"] = inl
    # normalised-coordinate homography: A(0..1) -> B(0..1)
    Na = np.diag([wa, ha, 1.0]); Nb = np.diag([1 / wb, 1 / hb, 1.0])
    Hn = Nb @ H @ Na
    quad_b_in_a = _project(np.linalg.inv(Hn), [(0, 0), (1, 0), (1, 1), (0, 1)])
    quad_a_in_b = _project(Hn, [(0, 0), (1, 0), (1, 1), (0, 1)])
    ok = inl >= 18 and inl / max(len(good), 1) >= 0.25 and _sane_quad(quad_b_in_a, wa / ha) and _sane_quad(quad_a_in_b, wb / hb)
    if not ok:
        return res
    m = mask.ravel().astype(bool)
    spread = float(np.prod(pa[m].max(0) - pa[m].min(0)))   # inliers must cover a real area, not one landmark
    if spread < 0.06:
        return res
    res.update(aligned=True, H=Hn, quad_b_in_a=quad_b_in_a, quad_a_in_b=quad_a_in_b,
               pts_a=pa[m][:150].tolist(), pts_b=pb[m][:150].tolist(), spread=spread)
    return res


def _project(H, pts):
    p = np.float64([[x, y, 1.0] for x, y in pts]) @ H.T
    return (p[:, :2] / p[:, 2:3]).tolist()


def _sane_quad(q, aspect):
    q = np.float64(q) * [aspect, 1.0]
    cross = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cross.append(np.cross(b - a, c - b))
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        return False
    area = 0.5 * abs(sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4))) / aspect
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    return 0.04 < area < 25 and max(sides) / (min(sides) + 1e-9) < 6


def framing(quad_other_in_me: list, my_aspect: float) -> dict:
    """Describe where the OTHER photo's frame sits inside MY frame.
    quad: other frame's corners (TL,TR,BR,BL) in my normalised coords."""
    q = np.float64(quad_other_in_me)
    area = 0.5 * abs(sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4)))
    cx, cy = q.mean(0)
    top = (q[1] - q[0]) * [my_aspect, 1.0]
    rot = math.degrees(math.atan2(top[1], top[0]))
    inside = float(np.mean((q >= -0.03) & (q <= 1.03)))
    return {
        "area_frac": float(area),            # share of my frame the other photo covers (<1: other is tighter)
        "zoom": float(1 / math.sqrt(area)),  # >1: other is zoomed in relative to me
        "dx": float(cx - 0.5), "dy": float(cy - 0.5),
        "rotation": float(rot),
        "inside": inside,
        "left": float(q[:, 0].min()), "right": float(q[:, 0].max()),
        "top": float(q[:, 1].min()), "bottom": float(q[:, 1].max()),
    }


def crop_to_other(rgb_me: np.ndarray, H_me_to_other: np.ndarray, other_shape, max_side=1024) -> np.ndarray:
    """Re-frame my photo the way the other one is framed (valid when the other frame lies inside mine)."""
    h, w = rgb_me.shape[:2]
    oh, ow = other_shape[:2]
    s = min(1.0, max_side / max(oh, ow))
    ow, oh = round(ow * s), round(oh * s)
    M = np.diag([ow, oh, 1.0]) @ H_me_to_other @ np.diag([1 / w, 1 / h, 1.0])
    return cv2.warpPerspective(rgb_me, M, (ow, oh), flags=cv2.INTER_AREA if ow < w else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def overlap_masks(shape_a, shape_b, H) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of the region each photo shares with the other."""
    def mask(shape_me, shape_other, Hmo):
        h, w = shape_me[:2]
        ys, xs = np.mgrid[0:h:4, 0:w:4]
        p = np.stack([(xs + 0.5) / w, (ys + 0.5) / h, np.ones_like(xs, float)], -1) @ Hmo.T
        u, v = p[..., 0] / p[..., 2], p[..., 1] / p[..., 2]
        m = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & (p[..., 2] > 0)
        return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask(shape_a, shape_b, H), mask(shape_b, shape_a, np.linalg.inv(H))


# ----------------------------------------------------------------------------
# "match the look": an explainable global edit from one photo toward another
# ----------------------------------------------------------------------------
_Q = [0.5, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99.5]


def _lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _Y(L):   # CIE L* -> relative luminance
    L = np.asarray(L, float)
    return np.where(L > 8, ((L + 16) / 116) ** 3, L / 903.3)


def look_recipe(src: np.ndarray, ref: np.ndarray, src_mask=None, ref_mask=None, color_ok: bool = True) -> dict:
    """Measure the tone/colour differences src -> ref as editing-slider style numbers."""
    ls, lr = _lab(src), _lab(ref)
    ms = src_mask if src_mask is not None and src_mask.mean() > 0.1 else np.ones(src.shape[:2], bool)
    mr = ref_mask if ref_mask is not None and ref_mask.mean() > 0.1 else np.ones(ref.shape[:2], bool)
    qs = np.percentile(ls[..., 0][ms], _Q)
    qr = np.percentile(lr[..., 0][mr], _Q)
    cs = np.hypot(ls[..., 1], ls[..., 2])[ms]
    cr = np.hypot(lr[..., 1], lr[..., 2])[mr]
    sat = float(np.clip((cr.mean() + 2.0) / (cs.mean() + 2.0), 0.5, 2.2))
    ev = float(np.clip(math.log2((_Y(qr[5]) + 0.01) / (_Y(qs[5]) + 0.01)), -3, 3))
    iqr_s, iqr_r = qs[6] - qs[4], qr[6] - qr[4]
    rec = {
        "exposure_ev": ev,
        "contrast_pct": float(np.clip((iqr_r + 4) / (iqr_s + 4) - 1, -0.6, 1.5) * 100),
        # highlight / shadow placement relative to the mid-tone (L* units)
        "highlights": float((qr[8] - qr[5]) - (qs[8] - qs[5])),
        "shadows": float((qs[5] - qs[2]) - (qr[5] - qr[2])),
        "blacks": float(qr[0] - qs[0]),
        "whites": float(qr[10] - qs[10]),
        "saturation_pct": (sat - 1) * 100,
        "temp": float(lr[..., 2][mr].mean() - ls[..., 2][ms].mean()) if color_ok else 0.0,   # + warmer
        "tint": float(lr[..., 1][mr].mean() - ls[..., 1][ms].mean()) if color_ok else 0.0,   # + magenta
        "_qs": qs.tolist(), "_qr": qr.tolist(), "_sat": sat,
        "color_ok": color_ok,
    }
    return rec


def apply_look(src: np.ndarray, rec: dict, strength: float = 0.85) -> np.ndarray:
    """Apply the recipe as a smooth global tone curve + white-balance shift + saturation scale."""
    lab = _lab(src)
    qs, qr = np.array(rec["_qs"]), np.array(rec["_qr"])
    tgt = qs + strength * (qr - qs)
    xs = np.concatenate([[0.0], qs, [100.0]])
    ys = np.concatenate([[max(0.0, tgt[0] - qs[0]) * 0.5], tgt, [min(100.0, 100.0 + (tgt[-1] - qs[-1]) * 0.5)]])
    xs, idx = np.unique(xs, return_index=True)
    ys = np.maximum.accumulate(ys[idx])
    grid = np.linspace(0, 100, 256)
    lut = np.interp(grid, xs, ys)
    lut = np.convolve(np.pad(lut, 8, mode="edge"), np.ones(17) / 17, mode="valid")   # no kinks / banding
    L = np.interp(lab[..., 0], grid, lut)
    sat = 1 + strength * (rec["_sat"] - 1)
    a = lab[..., 1] * sat + strength * rec["tint"]
    b = lab[..., 2] * sat + strength * rec["temp"]
    out = cv2.cvtColor(np.stack([L, a, b], -1).astype(np.float32), cv2.COLOR_LAB2RGB)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------
# EXIF
# ----------------------------------------------------------------------------
def exif(pil: Image.Image) -> dict:
    out = {}
    try:
        ex = pil.getexif()
        if not ex:
            return out
        base = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
        sub = {ExifTags.TAGS.get(k, k): v for k, v in ex.get_ifd(0x8769).items()}
        d = {**base, **sub}

        def f(x):
            try:
                return float(x)
            except Exception:
                return None
        cam = " ".join(str(d[k]).strip() for k in ("Make", "Model") if d.get(k))
        if cam: out["camera"] = cam
        if d.get("LensModel"): out["lens"] = str(d["LensModel"]).strip()
        if f(d.get("FNumber")): out["aperture"] = round(f(d["FNumber"]), 1)
        if f(d.get("ExposureTime")): out["shutter"] = f(d["ExposureTime"])
        iso = d.get("ISOSpeedRatings") or d.get("PhotographicSensitivity")
        if isinstance(iso, (tuple, list)): iso = iso[0]
        if f(iso): out["iso"] = int(f(iso))
        if f(d.get("FocalLength")): out["focal"] = round(f(d["FocalLength"]), 1)
        if f(d.get("FocalLengthIn35mmFilm")): out["focal35"] = int(f(d["FocalLengthIn35mmFilm"]))
        if f(d.get("ExposureBiasValue")) is not None and d.get("ExposureBiasValue") is not None: out["ev_comp"] = round(f(d["ExposureBiasValue"]), 1)
        if d.get("DateTimeOriginal"): out["taken"] = str(d["DateTimeOriginal"])
        if d.get("Flash") is not None:
            try: out["flash"] = bool(int(d["Flash"]) & 1)
            except Exception: pass
    except Exception:
        pass
    return out


def shutter_str(t: float) -> str:
    return f"1/{round(1 / t)}s" if t < 0.4 else f"{t:.1f}s"

"""What is in the frame: panoptic segmentation -> objects, subject, and composition geometry.

A panoptic model (Mask2Former, COCO: 80 countable 'thing' classes + 53 background 'stuff'
classes) gives one mask per object and per background region.  From the masks we pick the
main subject and measure where it sits, how big it is, whether the frame cuts it off and how
well it separates from its surroundings.  The twin network then scores the photo with each
object removed (see Engine.object_effects) to say which ones help and which distract.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch

SEG_MODEL = "facebook/mask2former-swin-small-coco-panoptic"
SHORT_SIDE = 640
MAX_LONG = 1152
N_THINGS = 80                      # COCO panoptic: label ids below this are countable objects

# how readily a class is "the subject" when nothing else decides it
PRIOR = {"person": 1.6, "bird": 1.4, "cat": 1.4, "dog": 1.4, "horse": 1.4, "sheep": 1.3, "cow": 1.3, "elephant": 1.4,
         "bear": 1.4, "zebra": 1.4, "giraffe": 1.4, "boat": 1.2, "house": 1.1, "bridge": 1.1, "tent": 1.1,
         "flower": 1.1, "airplane": 1.2, "train": 1.2, "bicycle": 1.1, "motorcycle": 1.1, "car": 1.0,
         # surfaces and furniture are usually what the subject sits on, not the subject
         "dining table": 0.45, "bed": 0.5, "couch": 0.55, "chair": 0.6, "bench": 0.6, "tv": 0.7}
# background classes that can still be a subject (a cabin, a bridge) rather than plain surroundings
STUFF_SUBJECTS = {"house", "bridge", "tent", "flower", "tower", "building", "lighthouse"}
RENAME = {"sky-other": "sky", "water-other": "water", "tree": "trees", "rock": "rocks", "dirt": "ground",
          "pavement": "pavement", "grass": "grass", "floor-other": "floor", "wall-other": "wall", "food-other": "food",
          "building-other": "building", "fence": "fence", "paper": "paper", "cabinet": "cabinet", "table": "table",
          "rug": "rug", "ceiling": "ceiling", "sea": "sea", "river": "water", "potted plant": "plant"}
WATER = {"water", "sea"}


def _name(raw: str) -> str:
    n = raw.replace("-merged", "").replace("-stuff", "")
    return RENAME.get(n, n.replace("-other", ""))


class Segmenter:
    def __init__(self, device: str | None = None):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(SEG_MODEL, use_fast=False)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_MODEL).to(self.device).eval()
        self.model.requires_grad_(False)
        self.id2label = self.model.config.id2label
        self.fuse = {i for i in self.id2label if i >= N_THINGS}

    @torch.inference_mode()
    def segment(self, rgb: np.ndarray) -> list:
        """rgb uint8 HxWx3 -> list of segments, largest first.  Masks are bool arrays at rgb's size."""
        h, w = rgb.shape[:2]
        s = min(SHORT_SIDE / min(h, w), MAX_LONG / max(h, w))
        tw, th = max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)
        im = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        x = self.proc(images=im, return_tensors="pt", do_resize=False).to(self.device)
        out = self.model(**x)
        res = self.proc.post_process_panoptic_segmentation(out, target_sizes=[(th, tw)], label_ids_to_fuse=self.fuse)[0]
        idmap = cv2.resize(res["segmentation"].cpu().numpy().astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        segs, seen = [], set()
        for info in res["segments_info"]:
            if info["id"] in seen:          # fused background classes are listed once per fused piece
                continue
            seen.add(info["id"])
            m = idmap == info["id"]
            area = float(m.mean())
            if area < 0.0008:
                continue
            segs.append(_describe(m, _name(self.id2label[info["label_id"]]), info["label_id"] < N_THINGS, float(info.get("score", 1.0))))
        segs.sort(key=lambda g: -g["area"])
        return segs


def _describe(m: np.ndarray, label: str, thing: bool, score: float) -> dict:
    h, w = m.shape
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = max(2, round(0.012 * w)), max(2, round(0.012 * h))
    # share of each frame edge that the object lies against
    on = dict(left=float(m[:, :bw].any(1).mean()), right=float(m[:, -bw:].any(1).mean()),
              top=float(m[:bh].any(0).mean()), bottom=float(m[-bh:].any(0).mean()))
    return dict(label=label, thing=thing, score=score, mask=m, area=float(m.mean()),
                cx=float(xs.mean() / w), cy=float(ys.mean() / h),
                box=[float(x0 / w), float(y0 / h), float(x1 / w), float(y1 / h)], edges=on)


def where(g: dict) -> str:
    """'lower left', 'centre', 'right edge' ..."""
    e = g["edges"]
    if g["area"] < 0.25:
        for side in ("left", "right", "top", "bottom"):
            if e[side] > 0.02 and (g["box"][2] - g["box"][0] < 0.5 or side in ("top", "bottom")) and (g["box"][3] - g["box"][1] < 0.5 or side in ("left", "right")):
                return f"{side} edge"
    v = "upper" if g["cy"] < 0.38 else ("lower" if g["cy"] > 0.62 else "")
    hz = "left" if g["cx"] < 0.38 else ("right" if g["cx"] > 0.62 else "")
    return (f"{v} {hz}".strip()) or "centre"


def cut_off(g: dict) -> list:
    """Frame edges that slice through a compact object (not background that naturally runs off-frame)."""
    if not g["thing"] and g["label"] not in STUFF_SUBJECTS:
        return []
    bw, bh = g["box"][2] - g["box"][0], g["box"][3] - g["box"][1]
    out = []
    for side, span in (("left", bh), ("right", bh), ("top", bw), ("bottom", bw)):
        # the contact has to be a real slice: a good part of the object's own extent along that edge
        extent = g["edges"][side]
        if extent >= 0.25 * max(span, 1e-6) and extent >= 0.03:
            out.append(side)
    return out


def pick_subject(segs: list, rgb_shape, sal_xy=None) -> int | None:
    """Index of the main subject, or None for a scene with no single object (open landscape)."""
    best, best_s = None, 0.0
    for i, g in enumerate(segs):
        cand = g["thing"] or g["label"] in STUFF_SUBJECTS
        if not cand or g["area"] < 0.008 or g["area"] > 0.7:
            continue
        central = 1.0 - min(1.0, math.hypot(g["cx"] - 0.5, g["cy"] - 0.5) / 0.6)
        s = math.sqrt(g["area"]) * (0.45 + central) * PRIOR.get(g["label"], 0.9 if g["thing"] else 0.8)
        if sal_xy is not None and g["box"][0] <= sal_xy[0] <= g["box"][2] and g["box"][1] <= sal_xy[1] <= g["box"][3]:
            s *= 1.3                                     # the strongest visual pull lands on it
        s *= 1.0 + 2.0 * max(0.0, g.get("lean", 0.0))    # the network's score depends on it
        if s > best_s:
            best, best_s = i, s
    return best if best_s >= 0.06 else None


def _thirds_dist(cx, cy):
    return min(math.hypot(cx - px, cy - py) for px in (1 / 3, 2 / 3) for py in (1 / 3, 2 / 3))


def geometry(rgb: np.ndarray, segs: list, subj: int | None, detail_map: np.ndarray | None = None) -> dict:
    """Composition facts read off the masks."""
    h, w = rgb.shape[:2]
    out = dict(subject=None, sky=0.0, water=0.0, horizon_y=None, n_things=sum(g["thing"] for g in segs))
    for g in segs:
        if g["label"] == "sky":
            out["sky"] += g["area"]
        if g["label"] in WATER:
            out["water"] += g["area"]
    sky = next((g for g in segs if g["label"] == "sky"), None)
    if sky is not None and sky["area"] >= 0.04:
        cols = sky["mask"].any(0)
        if cols.mean() >= 0.6:
            low = h - 1 - np.argmax(sky["mask"][::-1], axis=0)     # lowest sky pixel per column
            out["horizon_y"] = float(np.median(low[cols]) / h)     # where sky meets land, as a fraction of height
    if subj is None:
        return out
    g = segs[subj]
    m = g["mask"]
    small = cv2.resize(rgb, (max(64, w // 4), max(64, h // 4)), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    sep = None
    if ms.sum() >= 12:
        r = max(3, round(0.35 * math.sqrt(ms.sum())))
        ring = (cv2.dilate(ms.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0) & ~ms
        if ring.sum() >= 12:
            lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[..., 0] *= 100.0 / 255.0
            lab[..., 1:] -= 128.0
            sep = float(np.linalg.norm(lab[ms].mean(0) - lab[ring].mean(0)))     # CIE delta-E, subject vs surroundings
    sharp = None
    if detail_map is not None:
        d = cv2.resize(detail_map, (w, h), interpolation=cv2.INTER_LINEAR)
        if (~m).any():
            sharp = float(np.median(d[m]) / (np.median(d[~m]) + 1e-6))           # >1: subject crisper than the rest
    out["subject"] = dict(index=subj, label=g["label"], area=g["area"], cx=g["cx"], cy=g["cy"], where=where(g),
                          thirds_dist=float(_thirds_dist(g["cx"], g["cy"])), center_dist=float(math.hypot(g["cx"] - 0.5, g["cy"] - 0.5)),
                          cut=cut_off(g), separation=sep, sharp_ratio=sharp,
                          height=g["box"][3] - g["box"][1], width=g["box"][2] - g["box"][0])
    return out


def remove(rgb: np.ndarray, m: np.ndarray, area: float) -> np.ndarray:
    """The photo without this region.  Compact objects are painted out from their surroundings;
    large regions cannot be invented, so they are replaced by a heavy blur of themselves."""
    h, w = rgb.shape[:2]
    if area <= 0.12:
        s = 384 / max(h, w)
        sm = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        k = max(3, round(0.012 * 384)) * 2 + 1
        mm = cv2.dilate(cv2.resize(m.astype(np.uint8), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        fill = cv2.inpaint(sm, mm, 5, cv2.INPAINT_TELEA)
        fill = cv2.resize(fill, (w, h), interpolation=cv2.INTER_CUBIC)
        a = cv2.resize(mm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        a = cv2.GaussianBlur(a, (0, 0), 0.006 * max(h, w))[..., None]
    else:
        q = cv2.resize(rgb, (max(8, w // 6), max(8, h // 6)), interpolation=cv2.INTER_AREA)      # blur small: same look, far faster
        fill = cv2.resize(cv2.GaussianBlur(q, (0, 0), 0.01 * max(h, w)), (w, h), interpolation=cv2.INTER_CUBIC)
        a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.008 * max(h, w))[..., None]
    return np.clip(rgb * (1 - a) + fill * a, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# overlay
# ----------------------------------------------------------------------------
ROLE_COLOR = {"subject": (255, 200, 40), "distraction": (255, 80, 60), "helps": (40, 210, 150), "other": (235, 235, 235)}


def overlay(rgb: np.ndarray, segs: list) -> np.ndarray:
    """Numbered outlines: yellow = subject, red = the score rises without it, green = the score leans on it."""
    g_ = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = g_[..., None] * 0.35 + rgb.astype(np.float32) * 0.4
    h, w = rgb.shape[:2]
    t = max(1, round(max(h, w) / 420))
    for g in segs:
        role = g.get("role", "other")
        if role in ("distraction", "subject") or (role == "helps" and g["area"] <= 0.12):   # big regions: outline only
            c = np.array(ROLE_COLOR[role], np.float32)
            out[g["mask"]] = out[g["mask"]] * 0.62 + c * 0.38
    out = np.clip(out, 0, 255).astype(np.uint8)
    for g in sorted(segs, key=lambda g: g.get("role", "other") != "other"):          # important outlines on top
        role = g.get("role", "other")
        cs, _ = cv2.findContours(g["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = [c for c in cs if cv2.contourArea(c) >= 0.0004 * h * w]
        if role != "other":
            cv2.drawContours(out, cs, -1, (0, 0, 0), t + 2, cv2.LINE_AA)
        cv2.drawContours(out, cs, -1, ROLE_COLOR[role], t if role == "other" else t + 1, cv2.LINE_AA)
    fs = max(0.45, max(h, w) / 1500)
    for g in segs:
        if "n" not in g:
            continue
        txt = str(g["n"])
        # label at the point deepest inside the mask so it sits on the object
        dist = cv2.distanceTransform(g["mask"].astype(np.uint8), cv2.DIST_L2, 3)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, t + 1)
        x0, y0 = int(np.clip(x - tw / 2, 3, w - tw - 3)), int(np.clip(y + th / 2, th + 3, h - 3))
        cv2.rectangle(out, (x0 - 4, y0 - th - 4), (x0 + tw + 4, y0 + 5), (18, 18, 18), -1)
        cv2.putText(out, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fs, ROLE_COLOR[g.get("role", "other")], t + 1, cv2.LINE_AA)
    return out


def assign_roles(segs: list, subj: int | None) -> None:
    """subject / distraction / helps / other.  A distraction is a compact object that is not the
    subject and whose removal raises the aesthetic score clearly above the re-scoring noise.
    Large regions are only ever blurred, which says how much the score leans on them, not
    whether the photo would be better without them - so they are never called distractions."""
    for i, g in enumerate(segs):
        d, lean = g.get("delta", 0.0), g.get("lean", 0.0)
        if i == subj:
            g["role"] = "subject"
        elif g["area"] <= 0.12 and d >= 3.0 and lean <= -0.10:
            g["role"] = "distraction"
        elif lean >= 0.30:
            g["role"] = "helps"
        else:
            g["role"] = "other"


def public(g: dict) -> dict:
    return dict(n=g.get("n"), label=g["label"], thing=g["thing"], area=round(g["area"], 4), where=where(g),
                delta=round(g.get("delta", 0.0), 1), role=g.get("role", "other"), cut=cut_off(g), compact=g["area"] <= 0.12)

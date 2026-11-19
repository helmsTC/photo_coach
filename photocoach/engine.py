"""End-to-end comparison: two photos in, verdict + evidence + coaching out."""
from __future__ import annotations

import io
import math
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import coach, objects, scene, signals
from .backbone import Backbone, global_view, load_image, local_views
from .model import HEAD_NAMES, HIDX, K, load_judge
from .synth import EFFECT_HEAD

W_AES, W_TECH = 0.6, 0.4          # how the two overall scores combine into the verdict
MIN_RELIABLE_SRCC = 0.30          # attribute heads below this on held-out data are shown but not used for advice


class Photo:
    def __init__(self, pil: Image.Image, name: str):
        self.name = name
        self.exif = scene.exif(pil)
        self.full_size = pil.size
        self.rgb = signals.to_std(np.asarray(pil))
        self.sig = signals.measure(self.rgb)
        self.h = torch.tensor(signals.vector(self.sig))
        self.g = self.l = self.raw = self.pct = self.z = None


class Engine:
    def __init__(self, device: str | None = None):
        self.bb = Backbone(device)
        self.dev = self.bb.device
        self.judge, self.meta = load_judge(device=self.dev)
        self.lock = threading.Lock()
        self.seg = None                # panoptic model, loaded on first deep comparison
        self.reliable = self._reliability()
        # if the technical head has no training data yet, judge on aesthetics alone
        self.w_aes, self.w_tech = (W_AES, W_TECH) if self.reliable['technical']['ok'] else (1.0, 0.0)

    # ------------------------------------------------------------------
    def _reliability(self) -> dict:
        rep = self.meta.get("report", {})
        best = {}
        for k, v in rep.items():
            if k == "synth":
                continue
            head = k.split("/")[1]
            best[head] = max(best.get(head, -1), v["srcc"])
        syn = rep.get("synth", {})
        for eff, v in syn.items():
            head = EFFECT_HEAD.get(eff)
            if head and head not in best and v.get("attr_acc") is not None:
                best.setdefault("_syn_" + head, []).append(v["attr_acc"])
        out = {}
        for h in HEAD_NAMES:
            if h in best:
                out[h] = dict(metric="srcc", value=round(best[h], 3), ok=best[h] >= MIN_RELIABLE_SRCC)
            elif "_syn_" + h in best:
                a = float(np.mean(best["_syn_" + h]))
                out[h] = dict(metric="pair_acc", value=round(a, 3), ok=a >= 0.75)
            else:
                out[h] = dict(metric=None, value=None, ok=False)   # never validated -> never used for advice
        return out

    # ------------------------------------------------------------------
    def _score(self, g, l, h):
        with torch.inference_mode():
            n = g.shape[0]
            s = self.judge.score(g.to(self.dev), l.to(self.dev), h.to(self.dev), torch.ones(n, device=self.dev))
        return s.cpu().numpy()

    def featurize(self, p: Photo):
        p.g, p.l = self.bb.embed_photo(p.rgb)
        p.raw = self._score(p.g[None], p.l[None], p.h[None])[0]
        p.pct = self.judge.percentile(p.raw)
        p.z = self.judge.zscore(p.raw)

    def overall(self, p: Photo) -> float:
        return float(self.w_aes * p.pct[HIDX["aesthetic"]] + self.w_tech * p.pct[HIDX["technical"]])

    def win_prob(self, a: Photo, b: Photo) -> dict:
        sc = self.judge.log_scale.detach().exp().cpu().numpy()
        logit = (a.raw - b.raw) * sc
        v = self.w_aes * logit[HIDX["aesthetic"]] + self.w_tech * logit[HIDX["technical"]]
        return dict(overall=float(1 / (1 + math.exp(-v))), per_head=(1 / (1 + np.exp(-logit))).tolist())

    # ------------------------------------------------------------------
    def occlusion(self, p: Photo, head: str = "aesthetic", n: int = 5, win: float = 0.36) -> np.ndarray:
        """How much does each region carry the score?  Slide a window that replaces the
        region with a heavy blur; + = score drops without it (it helps), - = it hurts."""
        gv = global_view(p.rgb)
        H, W = gv.shape[:2]
        blur = cv2.GaussianBlur(gv, (0, 0), 0.09 * max(H, W))
        wh, ww = max(14, round(win * H)), max(14, round(win * W))
        ys = np.linspace(0, H - wh, n).round().astype(int)
        xs = np.linspace(0, W - ww, n).round().astype(int)
        batch, boxes = [], []
        for y in ys:
            for x in xs:
                im = gv.copy()
                im[y:y + wh, x:x + ww] = blur[y:y + wh, x:x + ww]
                batch.append(im); boxes.append((y, x))
        g = self.bb.embed_global(np.stack(batch))
        m = len(batch)
        s = self._score(g, p.l[None].expand(m, -1, -1), p.h[None].expand(m, -1))[:, HIDX[head]]
        d = (p.raw[HIDX[head]] - s) / float(self.judge.ref_std[HIDX[head]])
        heat = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
        for (y, x), v in zip(boxes, d):
            heat[y:y + wh, x:x + ww] += v; cnt[y:y + wh, x:x + ww] += 1
        return cv2.GaussianBlur(heat / np.maximum(cnt, 1), (0, 0), 0.04 * max(H, W))

    # ------------------------------------------------------------------
    def object_effects(self, p: Photo, max_objects: int = 10) -> list:
        """Find what is in the frame, then re-score the photo with each object removed.
        seg['delta'] = change in the aesthetic percentile without it (+ = the photo scores
        higher without it, i.e. it distracts; - = the score leans on it)."""
        if self.seg is None:
            self.seg = objects.Segmenter(self.dev)
        segs = self.seg.segment(p.rgb)
        cand = [g for g in segs if (g["area"] >= 0.002 if g["thing"] else g["area"] >= 0.015)]
        cand = sorted(cand, key=lambda g: -(g["area"] * (3.0 if g["thing"] else 1.0)))[:max_objects]
        if cand:
            mods = [objects.remove(p.rgb, g["mask"], g["area"]) for g in cand]
            gg = self.bb.embed_global(np.stack([global_view(m) for m in mods]))
            ll = self.bb.embed_local(np.concatenate([local_views(m) for m in mods])).view(len(mods), 4, -1)
            raw = self._score(gg, ll, p.h[None].expand(len(mods), -1))
            i = HIDX["aesthetic"]
            for g, r in zip(cand, raw):
                g["delta"] = float(self.judge.percentile(r)[i] - p.pct[i])
                g["lean"] = float((p.raw[i] - r[i]) / float(self.judge.ref_std[i]))
        cand.sort(key=lambda g: -g["area"])
        for n, g in enumerate(cand, 1):
            g["n"] = n
        return cand

    # ------------------------------------------------------------------
    def compare(self, src_a, src_b, name_a="Photo A", name_b="Photo B", deep: bool = True) -> tuple[dict, dict]:
        """Returns (json-able result, {image_name: jpeg bytes})."""
        with self.lock:
            return self._compare(src_a, src_b, name_a, name_b, deep)

    def _compare(self, src_a, src_b, name_a, name_b, deep):
        P = {"a": Photo(load_image(src_a), name_a), "b": Photo(load_image(src_b), name_b)}
        for p in P.values():
            self.featurize(p)
        a, b = P["a"], P["b"]
        wp = self.win_prob(a, b)
        win = "a" if wp["overall"] >= 0.5 else "b"
        lose = "b" if win == "a" else "a"
        conf = max(wp["overall"], 1 - wp["overall"])
        Wn, Ln = P[win], P[lose]

        # ---- relation between the two photos ----
        sim = float(F.cosine_similarity(a.g[None], b.g[None]).item())
        al = scene.align(a.rgb, b.rgb)
        relation = "aligned" if al["aligned"] else ("similar" if sim >= 0.80 else "different")
        frame = None
        masks = {"a": None, "b": None}
        if al["aligned"]:
            ma, mb = scene.overlap_masks(a.rgb.shape, b.rgb.shape, al["H"])
            masks = {"a": ma, "b": mb}
            quad = al["quad_b_in_a"] if lose == "a" else al["quad_a_in_b"]   # winner's frame inside the loser's
            frame = scene.framing(quad, Ln.rgb.shape[1] / Ln.rgb.shape[0])
            frame["quad"] = quad

        # ---- "edit toward the winner" previews, scored by the same network ----
        recipe = scene.look_recipe(Ln.rgb, Wn.rgb, masks[lose], masks[win], color_ok=relation != "different")
        previews = {}
        edited = scene.apply_look(Ln.rgb, recipe)
        pe = Photo(Image.fromarray(edited), "edit"); self.featurize(pe)
        previews["edit"] = pe
        if frame and frame["zoom"] >= 1.08 and frame["inside"] >= 0.85:
            Hlw = al["H"] if lose == "a" else np.linalg.inv(al["H"])
            crop = scene.crop_to_other(Ln.rgb, Hlw, Wn.rgb.shape)
            pc = Photo(Image.fromarray(scene.apply_look(crop, scene.look_recipe(crop, Wn.rgb, color_ok=True))), "crop+edit")
            self.featurize(pc)
            previews["crop_edit"] = pc
        o_w, o_l = self.overall(Wn), self.overall(Ln)

        def closure(p):
            return float(np.clip((self.overall(p) - o_l) / max(o_w - o_l, 1e-6), -1, 1.5))

        best_key = max(previews, key=lambda k: self.overall(previews[k]))
        prev_info = {k: dict(overall=round(self.overall(p), 1), closure=round(closure(p), 3),
                             aesthetic=round(float(p.pct[HIDX["aesthetic"]]), 1), technical=round(float(p.pct[HIDX["technical"]]), 1))
                     for k, p in previews.items()}

        # ---- coaching ----
        pctd = {k: {h: float(p.pct[i]) for i, h in enumerate(HEAD_NAMES)} for k, p in P.items()}
        zd = {k: {h: float(p.z[i]) for i, h in enumerate(HEAD_NAMES)} for k, p in P.items()}
        ctx = dict(names={"a": a.name, "b": b.name}, winner=win, loser=lose, pct=pctd, z=zd,
                   sig={k: p.sig for k, p in P.items()}, exif={k: p.exif for k, p in P.items()},
                   recipe=recipe, frame=frame, relation=relation,
                   reliable={h: v["ok"] for h, v in self.reliable.items()})
        # ---- what is in each frame, and what each object does to the score ----
        obj = {}
        if deep:
            for k, p in P.items():
                segs = self.object_effects(p)
                si = objects.pick_subject(segs, p.rgb.shape, (p.sig["subject_x"], p.sig["subject_y"]))
                objects.assign_roles(segs, si)
                obj[k] = dict(segs=segs, geo=objects.geometry(p.rgb, segs, si, p.sig["_detail_map"]))
        ctx["objects"] = obj
        advice = coach.build(ctx)

        # ---- images ----
        images = {}
        occ = {k: (self.occlusion(p) if deep else None) for k, p in P.items()}
        occ_scale = None
        det_scale = max(float(p.sig["_detail_map"].max()) for p in P.values())
        for k, p in P.items():
            images[f"{k}_photo"] = _jpg(p.rgb)
            images[f"{k}_clip"] = _jpg(_ov_clip(p.rgb))
            images[f"{k}_focus"] = _jpg(_ov_focus(p.rgb, p.sig["_detail_map"], det_scale))
            images[f"{k}_comp"] = _jpg(_ov_comp(p.rgb, p.sig))
            if occ[k] is not None:
                images[f"{k}_impact"] = _jpg(_ov_impact(p.rgb, occ[k], occ_scale))
        for k, o in obj.items():
            if o["segs"]:
                images[f"{k}_objects"] = _jpg(objects.overlay(P[k].rgb, o["segs"]))
        if frame:
            images[f"{lose}_frame"] = _jpg(_ov_frame(Ln.rgb, frame["quad"]))
            other_quad = al["quad_a_in_b"] if lose == "a" else al["quad_b_in_a"]
            images[f"{win}_frame"] = _jpg(_ov_frame(Wn.rgb, other_quad, color=(255, 255, 255)))
        for k, p in previews.items():
            images[f"preview_{k}"] = _jpg(p.rgb)

        # ---- assemble ----
        attrs = []
        for gname, heads in coach.GROUPS:
            for h in heads:
                i = HIDX[h]
                attrs.append(dict(key=h, group=gname, label=coach.LABEL[h], blurb=coach.BLURB[h],
                                  a=round(float(a.pct[i]), 1), b=round(float(b.pct[i]), 1),
                                  p_a=round(wp["per_head"][i], 3), reliable=self.reliable[h]["ok"],
                                  validity=self.reliable[h]))
        style = [dict(key=h, label=coach.LABEL[h], a=round(float(a.pct[HIDX[h]]), 1), b=round(float(b.pct[HIDX[h]]), 1)) for h in ("symmetry", "repetition")]

        def photo_json(k, p):
            s = p.sig
            return dict(
                name=p.name, size=list(p.full_size), exif=p.exif,
                overall=round(self.overall(p), 1),
                aesthetic=round(float(p.pct[HIDX["aesthetic"]]), 1), technical=round(float(p.pct[HIDX["technical"]]), 1),
                histogram=s["histogram"], palette=s["palette"],
                measured=dict(
                    clip_hi=s["clip_hi"], clip_lo=s["clip_lo"], luma_median=s["luma_median"], contrast=s["contrast_rms"],
                    dyn_range=s["dyn_range"], saturation=s["sat_mean"], colorfulness=s["colorfulness"],
                    cast_a=s["neutral_cast_a"], cast_b=s["neutral_cast_b"], crispness=s["edge_crispness"],
                    detail=s["sharp_top"], focus_spread=s["focus_spread"], noise=s["noise_luma"],
                    tilt=s["tilt_deg"], tilt_conf=s["tilt_conf"], subject=[s["subject_x"], s["subject_y"]],
                    edge_weight=s["edge_weight"], aspect=s["aspect"]),
                overlays=[o for o in ("clip", "focus", "comp", "objects", "impact", "frame") if f"{k}_{o}" in images],
                objects=[objects.public(g) for g in obj[k]["segs"]] if k in obj else [],
                scene=({kk: vv for kk, vv in obj[k]["geo"].items()} if k in obj else None),
            )

        dz_a = Wn.z[HIDX["aesthetic"]] - Ln.z[HIDX["aesthetic"]]
        dz_t = Wn.z[HIDX["technical"]] - Ln.z[HIDX["technical"]]
        if dz_a > 0.25 and dz_t > 0.25:
            split = "It is ahead on both aesthetic appeal and technical quality."
        elif dz_a > 0.25:
            split = (f"It wins on aesthetic appeal; technically the two are about level." if dz_t > -0.25
                     else f"It wins on aesthetic appeal; {Ln.name} is actually the cleaner capture technically.")
        elif dz_t > 0.25:
            split = (f"It wins on technical quality; aesthetically the two are about level." if dz_a > -0.25
                     else f"It wins on technical quality; {Ln.name} is the more appealing image aesthetically.")
        else:
            split = "The two are close on both overall scores."
        strength = "clearly" if conf >= 0.85 else ("fairly confidently" if conf >= 0.68 else "narrowly")
        result = dict(
            winner=win, loser=lose, confidence=round(conf, 3), p_a=round(wp["overall"], 3),
            headline=f"{Wn.name} wins {strength}",
            summary=split,
            close_call=conf < 0.56,
            relation=dict(kind=relation, similarity=round(sim, 3), inliers=al["inliers"],
                          text={"aligned": "Same scene - the two frames were matched point-to-point, so framing and tone are compared directly.",
                                "similar": "Similar subject - tone and colour are compared directly; framing is judged separately for each photo.",
                                "different": "Different scenes - each photo is judged on its own merits; colour-matching advice is limited to tone."}[relation]),
            photos={k: photo_json(k, p) for k, p in P.items()},
            attributes=attrs, style=style,
            reasons=advice["reasons"], loser_leads=advice["loser_leads"], coaching=advice["items"],
            frame={k: v for k, v in (frame or {}).items()} if frame else None,
            recipe=dict(lines=coach.recipe_lines(recipe), color_ok=recipe["color_ok"]),
            previews=prev_info, best_preview=best_key,
            weights=dict(aesthetic=self.w_aes, technical=self.w_tech),
            model=dict(trained=self.meta.get("trained"), datasets=self.meta.get("datasets"), pool=self.meta.get("pool_size")),
        )
        return _clean(result), images


# ----------------------------------------------------------------------------
# overlays
# ----------------------------------------------------------------------------
def _jpg(rgb: np.ndarray, q: int = 88) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=q)
    return buf.getvalue()


def _dim(rgb, keep=0.45):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return (g[..., None] * keep + rgb.astype(np.float32) * 0.15)


def _ov_clip(rgb):
    out = _dim(rgb, 0.55)
    y = signals._luma(rgb.astype(np.float32) / 255.0)
    hi = rgb.max(2) >= 252
    lo = y <= 0.016
    out[hi] = (255, 45, 60)
    out[lo] = (40, 110, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def _ov_focus(rgb, detail, scale):
    h, w = rgb.shape[:2]
    d = cv2.resize(np.clip(detail / (scale + 1e-9), 0, 1) ** 0.6, (w, h), interpolation=cv2.INTER_CUBIC)
    d = np.clip(cv2.GaussianBlur(d, (0, 0), 0.02 * max(h, w)), 0, 1)
    cm = cv2.cvtColor(cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.clip(_dim(rgb, 0.5) * 0.45 + cm * 0.6, 0, 255).astype(np.uint8)


def _ov_impact(rgb, heat, scale=None):
    """Relative map: green = the score leans on this region more than on the rest of the frame,
    red = less than the rest (or the score improves without it)."""
    h, w = rgb.shape[:2]
    rel = heat - np.median(heat)
    m = cv2.resize(rel, (w, h), interpolation=cv2.INTER_CUBIC) / (float(np.abs(rel).max()) + 1e-9)
    base = _dim(rgb, 0.6)
    pos = np.clip(m, 0, 1)[..., None] ** 0.8
    neg = np.clip(-m, 0, 1)[..., None] ** 0.8
    out = base * (1 - 0.75 * pos) + np.array([40, 210, 150], np.float32) * 0.75 * pos
    out = out * (1 - 0.75 * neg) + np.array([255, 80, 60], np.float32) * 0.75 * neg
    return np.clip(out, 0, 255).astype(np.uint8)


def _ov_comp(rgb, s):
    out = (rgb.astype(np.float32) * 0.8).astype(np.uint8).copy()
    h, w = out.shape[:2]
    t = max(1, round(max(h, w) / 500))
    for f in (1 / 3, 2 / 3):
        cv2.line(out, (round(f * w), 0), (round(f * w), h), (255, 255, 255), t, cv2.LINE_AA)
        cv2.line(out, (0, round(f * h)), (w, round(f * h)), (255, 255, 255), t, cv2.LINE_AA)
    r = round(0.035 * max(h, w))
    c = (round(s["subject_x"] * w), round(s["subject_y"] * h))
    cv2.circle(out, c, r, (0, 0, 0), 3 * t + 2, cv2.LINE_AA)
    cv2.circle(out, c, r, (255, 200, 40), 3 * t, cv2.LINE_AA)
    cw = (round(s["weight_cx"] * w), round(s["weight_cy"] * h))
    cv2.drawMarker(out, cw, (0, 0, 0), cv2.MARKER_CROSS, 2 * r, 3 * t + 2, cv2.LINE_AA)
    cv2.drawMarker(out, cw, (90, 220, 255), cv2.MARKER_CROSS, 2 * r, 2 * t, cv2.LINE_AA)
    if s.get("_horizon_line") and s["tilt_conf"] >= 0.3:
        x1, y1, x2, y2 = s["_horizon_line"]
        cy = (y1 + y2) / 2
        sl = math.tan(math.radians(s["tilt_deg"])) * w / h
        pa, pb = (0, round((cy - 0.5 * sl) * h)), (w, round((cy + 0.5 * sl) * h))
        col = (255, 70, 70) if abs(s["tilt_deg"]) >= 1.0 else (90, 255, 140)
        cv2.line(out, pa, pb, (0, 0, 0), 3 * t + 2, cv2.LINE_AA)
        cv2.line(out, pa, pb, col, 2 * t, cv2.LINE_AA)
    return out


def _ov_frame(rgb, quad, color=(255, 200, 40)):
    h, w = rgb.shape[:2]
    pts = np.int32([[round(x * w), round(y * h)] for x, y in quad])
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    out = rgb.astype(np.float32)
    out[mask == 0] *= 0.35
    out = out.astype(np.uint8)
    t = max(2, round(max(h, w) / 300))
    cv2.polylines(out, [pts], True, (0, 0, 0), t + 2, cv2.LINE_AA)
    cv2.polylines(out, [pts], True, color, t, cv2.LINE_AA)
    return out


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o

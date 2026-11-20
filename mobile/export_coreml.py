"""Package the on-device model for the iOS app.

    python -m mobile.export_coreml            # reads models/mobile_judge.pt

Writes into ios/PhotoCoach/Resources/:
    Encoder.mlpackage   image (256x256 RGB) -> 512-d embedding            (run 5x per photo)
    Tower.mlpackage     global embed + 4 crop embeds + signals -> 20 raw attribute scores
    judge.json          head names/labels, calibration tables, comparison scales, reliability, advice text
    parity.json         reference outputs for bundled test photos, so the app can check its own
                        preprocessing + Core ML results against the PyTorch model

Conversion runs on Linux; Core ML *prediction* only runs on Apple hardware, which is why the
parity file exists - the app's self-test is the real verification.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn

from photocoach import coach, signals
from photocoach.backbone import load_image
from photocoach.model import HEAD_NAMES, K, load_judge
from photocoach.signals import VEC_KEYS

from .extract import CANDIDATES, DISPLAY, RES, load_visual, views

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "models", "mobile_judge.pt")
OUT = os.path.join(ROOT, "ios", "PhotoCoach", "Resources")


class EncoderWrap(nn.Module):
    """uint8-range RGB image tensor (1,3,H,W in 0..255) -> embedding.  Normalisation lives inside
    the model so the app only has to hand over pixels."""

    def __init__(self, visual, mean, std):
        super().__init__()
        self.visual = visual
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255.0)

    def forward(self, image):
        return self.visual((image - self.mean) / self.std)


class TowerWrap(nn.Module):
    """Same arithmetic as photocoach.model.Tower.forward, written with ops that convert cleanly."""

    def __init__(self, tower):
        super().__init__()
        self.t = tower

    def forward(self, g, l, h):
        t = self.t
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-12) * (t.g_dim ** 0.5)
        l = l / l.norm(dim=-1, keepdim=True).clamp(min=1e-12) * (t.l_dim ** 0.5)
        mu = l.mean(1)
        sd = (((l - mu.unsqueeze(1)) ** 2).sum(1) / (l.shape[1] - 1)).sqrt()
        hz = ((h - t.h_mean) / t.h_std).clamp(-6, 6) * t.h_use
        one = torch.ones_like(g[:, :1])
        x = torch.cat([t.g_in(g), t.l_in(torch.cat([mu, sd], -1)), t.h_in(hz), one], -1)
        return t.out(t.trunk(x))


def store_8bit(ml):
    """Store the weights of a `_q8` encoder as 8-bit.  They are already on an 8-bit grid (see
    mobile.extract.quantize8), so this must not change them: any tensor Core ML would re-round along a
    different axis is left in 16-bit instead."""
    import coremltools.optimize.coreml as cto

    def consts(model, kind):
        return {op.name: op for f in model._mil_program.functions.values() for op in f.operations if op.op_type == kind}

    before = {n: op.val.val for n, op in consts(ml, "const").items() if isinstance(op.val.val, np.ndarray) and op.val.val.size >= 2048}
    skip = {}
    for _ in range(2):
        cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=2048),
                                     op_name_configs=skip)
        q = cto.linear_quantize_weights(ml, cfg)
        worst = 0.0
        for n, op in consts(q, "constexpr_affine_dequantize").items():
            n = n.removesuffix("_quantized")
            w = before[n].astype(np.float32)
            err = float(np.abs(np.asarray(op.materialized_val_inference(), np.float32) - w).max() / max(float(np.abs(w).max()), 1e-12))
            if err > 1e-3:
                skip[n] = None
            else:
                worst = max(worst, err)
        if not skip or _ == 1:
            break
    print(f"   8-bit weights: {len(consts(q, 'constexpr_affine_dequantize'))} tensors, worst change {worst:.1e} of a tensor's range; "
          f"{len(skip)} kept in 16-bit")
    return q


def _size(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e6


def main():
    import coremltools as ct
    from timm.utils import reparameterize_model

    judge, meta = load_judge(CKPT)
    arch = meta["arch"]
    enc_name = meta["encoder"]
    assert arch["g_key"] == "g_sq", "the app squashes the whole frame; export expects a model trained that way"
    os.makedirs(OUT, exist_ok=True)

    # ---- encoder ----
    visual, size, mean, std = load_visual(enc_name)
    res = int(meta.get("res", RES))                       # the view size the tower's features were cut at
    assert size == res, f"{enc_name} takes {size}px input but the features were extracted at {res}px"
    if enc_name.startswith("mobileclip"):
        visual = reparameterize_model(visual.eval())      # fold MobileCLIP's training-time branches
    enc_label = DISPLAY.get(enc_name, CANDIDATES[enc_name][0])
    enc = EncoderWrap(visual.eval(), mean, std).eval()
    ex = torch.rand(1, 3, res, res) * 255
    ml = ct.convert(torch.jit.trace(enc, ex),
                    inputs=[ct.ImageType(name="image", shape=(1, 3, res, res), scale=1.0, color_layout=ct.colorlayout.RGB)],
                    outputs=[ct.TensorType(name="embedding")], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17)
    if enc_name.endswith("_q8"):
        ml = store_8bit(ml)
    ml.short_description = f"Photo Coach image encoder ({enc_label})"
    p = os.path.join(OUT, "Encoder.mlpackage")
    shutil.rmtree(p, ignore_errors=True)
    ml.save(p)
    print(f"Encoder.mlpackage  {_size(p):.1f} MB  ({enc_label}, {res}px)")

    # ---- tower ----
    tw = TowerWrap(judge.tower).eval()
    eg, el, eh = torch.randn(1, arch["g_dim"]), torch.randn(1, 4, arch["l_dim"]), torch.zeros(1, len(VEC_KEYS))
    with torch.no_grad():
        ref = judge.tower(eg, el, eh, torch.ones(1))
        assert torch.allclose(tw(eg, el, eh), ref, atol=1e-4), "TowerWrap disagrees with Tower"
    mt = ct.convert(torch.jit.trace(tw, (eg, el, eh)),
                    inputs=[ct.TensorType(name="g", shape=eg.shape), ct.TensorType(name="l", shape=el.shape), ct.TensorType(name="h", shape=eh.shape)],
                    outputs=[ct.TensorType(name="scores")], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.iOS17)
    p = os.path.join(OUT, "Tower.mlpackage")
    shutil.rmtree(p, ignore_errors=True)
    mt.save(p)
    print(f"Tower.mlpackage    {_size(p):.1f} MB")

    # ---- everything the app needs besides weights ----
    rep = meta.get("report", {})
    best = {}
    for k, v in rep.items():
        if k != "synth":
            best[k.split("/")[1]] = max(best.get(k.split("/")[1], -1), v["srcc"])
    from photocoach.synth import EFFECT_HEAD
    syn = {}
    for eff, v in rep.get("synth", {}).items():
        if EFFECT_HEAD.get(eff) and v.get("attr_acc") is not None:
            syn.setdefault(EFFECT_HEAD[eff], []).append(v["attr_acc"])
    heads = []
    for i, h in enumerate(HEAD_NAMES):
        if h in best:
            rel, val, metric = best[h] >= 0.30, best[h], "srcc"
        elif h in syn:
            val = float(np.mean(syn[h])); rel, metric = val >= 0.75, "pair_acc"
        else:
            rel, val, metric = False, None, None
        grp = next((g for g, hs in coach.GROUPS if h in hs), "Overall" if h in ("aesthetic", "technical") else "Style")
        heads.append(dict(key=h, label=coach.LABEL[h], blurb=coach.BLURB.get(h, ""), group=grp, reliable=bool(rel),
                          validity=None if val is None else round(float(val), 3), metric=metric,
                          scale=float(judge.log_scale[i].exp()), refMean=float(judge.ref_mean[i]), refStd=float(judge.ref_std[i]),
                          quantiles=[round(float(q), 5) for q in judge.q_table[i]]))
    info = dict(encoder=enc_label, inputSize=res, stdLongSide=signals.STD_LONG_SIDE,
                weights=dict(aesthetic=0.6, technical=0.4), groups=[g for g, _ in coach.GROUPS],
                signalKeys=VEC_KEYS, signalsUsed=arch["h_keys"] if arch["h_keys"] is not None else VEC_KEYS,
                heads=heads, poolSize=meta.get("pool_size"), trained=meta.get("trained"), datasets=meta.get("datasets"),
                advice=ADVICE)
    json.dump(info, open(os.path.join(OUT, "judge.json"), "w"), indent=1)

    # ---- parity fixtures ----
    demo = os.path.join(ROOT, "data", "demo")
    par = []
    for f in ("louise_03.jpg", "louise_11.jpg"):
        shutil.copy(os.path.join(demo, f), os.path.join(OUT, "parity_" + f))
        rgb = signals.to_std(np.asarray(load_image(os.path.join(demo, f))))
        v = torch.tensor(views(rgb, res)).permute(0, 3, 1, 2).float()
        hv = torch.tensor(signals.vector(signals.measure(rgb, full=False)))[None]
        with torch.no_grad():
            e = enc(v)                                     # (6,D): squash, letterbox, 4 crops
            s = judge.tower(e[0:1], e[2:6][None], hv, torch.ones(1))[0].numpy()
        par.append(dict(file="parity_" + f, stdSize=[rgb.shape[1], rgb.shape[0]],
                        globalEmbedHead=[round(float(x), 4) for x in e[0, :8]],
                        signals={k: round(float(x), 5) for k, x in zip(VEC_KEYS, hv[0])},
                        scores={h: round(float(x), 4) for h, x in zip(HEAD_NAMES, s)},
                        percentiles={h: round(float(x), 1) for h, x in zip(HEAD_NAMES, judge.percentile(s[None])[0])}))
    json.dump(par, open(os.path.join(OUT, "parity.json"), "w"), indent=1)
    print("judge.json, parity.json written to", OUT)


# Per-attribute coaching used by the first on-device build: when the twin network puts the other
# photo clearly ahead on an attribute, this is what to do about it.  (The desktop app layers
# measured evidence and object analysis on top; those arrive on iOS in later steps.)
ADVICE = {
    "lighting": dict(title="The light itself is weaker", fix="reshoot",
        shoot=["Light is the one thing you cannot add later. Come back in the hour after sunrise or before sunset when it is low, warm and directional - or right after weather clears.",
               "Put the light to the side of or behind the subject for shape and depth; front-lit and overhead midday light flattens everything."],
        edit=["Dodging and burning can emphasise existing light, but cannot change its direction."]),
    "exposure": dict(title="Exposure is off", fix="edit",
        shoot=["Check the histogram after the shot: nothing piled against either edge. Use exposure compensation (+/- 0.7 EV) when the scene is mostly bright or mostly dark.",
               "Turn on highlight warnings; if the scene has more range than the sensor, bracket or expose for the highlights and shoot RAW."],
        edit=["Adjust Exposure until the mid-tones sit right, then recover Highlights and lift Shadows. Clipped white cannot be brought back."]),
    "contrast": dict(title="Weaker tonal separation", fix="edit",
        shoot=["Flat tone usually comes from flat light or haze: shoot with the sun to the side, avoid midday haze, use a lens hood and a clean lens."],
        edit=["Set a true black and white point (Blacks down, Whites up), add a gentle S-curve and a little Dehaze/Clarity."]),
    "color_harmony": dict(title="Less cohesive colour palette", fix="both",
        shoot=["Simplify the palette: frame so two or three related or complementary colours dominate, and exclude stray bright objects."],
        edit=["Use HSL / colour grading to pull distracting hues toward the main palette."]),
    "vivid_color": dict(title="Muted colour", fix="edit",
        shoot=["Colour is richest in low-angle light and just after rain; a polariser deepens skies, foliage and water by removing glare."],
        edit=["Add Vibrance rather than Saturation so already-strong colours don't go neon."]),
    "color_accuracy": dict(title="Colour looks off", fix="edit",
        shoot=["Set white balance for the light (Daylight / Cloudy / Shade) instead of Auto, or shoot RAW and set it afterwards.", "Turn off 'Vivid' picture styles."],
        edit=["Click the white-balance eyedropper on something that should be grey; pull Saturation back until skin, sky and foliage look believable."]),
    "sharpness": dict(title="Softer detail", fix="reshoot",
        shoot=["Focus deliberately: tap on the subject (for landscapes, about a third of the way into the scene) and check at 100% before leaving.",
               "Hand-held, keep the shutter at 1/(2x focal length) or faster; brace the camera and keep stabilisation on."],
        edit=["Capture sharpening helps slightly-soft files, but cannot rescue missed focus or shake."]),
    "noise": dict(title="Visible noise", fix="both",
        shoot=["Use the lowest ISO the light allows: add light, open the aperture, or use a tripod with a longer shutter.", "Expose generously - brightening a dark file afterwards multiplies noise."],
        edit=["Apply luminance + colour noise reduction, then sharpen lightly."]),
    "depth_of_field": dict(title="Less subject separation (depth of field)", fix="reshoot",
        shoot=["To blur the background: widest aperture, longer lens, get closer to the subject, and put distance between subject and background.",
               "On a phone, use the 2x/3x lens or Portrait mode and move in close."],
        edit=["Background blur added in software rarely looks natural around hair and fine edges - get it in camera."]),
    "subject_emphasis": dict(title="The subject doesn't stand out", fix="both",
        shoot=["Decide what the one subject is, then remove competition: get closer, simplify the background, or change angle so the subject stands against something plain.",
               "Run your eye around the border before pressing the shutter; step sideways or zoom to drop anything bright or cut-off at the edges."],
        edit=["A crop can tighten the frame and move the subject onto a stronger position."]),
    "framing": dict(title="Looser framing", fix="both",
        shoot=["Leave breathing room around the subject and avoid cutting things off at the frame edge; fill the frame with what matters."],
        edit=["Try a few crops before reshooting."]),
    "level": dict(title="Frame looks tilted", fix="edit",
        shoot=["Switch on the grid or electronic level and line the horizon up with it."],
        edit=["Straighten with the crop tool; fix converging verticals with perspective correction."]),
    "content": dict(title="Less engaging subject matter", fix="reshoot",
        shoot=["Give the viewer something to land on: a person for scale, a boat, wildlife, a striking foreground, or a moment (weather, reflection, light beam).",
               "Ask what the photo is about, and exclude everything that isn't that."],
        edit=[]),
}


if __name__ == "__main__":
    main()

"""Turn score gaps + measured evidence into a verdict explanation and coaching.

Every piece of advice is tied to (a) an attribute where the twin network rates
the winner higher and/or (b) a number measured on the pixels or read from EXIF.
Nothing is said that isn't backed by one of those.
"""
from __future__ import annotations

import math

from .scene import shutter_str

LABEL = {
    "aesthetic": "Aesthetic appeal", "technical": "Technical quality",
    "balance": "Visual balance", "thirds": "Subject placement", "subject_emphasis": "Subject emphasis",
    "framing": "Framing", "level": "Level horizon",
    "lighting": "Quality of light", "exposure": "Exposure", "contrast": "Contrast & tone",
    "color_harmony": "Colour harmony", "vivid_color": "Colour vividness", "color_accuracy": "White balance & colour fidelity",
    "depth_of_field": "Depth of field / bokeh", "sharpness": "Sharpness", "motion_blur": "Freedom from motion blur",
    "noise": "Low noise", "content": "Subject interest",
    "symmetry": "Symmetry", "repetition": "Pattern / repetition",
}
BLURB = {
    "aesthetic": "How much people would like the photo overall (learned from ~140k contest photos with hundreds of votes each).",
    "technical": "Capture quality: clean, sharp, well exposed (learned from human quality ratings of 21k real photos).",
    "balance": "Whether visual weight is distributed so the frame feels settled.",
    "thirds": "Whether the main subject sits at a strong position in the frame.",
    "subject_emphasis": "How clearly one subject stands out from everything else.",
    "framing": "Whether the frame edges are deliberate - nothing important cut off, no dead space.",
    "level": "Horizon and verticals straight.",
    "lighting": "Direction, softness and drama of the light itself.",
    "exposure": "Overall brightness; highlights and shadows holding detail.",
    "contrast": "Tonal separation - neither flat/hazy nor harsh.",
    "color_harmony": "Whether the colours work together.",
    "vivid_color": "Richness and saturation of colour.",
    "color_accuracy": "Neutral whites, no colour cast, believable saturation.",
    "depth_of_field": "Use of focus depth - subject separation, background blur.",
    "sharpness": "Crisp detail where it matters.",
    "motion_blur": "No smearing from camera shake or subject movement.",
    "noise": "Freedom from grain and colour speckle.",
    "content": "How interesting the subject matter is.",
}
GROUPS = [
    ("Composition", ["subject_emphasis", "thirds", "balance", "framing", "level"]),
    ("Light & tone", ["lighting", "exposure", "contrast"]),
    ("Colour", ["color_harmony", "vivid_color", "color_accuracy"]),
    ("Focus & detail", ["sharpness", "depth_of_field", "motion_blur", "noise"]),
    ("Subject", ["content"]),
]
JUDGED = [h for _, hs in GROUPS for h in hs]


def _ord(p):
    p = int(round(p))
    p = max(1, min(99, p))
    suf = "th" if 10 <= p % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(p % 10, "th")
    return f"{p}{suf}"


def _pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def _cast_words(a, b):
    w = []
    if abs(b) >= 0.07: w.append("yellow/warm" if b > 0 else "blue/cool")
    if abs(a) >= 0.07: w.append("magenta" if a > 0 else "green")
    return " and ".join(w)


def _where(g):
    from .objects import where
    return where(g)


def _cut(g):
    from .objects import cut_off
    return cut_off(g)


def build(ctx: dict) -> dict:
    """ctx: names, win/lose keys, pct, z, sig, exif, recipe, frame, similarity, reliable, preview."""
    W, L = ctx["winner"], ctx["loser"]
    nW, nL = ctx["names"][W], ctx["names"][L]
    pct, z, sig, ex = ctx["pct"], ctx["z"], ctx["sig"], ctx["exif"]
    rel = ctx["reliable"]
    sw, sl = sig[W], sig[L]
    rec = ctx.get("recipe")
    same = ctx["relation"] in ("aligned", "similar")

    def gap(h):  # winner minus loser, percentile points
        return pct[W][h] - pct[L][h]

    def zgap(h):
        return z[W][h] - z[L][h]

    def vs(h):
        return f"{_ord(pct[L][h])} vs {_ord(pct[W][h])} percentile"

    items = []

    def add(id, title, heads, evidence, shoot, edit, overlay=None, fix="both", boost=0.0, force=False):
        hs = [h for h in heads if rel.get(h, True)]
        g = max([gap(h) for h in hs], default=0.0)
        zg = max([zgap(h) for h in hs], default=0.0)
        if not force and g < 12:
            return
        pr = max(zg, 0) + boost
        ev = list(evidence)
        for h in hs:
            if gap(h) >= 6:
                ev.append(f"{LABEL[h]}: the network puts {nL} at the {_ord(pct[L][h])} percentile, {nW} at the {_ord(pct[W][h])}.")
        items.append(dict(id=id, title=title, heads=hs, gap=round(g, 1), priority=round(pr, 3),
                          evidence=ev, shoot=shoot, edit=edit, overlay=overlay, fix=fix))

    # ---------------- exposure / highlights / shadows ----------------
    if sl["clip_hi"] >= 0.01 and sl["clip_hi"] > 1.5 * sw["clip_hi"] + 0.005:
        add("highlights", "Blown-out highlights", ["exposure"],
            [f"{_pct(sl['clip_hi'])} of {nL} is clipped to pure white (red in the Clipping overlay); {nW} clips {_pct(sw['clip_hi'])}."],
            ["Dial in about -0.7 to -1.3 EV of exposure compensation and re-check the histogram - nothing should pile up against the right edge.",
             "Turn on highlight warnings ('blinkies'/zebras) or use highlight-weighted metering.",
             "If the scene has more range than the sensor (bright sky + dark land), bracket 3 frames and blend, use a graduated ND filter, or shoot RAW and expose for the sky."],
            ["Clipped areas hold no data, so this cannot be fully repaired - pulling Highlights/Whites down only turns white into flat grey. A RAW file may recover about a stop."],
            overlay="clip", fix="reshoot", boost=min(1.5, 12 * sl["clip_hi"]), force=sl["clip_hi"] >= 0.03)
    if sl["clip_lo"] >= 0.03 and sl["clip_lo"] > 1.5 * sw["clip_lo"] + 0.01:
        add("shadows", "Crushed shadows", ["exposure", "contrast"],
            [f"{_pct(sl['clip_lo'])} of {nL} is pure black with no detail (blue in the Clipping overlay); {nW}: {_pct(sw['clip_lo'])}."],
            ["Give the shot more exposure (+0.5 to +1 EV) or add fill light; at a low ISO you can lift shadows later with little penalty.",
             "Shoot RAW - JPEG throws away most shadow detail."],
            ["Raise Shadows and Blacks; if the file is a JPEG expect noise and banding to appear as you lift."],
            overlay="clip", boost=min(1.0, 5 * sl["clip_lo"]), force=sl["clip_lo"] >= 0.08)
    if rec and same and abs(rec["exposure_ev"]) >= 0.4 and (gap("exposure") >= 5 or abs(rec["exposure_ev"]) >= 0.8):
        ev_ = rec["exposure_ev"]
        dark = ev_ > 0
        add("brightness", f"{'Underexposed' if dark else 'Overexposed'} by about {abs(ev_):.1f} stops relative to the winner", ["exposure"],
            [f"Mid-tones of {nL} sit {abs(ev_):.1f} EV {'below' if dark else 'above'} those of {nW} (median brightness {sl['luma_median']:.2f} vs {sw['luma_median']:.2f} on a 0-1 scale)."],
            [f"Use {'+' if dark else '-'}{min(2.0, round(abs(ev_) * 3) / 3):.1f} EV exposure compensation." + (" The meter was fooled by a bright sky/water/snow - meter off a mid-tone instead." if dark else " The meter was fooled by a dark scene - meter off a mid-tone instead.")],
            [f"Exposure {ev_:+.1f} in your editor" + (" (watch that highlights don't clip as you brighten)." if dark else " (this won't bring back anything already clipped).")],
            overlay="hist", fix="edit", boost=0.3 * abs(ev_), force=abs(ev_) >= 0.8)
    elif not same and gap("exposure") >= 12:
        dark = sl["luma_median"] < 0.3
        bright = sl["luma_median"] > 0.68
        if dark or bright:
            add("brightness", "Too dark overall" if dark else "Too bright overall", ["exposure"],
                [f"Median brightness of {nL} is {sl['luma_median']:.2f} on a 0-1 scale."],
                [f"Use {'+' if dark else '-'}0.7 EV exposure compensation and check the histogram."],
                [f"{'Raise' if dark else 'Lower'} Exposure until the histogram is centred for this scene."], overlay="hist", fix="edit")

    # ---------------- contrast ----------------
    flat = sl["contrast_rms"] < 0.82 * sw["contrast_rms"] or sl["dyn_range"] < 0.8 * sw["dyn_range"]
    harsh = sl["contrast_rms"] > 1.25 * sw["contrast_rms"] and (sl["clip_hi"] + sl["clip_lo"]) > 0.04
    if flat:
        add("contrast", "Flat, hazy tones", ["contrast"],
            [f"Tonal spread of {nL} is {sl['contrast_rms']:.2f} vs {sw['contrast_rms']:.2f} for {nW}; its tones span {_pct(sl['dyn_range'], 0)} of the available range vs {_pct(sw['dyn_range'], 0)}."],
            ["Flat tone usually comes from flat light or haze: shoot with the sun to the side rather than behind you or overhead, and avoid midday haze.",
             "A polarising filter cuts atmospheric haze and reflections; a lens hood stops veiling flare. Clean the lens - a smeared phone lens causes exactly this."],
            [f"Set a true black and white point (Blacks down, Whites up), add Contrast{' about %+d' % round(rec['contrast_pct'] * 0.6) if rec and same else ''} and a little Dehaze/Clarity."],
            overlay="hist", fix="edit", boost=0.2)
    elif harsh:
        add("contrast", "Harsh contrast", ["contrast", "lighting"],
            [f"Tonal spread of {nL} is {sl['contrast_rms']:.2f} vs {sw['contrast_rms']:.2f}, with {_pct(sl['clip_hi'] + sl['clip_lo'])} of pixels clipped at either end."],
            ["Hard midday sun is the usual cause - shoot early or late, wait for thin cloud, or move the subject into open shade.",
             "Bracket exposures / use HDR mode when the scene's range is this wide."],
            ["Lower Contrast, pull Highlights down and Shadows up."], overlay="clip")
    elif gap("contrast") >= 12:
        add("contrast", "Weaker tonal separation", ["contrast"], [],
            ["Look for light that separates the subject from the background (side or back light)."],
            ["Shape the tone curve: gentle S-curve, set black and white points."], overlay="hist", fix="edit")

    # ---------------- light ----------------
    if gap("lighting") >= 10:
        ev = []
        if sw["local_contrast"] > 1.2 * sl["local_contrast"]:
            ev.append(f"{nW} has {sw['local_contrast'] / max(sl['local_contrast'], 1e-4):.1f}x the local light-and-shade modelling of {nL} - a sign of directional light.")
        if sw["cast_b"] - sl["cast_b"] > 0.08:
            ev.append(f"{nW} is lit noticeably warmer (golden-hour-like colour) than {nL}.")
        add("lighting", "The light itself is weaker", ["lighting"], ev,
            ["Light is the one thing you cannot add later. Come back in the hour after sunrise or before sunset when it is low, warm and directional - or right after weather clears.",
             "Put the light to the side of or behind the subject for shape and depth; front-lit and overhead midday light flattens everything.",
             "For a landmark view, check where the sun will be (PhotoPills / The Photographer's Ephemeris) and plan the time of day the scene is lit, not in shadow."],
            ["Dodging and burning (local brightening/darkening) can emphasise existing light, but cannot change its direction."],
            overlay="impact", fix="reshoot", boost=0.25)

    # ---------------- colour ----------------
    muted = sl["colorfulness"] < 0.78 * sw["colorfulness"]
    garish = sl["sat_mean"] > 0.62 and sl["sat_mean"] > 1.3 * sw["sat_mean"]
    if muted and (gap("vivid_color") >= 6 or gap("color_harmony") >= 8):
        add("muted", "Muted colour", ["vivid_color", "color_harmony"],
            [f"Colourfulness of {nL} measures {sl['colorfulness']:.2f} vs {sw['colorfulness']:.2f} for {nW}."],
            ["Colour is richest in low-angle light and just after rain; a polariser deepens skies, foliage and water by removing glare.",
             "Underexposing by 1/3 stop keeps bright colours from washing out."],
            [f"Add Vibrance (about +{min(60, max(10, round((rec['saturation_pct'] if rec else 25) * 0.8)))}) rather than Saturation so already-strong colours don't go neon."],
            overlay="palette", fix="edit", boost=0.15)
    elif garish:
        add("garish", "Over-saturated colour", ["color_accuracy", "color_harmony"],
            [f"Mean saturation of {nL} is {_pct(sl['sat_mean'], 0)} vs {_pct(sw['sat_mean'], 0)} for {nW}."],
            ["Turn off 'Vivid' picture styles / scene enhancers."],
            ["Reduce Saturation/Vibrance until skin, sky and foliage look believable; over-saturation destroys detail in the most colourful areas."],
            overlay="palette", fix="edit", force=True, boost=0.2)
    elif gap("vivid_color") >= 12 or gap("color_harmony") >= 12:
        add("palette", "Less cohesive colour palette", ["color_harmony", "vivid_color"], [],
            ["Simplify the palette: frame so two or three related or complementary colours dominate, and exclude stray bright objects."],
            ["Use HSL/colour-grading tools to pull distracting hues toward the main palette."], overlay="palette")
    ca, cb = sl["neutral_cast_a"], sl["neutral_cast_b"]
    wa, wb = sw["neutral_cast_a"], sw["neutral_cast_b"]
    if (abs(ca - wa) >= 0.09 or abs(cb - wb) >= 0.12) and (abs(ca) > abs(wa) + 0.04 or abs(cb) > abs(wb) + 0.06) and _cast_words(ca, cb):
        add("cast", f"Colour cast ({_cast_words(ca, cb)})", ["color_accuracy"],
            [f"Areas that should be neutral in {nL} lean {_cast_words(ca, cb)}; in {nW} they are closer to neutral."],
            ["Set white balance for the light (Daylight / Cloudy / Shade) instead of Auto, or shoot RAW and set it afterwards."],
            [f"Move Temperature toward {'blue' if cb > wb else 'yellow'}" + (f" and Tint toward {'green' if ca > wa else 'magenta'}" if abs(ca - wa) >= 0.07 else "") + ", or click the white-balance eyedropper on something that should be grey."],
            overlay="palette", fix="edit", force=gap("color_accuracy") >= 4, boost=0.15)

    # ---------------- sharpness / blur / noise ----------------
    soft = sl["edge_crispness"] < 0.85 * sw["edge_crispness"] or sl["sharp_top"] < 0.6 * sw["sharp_top"]
    eL = ex[L]
    if gap("sharpness") >= 12 or (soft and gap("sharpness") >= 6):
        ev = []
        if soft:
            ev.append(f"Edges in the sharpest part of {nL} are softer than in {nW} (edge crispness {sl['edge_crispness']:.2f} vs {sw['edge_crispness']:.2f}; see the Focus map).")
        shoot = []
        if eL.get("shutter") and (eL.get("focal35") or eL.get("focal")):
            f35 = eL.get("focal35") or eL["focal"]
            if eL["shutter"] > 1.0 / max(f35, 30):
                ev.append(f"EXIF: {shutter_str(eL['shutter'])} at {f35:g}mm - slower than the 1/focal-length hand-holding rule, so camera shake is likely.")
                shoot.append(f"Use 1/{int(max(2 * f35, 125))}s or faster hand-held, or put the camera on a tripod and use a 2-second timer.")
        if eL.get("aperture") and eL["aperture"] >= 16:
            ev.append(f"EXIF: f/{eL['aperture']:g} - diffraction softens the whole frame at apertures this small.")
            shoot.append("Stay around f/5.6-f/11, where most lenses are sharpest.")
        shoot += ["Focus deliberately: tap/half-press on the subject (for landscapes, about a third of the way into the scene) and check at 100% before leaving.",
                  "Brace the camera, squeeze the shutter gently, and keep image stabilisation on when hand-holding."]
        add("sharpness", "Softer detail", ["sharpness", "motion_blur"], ev, shoot,
            ["Capture sharpening helps slightly-soft files, but cannot rescue missed focus or shake."],
            overlay="focus", fix="reshoot", boost=0.25 if soft else 0.0)
    noisy = sl["noise_luma"] > 1.6 * sw["noise_luma"] and sl["noise_luma"] > 0.006
    if gap("noise") >= 14 or (noisy and gap("noise") >= 8):
        ev = [f"Grain in smooth areas of {nL} measures {1000 * sl['noise_luma']:.1f} vs {1000 * sw['noise_luma']:.1f} for {nW} (x1000, luminance noise)."] if noisy else []
        shoot = ["Use the lowest ISO the light allows: add light, open the aperture, or use a tripod with a longer shutter.",
                 "Expose generously - brightening a dark file afterwards multiplies noise."]
        if eL.get("iso") and eL["iso"] >= 800:
            ev.append(f"EXIF: ISO {eL['iso']}.")
        add("noise", "Visible noise", ["noise"], ev, shoot,
            ["Apply luminance + colour noise reduction (AI denoise tools work well), then sharpen lightly."],
            overlay="focus", boost=0.1 if noisy else 0.0)

    # ---------------- depth of field ----------------
    if gap("depth_of_field") >= 20 or (gap("depth_of_field") >= 12 and sw["focus_spread"] > sl["focus_spread"] + 0.4):
        ev = []
        if sw["focus_spread"] > sl["focus_spread"] + 0.4:
            ev.append(f"{nW} has a much stronger sharp-subject / soft-background split (focus spread {sw['focus_spread']:.1f} vs {sl['focus_spread']:.1f}; compare the Focus maps).")
        shoot = ["To blur the background: use the widest aperture (lowest f-number), zoom in or use a longer lens, get closer to the subject, and put distance between subject and background.",
                 "On a phone, use the 2x/3x lens or Portrait mode and move in close."]
        eW = ex[W]
        if eW.get("aperture") and eL.get("aperture"):
            ev.append(f"EXIF: {nW} was shot at f/{eW['aperture']:g}, {nL} at f/{eL['aperture']:g}.")
        add("dof", "Less subject separation (depth of field)", ["depth_of_field"], ev, shoot,
            ["Background blur added in software rarely looks natural around hair and fine edges - get it in camera."],
            overlay="focus", fix="reshoot")

    # ---------------- composition ----------------
    # A line detector alone is fooled by ridgelines and receding shorelines, so a tilt is only
    # reported when the learned level-horizon head (trained on deliberately tilted copies) agrees.
    tilt_ok = (1.0 <= abs(sl["tilt_deg"]) <= 8 and sl["tilt_conf"] >= 0.6 and abs(sl["tilt_deg"]) > abs(sw["tilt_deg"]) + 0.5
               and rel.get("level", False) and gap("level") >= 10)
    if tilt_ok:
        d = sl["tilt_deg"]
        add("level", f"Horizon tilted {abs(d):.1f} degrees", ["level"],
            [f"The dominant horizontal line in {nL} (horizon or far shoreline - see the Composition overlay) runs {abs(d):.1f} degrees off level, {'down to the right' if d > 0 else 'down to the left'}; {nW} is within {abs(sw['tilt_deg']):.1f} degrees. Ignore this if that line is a shoreline receding at an angle."],
            ["Switch on the grid or electronic level in the viewfinder and line the horizon up with it."],
            [f"Straighten by rotating {abs(d):.1f} degrees {'counter-clockwise' if d > 0 else 'clockwise'} (crop tool > Straighten)."],
            overlay="comp", fix="edit", force=True, boost=0.15 + 0.1 * abs(d))
    elif gap("level") >= 18:
        add("level", "Frame looks tilted", ["level"], [],
            ["Use the grid or electronic level; keep the sensor parallel to verticals for buildings."],
            ["Straighten with the crop tool; fix converging verticals with Upright/perspective correction."], overlay="comp", fix="edit")

    # ---------------- objects: distractions and the subject (from the panoptic masks) ----------------
    ob = ctx.get("objects") or {}
    oL, oW = ob.get(L), ob.get(W)
    obj_ev, obj_shoot = [], []
    if oL:
        dis = sorted([g for g in oL["segs"] if g.get("role") == "distraction"], key=lambda g: -g["delta"])[:3]
        if dis:
            ev, shoot, edit = [], [], []
            for g in dis:
                wh = _where(g)
                line = f"The {g['label']} ({wh}, no. {g['n']} in the Objects overlay): with it painted out, the network scores {nL} {g['delta']:.0f} points higher."
                cut = _cut(g)
                if cut:
                    line += f" It is also sliced by the {' and '.join(cut)} edge of the frame."
                ev.append(line)
            edge = any("edge" in _where(g) or _cut(g) for g in dis)
            shoot.append("Scan the frame for anything that isn't the subject before you press the shutter. A step sideways, a lower or higher viewpoint, or a tighter zoom usually hides it or pushes it out of frame.")
            if any(g["label"] in ("person", "car", "truck", "bus", "boat", "bird", "dog", "bicycle", "motorcycle") for g in dis):
                shoot.append("If it moves (people, vehicles), wait a few seconds for it to leave - or for it to reach a spot where it adds to the picture.")
            if edge:
                edit.append("It sits at the frame edge, so a small crop removes it.")
            edit.append("Otherwise remove it with the healing / clone / generative-remove tool; small objects against sky, water or grass heal cleanly.")
            one = len(dis) == 1
            add("distraction", f"Distracting {dis[0]['label']} at the {_where(dis[0])}" if one else "Distracting elements in the frame",
                ["subject_emphasis"], ev, shoot, edit, overlay="objects", fix="both", force=True,
                boost=min(1.2, 0.25 + sum(g["delta"] for g in dis) / 12))
    if oL and oW:
        sL_, sW_ = oL["geo"]["subject"], oW["geo"]["subject"]
        if sW_ and not sL_:
            obj_ev.append(f"{nW} has a clear subject (the {sW_['label']}, {sW_['where']} of the frame, {_pct(sW_['area'])} of the picture); no single object stands out in {nL}.")
        if sL_ and sW_:
            alike = sL_["label"] == sW_["label"]
            if (same or alike) and sW_["area"] >= 1.6 * sL_["area"] and sW_["area"] - sL_["area"] >= 0.01:
                r = math.sqrt(sW_["area"] / sL_["area"])
                obj_ev.append(f"The {sL_['label']} fills {_pct(sL_['area'])} of {nL} but the {sW_['label']} fills {_pct(sW_['area'])} of {nW}." if not alike else
                              f"The {sW_['label']} fills {_pct(sW_['area'])} of {nW} but only {_pct(sL_['area'])} of {nL}.")
                obj_shoot.append(f"Make the subject bigger: move closer or zoom in by about {min(r, 4):.1f}x.")
            if sL_["cut"] and not sW_["cut"]:
                obj_ev.append(f"The {sL_['label']} in {nL} is cut off by the {' and '.join(sL_['cut'])} edge of the frame; in {nW} the subject is whole.")
                obj_shoot.append("Either include the whole subject with a little breathing room, or crop in decisively - a subject clipped at the edge looks like an accident.")
            if sL_["separation"] is not None and sW_["separation"] is not None and sL_["separation"] < 12 and sL_["separation"] < 0.6 * sW_["separation"]:
                obj_ev.append(f"The {sL_['label']} in {nL} barely separates from what is behind it (tone/colour difference {sL_['separation']:.0f} vs {sW_['separation']:.0f} for {nW}'s subject; below about 10 a subject starts to merge).")
                obj_shoot.append("Move so the subject sits against something that contrasts with it - light against dark, or a different colour - or wait for light to pick it out.")
            if sL_["center_dist"] < 0.07 and sW_["thirds_dist"] < 0.10 and sL_["area"] < 0.2:
                obj_ev.append(f"The {sL_['label']} in {nL} sits dead-centre; in {nW} the subject is on a thirds point.")
        if same and abs(oL["geo"]["sky"] - oW["geo"]["sky"]) >= 0.12:
            more = oL["geo"]["sky"] > oW["geo"]["sky"]
            obj_ev.append(f"{nL} has no sky in the frame; {nW} gives it {_pct(oW['geo']['sky'], 0)}." if oL["geo"]["sky"] < 0.02 else
                          f"Sky takes up {_pct(oL['geo']['sky'], 0)} of {nL} against {_pct(oW['geo']['sky'], 0)} of {nW}.")
            if more and gap("lighting") + gap("content") > 0:
                obj_shoot.append("Tilt the camera down: unless the sky is the show, give that space to the land or water instead.")

    fr = ctx.get("frame")
    if fr:
        ev, shoot, edit = list(obj_ev), list(obj_shoot), []
        zf = fr["zoom"]
        if zf >= 1.12 and fr["inside"] >= 0.85:
            ev.append(f"Same scene, framed tighter: {nW} shows only the central {_pct(fr['area_frac'], 0)} of what {nL} includes (about {zf:.1f}x more zoom). See the Framing overlay.")
            shoot.append(f"Zoom in or walk closer by roughly {zf:.1f}x so the subject fills the frame and the empty margins go.")
            edit.append("You can get this framing by cropping to the box in the Framing overlay (the preview below does exactly that).")
        elif zf <= 0.9:
            ev.append(f"Same scene, framed wider: {nW} takes in about {fr['area_frac']:.1f}x the area of {nL}, giving the subject room and context.")
            shoot.append(f"Step back or use a wider lens (about {1 / zf:.1f}x wider field of view).")
            edit.append("This cannot be fixed by cropping - the missing surroundings were never captured.")
        if abs(fr["dx"]) >= 0.07 or abs(fr["dy"]) >= 0.07:
            hx = "right" if fr["dx"] > 0 else "left"
            hy = "down" if fr["dy"] > 0 else "up"
            parts = ([f"{abs(fr['dx']) * 100:.0f}% of the frame width to the {hx}"] if abs(fr["dx"]) >= 0.07 else []) + \
                    ([f"{abs(fr['dy']) * 100:.0f}% of the frame height {hy}"] if abs(fr["dy"]) >= 0.07 else [])
            ev.append(f"The winner's frame is centred {' and '.join(parts)} of yours.")
            shoot.append(f"Aim the camera further {' and '.join(([hx] if abs(fr['dx']) >= 0.07 else []) + ([hy] if abs(fr['dy']) >= 0.07 else []))}.")
        if abs(fr["rotation"]) >= 1.5:
            ev.append(f"Relative to {nW}, {nL} is rotated {abs(fr['rotation']):.1f} degrees.")
        if ev:
            add("framing", "Framing differs from the winner", ["framing", "subject_emphasis", "balance", "thirds"], ev, shoot, edit,
                overlay="frame", fix="both" if zf >= 1.12 else "reshoot", force=True, boost=0.3)
    comp_gap = max(gap("subject_emphasis"), gap("balance"), gap("thirds"), gap("framing"))
    if not any(i["id"] == "framing" for i in items) and (comp_gap >= 12 or (obj_ev and gap("subject_emphasis") >= 6)):
        ev, shoot = list(obj_ev), list(obj_shoot)
        if gap("subject_emphasis") >= 10:
            shoot.append("Decide what the one subject is, then remove competition: get closer, simplify the background, or change angle so the subject stands against something plain (sky, water, shadow).")
        if sl["edge_weight"] > 1.3 * sw["edge_weight"] and sl["edge_weight"] > 0.2:
            ev.append(f"{_pct(sl['edge_weight'], 0)} of the visual weight in {nL} sits right at the frame edges ({_pct(sw['edge_weight'], 0)} in {nW}) - edge clutter pulls the eye out of the picture.")
            shoot.append("Run your eye around the border before pressing the shutter; step sideways or zoom to drop anything bright or cut-off at the edges.")
        if gap("thirds") >= 10 or gap("balance") >= 10:
            if sl["center_dist"] < 0.1 and sw["center_dist"] > 0.14:
                ev.append(f"The main visual mass of {nL} is dead-centre; in {nW} it sits off-centre near a thirds line.")
            shoot.append("Try the subject on a thirds line with space in the direction it faces or moves, and balance it with a smaller secondary element on the other side.")
            shoot.append("For landscapes add a foreground anchor (rock, shoreline curve, canoe) to lead into the scene, and put the horizon on the upper or lower third rather than the middle.")
        if gap("framing") >= 10:
            shoot.append("Leave breathing room around the subject and avoid cutting things off at the frame edge.")
        add("composition", "Weaker composition", ["subject_emphasis", "thirds", "balance", "framing"], ev, shoot,
            ["A crop can tighten the frame and move the subject onto a stronger position - try a few crops before reshooting."],
            overlay="objects" if obj_ev else "comp", boost=0.1 + (0.15 if obj_ev else 0), force=bool(obj_ev))

    if gap("content") >= 14:
        add("content", "Less engaging subject matter", ["content"], [],
            ["Give the viewer something to land on: a person for scale, a boat, wildlife, a striking foreground, or a moment (weather, reflection, light beam) rather than the bare scene.",
             "Ask what the photo is about, and exclude everything that isn't that."],
            [], overlay="impact", fix="reshoot")

    items.sort(key=lambda i: -i["priority"])
    keep = items[:7]
    for it in items[7:]:            # a concrete, located distraction is always worth showing
        if it["id"] == "distraction":
            keep[-1] = it
    items = keep

    # ---------------- why it won / where the loser leads ----------------
    ranked = sorted([h for h in JUDGED if rel.get(h, True)], key=lambda h: -zgap(h))
    reasons = [dict(head=h, label=LABEL[h], gap=round(gap(h), 1), text=f"{LABEL[h]}: {_ord(pct[W][h])} percentile vs {_ord(pct[L][h])}. {BLURB[h]}")
               for h in ranked if gap(h) >= 8][:5]
    loser_leads = [dict(head=h, label=LABEL[h], gap=round(-gap(h), 1), text=f"{LABEL[h]}: {_ord(pct[L][h])} percentile vs {_ord(pct[W][h])}.")
                   for h in reversed(ranked) if gap(h) <= -8][:4]

    return dict(items=items, reasons=reasons, loser_leads=loser_leads)


def recipe_lines(rec: dict) -> list:
    """Human-readable slider list for the 'match the look' edit."""
    out = []

    def line(name, val, unit="", thr=0.0, fmt="{:+.0f}"):
        if abs(val) >= thr:
            out.append(dict(name=name, value=fmt.format(val) + unit))
    line("Exposure", rec["exposure_ev"], " EV", 0.15, "{:+.1f}")
    line("Contrast", rec["contrast_pct"] * 0.6, "", 6)
    line("Highlights", rec["highlights"] * 3.0, "", 8)
    line("Shadows", rec["shadows"] * 3.0, "", 8)
    line("Whites", rec["whites"] * 2.5, "", 8)
    line("Blacks", rec["blacks"] * 2.5, "", 8)
    line("Vibrance / saturation", rec["saturation_pct"] * 0.8, "", 6)
    if rec.get("color_ok"):
        if abs(rec["temp"]) >= 2: out.append(dict(name="Temperature", value=("warmer " if rec["temp"] > 0 else "cooler ") + f"{abs(rec['temp']) * 2.2:.0f}"))
        if abs(rec["tint"]) >= 2: out.append(dict(name="Tint", value=("magenta " if rec["tint"] > 0 else "green ") + f"{abs(rec['tint']) * 2.2:.0f}"))
    return out

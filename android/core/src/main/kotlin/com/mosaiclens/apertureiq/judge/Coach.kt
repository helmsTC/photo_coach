package com.mosaiclens.apertureiq.judge

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Turns score gaps + measured evidence into a verdict explanation and coaching.
 * A port of photocoach/coach.py: every piece of advice is tied to an attribute where the twin
 * network rates the winner higher and/or a number measured on the pixels or read from EXIF.
 */
object Coach {

    class Input(
        val info: JudgeInfo,
        val photos: List<ScoredPhoto>,                     // [a, b]
        val W: Int,
        val L: Int,
        val recipe: LookRecipe?,
        val frame: Framing?,
        val relation: String,
        val objects: List<Pair<List<Segment>, SceneGeometry?>>,
    )

    fun ordinal(p: Double): String {
        val n = max(1, min(99, rndInt(p)))
        val suffix = if (n % 100 in 10..20) "th" else when (n % 10) { 1 -> "st"; 2 -> "nd"; 3 -> "rd"; else -> "th" }
        return "$n$suffix"
    }

    fun pct(x: Double, d: Int = 1): String = usf("%.${d}f%%", 100 * x)

    private fun castWords(a: Double, b: Double): String {
        val w = ArrayList<String>()
        if (abs(b) >= 0.07) w.add(if (b > 0) "yellow/warm" else "blue/cool")
        if (abs(a) >= 0.07) w.add(if (a > 0) "magenta" else "green")
        return w.joinToString(" and ")
    }

    private val movers = setOf("person", "car", "truck", "bus", "boat", "bird", "dog", "bicycle", "motorcycle", "canoe", "kayak")

    private fun f1(v: Double) = usf("%.1f", v)
    private fun f2(v: Double) = usf("%.2f", v)
    private fun f0(v: Double) = usf("%.0f", v)

    // ---- coaching items

    fun items(c: Input): List<CoachItem> {
        val info = c.info
        val pw = c.photos[c.W]; val pl = c.photos[c.L]
        val nW = pw.name; val nL = pl.name
        val sw = pw.m; val sl = pl.m
        val eL = pl.exif; val eW = pw.exif
        val rec = c.recipe
        val same = c.relation == "aligned" || c.relation == "similar"

        fun idx(h: String): Int? = info.index(h)
        fun rel(h: String): Boolean = idx(h)?.let { info.heads[it].reliable } ?: true
        fun label(h: String): String = idx(h)?.let { info.heads[it].label } ?: h
        fun pW(h: String): Double = idx(h)?.let { pw.percentile[it] } ?: 50.0
        fun pL(h: String): Double = idx(h)?.let { pl.percentile[it] } ?: 50.0
        fun gap(h: String): Double = pW(h) - pL(h)
        fun zgap(h: String): Double = idx(h)?.let { pw.z[it] - pl.z[it] } ?: 0.0

        val items = ArrayList<CoachItem>()

        fun add(
            id: String, title: String, heads: List<String>, evidence: List<String>, shoot: List<String>, edit: List<String>,
            overlay: String? = null, fix: String = "both", boost: Double = 0.0, force: Boolean = false,
        ) {
            val hs = heads.filter { rel(it) }
            val g = hs.maxOfOrNull { gap(it) } ?: 0.0
            val zg = hs.maxOfOrNull { zgap(it) } ?: 0.0
            if (!force && g < 12) return
            val ev = ArrayList(evidence)
            for (h in hs) if (gap(h) >= 6) {
                ev.add("${label(h)}: the network puts $nL at the ${ordinal(pL(h))} percentile, $nW at the ${ordinal(pW(h))}.")
            }
            items.add(CoachItem(id, title, hs, g, max(zg, 0.0) + boost, ev, shoot, edit, overlay, fix))
        }

        // ---------------- exposure / highlights / shadows ----------------
        if (sl["clip_hi"] >= 0.01 && sl["clip_hi"] > 1.5 * sw["clip_hi"] + 0.005) {
            add(
                "highlights", "Blown-out highlights", listOf("exposure"),
                listOf("${pct(sl["clip_hi"])} of $nL is clipped to pure white (red in the Clipping view); $nW clips ${pct(sw["clip_hi"])}."),
                listOf(
                    "Dial in about -0.7 to -1.3 EV of exposure compensation and re-check the histogram - nothing should pile up against the right edge.",
                    "Turn on highlight warnings ('blinkies'/zebras) or use highlight-weighted metering.",
                    "If the scene has more range than the sensor (bright sky + dark land), bracket 3 frames and blend, use a graduated ND filter, or shoot RAW and expose for the sky.",
                ),
                listOf("Clipped areas hold no data, so this cannot be fully repaired - pulling Highlights/Whites down only turns white into flat grey. A RAW file may recover about a stop."),
                overlay = "clip", fix = "reshoot", boost = min(1.5, 12 * sl["clip_hi"]), force = sl["clip_hi"] >= 0.03,
            )
        }
        if (sl["clip_lo"] >= 0.03 && sl["clip_lo"] > 1.5 * sw["clip_lo"] + 0.01) {
            add(
                "shadows", "Crushed shadows", listOf("exposure", "contrast"),
                listOf("${pct(sl["clip_lo"])} of $nL is pure black with no detail (blue in the Clipping view); $nW: ${pct(sw["clip_lo"])}."),
                listOf(
                    "Give the shot more exposure (+0.5 to +1 EV) or add fill light; at a low ISO you can lift shadows later with little penalty.",
                    "Shoot RAW - JPEG throws away most shadow detail.",
                ),
                listOf("Raise Shadows and Blacks; if the file is a JPEG expect noise and banding to appear as you lift."),
                overlay = "clip", boost = min(1.0, 5 * sl["clip_lo"]), force = sl["clip_lo"] >= 0.08,
            )
        }
        if (rec != null && same && abs(rec.exposureEV) >= 0.4 && (gap("exposure") >= 5 || abs(rec.exposureEV) >= 0.8)) {
            val ev = rec.exposureEV
            val dark = ev > 0
            val comp = min(2.0, rnd(abs(ev) * 3) / 3)
            add(
                "brightness", "${if (dark) "Underexposed" else "Overexposed"} by about ${f1(abs(ev))} stops relative to the winner", listOf("exposure"),
                listOf("Mid-tones of $nL sit ${f1(abs(ev))} EV ${if (dark) "below" else "above"} those of $nW (median brightness ${f2(sl["luma_median"])} vs ${f2(sw["luma_median"])} on a 0-1 scale)."),
                listOf(
                    "Use ${if (dark) "+" else "-"}${f1(comp)} EV exposure compensation." +
                        (if (dark) " The meter was fooled by a bright sky/water/snow - meter off a mid-tone instead." else " The meter was fooled by a dark scene - meter off a mid-tone instead."),
                ),
                listOf(usf("Exposure %+.1f in your editor", ev) + (if (dark) " (watch that highlights don't clip as you brighten)." else " (this won't bring back anything already clipped).")),
                overlay = "hist", fix = "edit", boost = 0.3 * abs(ev), force = abs(ev) >= 0.8,
            )
        } else if (!same && gap("exposure") >= 12) {
            val dark = sl["luma_median"] < 0.3; val bright = sl["luma_median"] > 0.68
            if (dark || bright) {
                add(
                    "brightness", if (dark) "Too dark overall" else "Too bright overall", listOf("exposure"),
                    listOf("Median brightness of $nL is ${f2(sl["luma_median"])} on a 0-1 scale."),
                    listOf("Use ${if (dark) "+" else "-"}0.7 EV exposure compensation and check the histogram."),
                    listOf("${if (dark) "Raise" else "Lower"} Exposure until the histogram is centred for this scene."), overlay = "hist", fix = "edit",
                )
            }
        }

        // ---------------- contrast ----------------
        val flat = sl["contrast_rms"] < 0.82 * sw["contrast_rms"] || sl["dyn_range"] < 0.8 * sw["dyn_range"]
        val harsh = sl["contrast_rms"] > 1.25 * sw["contrast_rms"] && (sl["clip_hi"] + sl["clip_lo"]) > 0.04
        if (flat) {
            val amount = if (rec != null && same) usf(" about %+d", rndInt(rec.contrastPct * 0.6)) else ""
            add(
                "contrast", "Flat, hazy tones", listOf("contrast"),
                listOf("Tonal spread of $nL is ${f2(sl["contrast_rms"])} vs ${f2(sw["contrast_rms"])} for $nW; its tones span ${pct(sl["dyn_range"], 0)} of the available range vs ${pct(sw["dyn_range"], 0)}."),
                listOf(
                    "Flat tone usually comes from flat light or haze: shoot with the sun to the side rather than behind you or overhead, and avoid midday haze.",
                    "A polarising filter cuts atmospheric haze and reflections; a lens hood stops veiling flare. Clean the lens - a smeared phone lens causes exactly this.",
                ),
                listOf("Set a true black and white point (Blacks down, Whites up), add Contrast$amount and a little Dehaze/Clarity."),
                overlay = "hist", fix = "edit", boost = 0.2,
            )
        } else if (harsh) {
            add(
                "contrast", "Harsh contrast", listOf("contrast", "lighting"),
                listOf("Tonal spread of $nL is ${f2(sl["contrast_rms"])} vs ${f2(sw["contrast_rms"])}, with ${pct(sl["clip_hi"] + sl["clip_lo"])} of pixels clipped at either end."),
                listOf(
                    "Hard midday sun is the usual cause - shoot early or late, wait for thin cloud, or move the subject into open shade.",
                    "Bracket exposures / use HDR mode when the scene's range is this wide.",
                ),
                listOf("Lower Contrast, pull Highlights down and Shadows up."), overlay = "clip",
            )
        } else if (gap("contrast") >= 12) {
            add(
                "contrast", "Weaker tonal separation", listOf("contrast"), emptyList(),
                listOf("Look for light that separates the subject from the background (side or back light)."),
                listOf("Shape the tone curve: gentle S-curve, set black and white points."), overlay = "hist", fix = "edit",
            )
        }

        // ---------------- light ----------------
        if (gap("lighting") >= 10) {
            val ev = ArrayList<String>()
            if (sw["local_contrast"] > 1.2 * sl["local_contrast"]) {
                ev.add("$nW has ${f1(sw["local_contrast"] / max(sl["local_contrast"], 1e-4))}x the local light-and-shade modelling of $nL - a sign of directional light.")
            }
            if (sw["cast_b"] - sl["cast_b"] > 0.08) {
                ev.add("$nW is lit noticeably warmer (golden-hour-like colour) than $nL.")
            }
            add(
                "lighting", "The light itself is weaker", listOf("lighting"), ev,
                listOf(
                    "Light is the one thing you cannot add later. Come back in the hour after sunrise or before sunset when it is low, warm and directional - or right after weather clears.",
                    "Put the light to the side of or behind the subject for shape and depth; front-lit and overhead midday light flattens everything.",
                    "For a landmark view, check where the sun will be (PhotoPills / The Photographer's Ephemeris) and plan the time of day the scene is lit, not in shadow.",
                ),
                listOf("Dodging and burning (local brightening/darkening) can emphasise existing light, but cannot change its direction."),
                overlay = "impact", fix = "reshoot", boost = 0.25,
            )
        }

        // ---------------- colour ----------------
        val muted = sl["colorfulness"] < 0.78 * sw["colorfulness"]
        val garish = sl["sat_mean"] > 0.62 && sl["sat_mean"] > 1.3 * sw["sat_mean"]
        if (muted && (gap("vivid_color") >= 6 || gap("color_harmony") >= 8)) {
            val vib = min(60, max(10, rndInt((rec?.saturationPct ?: 25.0) * 0.8)))
            add(
                "muted", "Muted colour", listOf("vivid_color", "color_harmony"),
                listOf("Colourfulness of $nL measures ${f2(sl["colorfulness"])} vs ${f2(sw["colorfulness"])} for $nW."),
                listOf(
                    "Colour is richest in low-angle light and just after rain; a polariser deepens skies, foliage and water by removing glare.",
                    "Underexposing by 1/3 stop keeps bright colours from washing out.",
                ),
                listOf("Add Vibrance (about +$vib) rather than Saturation so already-strong colours don't go neon."),
                overlay = "palette", fix = "edit", boost = 0.15,
            )
        } else if (garish) {
            add(
                "garish", "Over-saturated colour", listOf("color_accuracy", "color_harmony"),
                listOf("Mean saturation of $nL is ${pct(sl["sat_mean"], 0)} vs ${pct(sw["sat_mean"], 0)} for $nW."),
                listOf("Turn off 'Vivid' picture styles / scene enhancers."),
                listOf("Reduce Saturation/Vibrance until skin, sky and foliage look believable; over-saturation destroys detail in the most colourful areas."),
                overlay = "palette", fix = "edit", boost = 0.2, force = true,
            )
        } else if (gap("vivid_color") >= 12 || gap("color_harmony") >= 12) {
            add(
                "palette", "Less cohesive colour palette", listOf("color_harmony", "vivid_color"), emptyList(),
                listOf("Simplify the palette: frame so two or three related or complementary colours dominate, and exclude stray bright objects."),
                listOf("Use HSL/colour-grading tools to pull distracting hues toward the main palette."), overlay = "palette",
            )
        }
        val ca = sl["neutral_cast_a"]; val cb = sl["neutral_cast_b"]; val wa = sw["neutral_cast_a"]; val wb = sw["neutral_cast_b"]
        val cw = castWords(ca, cb)
        if ((abs(ca - wa) >= 0.09 || abs(cb - wb) >= 0.12) && (abs(ca) > abs(wa) + 0.04 || abs(cb) > abs(wb) + 0.06) && cw.isNotEmpty()) {
            add(
                "cast", "Colour cast ($cw)", listOf("color_accuracy"),
                listOf("Areas that should be neutral in $nL lean $cw; in $nW they are closer to neutral."),
                listOf("Set white balance for the light (Daylight / Cloudy / Shade) instead of Auto, or shoot RAW and set it afterwards."),
                listOf(
                    "Move Temperature toward ${if (cb > wb) "blue" else "yellow"}" +
                        (if (abs(ca - wa) >= 0.07) " and Tint toward ${if (ca > wa) "green" else "magenta"}" else "") +
                        ", or click the white-balance eyedropper on something that should be grey.",
                ),
                overlay = "palette", fix = "edit", boost = 0.15, force = gap("color_accuracy") >= 4,
            )
        }

        // ---------------- sharpness / blur / noise ----------------
        val soft = sl["edge_crispness"] < 0.85 * sw["edge_crispness"] || sl["sharp_top"] < 0.6 * sw["sharp_top"]
        if (gap("sharpness") >= 12 || (soft && gap("sharpness") >= 6)) {
            val ev = ArrayList<String>(); val shoot = ArrayList<String>()
            if (soft) {
                ev.add("Edges in the sharpest part of $nL are softer than in $nW (edge crispness ${f2(sl["edge_crispness"])} vs ${f2(sw["edge_crispness"])}; see the Focus view).")
            }
            val sh = eL.shutter
            val f35 = eL.focal35?.toDouble() ?: eL.focal
            if (sh != null && f35 != null && sh > 1 / max(f35, 30.0)) {
                ev.add("EXIF: ${ExifInfo.shutterString(sh)} at ${fmt(f35)}mm - slower than the 1/focal-length hand-holding rule, so camera shake is likely.")
                shoot.add("Use 1/${max(2 * f35, 125.0).toInt()}s or faster hand-held, or put the camera on a tripod and use a 2-second timer.")
            }
            val ap = eL.aperture
            if (ap != null && ap >= 16) {
                ev.add("EXIF: f/${fmt(ap)} - diffraction softens the whole frame at apertures this small.")
                shoot.add("Stay around f/5.6-f/11, where most lenses are sharpest.")
            }
            shoot.add("Focus deliberately: tap/half-press on the subject (for landscapes, about a third of the way into the scene) and check at 100% before leaving.")
            shoot.add("Brace the camera, squeeze the shutter gently, and keep image stabilisation on when hand-holding.")
            add(
                "sharpness", "Softer detail", listOf("sharpness", "motion_blur"), ev, shoot,
                listOf("Capture sharpening helps slightly-soft files, but cannot rescue missed focus or shake."),
                overlay = "focus", fix = "reshoot", boost = if (soft) 0.25 else 0.0,
            )
        }
        val noisy = sl["noise_luma"] > 1.6 * sw["noise_luma"] && sl["noise_luma"] > 0.006
        if (gap("noise") >= 14 || (noisy && gap("noise") >= 8)) {
            val ev = ArrayList<String>()
            if (noisy) ev.add("Grain in smooth areas of $nL measures ${f1(1000 * sl["noise_luma"])} vs ${f1(1000 * sw["noise_luma"])} for $nW (x1000, luminance noise).")
            val iso = eL.iso
            if (iso != null && iso >= 800) ev.add("EXIF: ISO $iso.")
            add(
                "noise", "Visible noise", listOf("noise"), ev,
                listOf(
                    "Use the lowest ISO the light allows: add light, open the aperture, or use a tripod with a longer shutter.",
                    "Expose generously - brightening a dark file afterwards multiplies noise.",
                ),
                listOf("Apply luminance + colour noise reduction (AI denoise tools work well), then sharpen lightly."),
                overlay = "focus", boost = if (noisy) 0.1 else 0.0,
            )
        }

        // ---------------- depth of field ----------------
        if (gap("depth_of_field") >= 20 || (gap("depth_of_field") >= 12 && sw["focus_spread"] > sl["focus_spread"] + 0.4)) {
            val ev = ArrayList<String>()
            if (sw["focus_spread"] > sl["focus_spread"] + 0.4) {
                ev.add("$nW has a much stronger sharp-subject / soft-background split (focus spread ${f1(sw["focus_spread"])} vs ${f1(sl["focus_spread"])}; compare the Focus views).")
            }
            val aw = eW.aperture; val al = eL.aperture
            if (aw != null && al != null) ev.add("EXIF: $nW was shot at f/${fmt(aw)}, $nL at f/${fmt(al)}.")
            add(
                "dof", "Less subject separation (depth of field)", listOf("depth_of_field"), ev,
                listOf(
                    "To blur the background: use the widest aperture (lowest f-number), zoom in or use a longer lens, get closer to the subject, and put distance between subject and background.",
                    "On a phone, use the 2x/3x lens or Portrait mode and move in close.",
                ),
                listOf("Background blur added in software rarely looks natural around hair and fine edges - get it in camera."),
                overlay = "focus", fix = "reshoot",
            )
        }

        // ---------------- composition: level ----------------
        // A line detector alone is fooled by ridgelines and receding shorelines, so a tilt is only
        // reported when the learned level-horizon head agrees.
        val tl = sl["tilt_deg"]; val tw = sw["tilt_deg"]
        val tiltOK = abs(tl) >= 1 && abs(tl) <= 8 && sl["tilt_conf"] >= 0.6 && abs(tl) > abs(tw) + 0.5 && rel("level") && gap("level") >= 10
        if (tiltOK) {
            add(
                "level", "Horizon tilted ${f1(abs(tl))} degrees", listOf("level"),
                listOf("The horizon in $nL (see the Composition view) runs ${f1(abs(tl))} degrees off level, ${if (tl > 0) "down to the right" else "down to the left"}; $nW is within ${f1(abs(tw))} degrees. Ignore this if that line is a shoreline receding at an angle."),
                listOf("Switch on the grid or electronic level in the viewfinder and line the horizon up with it."),
                listOf("Straighten by rotating ${f1(abs(tl))} degrees ${if (tl > 0) "counter-clockwise" else "clockwise"} (crop tool > Straighten)."),
                overlay = "comp", fix = "edit", boost = 0.15 + 0.1 * abs(tl), force = true,
            )
        } else if (gap("level") >= 18) {
            add(
                "level", "Frame looks tilted", listOf("level"), emptyList(),
                listOf("Use the grid or electronic level; keep the sensor parallel to verticals for buildings."),
                listOf("Straighten with the crop tool; fix converging verticals with Upright/perspective correction."), overlay = "comp", fix = "edit",
            )
        }

        // ---------------- objects: distractions and the subject ----------------
        val oL = c.objects.getOrNull(c.L)
        val oW = c.objects.getOrNull(c.W)
        val objEv = ArrayList<String>(); val objShoot = ArrayList<String>()
        if (oL != null) {
            val dis = oL.first.filter { it.role == "distraction" }.sortedByDescending { it.delta }.take(3)
            if (dis.isNotEmpty()) {
                val ev = ArrayList<String>(); val shoot = ArrayList<String>(); val edit = ArrayList<String>()
                for (g in dis) {
                    var line = "The ${g.label} (${Objects.whereText(g)}, no. ${g.n} in the Objects view): with it painted out, the network scores $nL ${f0(g.delta)} points higher."
                    val cut = Objects.cutOff(g)
                    if (cut.isNotEmpty()) line += " It is also sliced by the ${cut.joinToString(" and ")} edge of the frame."
                    ev.add(line)
                }
                val edge = dis.any { Objects.whereText(it).contains("edge") || Objects.cutOff(it).isNotEmpty() }
                shoot.add("Scan the frame for anything that isn't the subject before you press the shutter. A step sideways, a lower or higher viewpoint, or a tighter zoom usually hides it or pushes it out of frame.")
                if (dis.any { movers.contains(it.label) }) {
                    shoot.add("If it moves (people, vehicles), wait a few seconds for it to leave - or for it to reach a spot where it adds to the picture.")
                }
                if (edge) edit.add("It sits at the frame edge, so a small crop removes it.")
                edit.add("Otherwise remove it with the healing / clone / generative-remove tool; small objects against sky, water or grass heal cleanly.")
                add(
                    "distraction",
                    if (dis.size == 1) "Distracting ${dis[0].label} at the ${Objects.whereText(dis[0])}" else "Distracting elements in the frame",
                    listOf("subject_emphasis"), ev, shoot, edit, overlay = "objects", fix = "both",
                    boost = min(1.2, 0.25 + dis.sumOf { it.delta } / 12), force = true,
                )
            }
        }
        val gL = oL?.second; val gW = oW?.second
        if (gL != null && gW != null) {
            val sL = gL.subject; val sW = gW.subject
            if (sW != null && sL == null) {
                objEv.add("$nW has a clear subject (the ${sW.label}, ${sW.whereText} of the frame, ${pct(sW.area)} of the picture); no single object stands out in $nL.")
            }
            if (sL != null && sW != null) {
                val alike = sL.label == sW.label
                if ((same || alike) && sW.area >= 1.6 * sL.area && sW.area - sL.area >= 0.01) {
                    val r = sqrt(sW.area / sL.area)
                    objEv.add(
                        if (alike) "The ${sW.label} fills ${pct(sW.area)} of $nW but only ${pct(sL.area)} of $nL."
                        else "The ${sL.label} fills ${pct(sL.area)} of $nL but the ${sW.label} fills ${pct(sW.area)} of $nW.",
                    )
                    objShoot.add("Make the subject bigger: move closer or zoom in by about ${f1(min(r, 4.0))}x.")
                }
                if (sL.cut.isNotEmpty() && sW.cut.isEmpty()) {
                    objEv.add("The ${sL.label} in $nL is cut off by the ${sL.cut.joinToString(" and ")} edge of the frame; in $nW the subject is whole.")
                    objShoot.add("Either include the whole subject with a little breathing room, or crop in decisively - a subject clipped at the edge looks like an accident.")
                }
                val a = sL.separation; val b = sW.separation
                if (a != null && b != null && a < 12 && a < 0.6 * b) {
                    objEv.add("The ${sL.label} in $nL barely separates from what is behind it (tone/colour difference ${f0(a)} vs ${f0(b)} for $nW's subject; below about 10 a subject starts to merge).")
                    objShoot.add("Move so the subject sits against something that contrasts with it - light against dark, or a different colour - or wait for light to pick it out.")
                }
                if (sL.centerDist < 0.07 && sW.thirdsDist < 0.10 && sL.area < 0.2) {
                    objEv.add("The ${sL.label} in $nL sits dead-centre; in $nW the subject is on a thirds point.")
                }
            }
            if (same && abs(gL.sky - gW.sky) >= 0.12) {
                val more = gL.sky > gW.sky
                objEv.add(
                    if (gL.sky < 0.02) "$nL has no sky in the frame; $nW gives it ${pct(gW.sky, 0)}."
                    else "Sky takes up ${pct(gL.sky, 0)} of $nL against ${pct(gW.sky, 0)} of $nW.",
                )
                if (more && gap("lighting") + gap("content") > 0) {
                    objShoot.add("Tilt the camera down: unless the sky is the show, give that space to the land or water instead.")
                }
            }
        }

        val fr = c.frame
        if (fr != null) {
            val ev = ArrayList(objEv); val shoot = ArrayList(objShoot); val edit = ArrayList<String>()
            val zf = fr.zoom
            if (zf >= 1.12 && fr.inside >= 0.85) {
                ev.add("Same scene, framed tighter: $nW shows only the central ${pct(fr.areaFrac, 0)} of what $nL includes (about ${f1(zf)}x more zoom). See the Framing view.")
                shoot.add("Zoom in or walk closer by roughly ${f1(zf)}x so the subject fills the frame and the empty margins go.")
                edit.add("You can get this framing by cropping to the box in the Framing view (the crop preview does exactly that).")
            } else if (zf <= 0.9) {
                ev.add("Same scene, framed wider: $nW takes in about ${f1(fr.areaFrac)}x the area of $nL, giving the subject room and context.")
                shoot.add("Step back or use a wider lens (about ${f1(1 / zf)}x wider field of view).")
                edit.add("This cannot be fixed by cropping - the missing surroundings were never captured.")
            }
            if (abs(fr.dx) >= 0.07 || abs(fr.dy) >= 0.07) {
                val hx = if (fr.dx > 0) "right" else "left"; val hy = if (fr.dy > 0) "down" else "up"
                val parts = ArrayList<String>(); val dirs = ArrayList<String>()
                if (abs(fr.dx) >= 0.07) { parts.add("${rndInt(abs(fr.dx) * 100)}% of the frame width to the $hx"); dirs.add(hx) }
                if (abs(fr.dy) >= 0.07) { parts.add("${rndInt(abs(fr.dy) * 100)}% of the frame height $hy"); dirs.add(hy) }
                ev.add("The winner's frame is centred ${parts.joinToString(" and ")} of yours.")
                shoot.add("Aim the camera further ${dirs.joinToString(" and ")}.")
            }
            if (abs(fr.rotation) >= 1.5) {
                ev.add("Relative to $nW, $nL is rotated ${f1(abs(fr.rotation))} degrees.")
            }
            if (ev.isNotEmpty()) {
                add(
                    "framing", "Framing differs from the winner", listOf("framing", "subject_emphasis", "balance", "thirds"), ev, shoot, edit,
                    overlay = "frame", fix = if (zf >= 1.12) "both" else "reshoot", boost = 0.3, force = true,
                )
            }
        }
        val compGap = listOf("subject_emphasis", "balance", "thirds", "framing").maxOf { gap(it) }
        if (items.none { it.id == "framing" } && (compGap >= 12 || (objEv.isNotEmpty() && gap("subject_emphasis") >= 6))) {
            val ev = ArrayList(objEv); val shoot = ArrayList(objShoot)
            if (gap("subject_emphasis") >= 10) {
                shoot.add("Decide what the one subject is, then remove competition: get closer, simplify the background, or change angle so the subject stands against something plain (sky, water, shadow).")
            }
            if (sl["edge_weight"] > 1.3 * sw["edge_weight"] && sl["edge_weight"] > 0.2) {
                ev.add("${pct(sl["edge_weight"], 0)} of the visual weight in $nL sits right at the frame edges (${pct(sw["edge_weight"], 0)} in $nW) - edge clutter pulls the eye out of the picture.")
                shoot.add("Run your eye around the border before pressing the shutter; step sideways or zoom to drop anything bright or cut-off at the edges.")
            }
            if (gap("thirds") >= 10 || gap("balance") >= 10) {
                if (sl["center_dist"] < 0.1 && sw["center_dist"] > 0.14) {
                    ev.add("The main visual mass of $nL is dead-centre; in $nW it sits off-centre near a thirds line.")
                }
                shoot.add("Try the subject on a thirds line with space in the direction it faces or moves, and balance it with a smaller secondary element on the other side.")
                shoot.add("For landscapes add a foreground anchor (rock, shoreline curve, canoe) to lead into the scene, and put the horizon on the upper or lower third rather than the middle.")
            }
            if (gap("framing") >= 10) {
                shoot.add("Leave breathing room around the subject and avoid cutting things off at the frame edge.")
            }
            add(
                "composition", "Weaker composition", listOf("subject_emphasis", "thirds", "balance", "framing"), ev, shoot,
                listOf("A crop can tighten the frame and move the subject onto a stronger position - try a few crops before reshooting."),
                overlay = if (objEv.isEmpty()) "comp" else "objects", boost = 0.1 + (if (objEv.isEmpty()) 0.0 else 0.15), force = objEv.isNotEmpty(),
            )
        }

        if (gap("content") >= 14) {
            add(
                "content", "Less engaging subject matter", listOf("content"), emptyList(),
                listOf(
                    "Give the viewer something to land on: a person for scale, a boat, wildlife, a striking foreground, or a moment (weather, reflection, light beam) rather than the bare scene.",
                    "Ask what the photo is about, and exclude everything that isn't that.",
                ),
                emptyList(), overlay = "impact", fix = "reshoot",
            )
        }

        val sorted = items.sortedByDescending { it.priority }
        val keep = ArrayList(sorted.take(7))
        for (it in sorted.drop(7)) if (it.id == "distraction") keep[keep.size - 1] = it   // a located distraction is always worth showing
        return keep
    }

    // ---- the whole result

    fun build(c: Input, pA: Double, extras: List<PhotoExtras>, similarity: Double, previews: List<Preview>, seconds: Double): Comparison {
        val info = c.info
        val a = c.photos[0]; val b = c.photos[1]
        val w = c.photos[c.W]; val l = c.photos[c.L]
        val conf = max(pA, 1 - pA)
        val ia = info.ia; val it = info.itech

        val dzA = w.z[ia] - l.z[ia]; val dzT = w.z[it] - l.z[it]
        val summary = when {
            dzA > 0.25 && dzT > 0.25 -> "It is ahead on both aesthetic appeal and technical quality."
            dzA > 0.25 -> if (dzT > -0.25) "It wins on aesthetic appeal; technically the two are about level."
            else "It wins on aesthetic appeal; ${l.name} is actually the cleaner capture technically."
            dzT > 0.25 -> if (dzA > -0.25) "It wins on technical quality; aesthetically the two are about level."
            else "It wins on technical quality; ${l.name} is the more appealing image aesthetically."
            else -> "The two are close on both overall scores."
        }
        val strength = if (conf >= 0.85) "clearly" else if (conf >= 0.68) "fairly confidently" else "narrowly"

        // why it won / where the loser leads
        val judged = info.heads.indices.filter { info.groups.contains(info.heads[it].group) && info.heads[it].reliable }
        val ranked = judged.sortedByDescending { w.z[it] - l.z[it] }
        fun gap(i: Int) = w.percentile[i] - l.percentile[i]
        val reasons = ranked.filter { gap(it) >= 8 }.take(5).map { i ->
            Reason(
                info.heads[i].key, info.heads[i].label, gap(i),
                "${info.heads[i].label}: ${ordinal(w.percentile[i])} percentile vs ${ordinal(l.percentile[i])}. ${info.heads[i].blurb}",
                w.percentile[i], l.percentile[i],
            )
        }
        val leads = ranked.reversed().filter { gap(it) <= -8 }.take(4).map { i ->
            Reason(
                info.heads[i].key, info.heads[i].label, -gap(i),
                "${info.heads[i].label}: ${ordinal(l.percentile[i])} percentile vs ${ordinal(w.percentile[i])}.",
                w.percentile[i], l.percentile[i],
            )
        }

        val rows = ArrayList<AttributeRow>()
        for (g in info.groups) {
            for (i in info.heads.indices) {
                val h = info.heads[i]
                if (h.group == g) rows.add(AttributeRow(h.key, h.label, h.blurb, g, h.reliable, a.percentile[i], b.percentile[i]))
            }
        }

        val relText = when (c.relation) {
            "aligned" -> "Same scene - the two frames were matched point-to-point, so framing and tone are compared directly."
            "similar" -> "Similar subject - tone and colour are compared directly; framing is judged separately for each photo."
            else -> "Different scenes - each photo is judged on its own merits; colour-matching advice is limited to tone."
        }

        return Comparison(
            a, b, extras, pA, c.W == 0,
            "${w.name} wins $strength", summary, conf < 0.56,
            reasons, leads, items(c), rows,
            c.relation, relText, similarity, c.frame, c.recipe, previews, seconds,
        )
    }
}

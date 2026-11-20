import Foundation

/// Turns score gaps + measured evidence into a verdict explanation and coaching.
/// A port of photocoach/coach.py: every piece of advice is tied to an attribute where the twin
/// network rates the winner higher and/or a number measured on the pixels or read from EXIF.
enum Coach {

    struct Input {
        let info: JudgeInfo
        let photos: [ScoredPhoto]                     // [a, b]
        let W: Int
        let L: Int
        let recipe: LookRecipe?
        let frame: Framing?
        let relation: String
        let objects: [([Segment], SceneGeometry?)]
    }

    static func ordinal(_ p: Double) -> String {
        let n = max(1, min(99, Int(p.rounded())))
        let suffix: String
        if (10...20).contains(n % 100) { suffix = "th" } else {
            switch n % 10 { case 1: suffix = "st"; case 2: suffix = "nd"; case 3: suffix = "rd"; default: suffix = "th" }
        }
        return "\(n)\(suffix)"
    }

    static func pct(_ x: Double, _ d: Int = 1) -> String { String(format: "%.\(d)f%%", 100 * x) }

    private static func castWords(_ a: Double, _ b: Double) -> String {
        var w: [String] = []
        if abs(b) >= 0.07 { w.append(b > 0 ? "yellow/warm" : "blue/cool") }
        if abs(a) >= 0.07 { w.append(a > 0 ? "magenta" : "green") }
        return w.joined(separator: " and ")
    }

    private static let movers: Set<String> = ["person", "car", "truck", "bus", "boat", "bird", "dog", "bicycle", "motorcycle", "canoe", "kayak"]

    // MARK: - coaching items

    static func items(_ c: Input) -> [CoachItem] {
        let info = c.info
        let pw = c.photos[c.W], pl = c.photos[c.L]
        let nW = pw.name, nL = pl.name
        let sw = pw.m, sl = pl.m
        let eL = pl.exif, eW = pw.exif
        let rec = c.recipe
        let same = c.relation == "aligned" || c.relation == "similar"

        func idx(_ h: String) -> Int? { info.index(of: h) }
        func rel(_ h: String) -> Bool { idx(h).map { info.heads[$0].reliable } ?? true }
        func label(_ h: String) -> String { idx(h).map { info.heads[$0].label } ?? h }
        func pW(_ h: String) -> Double { idx(h).map { pw.percentile[$0] } ?? 50 }
        func pL(_ h: String) -> Double { idx(h).map { pl.percentile[$0] } ?? 50 }
        func gap(_ h: String) -> Double { pW(h) - pL(h) }
        func zgap(_ h: String) -> Double { idx(h).map { pw.z[$0] - pl.z[$0] } ?? 0 }

        var items: [CoachItem] = []

        func add(_ id: String, _ title: String, _ heads: [String], _ evidence: [String], _ shoot: [String], _ edit: [String],
                 overlay: String? = nil, fix: String = "both", boost: Double = 0, force: Bool = false) {
            let hs = heads.filter { rel($0) }
            let g = hs.map(gap).max() ?? 0
            let zg = hs.map(zgap).max() ?? 0
            if !force && g < 12 { return }
            var ev = evidence
            for h in hs where gap(h) >= 6 {
                ev.append("\(label(h)): the network puts \(nL) at the \(ordinal(pL(h))) percentile, \(nW) at the \(ordinal(pW(h))).")
            }
            items.append(CoachItem(id: id, title: title, heads: hs, gap: g, priority: max(zg, 0) + boost,
                                   evidence: ev, shoot: shoot, edit: edit, overlay: overlay, fix: fix))
        }

        // ---------------- exposure / highlights / shadows ----------------
        if sl["clip_hi"] >= 0.01 && sl["clip_hi"] > 1.5 * sw["clip_hi"] + 0.005 {
            add("highlights", "Blown-out highlights", ["exposure"],
                ["\(pct(sl["clip_hi"])) of \(nL) is clipped to pure white (red in the Clipping view); \(nW) clips \(pct(sw["clip_hi"]))."],
                ["Dial in about -0.7 to -1.3 EV of exposure compensation and re-check the histogram - nothing should pile up against the right edge.",
                 "Turn on highlight warnings ('blinkies'/zebras) or use highlight-weighted metering.",
                 "If the scene has more range than the sensor (bright sky + dark land), bracket 3 frames and blend, use a graduated ND filter, or shoot RAW and expose for the sky."],
                ["Clipped areas hold no data, so this cannot be fully repaired - pulling Highlights/Whites down only turns white into flat grey. A RAW file may recover about a stop."],
                overlay: "clip", fix: "reshoot", boost: min(1.5, 12 * sl["clip_hi"]), force: sl["clip_hi"] >= 0.03)
        }
        if sl["clip_lo"] >= 0.03 && sl["clip_lo"] > 1.5 * sw["clip_lo"] + 0.01 {
            add("shadows", "Crushed shadows", ["exposure", "contrast"],
                ["\(pct(sl["clip_lo"])) of \(nL) is pure black with no detail (blue in the Clipping view); \(nW): \(pct(sw["clip_lo"]))."],
                ["Give the shot more exposure (+0.5 to +1 EV) or add fill light; at a low ISO you can lift shadows later with little penalty.",
                 "Shoot RAW - JPEG throws away most shadow detail."],
                ["Raise Shadows and Blacks; if the file is a JPEG expect noise and banding to appear as you lift."],
                overlay: "clip", boost: min(1.0, 5 * sl["clip_lo"]), force: sl["clip_lo"] >= 0.08)
        }
        if let rec, same, abs(rec.exposureEV) >= 0.4, gap("exposure") >= 5 || abs(rec.exposureEV) >= 0.8 {
            let ev = rec.exposureEV
            let dark = ev > 0
            let comp = min(2.0, (abs(ev) * 3).rounded() / 3)
            add("brightness", "\(dark ? "Underexposed" : "Overexposed") by about \(String(format: "%.1f", abs(ev))) stops relative to the winner", ["exposure"],
                ["Mid-tones of \(nL) sit \(String(format: "%.1f", abs(ev))) EV \(dark ? "below" : "above") those of \(nW) (median brightness \(String(format: "%.2f", sl["luma_median"])) vs \(String(format: "%.2f", sw["luma_median"])) on a 0-1 scale)."],
                ["Use \(dark ? "+" : "-")\(String(format: "%.1f", comp)) EV exposure compensation." + (dark ? " The meter was fooled by a bright sky/water/snow - meter off a mid-tone instead." : " The meter was fooled by a dark scene - meter off a mid-tone instead.")],
                [String(format: "Exposure %+.1f in your editor", ev) + (dark ? " (watch that highlights don't clip as you brighten)." : " (this won't bring back anything already clipped).")],
                overlay: "hist", fix: "edit", boost: 0.3 * abs(ev), force: abs(ev) >= 0.8)
        } else if !same && gap("exposure") >= 12 {
            let dark = sl["luma_median"] < 0.3, bright = sl["luma_median"] > 0.68
            if dark || bright {
                add("brightness", dark ? "Too dark overall" : "Too bright overall", ["exposure"],
                    ["Median brightness of \(nL) is \(String(format: "%.2f", sl["luma_median"])) on a 0-1 scale."],
                    ["Use \(dark ? "+" : "-")0.7 EV exposure compensation and check the histogram."],
                    ["\(dark ? "Raise" : "Lower") Exposure until the histogram is centred for this scene."], overlay: "hist", fix: "edit")
            }
        }

        // ---------------- contrast ----------------
        let flat = sl["contrast_rms"] < 0.82 * sw["contrast_rms"] || sl["dyn_range"] < 0.8 * sw["dyn_range"]
        let harsh = sl["contrast_rms"] > 1.25 * sw["contrast_rms"] && (sl["clip_hi"] + sl["clip_lo"]) > 0.04
        if flat {
            let amount = (rec != nil && same) ? String(format: " about %+d", Int((rec!.contrastPct * 0.6).rounded())) : ""
            add("contrast", "Flat, hazy tones", ["contrast"],
                ["Tonal spread of \(nL) is \(String(format: "%.2f", sl["contrast_rms"])) vs \(String(format: "%.2f", sw["contrast_rms"])) for \(nW); its tones span \(pct(sl["dyn_range"], 0)) of the available range vs \(pct(sw["dyn_range"], 0))."],
                ["Flat tone usually comes from flat light or haze: shoot with the sun to the side rather than behind you or overhead, and avoid midday haze.",
                 "A polarising filter cuts atmospheric haze and reflections; a lens hood stops veiling flare. Clean the lens - a smeared phone lens causes exactly this."],
                ["Set a true black and white point (Blacks down, Whites up), add Contrast\(amount) and a little Dehaze/Clarity."],
                overlay: "hist", fix: "edit", boost: 0.2)
        } else if harsh {
            add("contrast", "Harsh contrast", ["contrast", "lighting"],
                ["Tonal spread of \(nL) is \(String(format: "%.2f", sl["contrast_rms"])) vs \(String(format: "%.2f", sw["contrast_rms"])), with \(pct(sl["clip_hi"] + sl["clip_lo"])) of pixels clipped at either end."],
                ["Hard midday sun is the usual cause - shoot early or late, wait for thin cloud, or move the subject into open shade.",
                 "Bracket exposures / use HDR mode when the scene's range is this wide."],
                ["Lower Contrast, pull Highlights down and Shadows up."], overlay: "clip")
        } else if gap("contrast") >= 12 {
            add("contrast", "Weaker tonal separation", ["contrast"], [],
                ["Look for light that separates the subject from the background (side or back light)."],
                ["Shape the tone curve: gentle S-curve, set black and white points."], overlay: "hist", fix: "edit")
        }

        // ---------------- light ----------------
        if gap("lighting") >= 10 {
            var ev: [String] = []
            if sw["local_contrast"] > 1.2 * sl["local_contrast"] {
                ev.append("\(nW) has \(String(format: "%.1f", sw["local_contrast"] / max(sl["local_contrast"], 1e-4)))x the local light-and-shade modelling of \(nL) - a sign of directional light.")
            }
            if sw["cast_b"] - sl["cast_b"] > 0.08 {
                ev.append("\(nW) is lit noticeably warmer (golden-hour-like colour) than \(nL).")
            }
            add("lighting", "The light itself is weaker", ["lighting"], ev,
                ["Light is the one thing you cannot add later. Come back in the hour after sunrise or before sunset when it is low, warm and directional - or right after weather clears.",
                 "Put the light to the side of or behind the subject for shape and depth; front-lit and overhead midday light flattens everything.",
                 "For a landmark view, check where the sun will be (PhotoPills / The Photographer's Ephemeris) and plan the time of day the scene is lit, not in shadow."],
                ["Dodging and burning (local brightening/darkening) can emphasise existing light, but cannot change its direction."],
                overlay: "impact", fix: "reshoot", boost: 0.25)
        }

        // ---------------- colour ----------------
        let muted = sl["colorfulness"] < 0.78 * sw["colorfulness"]
        let garish = sl["sat_mean"] > 0.62 && sl["sat_mean"] > 1.3 * sw["sat_mean"]
        if muted && (gap("vivid_color") >= 6 || gap("color_harmony") >= 8) {
            let vib = min(60, max(10, Int(((rec?.saturationPct ?? 25) * 0.8).rounded())))
            add("muted", "Muted colour", ["vivid_color", "color_harmony"],
                ["Colourfulness of \(nL) measures \(String(format: "%.2f", sl["colorfulness"])) vs \(String(format: "%.2f", sw["colorfulness"])) for \(nW)."],
                ["Colour is richest in low-angle light and just after rain; a polariser deepens skies, foliage and water by removing glare.",
                 "Underexposing by 1/3 stop keeps bright colours from washing out."],
                ["Add Vibrance (about +\(vib)) rather than Saturation so already-strong colours don't go neon."],
                overlay: "palette", fix: "edit", boost: 0.15)
        } else if garish {
            add("garish", "Over-saturated colour", ["color_accuracy", "color_harmony"],
                ["Mean saturation of \(nL) is \(pct(sl["sat_mean"], 0)) vs \(pct(sw["sat_mean"], 0)) for \(nW)."],
                ["Turn off 'Vivid' picture styles / scene enhancers."],
                ["Reduce Saturation/Vibrance until skin, sky and foliage look believable; over-saturation destroys detail in the most colourful areas."],
                overlay: "palette", fix: "edit", boost: 0.2, force: true)
        } else if gap("vivid_color") >= 12 || gap("color_harmony") >= 12 {
            add("palette", "Less cohesive colour palette", ["color_harmony", "vivid_color"], [],
                ["Simplify the palette: frame so two or three related or complementary colours dominate, and exclude stray bright objects."],
                ["Use HSL/colour-grading tools to pull distracting hues toward the main palette."], overlay: "palette")
        }
        let ca = sl["neutral_cast_a"], cb = sl["neutral_cast_b"], wa = sw["neutral_cast_a"], wb = sw["neutral_cast_b"]
        let cw = castWords(ca, cb)
        if (abs(ca - wa) >= 0.09 || abs(cb - wb) >= 0.12) && (abs(ca) > abs(wa) + 0.04 || abs(cb) > abs(wb) + 0.06) && !cw.isEmpty {
            add("cast", "Colour cast (\(cw))", ["color_accuracy"],
                ["Areas that should be neutral in \(nL) lean \(cw); in \(nW) they are closer to neutral."],
                ["Set white balance for the light (Daylight / Cloudy / Shade) instead of Auto, or shoot RAW and set it afterwards."],
                ["Move Temperature toward \(cb > wb ? "blue" : "yellow")" + (abs(ca - wa) >= 0.07 ? " and Tint toward \(ca > wa ? "green" : "magenta")" : "") + ", or click the white-balance eyedropper on something that should be grey."],
                overlay: "palette", fix: "edit", boost: 0.15, force: gap("color_accuracy") >= 4)
        }

        // ---------------- sharpness / blur / noise ----------------
        let soft = sl["edge_crispness"] < 0.85 * sw["edge_crispness"] || sl["sharp_top"] < 0.6 * sw["sharp_top"]
        if gap("sharpness") >= 12 || (soft && gap("sharpness") >= 6) {
            var ev: [String] = [], shoot: [String] = []
            if soft {
                ev.append("Edges in the sharpest part of \(nL) are softer than in \(nW) (edge crispness \(String(format: "%.2f", sl["edge_crispness"])) vs \(String(format: "%.2f", sw["edge_crispness"])); see the Focus view).")
            }
            if let sh = eL.shutter, let f35 = eL.focal35.map(Double.init) ?? eL.focal {
                if sh > 1 / max(f35, 30) {
                    ev.append("EXIF: \(ExifInfo.shutterString(sh)) at \(fmt(f35))mm - slower than the 1/focal-length hand-holding rule, so camera shake is likely.")
                    shoot.append("Use 1/\(Int(max(2 * f35, 125)))s or faster hand-held, or put the camera on a tripod and use a 2-second timer.")
                }
            }
            if let ap = eL.aperture, ap >= 16 {
                ev.append("EXIF: f/\(fmt(ap)) - diffraction softens the whole frame at apertures this small.")
                shoot.append("Stay around f/5.6-f/11, where most lenses are sharpest.")
            }
            shoot += ["Focus deliberately: tap/half-press on the subject (for landscapes, about a third of the way into the scene) and check at 100% before leaving.",
                      "Brace the camera, squeeze the shutter gently, and keep image stabilisation on when hand-holding."]
            add("sharpness", "Softer detail", ["sharpness", "motion_blur"], ev, shoot,
                ["Capture sharpening helps slightly-soft files, but cannot rescue missed focus or shake."],
                overlay: "focus", fix: "reshoot", boost: soft ? 0.25 : 0)
        }
        let noisy = sl["noise_luma"] > 1.6 * sw["noise_luma"] && sl["noise_luma"] > 0.006
        if gap("noise") >= 14 || (noisy && gap("noise") >= 8) {
            var ev = noisy ? ["Grain in smooth areas of \(nL) measures \(String(format: "%.1f", 1000 * sl["noise_luma"])) vs \(String(format: "%.1f", 1000 * sw["noise_luma"])) for \(nW) (x1000, luminance noise)."] : []
            if let iso = eL.iso, iso >= 800 { ev.append("EXIF: ISO \(iso).") }
            add("noise", "Visible noise", ["noise"], ev,
                ["Use the lowest ISO the light allows: add light, open the aperture, or use a tripod with a longer shutter.",
                 "Expose generously - brightening a dark file afterwards multiplies noise."],
                ["Apply luminance + colour noise reduction (AI denoise tools work well), then sharpen lightly."],
                overlay: "focus", boost: noisy ? 0.1 : 0)
        }

        // ---------------- depth of field ----------------
        if gap("depth_of_field") >= 20 || (gap("depth_of_field") >= 12 && sw["focus_spread"] > sl["focus_spread"] + 0.4) {
            var ev: [String] = []
            if sw["focus_spread"] > sl["focus_spread"] + 0.4 {
                ev.append("\(nW) has a much stronger sharp-subject / soft-background split (focus spread \(String(format: "%.1f", sw["focus_spread"])) vs \(String(format: "%.1f", sl["focus_spread"])); compare the Focus views).")
            }
            if let aw = eW.aperture, let al = eL.aperture {
                ev.append("EXIF: \(nW) was shot at f/\(fmt(aw)), \(nL) at f/\(fmt(al)).")
            }
            add("dof", "Less subject separation (depth of field)", ["depth_of_field"], ev,
                ["To blur the background: use the widest aperture (lowest f-number), zoom in or use a longer lens, get closer to the subject, and put distance between subject and background.",
                 "On a phone, use the 2x/3x lens or Portrait mode and move in close."],
                ["Background blur added in software rarely looks natural around hair and fine edges - get it in camera."],
                overlay: "focus", fix: "reshoot")
        }

        // ---------------- composition: level ----------------
        // A line detector alone is fooled by ridgelines and receding shorelines, so a tilt is only
        // reported when the learned level-horizon head agrees.
        let tl = sl["tilt_deg"], tw = sw["tilt_deg"]
        let tiltOK = abs(tl) >= 1 && abs(tl) <= 8 && sl["tilt_conf"] >= 0.6 && abs(tl) > abs(tw) + 0.5 && rel("level") && gap("level") >= 10
        if tiltOK {
            add("level", "Horizon tilted \(String(format: "%.1f", abs(tl))) degrees", ["level"],
                ["The horizon in \(nL) (see the Composition view) runs \(String(format: "%.1f", abs(tl))) degrees off level, \(tl > 0 ? "down to the right" : "down to the left"); \(nW) is within \(String(format: "%.1f", abs(tw))) degrees. Ignore this if that line is a shoreline receding at an angle."],
                ["Switch on the grid or electronic level in the viewfinder and line the horizon up with it."],
                ["Straighten by rotating \(String(format: "%.1f", abs(tl))) degrees \(tl > 0 ? "counter-clockwise" : "clockwise") (crop tool > Straighten)."],
                overlay: "comp", fix: "edit", boost: 0.15 + 0.1 * abs(tl), force: true)
        } else if gap("level") >= 18 {
            add("level", "Frame looks tilted", ["level"], [],
                ["Use the grid or electronic level; keep the sensor parallel to verticals for buildings."],
                ["Straighten with the crop tool; fix converging verticals with Upright/perspective correction."], overlay: "comp", fix: "edit")
        }

        // ---------------- objects: distractions and the subject ----------------
        let oL = c.objects.indices.contains(c.L) ? c.objects[c.L] : nil
        let oW = c.objects.indices.contains(c.W) ? c.objects[c.W] : nil
        var objEv: [String] = [], objShoot: [String] = []
        if let oL {
            let dis = Array(oL.0.filter { $0.role == "distraction" }.sorted { $0.delta > $1.delta }.prefix(3))
            if !dis.isEmpty {
                var ev: [String] = [], shoot: [String] = [], edit: [String] = []
                for g in dis {
                    var line = "The \(g.label) (\(Objects.whereText(g)), no. \(g.n) in the Objects view): with it painted out, the network scores \(nL) \(String(format: "%.0f", g.delta)) points higher."
                    let cut = Objects.cutOff(g)
                    if !cut.isEmpty { line += " It is also sliced by the \(cut.joined(separator: " and ")) edge of the frame." }
                    ev.append(line)
                }
                let edge = dis.contains { Objects.whereText($0).contains("edge") || !Objects.cutOff($0).isEmpty }
                shoot.append("Scan the frame for anything that isn't the subject before you press the shutter. A step sideways, a lower or higher viewpoint, or a tighter zoom usually hides it or pushes it out of frame.")
                if dis.contains(where: { movers.contains($0.label) }) {
                    shoot.append("If it moves (people, vehicles), wait a few seconds for it to leave - or for it to reach a spot where it adds to the picture.")
                }
                if edge { edit.append("It sits at the frame edge, so a small crop removes it.") }
                edit.append("Otherwise remove it with the healing / clone / generative-remove tool; small objects against sky, water or grass heal cleanly.")
                add("distraction", dis.count == 1 ? "Distracting \(dis[0].label) at the \(Objects.whereText(dis[0]))" : "Distracting elements in the frame",
                    ["subject_emphasis"], ev, shoot, edit, overlay: "objects", fix: "both",
                    boost: min(1.2, 0.25 + dis.map(\.delta).reduce(0, +) / 12), force: true)
            }
        }
        if let gL = oL?.1, let gW = oW?.1 {
            let sL = gL.subject, sW = gW.subject
            if let sW, sL == nil {
                objEv.append("\(nW) has a clear subject (the \(sW.label), \(sW.whereText) of the frame, \(pct(sW.area)) of the picture); no single object stands out in \(nL).")
            }
            if let sL, let sW {
                let alike = sL.label == sW.label
                if (same || alike) && sW.area >= 1.6 * sL.area && sW.area - sL.area >= 0.01 {
                    let r = (sW.area / sL.area).squareRoot()
                    objEv.append(alike ? "The \(sW.label) fills \(pct(sW.area)) of \(nW) but only \(pct(sL.area)) of \(nL)."
                                       : "The \(sL.label) fills \(pct(sL.area)) of \(nL) but the \(sW.label) fills \(pct(sW.area)) of \(nW).")
                    objShoot.append("Make the subject bigger: move closer or zoom in by about \(String(format: "%.1f", min(r, 4)))x.")
                }
                if !sL.cut.isEmpty && sW.cut.isEmpty {
                    objEv.append("The \(sL.label) in \(nL) is cut off by the \(sL.cut.joined(separator: " and ")) edge of the frame; in \(nW) the subject is whole.")
                    objShoot.append("Either include the whole subject with a little breathing room, or crop in decisively - a subject clipped at the edge looks like an accident.")
                }
                if let a = sL.separation, let b = sW.separation, a < 12, a < 0.6 * b {
                    objEv.append("The \(sL.label) in \(nL) barely separates from what is behind it (tone/colour difference \(String(format: "%.0f", a)) vs \(String(format: "%.0f", b)) for \(nW)'s subject; below about 10 a subject starts to merge).")
                    objShoot.append("Move so the subject sits against something that contrasts with it - light against dark, or a different colour - or wait for light to pick it out.")
                }
                if sL.centerDist < 0.07 && sW.thirdsDist < 0.10 && sL.area < 0.2 {
                    objEv.append("The \(sL.label) in \(nL) sits dead-centre; in \(nW) the subject is on a thirds point.")
                }
            }
            if same && abs(gL.sky - gW.sky) >= 0.12 {
                let more = gL.sky > gW.sky
                objEv.append(gL.sky < 0.02 ? "\(nL) has no sky in the frame; \(nW) gives it \(pct(gW.sky, 0))."
                                          : "Sky takes up \(pct(gL.sky, 0)) of \(nL) against \(pct(gW.sky, 0)) of \(nW).")
                if more && gap("lighting") + gap("content") > 0 {
                    objShoot.append("Tilt the camera down: unless the sky is the show, give that space to the land or water instead.")
                }
            }
        }

        if let fr = c.frame {
            var ev = objEv, shoot = objShoot, edit: [String] = []
            let zf = fr.zoom
            if zf >= 1.12 && fr.inside >= 0.85 {
                ev.append("Same scene, framed tighter: \(nW) shows only the central \(pct(fr.areaFrac, 0)) of what \(nL) includes (about \(String(format: "%.1f", zf))x more zoom). See the Framing view.")
                shoot.append("Zoom in or walk closer by roughly \(String(format: "%.1f", zf))x so the subject fills the frame and the empty margins go.")
                edit.append("You can get this framing by cropping to the box in the Framing view (the crop preview does exactly that).")
            } else if zf <= 0.9 {
                ev.append("Same scene, framed wider: \(nW) takes in about \(String(format: "%.1f", fr.areaFrac))x the area of \(nL), giving the subject room and context.")
                shoot.append("Step back or use a wider lens (about \(String(format: "%.1f", 1 / zf))x wider field of view).")
                edit.append("This cannot be fixed by cropping - the missing surroundings were never captured.")
            }
            if abs(fr.dx) >= 0.07 || abs(fr.dy) >= 0.07 {
                let hx = fr.dx > 0 ? "right" : "left", hy = fr.dy > 0 ? "down" : "up"
                var parts: [String] = [], dirs: [String] = []
                if abs(fr.dx) >= 0.07 { parts.append("\(Int((abs(fr.dx) * 100).rounded()))% of the frame width to the \(hx)"); dirs.append(hx) }
                if abs(fr.dy) >= 0.07 { parts.append("\(Int((abs(fr.dy) * 100).rounded()))% of the frame height \(hy)"); dirs.append(hy) }
                ev.append("The winner's frame is centred \(parts.joined(separator: " and ")) of yours.")
                shoot.append("Aim the camera further \(dirs.joined(separator: " and ")).")
            }
            if abs(fr.rotation) >= 1.5 {
                ev.append("Relative to \(nW), \(nL) is rotated \(String(format: "%.1f", abs(fr.rotation))) degrees.")
            }
            if !ev.isEmpty {
                add("framing", "Framing differs from the winner", ["framing", "subject_emphasis", "balance", "thirds"], ev, shoot, edit,
                    overlay: "frame", fix: zf >= 1.12 ? "both" : "reshoot", boost: 0.3, force: true)
            }
        }
        let compGap = ["subject_emphasis", "balance", "thirds", "framing"].map(gap).max() ?? 0
        if !items.contains(where: { $0.id == "framing" }) && (compGap >= 12 || (!objEv.isEmpty && gap("subject_emphasis") >= 6)) {
            var ev = objEv, shoot = objShoot
            if gap("subject_emphasis") >= 10 {
                shoot.append("Decide what the one subject is, then remove competition: get closer, simplify the background, or change angle so the subject stands against something plain (sky, water, shadow).")
            }
            if sl["edge_weight"] > 1.3 * sw["edge_weight"] && sl["edge_weight"] > 0.2 {
                ev.append("\(pct(sl["edge_weight"], 0)) of the visual weight in \(nL) sits right at the frame edges (\(pct(sw["edge_weight"], 0)) in \(nW)) - edge clutter pulls the eye out of the picture.")
                shoot.append("Run your eye around the border before pressing the shutter; step sideways or zoom to drop anything bright or cut-off at the edges.")
            }
            if gap("thirds") >= 10 || gap("balance") >= 10 {
                if sl["center_dist"] < 0.1 && sw["center_dist"] > 0.14 {
                    ev.append("The main visual mass of \(nL) is dead-centre; in \(nW) it sits off-centre near a thirds line.")
                }
                shoot.append("Try the subject on a thirds line with space in the direction it faces or moves, and balance it with a smaller secondary element on the other side.")
                shoot.append("For landscapes add a foreground anchor (rock, shoreline curve, canoe) to lead into the scene, and put the horizon on the upper or lower third rather than the middle.")
            }
            if gap("framing") >= 10 {
                shoot.append("Leave breathing room around the subject and avoid cutting things off at the frame edge.")
            }
            add("composition", "Weaker composition", ["subject_emphasis", "thirds", "balance", "framing"], ev, shoot,
                ["A crop can tighten the frame and move the subject onto a stronger position - try a few crops before reshooting."],
                overlay: objEv.isEmpty ? "comp" : "objects", boost: 0.1 + (objEv.isEmpty ? 0 : 0.15), force: !objEv.isEmpty)
        }

        if gap("content") >= 14 {
            add("content", "Less engaging subject matter", ["content"], [],
                ["Give the viewer something to land on: a person for scale, a boat, wildlife, a striking foreground, or a moment (weather, reflection, light beam) rather than the bare scene.",
                 "Ask what the photo is about, and exclude everything that isn't that."],
                [], overlay: "impact", fix: "reshoot")
        }

        items.sort { $0.priority > $1.priority }
        var keep = Array(items.prefix(7))
        for it in items.dropFirst(7) where it.id == "distraction" { keep[keep.count - 1] = it }   // a located distraction is always worth showing
        return keep
    }

    // MARK: - the whole result

    static func build(_ c: Input, pA: Double, extras: [PhotoExtras], similarity: Double, previews: [Preview], seconds: Double) -> Comparison {
        let info = c.info
        let a = c.photos[0], b = c.photos[1]
        let w = c.photos[c.W], l = c.photos[c.L]
        let conf = max(pA, 1 - pA)
        let ia = info.index(of: "aesthetic") ?? 0, it = info.index(of: "technical") ?? 1

        let dzA = w.z[ia] - l.z[ia], dzT = w.z[it] - l.z[it]
        let summary: String
        if dzA > 0.25 && dzT > 0.25 {
            summary = "It is ahead on both aesthetic appeal and technical quality."
        } else if dzA > 0.25 {
            summary = dzT > -0.25 ? "It wins on aesthetic appeal; technically the two are about level."
                                  : "It wins on aesthetic appeal; \(l.name) is actually the cleaner capture technically."
        } else if dzT > 0.25 {
            summary = dzA > -0.25 ? "It wins on technical quality; aesthetically the two are about level."
                                  : "It wins on technical quality; \(l.name) is the more appealing image aesthetically."
        } else {
            summary = "The two are close on both overall scores."
        }
        let strength = conf >= 0.85 ? "clearly" : (conf >= 0.68 ? "fairly confidently" : "narrowly")

        // why it won / where the loser leads
        let judged = info.heads.indices.filter { info.groups.contains(info.heads[$0].group) && info.heads[$0].reliable }
        let ranked = judged.sorted { (w.z[$0] - l.z[$0]) > (w.z[$1] - l.z[$1]) }
        func gap(_ i: Int) -> Double { w.percentile[i] - l.percentile[i] }
        let reasons = ranked.filter { gap($0) >= 8 }.prefix(5).map { i in
            Reason(head: info.heads[i].key, label: info.heads[i].label, gap: gap(i),
                   text: "\(info.heads[i].label): \(ordinal(w.percentile[i])) percentile vs \(ordinal(l.percentile[i])). \(info.heads[i].blurb)",
                   winnerPct: w.percentile[i], loserPct: l.percentile[i])
        }
        let leads = ranked.reversed().filter { gap($0) <= -8 }.prefix(4).map { i in
            Reason(head: info.heads[i].key, label: info.heads[i].label, gap: -gap(i),
                   text: "\(info.heads[i].label): \(ordinal(l.percentile[i])) percentile vs \(ordinal(w.percentile[i])).",
                   winnerPct: w.percentile[i], loserPct: l.percentile[i])
        }

        var rows: [AttributeRow] = []
        for g in info.groups {
            for i in info.heads.indices where info.heads[i].group == g {
                let h = info.heads[i]
                rows.append(AttributeRow(id: h.key, label: h.label, blurb: h.blurb, group: g, reliable: h.reliable,
                                         a: a.percentile[i], b: b.percentile[i]))
            }
        }

        let relText: String
        switch c.relation {
        case "aligned": relText = "Same scene - the two frames were matched point-to-point, so framing and tone are compared directly."
        case "similar": relText = "Similar subject - tone and colour are compared directly; framing is judged separately for each photo."
        default: relText = "Different scenes - each photo is judged on its own merits; colour-matching advice is limited to tone."
        }

        return Comparison(a: a, b: b, extras: extras, pA: pA, winnerIsA: c.W == 0,
                          headline: "\(w.name) wins \(strength)", summary: summary, closeCall: conf < 0.56,
                          reasons: Array(reasons), loserLeads: Array(leads), coaching: items(c), attributes: rows,
                          relation: c.relation, relationText: relText, similarity: similarity, frame: c.frame,
                          recipe: c.recipe, previews: previews, seconds: seconds)
    }
}

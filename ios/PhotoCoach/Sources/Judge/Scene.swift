import Foundation
import simd
import Vision

/// 3x3 homography on normalised (0...1) image coordinates, row-major.
struct H3 {
    var m: [Double]

    static let identity = H3(m: [1, 0, 0, 0, 1, 0, 0, 0, 1])

    func apply(_ x: Double, _ y: Double) -> (x: Double, y: Double, w: Double) {
        let X = m[0] * x + m[1] * y + m[2]
        let Y = m[3] * x + m[4] * y + m[5]
        let W = m[6] * x + m[7] * y + m[8]
        return (X / W, Y / W, W)
    }

    func project(_ pts: [(Double, Double)]) -> [(Double, Double)] {
        pts.map { let p = apply($0.0, $0.1); return (p.x, p.y) }
    }

    static func * (a: H3, b: H3) -> H3 {
        var r = [Double](repeating: 0, count: 9)
        for i in 0..<3 { for j in 0..<3 { for k in 0..<3 { r[i * 3 + j] += a.m[i * 3 + k] * b.m[k * 3 + j] } } }
        return H3(m: r)
    }

    var inverse: H3? {
        let a = m
        let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6])
        guard abs(det) > 1e-12 else { return nil }
        let inv = [
            a[4] * a[8] - a[5] * a[7], a[2] * a[7] - a[1] * a[8], a[1] * a[5] - a[2] * a[4],
            a[5] * a[6] - a[3] * a[8], a[0] * a[8] - a[2] * a[6], a[2] * a[3] - a[0] * a[5],
            a[3] * a[7] - a[4] * a[6], a[1] * a[6] - a[0] * a[7], a[0] * a[4] - a[1] * a[3],
        ]
        return H3(m: inv.map { $0 / det })
    }
}

typealias Quad = [(Double, Double)]   // TL, TR, BR, BL

struct Alignment {
    var aligned = false
    var H = H3.identity                 // A (0..1) -> B (0..1)
    var quadBinA: Quad = []
    var quadAinB: Quad = []
    var match = 0.0                     // structure correlation of the matched frames
    var coverage = 0.0
}

/// Where the OTHER photo's frame sits inside MY frame (scene.framing).
struct Framing {
    var areaFrac: Double
    var zoom: Double
    var dx: Double, dy: Double
    var rotation: Double
    var inside: Double
    var quad: Quad
}

/// Editing-slider style numbers that move one photo's tone / colour toward another's.
struct LookRecipe {
    var exposureEV: Double
    var contrastPct: Double
    var highlights: Double
    var shadows: Double
    var blacks: Double
    var whites: Double
    var saturationPct: Double
    var temp: Double
    var tint: Double
    var qs: [Double], qr: [Double]
    var sat: Double
    var colorOK: Bool

    struct Line: Identifiable {
        let name: String
        let value: String
        let amount: Double          // signed, roughly -100...100, for drawing a slider
        var id: String { name }
    }

    /// coach.recipe_lines
    var lines: [Line] {
        var out: [Line] = []
        func line(_ name: String, _ v: Double, _ unit: String, _ thr: Double, _ f: String, scale: Double) {
            if abs(v) >= thr { out.append(Line(name: name, value: String(format: f, v) + unit, amount: max(-100, min(100, v * scale)))) }
        }
        line("Exposure", exposureEV, " EV", 0.15, "%+.1f", scale: 50)
        line("Contrast", contrastPct * 0.6, "", 6, "%+.0f", scale: 1)
        line("Highlights", highlights * 3, "", 8, "%+.0f", scale: 1)
        line("Shadows", shadows * 3, "", 8, "%+.0f", scale: 1)
        line("Whites", whites * 2.5, "", 8, "%+.0f", scale: 1)
        line("Blacks", blacks * 2.5, "", 8, "%+.0f", scale: 1)
        line("Vibrance", saturationPct * 0.8, "", 6, "%+.0f", scale: 1)
        if colorOK {
            if abs(temp) >= 2 { out.append(Line(name: "Temperature", value: (temp > 0 ? "warmer " : "cooler ") + String(format: "%.0f", abs(temp) * 2.2), amount: max(-100, min(100, temp * 2.2)))) }
            if abs(tint) >= 2 { out.append(Line(name: "Tint", value: (tint > 0 ? "magenta " : "green ") + String(format: "%.0f", abs(tint) * 2.2), amount: max(-100, min(100, tint * 2.2)))) }
        }
        return out
    }
}

/// Per-pixel CIE Lab of a photo (cached: several analyses need it).
struct LabImage {
    let w: Int, h: Int
    var L: [Float], a: [Float], b: [Float]

    init(_ img: RGBImage) {
        w = img.width; h = img.height
        let n = w * h
        L = [Float](repeating: 0, count: n); a = L; b = L
        img.data.withUnsafeBufferPointer { p in
            for i in 0..<n {
                let v = Imaging.lab(p[i * 3], p[i * 3 + 1], p[i * 3 + 2])
                L[i] = v.0; a[i] = v.1; b[i] = v.2
            }
        }
    }
}

/// Pair-specific analysis: same scene?  how do the framings differ?  what edit moves one toward the other?
enum SceneMatch {

    // MARK: - alignment

    /// Structure image for matching: grey at 128 px, high-passed so only edges / texture count,
    /// which makes the correlation immune to exposure and colour differences.
    private static func structure(_ img: RGBImage, size: Int = 128) -> Plane {
        let g = Imaging.gray(ImagePrep.resizeArea(img, width: size, height: size))
        let a = Imaging.gaussianBlur(g, sigma: 1.0)
        let b = Imaging.gaussianBlur(g, sigma: 6.0)
        var out = a
        for i in out.d.indices { out.d[i] = a.d[i] - b.d[i] }
        return out
    }

    @inline(__always) private static func sample(_ p: Plane, _ u: Double, _ v: Double) -> Float {
        let fx = max(0, min(Double(p.w - 1), u * Double(p.w) - 0.5))
        let fy = max(0, min(Double(p.h - 1), v * Double(p.h) - 0.5))
        let x0 = Int(fx), y0 = Int(fy), x1 = min(p.w - 1, x0 + 1), y1 = min(p.h - 1, y0 + 1)
        let tx = Float(fx - Double(x0)), ty = Float(fy - Double(y0))
        return (p[x0, y0] * (1 - tx) + p[x1, y0] * tx) * (1 - ty) + (p[x0, y1] * (1 - tx) + p[x1, y1] * tx) * ty
    }

    /// Normalised cross-correlation of A's structure with B's structure seen through H (A -> B).
    private static func correlation(_ sa: Plane, _ sb: Plane, _ H: H3, n: Int = 48) -> (ncc: Double, coverage: Double) {
        var xs: [Float] = [], ys: [Float] = []
        for j in 0..<n {
            for i in 0..<n {
                let u = (Double(i) + 0.5) / Double(n), v = (Double(j) + 0.5) / Double(n)
                let p = H.apply(u, v)
                guard p.w > 0, p.x >= 0, p.x <= 1, p.y >= 0, p.y <= 1 else { continue }
                xs.append(sample(sa, u, v)); ys.append(sample(sb, p.x, p.y))
            }
        }
        let cov = Double(xs.count) / Double(n * n)
        guard xs.count > n * n / 5 else { return (0, cov) }
        let mx = xs.reduce(0, +) / Float(xs.count), my = ys.reduce(0, +) / Float(ys.count)
        var sxy: Float = 0, sxx: Float = 0, syy: Float = 0
        for k in xs.indices {
            let dx = xs[k] - mx, dy = ys[k] - my
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy
        }
        return (Double(sxy / ((sxx * syy).squareRoot() + 1e-9)), cov)
    }

    private static func saneQuad(_ q: Quad, aspect: Double) -> Bool {
        guard q.count == 4, q.allSatisfy({ $0.0.isFinite && $0.1.isFinite }) else { return false }
        let p = q.map { ($0.0 * aspect, $0.1) }
        var signs: [Double] = []
        for i in 0..<4 {
            let a = p[i], b = p[(i + 1) % 4], c = p[(i + 2) % 4]
            signs.append((b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0))
        }
        guard signs.allSatisfy({ $0 > 0 }) || signs.allSatisfy({ $0 < 0 }) else { return false }
        var area = 0.0
        for i in 0..<4 { area += p[i].0 * p[(i + 1) % 4].1 - p[(i + 1) % 4].0 * p[i].1 }
        area = abs(area) / 2 / aspect
        let sides = (0..<4).map { hypot(p[$0].0 - p[($0 + 1) % 4].0, p[$0].1 - p[($0 + 1) % 4].1) }
        return area > 0.04 && area < 25 && (sides.max() ?? 0) / ((sides.min() ?? 0) + 1e-9) < 6
    }

    static let corners: Quad = [(0, 0), (1, 0), (1, 1), (0, 1)]

    /// Find the mapping A -> B.  Two photos of the same scene mostly differ by zoom and aim, so a direct
    /// search over B's frame as a rectangle inside A (scale + offset), scored by structure correlation,
    /// finds it robustly; Vision's image registration adds a perspective-capable candidate.  Whichever
    /// lines the structure up best wins, and it only counts as "aligned" when that correlation is strong.
    static func align(_ a: RGBImage, _ b: RGBImage) -> Alignment {
        var res = Alignment()
        let aspA = Double(a.width) / Double(a.height), aspB = Double(b.width) / Double(b.height)
        let sa64 = structure(a, size: 64), sb64 = structure(b, size: 64)
        let sa = structure(a), sb = structure(b)

        func rect(_ s: Double, _ x0: Double, _ y0: Double) -> H3 {
            let sw = s, sh = s * aspA / aspB
            return H3(m: [1 / sw, 0, -x0 / sw, 0, 1 / sh, -y0 / sh, 0, 0, 1])
        }
        // coarse: B's frame = [x0, x0+s] x [y0, y0+s*aspA/aspB] in A's normalised coordinates
        var coarse: [(c: Double, s: Double, x: Double, y: Double)] = []
        for k in 0..<28 {
            let s = exp(log(0.45) + (log(2.2) - log(0.45)) * Double(k) / 27)
            let sw = s, sh = s * aspA / aspB
            guard sh > 0.3, sh < 3 else { continue }
            for x0 in stride(from: -sw + 0.3, through: 0.7, by: 0.03) {
                for y0 in stride(from: -sh + 0.3, through: 0.7, by: 0.03) {
                    let ow = max(0, min(1, x0 + sw) - max(0, x0)), oh = max(0, min(1, y0 + sh) - max(0, y0))
                    guard ow * oh >= 0.3, ow * oh / (sw * sh) >= 0.25 else { continue }
                    let (c, _) = correlation(sa64, sb64, rect(s, x0, y0), n: 24)
                    coarse.append((c, s, x0, y0))
                }
            }
        }
        coarse.sort { $0.c > $1.c }
        var cands: [H3] = []
        var seeds: [(c: Double, s: Double, x: Double, y: Double)] = []
        for c in coarse where !seeds.contains(where: { abs(log($0.s / c.s)) < 0.08 && abs($0.x - c.x) < 0.06 && abs($0.y - c.y) < 0.06 }) {
            seeds.append(c)
            if seeds.count == 4 { break }
        }
        // refine each seed at 128 px
        for sd in seeds {
            var bestC = -2.0, bestH = rect(sd.s, sd.x, sd.y)
            for ds in stride(from: -0.04, through: 0.0401, by: 0.01) {
                for dx in stride(from: -0.03, through: 0.0301, by: 0.006) {
                    for dy in stride(from: -0.03, through: 0.0301, by: 0.006) {
                        let H = rect(sd.s * (1 + ds), sd.x + dx, sd.y + dy)
                        let (c, cov) = correlation(sa, sb, H, n: 32)
                        if cov >= 0.25 && c > bestC { bestC = c; bestH = H }
                    }
                }
            }
            cands.append(bestH)
        }
        cands += visionCandidates(a, b)

        var best: (H: H3, c: Double, cov: Double)?
        for H in cands {
            let (c, cov) = correlation(sa, sb, H)
            if cov >= 0.25 && (best == nil || c > best!.c) { best = (H, c, cov) }
        }
        guard let (H, c, cov) = best, let Hinv = H.inverse else { return res }
        res.match = c
        res.coverage = cov
        let qBinA = Hinv.project(corners), qAinB = H.project(corners)
        if c >= alignThreshold && saneQuad(qBinA, aspect: aspA) && saneQuad(qAinB, aspect: aspB) {
            res.aligned = true
            res.H = H
            res.quadBinA = qBinA
            res.quadAinB = qAinB
        }
        return res
    }

    /// Structure correlation above which two frames count as the same scene, point for point.
    static let alignThreshold = 0.45

    /// Vision's homography, read every way its convention could go (the scoring picks the right one).
    private static func visionCandidates(_ a: RGBImage, _ b: RGBImage) -> [H3] {
        let S = 512.0
        guard let ca = Imaging.cgImage(ImagePrep.resizeArea(a, width: Int(S), height: Int(S))),
              let cb = Imaging.cgImage(ImagePrep.resizeArea(b, width: Int(S), height: Int(S))) else { return [] }
        let req = VNHomographicImageRegistrationRequest(targetedCGImage: cb, options: [:]).cpuOnSimulator()
        guard (try? VNImageRequestHandler(cgImage: ca, options: [:]).perform([req])) != nil,
              let o = req.results?.first as? VNImageHomographicAlignmentObservation else { return [] }
        let t = o.warpTransform
        let M = H3(m: [Double(t.columns.0.x), Double(t.columns.1.x), Double(t.columns.2.x),
                       Double(t.columns.0.y), Double(t.columns.1.y), Double(t.columns.2.y),
                       Double(t.columns.0.z), Double(t.columns.1.z), Double(t.columns.2.z)])
        let N = H3(m: [S, 0, 0, 0, S, 0, 0, 0, 1])              // normalised -> pixels, y down
        let Nf = H3(m: [S, 0, 0, 0, -S, S, 0, 0, 1])            // normalised -> pixels, y up
        var out: [H3] = []
        for P in [M, M.inverse].compactMap({ $0 }) {
            for n in [N, Nf] { if let ni = n.inverse { out.append(ni * P * n) } }
        }
        return out
    }

    static func framing(_ q: Quad, myAspect: Double) -> Framing {
        var area = 0.0
        for i in 0..<4 { area += q[i].0 * q[(i + 1) % 4].1 - q[(i + 1) % 4].0 * q[i].1 }
        area = abs(area) / 2
        let cx = q.map(\.0).reduce(0, +) / 4, cy = q.map(\.1).reduce(0, +) / 4
        let rot = atan2(q[1].1 - q[0].1, (q[1].0 - q[0].0) * myAspect) * 180 / .pi
        let coords = q.flatMap { [$0.0, $0.1] }
        let inside = Double(coords.filter { $0 >= -0.03 && $0 <= 1.03 }.count) / Double(coords.count)
        return Framing(areaFrac: area, zoom: 1 / max(area, 1e-6).squareRoot(), dx: cx - 0.5, dy: cy - 0.5,
                       rotation: rot, inside: inside, quad: q)
    }

    /// Pixels of "me" that the other photo also shows.
    static func overlapMask(w: Int, h: Int, meToOther H: H3) -> [Bool] {
        var m = [Bool](repeating: false, count: w * h)
        let step = 4
        for y in stride(from: 0, to: h, by: step) {
            for x in stride(from: 0, to: w, by: step) {
                let p = H.apply((Double(x) + 0.5) / Double(w), (Double(y) + 0.5) / Double(h))
                let inside = p.w > 0 && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1
                guard inside else { continue }
                for yy in y..<min(h, y + step) { for xx in x..<min(w, x + step) { m[yy * w + xx] = true } }
            }
        }
        return m
    }

    // MARK: - match the look

    static let qLevels: [Double] = [0.5, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99.5]

    private static func relY(_ L: Double) -> Double { L > 8 ? pow((L + 16) / 116, 3) : L / 903.3 }

    static func lookRecipe(src: LabImage, ref: LabImage, srcMask: [Bool]? = nil, refMask: [Bool]? = nil, colorOK: Bool = true) -> LookRecipe {
        func usable(_ m: [Bool]?, _ n: Int) -> [Bool]? {
            guard let m, m.count == n else { return nil }
            return Double(m.filter { $0 }.count) / Double(n) > 0.1 ? m : nil
        }
        let ms = usable(srcMask, src.L.count), mr = usable(refMask, ref.L.count)
        func stats(_ lab: LabImage, _ mask: [Bool]?) -> ([Double], Double, Double, Double) {
            var Ls: [Float] = []
            Ls.reserveCapacity(lab.L.count)
            var chroma = 0.0, sa = 0.0, sb = 0.0
            for i in lab.L.indices where mask?[i] ?? true {
                Ls.append(lab.L[i])
                chroma += Double((lab.a[i] * lab.a[i] + lab.b[i] * lab.b[i]).squareRoot())
                sa += Double(lab.a[i]); sb += Double(lab.b[i])
            }
            let n = Double(max(Ls.count, 1))
            Ls.sort()
            return (qLevels.map { Imaging.percentile(sorted: Ls, $0) }, chroma / n, sa / n, sb / n)
        }
        let (qs, cs, as_, bs) = stats(src, ms)
        let (qr, cr, ar, br) = stats(ref, mr)
        let sat = max(0.5, min(2.2, (cr + 2) / (cs + 2)))
        let ev = max(-3, min(3, log2((relY(qr[5]) + 0.01) / (relY(qs[5]) + 0.01))))
        let iqrS = qs[6] - qs[4], iqrR = qr[6] - qr[4]
        return LookRecipe(
            exposureEV: ev,
            contrastPct: max(-0.6, min(1.5, (iqrR + 4) / (iqrS + 4) - 1)) * 100,
            highlights: (qr[8] - qr[5]) - (qs[8] - qs[5]),
            shadows: (qs[5] - qs[2]) - (qr[5] - qr[2]),
            blacks: qr[0] - qs[0],
            whites: qr[10] - qs[10],
            saturationPct: (sat - 1) * 100,
            temp: colorOK ? br - bs : 0,
            tint: colorOK ? ar - as_ : 0,
            qs: qs, qr: qr, sat: sat, colorOK: colorOK)
    }

    /// numpy.interp for increasing xs.
    private static func interp(_ x: Double, _ xs: [Double], _ ys: [Double]) -> Double {
        if x <= xs[0] { return ys[0] }
        if x >= xs[xs.count - 1] { return ys[ys.count - 1] }
        var lo = 0, hi = xs.count - 1
        while hi - lo > 1 { let mid = (lo + hi) / 2; if xs[mid] <= x { lo = mid } else { hi = mid } }
        let span = xs[hi] - xs[lo]
        return span > 0 ? ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / span : ys[lo]
    }

    /// The recipe as a smooth global tone curve + white-balance shift + saturation scale.
    static func applyLook(_ lab: LabImage, _ rec: LookRecipe, strength s: Double = 0.85) -> RGBImage {
        let qs = rec.qs, qr = rec.qr
        let tgt = zip(qs, qr).map { $0 + s * ($1 - $0) }
        var xs = [0.0] + qs + [100.0]
        var ys = [max(0, tgt[0] - qs[0]) * 0.5] + tgt + [min(100, 100 + (tgt[tgt.count - 1] - qs[qs.count - 1]) * 0.5)]
        // np.unique(return_index): sorted unique xs, first occurrence
        let order = xs.indices.sorted { xs[$0] < xs[$1] || (xs[$0] == xs[$1] && $0 < $1) }
        var ux: [Double] = [], uy: [Double] = []
        for i in order where ux.last != xs[i] { ux.append(xs[i]); uy.append(ys[i]) }
        for i in 1..<uy.count { uy[i] = max(uy[i], uy[i - 1]) }
        xs = ux; ys = uy
        let grid = (0..<256).map { Double($0) * 100 / 255 }
        let lut0 = grid.map { interp($0, xs, ys) }
        let padded = [Double](repeating: lut0[0], count: 8) + lut0 + [Double](repeating: lut0[255], count: 8)
        let lut = (0..<256).map { i in padded[i..<(i + 17)].reduce(0, +) / 17 }
        let sat = Float(1 + s * (rec.sat - 1))
        let da = Float(s * rec.tint), db = Float(s * rec.temp)
        var out = [UInt8](repeating: 0, count: lab.w * lab.h * 3)
        for i in 0..<(lab.w * lab.h) {
            let L = Float(interp(Double(lab.L[i]), grid, lut))
            let (r, g, b) = Imaging.rgb(L: L, a: lab.a[i] * sat + da, b: lab.b[i] * sat + db)
            out[i * 3] = r; out[i * 3 + 1] = g; out[i * 3 + 2] = b
        }
        return RGBImage(width: lab.w, height: lab.h, data: out)
    }

    /// Re-frame my photo the way the other one is framed (valid when the other frame lies inside mine).
    static func cropToOther(_ me: RGBImage, meToOther H: H3, otherW: Int, otherH: Int, maxSide: Int = 1024) -> RGBImage? {
        guard let inv = H.inverse else { return nil }
        let s = min(1.0, Double(maxSide) / Double(max(otherW, otherH)))
        let ow = max(1, Int((Double(otherW) * s).rounded())), oh = max(1, Int((Double(otherH) * s).rounded()))
        let c = Imaging.planes(me)
        var out = [UInt8](repeating: 0, count: ow * oh * 3)
        for y in 0..<oh {
            for x in 0..<ow {
                let p = inv.apply((Double(x) + 0.5) / Double(ow), (Double(y) + 0.5) / Double(oh))
                let u = min(1, max(0, p.x)), v = min(1, max(0, p.y))
                for k in 0..<3 {
                    out[(y * ow + x) * 3 + k] = UInt8(max(0, min(255, sample(c[k], u, v).rounded())))
                }
            }
        }
        return RGBImage(width: ow, height: oh, data: out)
    }
}

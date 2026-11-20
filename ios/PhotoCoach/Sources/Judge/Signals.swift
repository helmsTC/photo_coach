import Foundation
import UIKit
import Vision

/// Dominant colour and its share of the frame.
struct Swatch: Hashable {
    let r: UInt8, g: UInt8, b: UInt8
    let share: Double
    var color: UIColor { UIColor(red: CGFloat(r) / 255, green: CGFloat(g) / 255, blue: CGFloat(b) / 255, alpha: 1) }
}

/// 64-bin histograms, each summing to 1.
struct Histogram {
    var r: [Double], g: [Double], b: [Double], l: [Double]
}

/// Everything measured on the pixels of one photo.
struct Measurement {
    /// Scalars, keyed exactly as photocoach/signals.py.
    var s: [String: Double]
    /// 12x12 maps of fine-edge strength and edge crispness.
    var detail: Plane
    var crisp: Plane
    /// 96x96 visual-weight map, normalised to max 1.
    var saliency: Plane
    var palette: [Swatch]
    var histogram: Histogram

    subscript(_ k: String) -> Double { s[k] ?? 0 }
}

/// Measurements read directly off the pixels of the standard image (long side <= 1024).
/// A port of photocoach/signals.py.  The tone / colour numbers are tower inputs, so their
/// definitions must not drift from the Python; focus / noise / composition numbers feed the
/// coaching evidence and overlays.  Horizon and the attention part of saliency use Apple's
/// Vision framework in place of OpenCV's Hough lines and spectral residual.
enum Signals {
    static let tileGrid = 12

    static func measure(_ img: RGBImage) -> Measurement {
        var s = tone(img)
        s.merge(color(img)) { $1 }
        let (f, detail, crisp) = focus(img)
        s.merge(f) { $1 }
        s.merge(noise(img)) { $1 }
        s["log_long_side"] = log(Double(max(img.width, img.height)) / 1024.0 + 1e-6)
        s["aspect"] = Double(img.width) / Double(img.height)
        let cg = Imaging.cgImage(img)
        s.merge(horizon(img)) { $1 }
        let (sal, salMap) = saliency(img, cg: cg, detail: detail)
        s.merge(sal) { $1 }
        return Measurement(s: s, detail: detail, crisp: crisp, saliency: salMap, palette: palette(img), histogram: histogram(img))
    }

    // MARK: - tone + colour (tower inputs: keep identical to the Python)

    static func tone(_ img: RGBImage) -> [String: Double] {
        let n = img.width * img.height
        var out: [String: Double] = [:]
        guard n > 0 else { return out }
        var luma = [Float](repeating: 0, count: n)
        var sumY = 0.0, sumY2 = 0.0
        var clipHi = 0, clipLo = 0, nearHi = 0, nearLo = 0
        img.data.withUnsafeBufferPointer { p in
            for i in 0..<n {
                let r8 = p[i * 3], g8 = p[i * 3 + 1], b8 = p[i * 3 + 2]
                let r = Float(r8) / 255, g = Float(g8) / 255, b = Float(b8) / 255
                let y = r * 0.2126 + g * 0.7152 + b * 0.0722
                luma[i] = y
                sumY += Double(y); sumY2 += Double(y) * Double(y)
                if max(r8, max(g8, b8)) >= 252 { clipHi += 1 }
                if y <= 0.016 { clipLo += 1 }
                if y >= 0.92 { nearHi += 1 }
                if y <= 0.06 { nearLo += 1 }
            }
        }
        let N = Double(n)
        let sorted = luma.sorted()
        let p01 = Imaging.percentile(sorted: sorted, 1), p99 = Imaging.percentile(sorted: sorted, 99)
        let meanY = sumY / N
        out["luma_mean"] = meanY
        out["luma_median"] = Imaging.percentile(sorted: sorted, 50)
        out["luma_p01"] = p01
        out["luma_p05"] = Imaging.percentile(sorted: sorted, 5)
        out["luma_p95"] = Imaging.percentile(sorted: sorted, 95)
        out["luma_p99"] = p99
        out["clip_hi"] = Double(clipHi) / N
        out["clip_lo"] = Double(clipLo) / N
        out["near_hi"] = Double(nearHi) / N
        out["near_lo"] = Double(nearLo) / N
        out["contrast_rms"] = max(0, sumY2 / N - meanY * meanY).squareRoot()
        out["dyn_range"] = p99 - p01

        // local contrast: mean |luma - heavily blurred luma|
        let y = Plane(w: img.width, h: img.height, d: luma)
        let blur = Imaging.gaussianBlur(y, sigma: max(2.0, 0.02 * Double(max(img.width, img.height))))
        var lc = 0.0
        for i in 0..<n { lc += Double(abs(y.d[i] - blur.d[i])) }
        out["local_contrast"] = lc / N
        return out
    }

    static func color(_ img: RGBImage) -> [String: Double] {
        let n = img.width * img.height
        var out: [String: Double] = [:]
        guard n > 0 else { return out }
        var chroma = [Float](repeating: 0, count: n)
        var labL = [Float](repeating: 0, count: n)
        var labA = [Float](repeating: 0, count: n)
        var labB = [Float](repeating: 0, count: n)
        var sumSat = 0.0
        var sumRG = 0.0, sumRG2 = 0.0, sumYB = 0.0, sumYB2 = 0.0
        var sumA = 0.0, sumB = 0.0
        img.data.withUnsafeBufferPointer { p in
            for i in 0..<n {
                let r8 = p[i * 3], g8 = p[i * 3 + 1], b8 = p[i * 3 + 2]
                let r = Float(r8) / 255, g = Float(g8) / 255, b = Float(b8) / 255
                let mx = max(r, max(g, b)), mn = min(r, min(g, b))
                sumSat += mx > 0 ? Double((mx - mn) / mx) : 0
                let rg = Double(r - g), yb = Double(0.5 * (r + g) - b)
                sumRG += rg; sumRG2 += rg * rg; sumYB += yb; sumYB2 += yb * yb
                let (L, a, bb) = Imaging.lab(r8, g8, b8)
                labL[i] = L; labA[i] = a; labB[i] = bb
                chroma[i] = (a * a + bb * bb).squareRoot()
                sumA += Double(a); sumB += Double(bb)
            }
        }
        let N = Double(n)
        let mRG = sumRG / N, mYB = sumYB / N
        let vRG = max(0, sumRG2 / N - mRG * mRG), vYB = max(0, sumYB2 / N - mYB * mYB)
        out["sat_mean"] = sumSat / N
        out["colorfulness"] = (vRG + vYB).squareRoot() + 0.3 * (mRG * mRG + mYB * mYB).squareRoot()
        out["cast_a"] = sumA / N / 50
        out["cast_b"] = sumB / N / 50
        // "should-be-neutral" pixels: the least colourful quarter of the mid-tones; their mean a/b is the cast
        let c25 = Float(Imaging.percentile(chroma, 25))
        var na = 0.0, nb = 0.0, cnt = 0
        for i in 0..<n where chroma[i] < c25 && labL[i] > 20 && labL[i] < 92 {
            na += Double(labA[i]); nb += Double(labB[i]); cnt += 1
        }
        if cnt < 50 { na = sumA; nb = sumB; cnt = n }
        out["neutral_cast_a"] = na / Double(cnt) / 50
        out["neutral_cast_b"] = nb / Double(cnt) / 50
        return out
    }

    // MARK: - tiles

    /// Tile edges as numpy.linspace(0, n, grid + 1).astype(int).
    static func tileEdges(_ n: Int, _ grid: Int = tileGrid) -> [Int] {
        (0...grid).map { Int(Double(n) / Double(grid) * Double($0)) }
    }

    // MARK: - focus

    private static func gradientMagnitude(_ gray: Plane, sigma: Double) -> Plane {
        let g = Imaging.gaussianBlur(gray, sigma: sigma)
        var out = Plane(w: g.w, h: g.h)
        let w = g.w, h = g.h
        g.d.withUnsafeBufferPointer { p in
            for y in 0..<h {
                let ym = Imaging.reflect101(y - 1, h) * w, y0 = y * w, yp = Imaging.reflect101(y + 1, h) * w
                for x in 0..<w {
                    let xm = Imaging.reflect101(x - 1, w), xp = Imaging.reflect101(x + 1, w)
                    let gx = (p[ym + xp] - p[ym + xm]) + 2 * (p[y0 + xp] - p[y0 + xm]) + (p[yp + xp] - p[yp + xm])
                    let gy = (p[yp + xm] - p[ym + xm]) + 2 * (p[yp + x] - p[ym + x]) + (p[yp + xp] - p[ym + xp])
                    out.d[y0 + x] = (gx * gx + gy * gy).squareRoot() / 8
                }
            }
        }
        return out
    }

    /// Where is the image sharp, and how sharp?  (signals.focus)
    static func focus(_ img: RGBImage) -> ([String: Double], Plane, Plane) {
        let gray = Imaging.gray(img)
        let g1 = gradientMagnitude(gray, sigma: 0.8)
        let g2 = gradientMagnitude(gray, sigma: 2.4)
        let n = tileGrid
        let ys = tileEdges(img.height), xs = tileEdges(img.width)
        var detail = Plane(w: n, h: n), crisp = Plane(w: n, h: n)
        for i in 0..<n {
            for j in 0..<n {
                var t1: [Float] = [], t2: [Float] = []
                t1.reserveCapacity((ys[i + 1] - ys[i]) * (xs[j + 1] - xs[j]))
                t2.reserveCapacity(t1.capacity)
                for y in ys[i]..<ys[i + 1] {
                    for x in xs[j]..<xs[j + 1] {
                        t1.append(g1[x, y]); t2.append(g2[x, y])
                    }
                }
                let a = Float(Imaging.percentile(t1, 97))
                let b = Float(Imaging.percentile(t2, 97))
                detail[j, i] = a
                crisp[j, i] = b > 0.004 ? a / (b + 1e-3) : 0
            }
        }
        let flat = detail.d.sorted()
        let top = Double(flat[Int(0.9 * Double(flat.count))])
        let med = Imaging.percentile(sorted: flat, 50)
        let low = Double(flat[Int(0.2 * Double(flat.count))])
        let topThr = flat[Int(0.8 * Double(flat.count))]
        var crispSum = 0.0, crispN = 0
        var wsum = 0.0, wx = 0.0, wy = 0.0, sharpN = 0
        for i in 0..<n {
            for j in 0..<n {
                let d = detail[j, i]
                if d >= topThr { crispSum += Double(crisp[j, i]); crispN += 1 }
                if Double(d) >= max(0.5 * top, 1e-4) {
                    sharpN += 1
                    wsum += Double(d); wx += Double(d) * (Double(j) + 0.5); wy += Double(d) * (Double(i) + 0.5)
                }
            }
        }
        let tot = wsum + 1e-9
        let out: [String: Double] = [
            "sharp_top": top,
            "sharp_median": med,
            "edge_crispness": crispN > 0 ? crispSum / Double(crispN) : 0,
            "focus_spread": log((top + 1e-3) / (low + 1e-3)),
            "sharp_area": Double(sharpN) / Double(n * n),
            "sharp_cx": wx / tot / Double(n),
            "sharp_cy": wy / tot / Double(n),
        ]
        return (out, detail, crisp)
    }

    // MARK: - noise (Immerkaer, flattest tiles)

    static func noise(_ img: RGBImage) -> [String: Double] {
        let w = img.width, h = img.height, n = w * h
        var Y = [Float](repeating: 0, count: n), C = [Float](repeating: 0, count: n)
        img.data.withUnsafeBufferPointer { p in
            for i in 0..<n {
                let r = Float(p[i * 3]), g = Float(p[i * 3 + 1]), b = Float(p[i * 3 + 2])
                let y = 0.299 * r + 0.587 * g + 0.114 * b
                let cr = max(0, min(255, ((r - y) * 0.713 + 128).rounded()))
                let cb = max(0, min(255, ((b - y) * 0.564 + 128).rounded()))
                Y[i] = max(0, min(255, y.rounded())) / 255
                C[i] = 0.5 * (cr + cb) / 255
            }
        }
        let ys = tileEdges(h), xs = tileEdges(w)
        func sigma(_ ch: [Float]) -> Double {
            var sig: [Float] = []
            ch.withUnsafeBufferPointer { p in
                for i in 0..<tileGrid {
                    for j in 0..<tileGrid {
                        var s = 0.0, c = 0
                        for y in ys[i]..<ys[i + 1] {
                            let ym = Imaging.reflect101(y - 1, h) * w, y0 = y * w, yp = Imaging.reflect101(y + 1, h) * w
                            for x in xs[j]..<xs[j + 1] {
                                let xm = Imaging.reflect101(x - 1, w), xp = Imaging.reflect101(x + 1, w)
                                let v = (p[ym + xm] - 2 * p[ym + x] + p[ym + xp])
                                    - 2 * (p[y0 + xm] - 2 * p[y0 + x] + p[y0 + xp])
                                    + (p[yp + xm] - 2 * p[yp + x] + p[yp + xp])
                                s += Double(abs(v)); c += 1
                            }
                        }
                        if c > 0 { sig.append(Float((Double.pi / 2).squareRoot() / 6 * s / Double(c))) }
                    }
                }
            }
            return sig.isEmpty ? 0 : Imaging.percentile(sig, 10)
        }
        return ["noise_luma": sigma(Y), "noise_chroma": sigma(C)]
    }

    // MARK: - horizon

    /// Angle of the horizon / main waterline in degrees (positive = clockwise: the right end sits lower).
    /// A port of signals.horizon: a candidate must be a long near-horizontal edge that actually separates
    /// two different regions (sky|land, land|water) - ripples and rock strata make long lines too, but
    /// with the same colour on both sides.  OpenCV's Canny + probabilistic Hough is replaced by a
    /// gradient edge map and a Hough vote restricted to +-10 degrees.
    static func horizon(_ img: RGBImage) -> [String: Double] {
        var out: [String: Double] = ["tilt_deg": 0, "tilt_conf": 0, "horizon_y": -1]
        let sc = 512.0 / Double(max(img.width, img.height))
        let small = sc < 1 ? ImagePrep.resizeArea(img, width: max(1, Int((Double(img.width) * sc).rounded())),
                                                  height: max(1, Int((Double(img.height) * sc).rounded()))) : img
        let w = small.width, h = small.height
        guard w >= 32, h >= 32 else { return out }
        var gray = Imaging.gray(small)
        for i in gray.d.indices { gray.d[i] *= 255 }
        gray = Imaging.gaussianBlur(gray, sigma: 1.5)

        // thin, mostly-horizontal edges: vertical-gradient maxima above a Canny-like threshold
        var mag = [Float](repeating: 0, count: w * h), vert = [Bool](repeating: false, count: w * h)
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let p = { (dx: Int, dy: Int) in gray.d[(y + dy) * w + x + dx] }
                let gx = (p(1, -1) + 2 * p(1, 0) + p(1, 1)) - (p(-1, -1) + 2 * p(-1, 0) + p(-1, 1))
                let gy = (p(-1, 1) + 2 * p(0, 1) + p(1, 1)) - (p(-1, -1) + 2 * p(0, -1) + p(1, -1))
                mag[y * w + x] = (gx * gx + gy * gy).squareRoot()
                vert[y * w + x] = abs(gy) >= abs(gx)
            }
        }
        var pts: [(Int, Int)] = []
        for y in 2..<(h - 2) {
            for x in 1..<(w - 1) {
                let i = y * w + x, m = mag[i]
                if vert[i] && m >= 70 && m >= mag[i - w] && m > mag[i + w] { pts.append((x, y)) }
            }
        }
        guard pts.count >= w / 4 else { return out }

        // Hough vote: line y = r + tan(theta) * (x - cx)
        let angles = (0...80).map { -10.0 + 0.25 * Double($0) }
        let tans = angles.map { tan($0 * .pi / 180) }
        let cx = Double(w) / 2
        let off = Int(0.2 * Double(h)), nR = Int(1.4 * Double(h)) + 1
        var acc = [Int32](repeating: 0, count: angles.count * nR)
        for (x, y) in pts {
            for a in angles.indices {
                let r = Int((Double(y) - tans[a] * (Double(x) - cx)).rounded()) + off
                if r >= 0 && r < nR { acc[a * nR + r] += 1 }
            }
        }
        var peaks: [(Int32, Int, Int)] = []
        for a in angles.indices { for r in 0..<nR where acc[a * nR + r] >= Int32(w / 4) { peaks.append((acc[a * nR + r], a, r)) } }
        peaks.sort { $0.0 > $1.0 }
        var chosen: [(Int, Int)] = []
        for (_, a, r) in peaks {
            if chosen.contains(where: { abs($0.0 - a) <= 4 && abs($0.1 - r) <= 6 }) { continue }
            chosen.append((a, r))
            if chosen.count == 20 { break }
        }

        // Lab, lightly blurred, with a/b weighted as in 8-bit OpenCV Lab
        var Lp = Plane(w: w, h: h), Ap = Lp, Bp = Lp
        for i in 0..<(w * h) {
            let (l, a, b) = Imaging.lab(small.data[i * 3], small.data[i * 3 + 1], small.data[i * 3 + 2])
            Lp.d[i] = l; Ap.d[i] = a / 2.55; Bp.d[i] = b / 2.55
        }
        Lp = Imaging.gaussianBlur(Lp, sigma: 2); Ap = Imaging.gaussianBlur(Ap, sigma: 2); Bp = Imaging.gaussianBlur(Bp, sigma: 2)
        let band = max(4, Int(0.035 * Double(h)))
        let maxGap = max(2, Int(0.04 * Double(w)))
        var best: (score: Double, deg: Double, frac: Double, dE: Double, y: Double)?
        for (a, r) in chosen {
            let t = tans[a], r0 = Double(r - off)
            func lineY(_ x: Double) -> Double { r0 + t * (x - cx) }
            var cols = [Bool](repeating: false, count: w)
            for (x, y) in pts where abs(Double(y) - lineY(Double(x))) <= 1.5 { cols[x] = true }
            // longest run of supported columns, bridging gaps up to maxGap
            var bestRun = (0, -1), start = -1, last = -1
            for x in 0..<w where cols[x] {
                if start < 0 || x - last > maxGap { start = x }
                last = x
                if last - start > bestRun.1 - bestRun.0 { bestRun = (start, last) }
            }
            let x1 = bestRun.0, x2 = bestRun.1
            let len = Double(x2 - x1)
            guard len >= 0.25 * Double(w) else { continue }
            let ya = lineY(Double(x1)), yb = lineY(Double(x2))
            guard min(ya, yb) - Double(band) >= 0, max(ya, yb) + Double(band) <= Double(h - 1) else { continue }
            var diffs: [Float] = []
            for k in 0..<40 {
                let x = Double(x1) + len * Double(k) / 39
                let y = lineY(x)
                let xi = min(w - 1, max(0, Int(x)))
                let up = min(h - 1, max(0, Int(y - Double(band)))) * w + xi
                let dn = min(h - 1, max(0, Int(y + Double(band)))) * w + xi
                let dl = Lp.d[up] - Lp.d[dn], da = Ap.d[up] - Ap.d[dn], db = Bp.d[up] - Bp.d[dn]
                diffs.append((dl * dl + da * da + db * db).squareRoot())
            }
            let dE = Imaging.median(diffs)
            guard dE >= 9 else { continue }
            let score = len / Double(w) * min(dE, 40)
            if best == nil || score > best!.score {
                best = (score, angles[a], len / Double(w), dE, lineY(cx) / Double(h))
            }
        }
        guard let b = best else { return out }
        out["tilt_deg"] = b.deg
        out["tilt_conf"] = min(1, b.frac / 0.55) * min(1, b.dE / 18)
        out["horizon_y"] = b.y
        return out
    }

    // MARK: - saliency / visual weight

    /// Vision attention saliency blended with colour contrast and the focus map, then the same
    /// statistics as signals.saliency on a 96x96 grid.
    static func saliency(_ img: RGBImage, cg: CGImage?, detail: Plane) -> ([String: Double], Plane) {
        let S = 96
        let small = ImagePrep.resizeArea(img, width: S, height: S)

        var sal = Plane(w: S, h: S)
        if let cg {
            let req = VNGenerateAttentionBasedSaliencyImageRequest().cpuOnSimulator()
            if (try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])) != nil,
               let o = req.results?.first {
                let pb = o.pixelBuffer
                CVPixelBufferLockBaseAddress(pb, .readOnly)
                let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
                let stride = CVPixelBufferGetBytesPerRow(pb) / MemoryLayout<Float>.size
                if let base = CVPixelBufferGetBaseAddress(pb)?.assumingMemoryBound(to: Float.self) {
                    var p = Plane(w: w, h: h)
                    for y in 0..<h { for x in 0..<w { p[x, y] = base[y * stride + x] } }
                    sal = Imaging.resize(p, w: S, h: S)
                }
                CVPixelBufferUnlockBaseAddress(pb, .readOnly)
            }
        }
        let smax = sal.max
        if smax > 0 { for i in sal.d.indices { sal.d[i] /= smax } }

        // colour contrast against the frame mean (8-bit Lab scaling, as cv2 on uint8)
        var L = Plane(w: S, h: S), A = L, B = L
        for i in 0..<(S * S) {
            let (l, a, b) = Imaging.lab(small.data[i * 3], small.data[i * 3 + 1], small.data[i * 3 + 2])
            L.d[i] = l * 2.55; A.d[i] = a; B.d[i] = b
        }
        let mL = Float(L.sum / Double(S * S)), mA = Float(A.sum / Double(S * S)), mB = Float(B.sum / Double(S * S))
        let bl = Imaging.gaussianBlur(L, sigma: 2), ba = Imaging.gaussianBlur(A, sigma: 2), bb = Imaging.gaussianBlur(B, sigma: 2)
        var cc = Plane(w: S, h: S)
        for i in 0..<(S * S) {
            let dl = bl.d[i] - mL, da = ba.d[i] - mA, db = bb.d[i] - mB
            cc.d[i] = (dl * dl + da * da + db * db).squareRoot()
        }
        let cmax = cc.max + 1e-9
        let d = Imaging.resize(detail, w: S, h: S)
        let dmax = d.max + 1e-9
        var m = Plane(w: S, h: S)
        for i in 0..<(S * S) {
            m.d[i] = 0.5 * sal.d[i] + 0.3 * cc.d[i] / cmax + 0.35 * max(0, min(1, d.d[i] / dmax))
        }
        let p60 = Float(Imaging.percentile(m.d, 60))
        for i in m.d.indices { m.d[i] = max(0, m.d[i] - p60) }
        let tot = Float(m.sum) + 1e-9
        for i in m.d.indices { m.d[i] /= tot }

        var cx = 0.0, cy = 0.0, right = 0.0, left = 0.0, border = 0.0
        for y in 0..<S {
            for x in 0..<S {
                let v = Double(m[x, y])
                cx += v * (Double(x) + 0.5); cy += v * (Double(y) + 0.5)
                if x >= 48 { right += v } else { left += v }
                if x < 8 || x >= S - 8 || y < 8 || y >= S - 8 { border += v }
            }
        }
        let pk = Imaging.gaussianBlur(m, sigma: 5)
        var best = 0
        for i in pk.d.indices where pk.d[i] > pk.d[best] { best = i }
        let px = (Double(best % S) + 0.5) / Double(S), py = (Double(best / S) + 0.5) / Double(S)
        let t1: Double = 1.0 / 3, t2: Double = 2.0 / 3
        let thirds: [(Double, Double)] = [(t1, t1), (t2, t1), (t1, t2), (t2, t2)]
        let dThird = thirds.map { hypot(px - $0.0, py - $0.1) }.min() ?? 0
        let mmax = m.max + 1e-9
        var norm = m
        for i in norm.d.indices { norm.d[i] /= mmax }
        return ([
            "weight_cx": cx / Double(S), "weight_cy": cy / Double(S),
            "subject_x": px, "subject_y": py,
            "thirds_dist": dThird,
            "center_dist": hypot(px - 0.5, py - 0.5),
            "lr_balance": right - left,
            "edge_weight": border,
        ], norm)
    }

    // MARK: - palette (k-means in Lab on a 64x64 thumbnail)

    static func palette(_ img: RGBImage, k: Int = 6) -> [Swatch] {
        let small = ImagePrep.resizeArea(img, width: 64, height: 64)
        let n = 64 * 64
        var pts = [SIMD3<Float>](repeating: .zero, count: n)
        for i in 0..<n {
            let (l, a, b) = Imaging.lab(small.data[i * 3], small.data[i * 3 + 1], small.data[i * 3 + 2])
            pts[i] = SIMD3(l, a, b)
        }
        // deterministic k-means++ seeding
        var rng: UInt64 = 0x9E3779B97F4A7C15
        func rand() -> Double {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return Double(rng >> 11) / Double(1 << 53)
        }
        var centers: [SIMD3<Float>] = [pts[n / 2]]
        var dist = pts.map { simd_length_squared($0 - centers[0]) }
        while centers.count < k {
            let tot = dist.reduce(0, +)
            var r = Float(rand()) * tot
            var pick = 0
            for i in 0..<n { r -= dist[i]; if r <= 0 { pick = i; break } }
            centers.append(pts[pick])
            for i in 0..<n { dist[i] = min(dist[i], simd_length_squared(pts[i] - pts[pick])) }
        }
        var labels = [Int](repeating: 0, count: n)
        for _ in 0..<20 {
            for i in 0..<n {
                var bi = 0, bd = Float.greatestFiniteMagnitude
                for c in 0..<k {
                    let d = simd_length_squared(pts[i] - centers[c])
                    if d < bd { bd = d; bi = c }
                }
                labels[i] = bi
            }
            var sums = [SIMD3<Float>](repeating: .zero, count: k), cnt = [Float](repeating: 0, count: k)
            for i in 0..<n { sums[labels[i]] += pts[i]; cnt[labels[i]] += 1 }
            for c in 0..<k where cnt[c] > 0 { centers[c] = sums[c] / cnt[c] }
        }
        var counts = [Int](repeating: 0, count: k)
        for l in labels { counts[l] += 1 }
        return (0..<k).sorted { counts[$0] > counts[$1] }.filter { counts[$0] > 0 }.map { c in
            let (r, g, b) = Imaging.rgb(L: centers[c].x, a: centers[c].y, b: centers[c].z)
            return Swatch(r: r, g: g, b: b, share: Double(counts[c]) / Double(n))
        }
    }

    // MARK: - histogram

    static func histogram(_ img: RGBImage, bins: Int = 64) -> Histogram {
        var r = [Double](repeating: 0, count: bins), g = r, b = r, l = r
        let n = img.width * img.height
        img.data.withUnsafeBufferPointer { p in
            for i in 0..<n {
                let R = Int(p[i * 3]), G = Int(p[i * 3 + 1]), B = Int(p[i * 3 + 2])
                r[(R * bins) >> 8] += 1; g[(G * bins) >> 8] += 1; b[(B * bins) >> 8] += 1
                let y = Double(R) * 0.2126 + Double(G) * 0.7152 + Double(B) * 0.0722
                l[min(bins - 1, Int(y * Double(bins) / 256))] += 1
            }
        }
        let N = Double(max(n, 1))
        return Histogram(r: r.map { $0 / N }, g: g.map { $0 / N }, b: b.map { $0 / N }, l: l.map { $0 / N })
    }
}

import UIKit

/// Pixel overlays drawn on the standard image (engine.py _ov_*).  Vector annotations
/// (thirds grid, horizon, framing box, object numbers) are drawn live by the UI instead.
enum Overlays {

    /// grey * keep + colour * 0.15: the photo pushed back so the overlay reads.
    private static func dim(_ img: RGBImage, keep: Float) -> [Float] {
        let n = img.width * img.height
        var out = [Float](repeating: 0, count: n * 3)
        for i in 0..<n {
            let r = Float(img.data[i * 3]), g = Float(img.data[i * 3 + 1]), b = Float(img.data[i * 3 + 2])
            let gray = (0.299 * r + 0.587 * g + 0.114 * b).rounded()
            out[i * 3] = gray * keep + r * 0.15
            out[i * 3 + 1] = gray * keep + g * 0.15
            out[i * 3 + 2] = gray * keep + b * 0.15
        }
        return out
    }

    private static func image(_ f: [Float], w: Int, h: Int) -> UIImage {
        Imaging.uiImage(RGBImage(width: w, height: h, data: f.map { UInt8(max(0, min(255, $0))) }))
    }

    /// Red = any channel blown to white, blue = crushed to black.
    static func clipping(_ img: RGBImage) -> UIImage {
        var out = dim(img, keep: 0.55)
        for i in 0..<(img.width * img.height) {
            let r = img.data[i * 3], g = img.data[i * 3 + 1], b = img.data[i * 3 + 2]
            let y = (Float(r) * 0.2126 + Float(g) * 0.7152 + Float(b) * 0.0722) / 255
            if max(r, max(g, b)) >= 252 { out[i * 3] = 255; out[i * 3 + 1] = 45; out[i * 3 + 2] = 60 }
            if y <= 0.016 { out[i * 3] = 40; out[i * 3 + 1] = 110; out[i * 3 + 2] = 255 }
        }
        return image(out, w: img.width, h: img.height)
    }

    private static let inferno: [(Float, SIMD3<Float>)] = [
        (0.00, SIMD3(0, 0, 4)), (0.13, SIMD3(31, 12, 72)), (0.25, SIMD3(85, 15, 109)), (0.38, SIMD3(136, 34, 106)),
        (0.50, SIMD3(186, 54, 85)), (0.63, SIMD3(227, 89, 51)), (0.75, SIMD3(249, 140, 10)), (0.88, SIMD3(249, 201, 50)),
        (1.00, SIMD3(252, 255, 164)),
    ]

    static func infernoColor(_ t: Float) -> SIMD3<Float> {
        let v = max(0, min(1, t))
        for k in 1..<inferno.count where v <= inferno[k].0 {
            let (t0, c0) = inferno[k - 1], (t1, c1) = inferno[k]
            let f = (v - t0) / (t1 - t0)
            return c0 + (c1 - c0) * f
        }
        return inferno.last!.1
    }

    /// Where the photo is sharp: bright = crisp fine detail.  `scale` is shared by both photos so they compare.
    static func focus(_ img: RGBImage, detail: Plane, scale: Float) -> UIImage {
        let w = img.width, h = img.height
        var d = detail
        for i in d.d.indices { d.d[i] = powf(max(0, min(1, d.d[i] / (scale + 1e-9))), 0.6) }
        var big = Imaging.resize(d, w: w, h: h)
        big = Imaging.gaussianBlur(big, sigma: 0.02 * Double(max(w, h)))
        var out = dim(img, keep: 0.5)
        for i in 0..<(w * h) {
            let c = infernoColor(big.d[i])
            out[i * 3] = out[i * 3] * 0.45 + c.x * 0.6
            out[i * 3 + 1] = out[i * 3 + 1] * 0.45 + c.y * 0.6
            out[i * 3 + 2] = out[i * 3 + 2] * 0.45 + c.z * 0.6
        }
        return image(out, w: w, h: h)
    }

    /// Green = the score leans on this region more than on the rest of the frame; red = less (or it hurts).
    static func impact(_ img: RGBImage, heat: Plane) -> UIImage {
        let w = img.width, h = img.height
        let med = Float(Imaging.median(heat.d))
        var rel = heat
        for i in rel.d.indices { rel.d[i] -= med }
        let mx = (rel.d.map { abs($0) }.max() ?? 0) + 1e-9
        let m = Imaging.resize(rel, w: w, h: h)
        var out = dim(img, keep: 0.6)
        for i in 0..<(w * h) {
            let v = m.d[i] / mx
            let pos = powf(max(0, min(1, v)), 0.8), neg = powf(max(0, min(1, -v)), 0.8)
            for (k, (gc, rc)) in [(40, 255), (210, 80), (150, 60)].enumerated() {
                var o = out[i * 3 + k] * (1 - 0.75 * pos) + Float(gc) * 0.75 * pos
                o = o * (1 - 0.75 * neg) + Float(rc) * 0.75 * neg
                out[i * 3 + k] = o
            }
        }
        return image(out, w: w, h: h)
    }

    static func roleColor(_ role: String) -> SIMD3<Float> {
        switch role {
        case "subject": return SIMD3(255, 200, 40)
        case "distraction": return SIMD3(255, 80, 60)
        case "helps": return SIMD3(40, 210, 150)
        default: return SIMD3(235, 235, 235)
        }
    }

    /// Non-mask pixels connected to the frame border.
    private static func reachableFromBorder(_ m: [Bool], w: Int, h: Int) -> [Bool] {
        var out = [Bool](repeating: false, count: w * h)
        var stack: [Int] = []
        func push(_ i: Int) { if !m[i] && !out[i] { out[i] = true; stack.append(i) } }
        for x in 0..<w { push(x); push((h - 1) * w + x) }
        for y in 0..<h { push(y * w); push(y * w + w - 1) }
        while let i = stack.popLast() {
            let x = i % w, y = i / w
            if x > 0 { push(i - 1) }
            if x < w - 1 { push(i + 1) }
            if y > 0 { push(i - w) }
            if y < h - 1 { push(i + w) }
        }
        return out
    }

    /// Tinted masks + outlines; yellow = subject, red = distraction, green = the score leans on it.
    static func objects(_ img: RGBImage, segs: [Segment]) -> UIImage {
        let w = img.width, h = img.height
        var out = [Float](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            let r = Float(img.data[i * 3]), g = Float(img.data[i * 3 + 1]), b = Float(img.data[i * 3 + 2])
            let gray = 0.299 * r + 0.587 * g + 0.114 * b
            out[i * 3] = gray * 0.35 + r * 0.4; out[i * 3 + 1] = gray * 0.35 + g * 0.4; out[i * 3 + 2] = gray * 0.35 + b * 0.4
        }
        for g in segs where g.role == "distraction" || g.role == "subject" || (g.role == "helps" && g.compact) {
            let c = roleColor(g.role)
            for i in 0..<(w * h) where g.mask[i] {
                out[i * 3] = out[i * 3] * 0.62 + c.x * 0.38
                out[i * 3 + 1] = out[i * 3 + 1] * 0.62 + c.y * 0.38
                out[i * 3 + 2] = out[i * 3 + 2] * 0.62 + c.z * 0.38
            }
        }
        let t = max(1, Int((Double(max(w, h)) / 420).rounded()))
        for g in segs.sorted(by: { ($0.role != "other" ? 1 : 0) < ($1.role != "other" ? 1 : 0) }) {
            let c = roleColor(g.role)
            let thick = g.role == "other" ? t : t + 1
            // outer contour only (cv2.RETR_EXTERNAL): holes inside the mask are not outlined
            let outside = reachableFromBorder(g.mask, w: w, h: h)
            for y in 0..<h {
                for x in 0..<w where g.mask[y * w + x] {
                    let edge = x == 0 || y == 0 || x == w - 1 || y == h - 1
                        || outside[y * w + x - 1] || outside[y * w + x + 1] || outside[(y - 1) * w + x] || outside[(y + 1) * w + x]
                    guard edge else { continue }
                    for yy in max(0, y - thick / 2)...min(h - 1, y + thick / 2) {
                        for xx in max(0, x - thick / 2)...min(w - 1, x + thick / 2) {
                            let o = (yy * w + xx) * 3
                            out[o] = c.x; out[o + 1] = c.y; out[o + 2] = c.z
                        }
                    }
                }
            }
        }
        return image(out, w: w, h: h)
    }
}

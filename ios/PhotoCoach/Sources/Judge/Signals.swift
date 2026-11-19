import Foundation

/// Measurements read directly off the pixels of the standard image (long side <= 1024).
/// A port of the tone / colour / size parts of photocoach/signals.py: the same numbers the
/// network saw during training, so the definitions must not drift from the Python.
enum Signals {

    /// numpy.percentile (linear interpolation) on an already-sorted array.
    private static func percentile(_ sorted: [Float], _ q: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let pos = q / 100 * Double(sorted.count - 1)
        let lo = Int(pos.rounded(.down))
        let hi = min(sorted.count - 1, lo + 1)
        let f = pos - Double(lo)
        return Double(sorted[lo]) * (1 - f) + Double(sorted[hi]) * f
    }

    private static func srgbToLinear(_ c: Float) -> Float {
        c <= 0.04045 ? c / 12.92 : powf((c + 0.055) / 1.055, 2.4)
    }

    private static func labF(_ t: Float) -> Float {
        t > 0.008856 ? cbrtf(t) : 7.787 * t + 16.0 / 116.0
    }

    static func measure(_ img: RGBImage) -> [String: Double] {
        let n = img.width * img.height
        var out: [String: Double] = [:]
        guard n > 0 else { return out }

        var luma = [Float](repeating: 0, count: n)
        var chroma = [Float](repeating: 0, count: n)
        var labL = [Float](repeating: 0, count: n)
        var labA = [Float](repeating: 0, count: n)
        var labB = [Float](repeating: 0, count: n)

        // sRGB -> linear lookup (256 entries) keeps the Lab conversion cheap
        let lin: [Float] = (0..<256).map { srgbToLinear(Float($0) / 255) }

        var sumY = 0.0, sumY2 = 0.0
        var clipHi = 0, clipLo = 0, nearHi = 0, nearLo = 0
        var sumSat = 0.0
        var sumRG = 0.0, sumRG2 = 0.0, sumYB = 0.0, sumYB2 = 0.0
        var sumA = 0.0, sumB = 0.0

        img.data.withUnsafeBufferPointer { p in
            for i in 0..<n {
                let r8 = p[i * 3], g8 = p[i * 3 + 1], b8 = p[i * 3 + 2]
                let r = Float(r8) / 255, g = Float(g8) / 255, b = Float(b8) / 255

                let y = r * 0.2126 + g * 0.7152 + b * 0.0722
                luma[i] = y
                sumY += Double(y); sumY2 += Double(y) * Double(y)
                let mx8 = max(r8, max(g8, b8))
                if mx8 >= 252 { clipHi += 1 }
                if y <= 0.016 { clipLo += 1 }
                if y >= 0.92 { nearHi += 1 }
                if y <= 0.06 { nearLo += 1 }

                let mx = max(r, max(g, b)), mn = min(r, min(g, b))
                sumSat += mx > 0 ? Double((mx - mn) / mx) : 0

                let rg = Double(r - g), yb = Double(0.5 * (r + g) - b)
                sumRG += rg; sumRG2 += rg * rg; sumYB += yb; sumYB2 += yb * yb

                // CIE Lab (D65), as OpenCV's float RGB2LAB
                let lr = lin[Int(r8)], lg = lin[Int(g8)], lb = lin[Int(b8)]
                let X = (0.412453 * lr + 0.357580 * lg + 0.180423 * lb) / 0.950456
                let Y = 0.212671 * lr + 0.715160 * lg + 0.072169 * lb
                let Z = (0.019334 * lr + 0.119193 * lg + 0.950227 * lb) / 1.088754
                let fx = labF(X), fy = labF(Y), fz = labF(Z)
                let L = Y > 0.008856 ? 116 * fy - 16 : 903.3 * Y
                let a = 500 * (fx - fy), bb = 200 * (fy - fz)
                labL[i] = L; labA[i] = a; labB[i] = bb
                chroma[i] = (a * a + bb * bb).squareRoot()
                sumA += Double(a); sumB += Double(bb)
            }
        }

        let N = Double(n)
        let sortedY = luma.sorted()
        let p01 = percentile(sortedY, 1), p05 = percentile(sortedY, 5), p50 = percentile(sortedY, 50)
        let p95 = percentile(sortedY, 95), p99 = percentile(sortedY, 99)
        let meanY = sumY / N
        out["luma_mean"] = meanY
        out["luma_median"] = p50
        out["luma_p01"] = p01
        out["luma_p05"] = p05
        out["luma_p95"] = p95
        out["luma_p99"] = p99
        out["clip_hi"] = Double(clipHi) / N
        out["clip_lo"] = Double(clipLo) / N
        out["near_hi"] = Double(nearHi) / N
        out["near_lo"] = Double(nearLo) / N
        out["contrast_rms"] = max(0, sumY2 / N - meanY * meanY).squareRoot()
        out["dyn_range"] = p99 - p01

        let mRG = sumRG / N, mYB = sumYB / N
        let vRG = max(0, sumRG2 / N - mRG * mRG), vYB = max(0, sumYB2 / N - mYB * mYB)
        out["sat_mean"] = sumSat / N
        out["colorfulness"] = (vRG + vYB).squareRoot() + 0.3 * (mRG * mRG + mYB * mYB).squareRoot()
        out["cast_a"] = sumA / N / 50
        out["cast_b"] = sumB / N / 50

        // "should-be-neutral" pixels: the least colourful quarter of the mid-tones; their mean a/b is the cast
        let c25 = Float(percentile(chroma.sorted(), 25))
        var na = 0.0, nb = 0.0, cnt = 0
        for i in 0..<n where chroma[i] < c25 && labL[i] > 20 && labL[i] < 92 {
            na += Double(labA[i]); nb += Double(labB[i]); cnt += 1
        }
        if cnt < 50 { na = sumA; nb = sumB; cnt = n }
        out["neutral_cast_a"] = na / Double(cnt) / 50
        out["neutral_cast_b"] = nb / Double(cnt) / 50

        out["log_long_side"] = log(Double(max(img.width, img.height)) / 1024.0 + 1e-6)
        out["aspect"] = Double(img.width) / Double(img.height)
        return out
    }
}

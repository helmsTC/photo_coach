import Accelerate
import CoreGraphics
import UIKit

/// A single-channel float image (row-major, top row first).
struct Plane {
    var w: Int
    var h: Int
    var d: [Float]

    init(w: Int, h: Int, fill: Float = 0) {
        self.w = w
        self.h = h
        d = [Float](repeating: fill, count: w * h)
    }

    init(w: Int, h: Int, d: [Float]) {
        precondition(d.count == w * h)
        self.w = w
        self.h = h
        self.d = d
    }

    @inline(__always) subscript(x: Int, y: Int) -> Float {
        get { d[y * w + x] }
        set { d[y * w + x] = newValue }
    }

    var max: Float { d.max() ?? 0 }
    var sum: Double { d.reduce(0.0) { $0 + Double($1) } }
}

/// Small numeric helpers that mirror the OpenCV / numpy calls used by the desktop app.
enum Imaging {

    // MARK: - percentiles (numpy linear interpolation)

    static func percentile(sorted s: [Float], _ q: Double) -> Double {
        guard !s.isEmpty else { return 0 }
        let pos = q / 100 * Double(s.count - 1)
        let lo = Int(pos.rounded(.down))
        let hi = Swift.min(s.count - 1, lo + 1)
        let f = pos - Double(lo)
        return Double(s[lo]) * (1 - f) + Double(s[hi]) * f
    }

    static func percentile(_ v: [Float], _ q: Double) -> Double { percentile(sorted: v.sorted(), q) }

    static func median(_ v: [Float]) -> Double { percentile(v, 50) }

    // MARK: - borders

    /// OpenCV BORDER_REFLECT_101 index.
    @inline(__always) static func reflect101(_ i: Int, _ n: Int) -> Int {
        if n == 1 { return 0 }
        var x = i
        while x < 0 || x >= n {
            if x < 0 { x = -x }
            if x >= n { x = 2 * n - 2 - x }
        }
        return x
    }

    // MARK: - Gaussian blur

    /// Same kernel size rule as cv2.GaussianBlur for float images: round(sigma * 8 + 1) | 1.
    static func gaussianKernel(sigma: Double) -> [Float] {
        let k = Int((sigma * 8 + 1).rounded()) | 1
        let c = k / 2
        var w = (0..<k).map { i -> Float in
            let x = Double(i - c)
            return Float(exp(-x * x / (2 * sigma * sigma)))
        }
        let s = w.reduce(0, +)
        for i in w.indices { w[i] /= s }
        return w
    }

    private static func convolveRows(_ p: Plane, _ kernel: [Float]) -> Plane {
        let r = kernel.count / 2
        var out = [Float](repeating: 0, count: p.w * p.h)
        var buf = [Float](repeating: 0, count: p.w + 2 * r)
        let idx = (0..<(p.w + 2 * r)).map { reflect101($0 - r, p.w) }
        p.d.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<p.h {
                    let row = y * p.w
                    for i in 0..<buf.count { buf[i] = src[row + idx[i]] }
                    vDSP_conv(buf, 1, kernel, 1, dst.baseAddress! + row, 1, vDSP_Length(p.w), vDSP_Length(kernel.count))
                }
            }
        }
        return Plane(w: p.w, h: p.h, d: out)
    }

    static func transpose(_ p: Plane) -> Plane {
        var out = [Float](repeating: 0, count: p.w * p.h)
        p.d.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { o in
                for y in 0..<p.h {
                    for x in 0..<p.w { o[x * p.h + y] = s[y * p.w + x] }
                }
            }
        }
        return Plane(w: p.h, h: p.w, d: out)
    }

    static func gaussianBlur(_ p: Plane, sigma: Double) -> Plane {
        guard sigma > 0, p.w > 0, p.h > 0 else { return p }
        let k = gaussianKernel(sigma: sigma)
        let a = convolveRows(p, k)
        return transpose(convolveRows(transpose(a), k))
    }

    // MARK: - resizing float maps

    /// Bilinear resize with pixel-centre alignment (as cv2.resize INTER_LINEAR).
    static func resize(_ p: Plane, w ow: Int, h oh: Int) -> Plane {
        if ow == p.w && oh == p.h { return p }
        var out = Plane(w: ow, h: oh)
        let sx = Double(p.w) / Double(ow), sy = Double(p.h) / Double(oh)
        for y in 0..<oh {
            let fy = Swift.max(0, Swift.min(Double(p.h - 1), (Double(y) + 0.5) * sy - 0.5))
            let y0 = Int(fy), y1 = Swift.min(p.h - 1, y0 + 1)
            let ty = Float(fy - Double(y0))
            for x in 0..<ow {
                let fx = Swift.max(0, Swift.min(Double(p.w - 1), (Double(x) + 0.5) * sx - 0.5))
                let x0 = Int(fx), x1 = Swift.min(p.w - 1, x0 + 1)
                let tx = Float(fx - Double(x0))
                let a = p[x0, y0] * (1 - tx) + p[x1, y0] * tx
                let b = p[x0, y1] * (1 - tx) + p[x1, y1] * tx
                out[x, y] = a * (1 - ty) + b * ty
            }
        }
        return out
    }

    /// Area-average shrink of a float plane (INTER_AREA when shrinking).
    static func shrink(_ p: Plane, w ow: Int, h oh: Int) -> Plane {
        if ow >= p.w || oh >= p.h { return resize(p, w: ow, h: oh) }
        var out = Plane(w: ow, h: oh)
        var cnt = Plane(w: ow, h: oh)
        for y in 0..<p.h {
            let oy = Swift.min(oh - 1, y * oh / p.h)
            for x in 0..<p.w {
                let ox = Swift.min(ow - 1, x * ow / p.w)
                out[ox, oy] += p[x, y]
                cnt[ox, oy] += 1
            }
        }
        for i in out.d.indices { out.d[i] /= Swift.max(cnt.d[i], 1) }
        return out
    }

    // MARK: - colour planes

    /// cv2.cvtColor(RGB2GRAY) on 8-bit: rounded 0.299 R + 0.587 G + 0.114 B, scaled to 0...1.
    static func gray(_ img: RGBImage) -> Plane {
        var out = [Float](repeating: 0, count: img.width * img.height)
        img.data.withUnsafeBufferPointer { p in
            for i in 0..<(img.width * img.height) {
                let v = 0.299 * Float(p[i * 3]) + 0.587 * Float(p[i * 3 + 1]) + 0.114 * Float(p[i * 3 + 2])
                out[i] = v.rounded() / 255
            }
        }
        return Plane(w: img.width, h: img.height, d: out)
    }

    /// Rec.709 luma on 0...1 floats (signals._luma).
    static func luma(_ img: RGBImage) -> Plane {
        var out = [Float](repeating: 0, count: img.width * img.height)
        img.data.withUnsafeBufferPointer { p in
            for i in 0..<(img.width * img.height) {
                out[i] = (Float(p[i * 3]) * 0.2126 + Float(p[i * 3 + 1]) * 0.7152 + Float(p[i * 3 + 2]) * 0.0722) / 255
            }
        }
        return Plane(w: img.width, h: img.height, d: out)
    }

    /// Split into three 0...255 float planes.
    static func planes(_ img: RGBImage) -> [Plane] {
        let n = img.width * img.height
        var r = [Float](repeating: 0, count: n), g = r, b = r
        img.data.withUnsafeBufferPointer { p in
            for i in 0..<n {
                r[i] = Float(p[i * 3]); g[i] = Float(p[i * 3 + 1]); b[i] = Float(p[i * 3 + 2])
            }
        }
        return [Plane(w: img.width, h: img.height, d: r), Plane(w: img.width, h: img.height, d: g), Plane(w: img.width, h: img.height, d: b)]
    }

    static func merge(_ c: [Plane]) -> RGBImage {
        let w = c[0].w, h = c[0].h
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            for k in 0..<3 { out[i * 3 + k] = UInt8(Swift.max(0, Swift.min(255, c[k].d[i].rounded()))) }
        }
        return RGBImage(width: w, height: h, data: out)
    }

    static func blurRGB(_ img: RGBImage, sigma: Double) -> RGBImage {
        merge(planes(img).map { gaussianBlur($0, sigma: sigma) })
    }

    // MARK: - CIE Lab (OpenCV float RGB2LAB conventions: L 0...100, a/b roughly -127...127)

    static let srgbToLinear: [Float] = (0..<256).map { i in
        let c = Float(i) / 255
        return c <= 0.04045 ? c / 12.92 : powf((c + 0.055) / 1.055, 2.4)
    }

    @inline(__always) static func labF(_ t: Float) -> Float { t > 0.008856 ? cbrtf(t) : 7.787 * t + 16.0 / 116.0 }

    @inline(__always) static func lab(_ r8: UInt8, _ g8: UInt8, _ b8: UInt8) -> (Float, Float, Float) {
        let lr = srgbToLinear[Int(r8)], lg = srgbToLinear[Int(g8)], lb = srgbToLinear[Int(b8)]
        let X = (0.412453 * lr + 0.357580 * lg + 0.180423 * lb) / 0.950456
        let Y = 0.212671 * lr + 0.715160 * lg + 0.072169 * lb
        let Z = (0.019334 * lr + 0.119193 * lg + 0.950227 * lb) / 1.088754
        let fx = labF(X), fy = labF(Y), fz = labF(Z)
        let L = Y > 0.008856 ? 116 * fy - 16 : 903.3 * Y
        return (L, 500 * (fx - fy), 200 * (fy - fz))
    }

    /// Lab -> 8-bit sRGB (inverse of `lab`), clipped.
    @inline(__always) static func rgb(L: Float, a: Float, b: Float) -> (UInt8, UInt8, UInt8) {
        let fy = (L + 16) / 116
        let fx = fy + a / 500
        let fz = fy - b / 200
        func finv(_ t: Float) -> Float { t > 0.206893 ? t * t * t : (t - 16.0 / 116.0) / 7.787 }
        let Y = L > 7.9996 ? fy * fy * fy : L / 903.3
        let X = finv(fx) * 0.950456
        let Z = finv(fz) * 1.088754
        var r = 3.240479 * X - 1.537150 * Y - 0.498535 * Z
        var g = -0.969256 * X + 1.875992 * Y + 0.041556 * Z
        var bl = 0.055648 * X - 0.204043 * Y + 1.057311 * Z
        func gamma(_ c: Float) -> Float {
            let v = Swift.max(0, Swift.min(1, c))
            return v <= 0.0031308 ? 12.92 * v : 1.055 * powf(v, 1 / 2.4) - 0.055
        }
        r = gamma(r); g = gamma(g); bl = gamma(bl)
        return (UInt8(r * 255 + 0.5), UInt8(g * 255 + 0.5), UInt8(bl * 255 + 0.5))
    }

    // MARK: - CoreGraphics bridges

    static func cgImage(_ img: RGBImage) -> CGImage? {
        var rgba = [UInt8](repeating: 255, count: img.width * img.height * 4)
        img.data.withUnsafeBufferPointer { s in
            for i in 0..<(img.width * img.height) {
                rgba[i * 4] = s[i * 3]; rgba[i * 4 + 1] = s[i * 3 + 1]; rgba[i * 4 + 2] = s[i * 3 + 2]
            }
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: img.width, height: img.height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: img.width * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    static func uiImage(_ img: RGBImage) -> UIImage {
        guard let cg = cgImage(img) else { return UIImage() }
        return UIImage(cgImage: cg)
    }

    /// RGBA overlay (premultiplied not required: alpha is 0 or 255 except where noted).
    static func uiImage(rgba: [UInt8], w: Int, h: Int) -> UIImage {
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                               space: CGColorSpace(name: CGColorSpace.sRGB)!,
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return UIImage() }
        return UIImage(cgImage: cg)
    }

    /// Resize an RGB image with bilinear sampling (used for enlarging; shrinking uses ImagePrep.resizeArea).
    static func resizeBilinear(_ src: RGBImage, w ow: Int, h oh: Int) -> RGBImage {
        let c = planes(src).map { resize($0, w: ow, h: oh) }
        return merge(c)
    }
}

import CoreVideo
import UIKit

/// 8-bit RGB pixels, row-major, top row first, 3 bytes per pixel.
struct RGBImage {
    let width: Int
    let height: Int
    var data: [UInt8]

    init(width: Int, height: Int, data: [UInt8]) {
        precondition(data.count == width * height * 3)
        self.width = width
        self.height = height
        self.data = data
    }
}

/// Mirrors the Python preprocessing (photocoach/signals.py `to_std`, mobile/extract.py `views`)
/// so the on-device model sees what it was trained on.
enum ImagePrep {

    /// Decode, apply the EXIF orientation, and shrink so the long side is at most `longSide`
    /// (never enlarges).  This is the "standard" image every measurement and crop is taken from.
    static func standardImage(from data: Data, longSide: Int) throws -> (RGBImage, UIImage) {
        guard let ui = UIImage(data: data), ui.size.width >= 1, ui.size.height >= 1 else { throw JudgeError.badImage }
        let pxW = ui.size.width * ui.scale
        let pxH = ui.size.height * ui.scale
        let s = min(1.0, CGFloat(longSide) / max(pxW, pxH))
        let w = max(1, Int((pxW * s).rounded()))
        let h = max(1, Int((pxH * s).rounded()))
        return (try render(ui, width: w, height: h), ui)
    }

    /// Draw a UIImage (orientation applied) into a w x h RGB buffer.
    static func render(_ ui: UIImage, width w: Int, height h: Int) throws -> RGBImage {
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        let ok: Bool = rgba.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            ctx.interpolationQuality = .high
            ctx.setFillColor(UIColor.white.cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            // UIKit draws top-down; flip so UIImage.draw applies the orientation the right way up
            ctx.translateBy(x: 0, y: CGFloat(h))
            ctx.scaleBy(x: 1, y: -1)
            UIGraphicsPushContext(ctx)
            ui.draw(in: CGRect(x: 0, y: 0, width: w, height: h))
            UIGraphicsPopContext()
            return true
        }
        guard ok else { throw JudgeError.badImage }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        var j = 0
        for i in stride(from: 0, to: rgba.count, by: 4) {
            rgb[j] = rgba[i]
            rgb[j + 1] = rgba[i + 1]
            rgb[j + 2] = rgba[i + 2]
            j += 3
        }
        return RGBImage(width: w, height: h, data: rgb)
    }

    /// Area-averaging resize (what OpenCV's INTER_AREA does when shrinking): every output pixel is the
    /// mean of the source rectangle it covers, with fractional coverage at the borders.
    static func resizeArea(_ src: RGBImage, width ow: Int, height oh: Int) -> RGBImage {
        if ow == src.width && oh == src.height { return src }
        let sw = src.width, sh = src.height

        func spans(_ n: Int, _ m: Int) -> [[(Int, Float)]] {
            // for each of m outputs: the source indices it covers and their weights (sum 1)
            let scale = Double(n) / Double(m)
            return (0..<m).map { o in
                let a = Double(o) * scale
                let b = Double(o + 1) * scale
                var out: [(Int, Float)] = []
                var i = Int(a.rounded(.down))
                while Double(i) < b && i < n {
                    let cover = min(b, Double(i + 1)) - max(a, Double(i))
                    if cover > 1e-9 { out.append((i, Float(cover / scale))) }
                    i += 1
                }
                if out.isEmpty { out.append((min(n - 1, max(0, Int(a))), 1)) }
                return out
            }
        }

        let xs = spans(sw, ow)
        let ys = spans(sh, oh)
        var tmp = [Float](repeating: 0, count: sh * ow * 3)          // horizontal pass
        src.data.withUnsafeBufferPointer { s in
            for y in 0..<sh {
                let row = y * sw * 3
                for x in 0..<ow {
                    var r: Float = 0, g: Float = 0, b: Float = 0
                    for (i, wgt) in xs[x] {
                        let p = row + i * 3
                        r += Float(s[p]) * wgt
                        g += Float(s[p + 1]) * wgt
                        b += Float(s[p + 2]) * wgt
                    }
                    let o = (y * ow + x) * 3
                    tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b
                }
            }
        }
        var out = [UInt8](repeating: 0, count: oh * ow * 3)          // vertical pass
        for y in 0..<oh {
            for x in 0..<ow {
                var r: Float = 0, g: Float = 0, b: Float = 0
                for (i, wgt) in ys[y] {
                    let p = (i * ow + x) * 3
                    r += tmp[p] * wgt
                    g += tmp[p + 1] * wgt
                    b += tmp[p + 2] * wgt
                }
                let o = (y * ow + x) * 3
                out[o] = UInt8(max(0, min(255, r.rounded())))
                out[o + 1] = UInt8(max(0, min(255, g.rounded())))
                out[o + 2] = UInt8(max(0, min(255, b.rounded())))
            }
        }
        return RGBImage(width: ow, height: oh, data: out)
    }

    static func crop(_ src: RGBImage, x0: Int, y0: Int, size: Int) -> RGBImage {
        var out = [UInt8](repeating: 0, count: size * size * 3)
        src.data.withUnsafeBufferPointer { s in
            for y in 0..<size {
                let from = ((y0 + y) * src.width + x0) * 3
                let to = y * size * 3
                for k in 0..<(size * 3) { out[to + k] = s[from + k] }
            }
        }
        return RGBImage(width: size, height: size, data: out)
    }

    /// The five model inputs: the whole frame squashed to a square, then four native-resolution
    /// crops centred on the rule-of-thirds intersections (top-left, top-right, bottom-left, bottom-right).
    static func views(std: RGBImage, original: UIImage, size: Int) throws -> [RGBImage] {
        var out = [resizeArea(std, width: size, height: size)]
        var base = std
        if min(std.width, std.height) < size {                       // tiny photo: enlarge so a crop fits
            let s = Double(size) / Double(min(std.width, std.height))
            base = try render(original, width: max(size, Int((Double(std.width) * s).rounded())),
                              height: max(size, Int((Double(std.height) * s).rounded())))
        }
        for fy in [1.0 / 3.0, 2.0 / 3.0] {
            for fx in [1.0 / 3.0, 2.0 / 3.0] {
                let y0 = Int(min(max(fy * Double(base.height) - Double(size) / 2, 0), Double(base.height - size)))
                let x0 = Int(min(max(fx * Double(base.width) - Double(size) / 2, 0), Double(base.width - size)))
                out.append(crop(base, x0: x0, y0: y0, size: size))
            }
        }
        return out
    }

    static func pixelBuffer(_ img: RGBImage) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferCGImageCompatibilityKey: true,
                                      kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                                      kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        let st = CVPixelBufferCreate(kCFAllocatorDefault, img.width, img.height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard st == kCVReturnSuccess, let buf = pb else { throw JudgeError.model("could not allocate pixel buffer") }
        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }
        guard let base = CVPixelBufferGetBaseAddress(buf) else { throw JudgeError.model("pixel buffer has no memory") }
        let stride = CVPixelBufferGetBytesPerRow(buf)
        let dst = base.assumingMemoryBound(to: UInt8.self)
        img.data.withUnsafeBufferPointer { s in
            for y in 0..<img.height {
                for x in 0..<img.width {
                    let p = (y * img.width + x) * 3
                    let o = y * stride + x * 4
                    dst[o] = s[p + 2]          // B
                    dst[o + 1] = s[p + 1]      // G
                    dst[o + 2] = s[p]          // R
                    dst[o + 3] = 255
                }
            }
        }
        return buf
    }
}

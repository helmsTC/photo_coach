import Foundation
import ImageIO

/// Capture settings read from the file (scene.exif in the desktop app).
struct ExifInfo {
    var camera: String?
    var lens: String?
    var aperture: Double?
    var shutter: Double?
    var iso: Int?
    var focal: Double?
    var focal35: Int?
    var evComp: Double?
    var taken: String?
    var flash: Bool?

    var isEmpty: Bool { camera == nil && aperture == nil && shutter == nil && iso == nil && focal == nil }

    static func read(_ data: Data) -> ExifInfo {
        var out = ExifInfo()
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any] else { return out }
        let ex = props[kCGImagePropertyExifDictionary] as? [CFString: Any] ?? [:]
        let tiff = props[kCGImagePropertyTIFFDictionary] as? [CFString: Any] ?? [:]
        let aux = props[kCGImagePropertyExifAuxDictionary] as? [CFString: Any] ?? [:]

        func num(_ v: Any?) -> Double? {
            if let n = v as? NSNumber { return n.doubleValue }
            if let a = v as? [NSNumber], let f = a.first { return f.doubleValue }
            return nil
        }
        func str(_ v: Any?) -> String? {
            guard let s = (v as? String)?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty else { return nil }
            return s
        }

        let cam = [str(tiff[kCGImagePropertyTIFFMake]), str(tiff[kCGImagePropertyTIFFModel])].compactMap { $0 }
        if !cam.isEmpty {
            // "Apple iPhone 15 Pro" rather than "Apple Apple iPhone…" when the model repeats the make
            out.camera = cam.count == 2 && cam[1].hasPrefix(cam[0]) ? cam[1] : cam.joined(separator: " ")
        }
        out.lens = str(ex[kCGImagePropertyExifLensModel]) ?? str(aux[kCGImagePropertyExifAuxLensModel])
        if let f = num(ex[kCGImagePropertyExifFNumber]), f > 0 { out.aperture = (f * 10).rounded() / 10 }
        if let t = num(ex[kCGImagePropertyExifExposureTime]), t > 0 { out.shutter = t }
        if let i = num(ex[kCGImagePropertyExifISOSpeedRatings]), i > 0 { out.iso = Int(i) }
        if let f = num(ex[kCGImagePropertyExifFocalLength]), f > 0 { out.focal = (f * 10).rounded() / 10 }
        if let f = num(ex[kCGImagePropertyExifFocalLenIn35mmFilm]), f > 0 { out.focal35 = Int(f) }
        if let e = num(ex[kCGImagePropertyExifExposureBiasValue]) { out.evComp = (e * 10).rounded() / 10 }
        out.taken = str(ex[kCGImagePropertyExifDateTimeOriginal])
        if let fl = num(ex[kCGImagePropertyExifFlash]) { out.flash = Int(fl) & 1 == 1 }
        return out
    }

    static func shutterString(_ t: Double) -> String {
        t < 0.4 ? "1/\(Int((1 / t).rounded()))s" : String(format: "%.1fs", t)
    }

    /// Short chips for the UI: "ƒ/1.8", "1/120s", "ISO 400", "26mm".
    var chips: [String] {
        var c: [String] = []
        if let f = focal35 { c.append("\(f)mm") } else if let f = focal { c.append("\(fmt(f))mm") }
        if let a = aperture { c.append("ƒ/\(fmt(a))") }
        if let s = shutter { c.append(Self.shutterString(s)) }
        if let i = iso { c.append("ISO \(i)") }
        if let e = evComp, e != 0 { c.append(String(format: "%+.1f EV", e)) }
        if flash == true { c.append("Flash") }
        return c
    }
}

/// "%g"-style: 1.8 -> "1.8", 26.0 -> "26".
func fmt(_ v: Double) -> String {
    v == v.rounded() ? String(Int(v)) : String(format: "%.1f", v)
}

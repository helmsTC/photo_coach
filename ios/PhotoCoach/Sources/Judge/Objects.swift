import CoreML
import Foundation
import UIKit
import Vision

/// One thing in the frame: its mask, where it sits, and (after re-scoring) what it does to the score.
struct Segment: Identifiable {
    var id: Int { n }
    var n = 0
    var label: String
    var score: Double
    var mask: [Bool]                 // std resolution
    var w: Int, h: Int
    var area: Double
    var cx: Double, cy: Double
    var box: (Double, Double, Double, Double)        // x0, y0, x1, y1 (0...1)
    var edges: [String: Double]      // share of each frame edge the object lies against
    var delta = 0.0                  // aesthetic percentile change with it painted out (+ = it distracts)
    var lean = 0.0                   // how much the score leans on it, in reference standard deviations
    var role = "other"               // subject / distraction / helps / other
    var labelPoint = (0.5, 0.5)      // deepest point inside the mask, for the number badge
    var thing = true                 // countable object (vs background "stuff": sky, water, trees...)

    var compact: Bool { area <= 0.12 }
}

struct SubjectInfo {
    var label: String
    var area: Double
    var cx: Double, cy: Double
    var whereText: String
    var thirdsDist: Double
    var centerDist: Double
    var cut: [String]
    var separation: Double?
    var sharpRatio: Double?
}

struct SceneGeometry {
    var subject: SubjectInfo?
    var subjectIndex: Int?
    var nThings: Int
    var sky = 0.0                    // share of the frame
    var water = 0.0
    var horizonY: Double?            // where sky meets land, as a fraction of height
}

/// What is in the frame: a COCO-panoptic segmenter (Apple's Core ML DETR, the same 133 classes as the
/// desktop's Mask2Former: 80 countable things + background stuff such as sky, water and trees), with
/// Vision's person masks splitting groups of people, then the subject / distraction logic of
/// photocoach/objects.py.
enum Objects {

    // MARK: - detection

    /// Read a Vision instance-label buffer (value = instance id, 0 = background) at std resolution.
    private static func labelMap(_ pb: CVPixelBuffer, w: Int, h: Int) -> [UInt8] {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let mw = CVPixelBufferGetWidth(pb), mh = CVPixelBufferGetHeight(pb)
        let row = CVPixelBufferGetBytesPerRow(pb)
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return [] }
        var out = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            let sy = min(mh - 1, y * mh / h)
            for x in 0..<w {
                let sx = min(mw - 1, x * mw / w)
                let v: UInt8
                if fmt == kCVPixelFormatType_OneComponent32Float {
                    v = UInt8(max(0, min(255, base.advanced(by: sy * row).assumingMemoryBound(to: Float.self)[sx].rounded())))
                } else {
                    v = base.advanced(by: sy * row).assumingMemoryBound(to: UInt8.self)[sx]
                }
                out[y * w + x] = v
            }
        }
        return out
    }

    private static func instances(_ obs: VNInstanceMaskObservation?, w: Int, h: Int) -> [[Bool]] {
        guard let obs else { return [] }
        let map = labelMap(obs.instanceMask, w: w, h: h)
        guard !map.isEmpty else { return [] }
        return obs.allInstances.sorted().map { id in map.map { Int($0) == id } }
    }

    // MARK: segmenter

    private static let segSize = 448

    /// The Core ML segmenter and its class names (read from the model's own metadata).
    private static let segmenter: (model: MLModel, labels: [String])? = {
        guard let url = Bundle.main.url(forResource: "Segmenter", withExtension: "mlmodelc") else { return nil }
        let cfg = MLModelConfiguration()
        #if targetEnvironment(simulator)
        cfg.computeUnits = .cpuOnly                     // the simulator's GPU path returns zeros
        #endif
        guard let m = try? MLModel(contentsOf: url, configuration: cfg) else { return nil }
        let meta = m.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
        guard let json = meta?["com.apple.coreml.model.preview.params"]?.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let labels = obj["labels"] as? [String] else { return nil }
        return (m, labels)
    }()

    /// COCO panoptic: label ids 1...90 are countable things, later ids are background stuff.
    private static func isThing(_ id: Int) -> Bool { id >= 1 && id <= 90 }

    /// Desktop naming: "sky (other)" -> "sky", "tree" -> "trees", "river" -> "water" ...
    private static let rename: [String: String] = [
        "sky": "sky", "water": "water", "river": "water", "tree": "trees", "rock": "rocks", "dirt": "ground",
        "potted plant": "plant", "floor": "floor", "wall": "wall", "window": "window", "food": "food",
        "building": "building", "playingfield": "playing field", "tv": "TV",
    ]

    static func displayName(_ raw: String) -> String {
        var n = raw
        if let p = n.range(of: " (") { n = String(n[..<p.lowerBound]) }
        return rename[n] ?? n
    }

    /// Class id per pixel on a 448 x 448 grid (the whole frame, squashed), or nil if unavailable.
    private static func semanticMap(_ img: RGBImage) -> [Int32]? {
        guard let seg = segmenter,
              let pb = try? ImagePrep.pixelBuffer(ImagePrep.resizeArea(img, width: segSize, height: segSize)),
              let out = try? seg.model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pb)])),
              let arr = out.featureValue(for: "semanticPredictions")?.multiArrayValue,
              arr.count == segSize * segSize else { return nil }
        let p = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        return Array(UnsafeBufferPointer(start: p, count: arr.count))
    }

    /// 8-connected components of `cls` in the label grid; returns one id map (0 = none) and the count.
    private static func components(_ map: [Int32], cls: Int32, n: Int) -> ([Int32], Int) {
        var comp = [Int32](repeating: 0, count: n * n)
        var next: Int32 = 0
        var stack: [Int] = []
        for start in 0..<(n * n) where map[start] == cls && comp[start] == 0 {
            next += 1
            comp[start] = next
            stack.append(start)
            while let i = stack.popLast() {
                let x = i % n, y = i / n
                for dy in -1...1 {
                    for dx in -1...1 where dx != 0 || dy != 0 {
                        let xx = x + dx, yy = y + dy
                        guard xx >= 0, yy >= 0, xx < n, yy < n else { continue }
                        let j = yy * n + xx
                        if map[j] == cls && comp[j] == 0 { comp[j] = next; stack.append(j) }
                    }
                }
            }
        }
        return (comp, Int(next))
    }

    /// Upsample a 448-grid selection to a std-resolution mask.
    private static func upsample(_ sel: (Int) -> Bool, w: Int, h: Int) -> [Bool] {
        let n = segSize
        var m = [Bool](repeating: false, count: w * h)
        for y in 0..<h {
            let row = min(n - 1, y * n / h) * n
            for x in 0..<w where sel(row + min(n - 1, x * n / w)) { m[y * w + x] = true }
        }
        return m
    }

    static func detect(_ img: RGBImage) -> [Segment] {
        let w = img.width, h = img.height, n = segSize
        guard let map = semanticMap(img) else { return [] }
        let minPix = Int(0.002 * Double(n * n))

        // Vision person instances (device only) split a crowd into individuals
        var personMasks: [[Bool]] = []
        if let cg = Imaging.cgImage(img) {
            let people = VNGeneratePersonInstanceMaskRequest().cpuOnSimulator()
            try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([people])
            personMasks = instances(people.results?.first, w: w, h: h).filter { m in
                let a = Double(m.filter { $0 }.count) / Double(max(1, m.count))
                return a >= 0.002 && a <= 0.85
            }
        }

        var counts = [Int32: Int]()
        for c in map { counts[c, default: 0] += 1 }
        var segs: [Segment] = []
        var stuff: [String: [Int32]] = [:]           // fused background classes by display name
        for (cls, cnt) in counts where cls > 0 && Int(cls) < (segmenter?.labels.count ?? 0) {
            let raw = segmenter!.labels[Int(cls)]
            guard raw != "--" else { continue }
            if isThing(Int(cls)) {
                if raw == "person" && !personMasks.isEmpty {
                    for m in personMasks { if let s = describe(m, w: w, h: h, label: "person", score: 1, thing: true) { segs.append(s) } }
                    continue
                }
                guard cnt >= minPix else { continue }
                let (comp, k) = components(map, cls: cls, n: n)
                var sizes = [Int](repeating: 0, count: k + 1)
                for c in comp where c > 0 { sizes[Int(c)] += 1 }
                for id in 1...max(1, k) where k > 0 && sizes[id] >= minPix {
                    let m = upsample({ comp[$0] == Int32(id) }, w: w, h: h)
                    if let s = describe(m, w: w, h: h, label: displayName(raw), score: 1, thing: true) { segs.append(s) }
                }
            } else {
                stuff[displayName(raw), default: []].append(cls)
            }
        }
        for (name, classes) in stuff {
            // fuse the classes, then drop specks (the desktop overlay ignores contours under 0.04% of the frame)
            let first = classes[0]
            let fused = map.map { classes.contains($0) ? first : -1 }
            let (comp, k) = components(fused, cls: first, n: n)
            var sizes = [Int](repeating: 0, count: k + 1)
            for c in comp where c > 0 { sizes[Int(c)] += 1 }
            let keep = sizes.map { $0 >= Int(0.0004 * Double(n * n)) }
            let m = upsample({ comp[$0] > 0 && keep[Int(comp[$0])] }, w: w, h: h)
            if let s = describe(m, w: w, h: h, label: name, score: 1, thing: false), s.area >= 0.0008 { segs.append(s) }
        }
        // a region covering nearly the whole frame is not something to analyse
        return segs.filter { $0.area <= 0.95 }.sorted { $0.area > $1.area }
    }

    // MARK: - geometry of one mask

    static func describe(_ m: [Bool], w: Int, h: Int, label: String, score: Double, thing: Bool = true) -> Segment? {
        var n = 0, sx = 0.0, sy = 0.0
        var x0 = w, x1 = 0, y0 = h, y1 = 0
        for y in 0..<h {
            for x in 0..<w where m[y * w + x] {
                n += 1; sx += Double(x); sy += Double(y)
                x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
            }
        }
        guard n > 0 else { return nil }
        let bw = max(2, Int((0.012 * Double(w)).rounded())), bh = max(2, Int((0.012 * Double(h)).rounded()))
        var left = 0, right = 0, top = 0, bottom = 0
        for y in 0..<h {
            if (0..<bw).contains(where: { m[y * w + $0] }) { left += 1 }
            if ((w - bw)..<w).contains(where: { m[y * w + $0] }) { right += 1 }
        }
        for x in 0..<w {
            if (0..<bh).contains(where: { m[$0 * w + x] }) { top += 1 }
            if ((h - bh)..<h).contains(where: { m[$0 * w + x] }) { bottom += 1 }
        }
        var s = Segment(label: label, score: score, mask: m, w: w, h: h, area: Double(n) / Double(w * h),
                        cx: sx / Double(n) / Double(w), cy: sy / Double(n) / Double(h),
                        box: (Double(x0) / Double(w), Double(y0) / Double(h), Double(x1 + 1) / Double(w), Double(y1 + 1) / Double(h)),
                        edges: ["left": Double(left) / Double(h), "right": Double(right) / Double(h),
                                "top": Double(top) / Double(w), "bottom": Double(bottom) / Double(w)])
        s.labelPoint = deepestPoint(m, w: w, h: h)
        s.thing = thing
        return s
    }

    /// "lower left", "centre", "right edge" …
    static func whereText(_ g: Segment) -> String {
        let e = g.edges
        let bw = g.box.2 - g.box.0, bh = g.box.3 - g.box.1
        if g.area < 0.25 {
            for side in ["left", "right", "top", "bottom"] {
                let horizontalSide = side == "left" || side == "right"
                if (e[side] ?? 0) > 0.02 && (bw < 0.5 || !horizontalSide) && (bh < 0.5 || horizontalSide) {
                    return "\(side) edge"
                }
            }
        }
        let v = g.cy < 0.38 ? "upper" : (g.cy > 0.62 ? "lower" : "")
        let hz = g.cx < 0.38 ? "left" : (g.cx > 0.62 ? "right" : "")
        let s = "\(v) \(hz)".trimmingCharacters(in: .whitespaces)
        return s.isEmpty ? "centre" : s
    }

    /// Frame edges that slice through the object.
    /// Background classes that can still be a subject (a cabin, a bridge) rather than plain surroundings.
    static let stuffSubjects: Set<String> = ["house", "bridge", "tent", "flower", "tower", "building", "lighthouse"]
    static let waterNames: Set<String> = ["water", "sea"]

    static func cutOff(_ g: Segment) -> [String] {
        guard g.thing || stuffSubjects.contains(g.label) else { return [] }
        let bw = g.box.2 - g.box.0, bh = g.box.3 - g.box.1
        var out: [String] = []
        for (side, span) in [("left", bh), ("right", bh), ("top", bw), ("bottom", bw)] {
            let extent = g.edges[side] ?? 0
            if extent >= 0.25 * max(span, 1e-6) && extent >= 0.03 { out.append(side) }
        }
        return out
    }

    private static let prior: [String: Double] = [
        "person": 1.6, "bird": 1.4, "cat": 1.4, "dog": 1.4, "horse": 1.4, "sheep": 1.3, "cow": 1.3, "elephant": 1.4,
        "bear": 1.4, "zebra": 1.4, "giraffe": 1.4, "boat": 1.2, "house": 1.1, "bridge": 1.1, "tent": 1.1,
        "flower": 1.1, "airplane": 1.2, "train": 1.2, "bicycle": 1.1, "motorcycle": 1.1, "car": 1.0,
        "building": 1.1, "lighthouse": 1.2, "canoe": 1.2, "kayak": 1.2,
        "dining table": 0.45, "table": 0.45, "bed": 0.5, "sofa": 0.55, "couch": 0.55, "chair": 0.6, "bench": 0.6, "TV": 0.7,
    ]

    /// Index of the main subject, or nil for a scene with no single object.
    static func pickSubject(_ segs: [Segment], salXY: (Double, Double)) -> Int? {
        var best: Int?, bestS = 0.0
        for (i, g) in segs.enumerated() {
            guard g.thing || stuffSubjects.contains(g.label), g.area >= 0.008, g.area <= 0.7 else { continue }
            let central = 1 - min(1, hypot(g.cx - 0.5, g.cy - 0.5) / 0.6)
            var s = g.area.squareRoot() * (0.45 + central) * (prior[g.label] ?? (g.thing ? 0.9 : 0.8))
            if g.box.0 <= salXY.0 && salXY.0 <= g.box.2 && g.box.1 <= salXY.1 && salXY.1 <= g.box.3 { s *= 1.3 }
            s *= 1 + 2 * max(0, g.lean)
            if s > bestS { best = i; bestS = s }
        }
        return bestS >= 0.06 ? best : nil
    }

    static func assignRoles(_ segs: inout [Segment], subject: Int?) {
        for i in segs.indices {
            let d = segs[i].delta, lean = segs[i].lean
            if i == subject { segs[i].role = "subject" }
            // landmarks (a cabin, a bridge) are never "distractions": you cannot step around them
            else if segs[i].area <= 0.12 && d >= 3 && lean <= -0.10 && !stuffSubjects.contains(segs[i].label) { segs[i].role = "distraction" }
            else if lean >= 0.30 { segs[i].role = "helps" }
            else { segs[i].role = "other" }
        }
    }

    private static func thirdsDist(_ x: Double, _ y: Double) -> Double {
        let t: [Double] = [1.0 / 3, 2.0 / 3]
        var best = Double.infinity
        for px in t { for py in t { best = min(best, hypot(x - px, y - py)) } }
        return best
    }

    static func geometry(_ img: RGBImage, segs: [Segment], subject: Int?, detail: Plane) -> SceneGeometry {
        let w = img.width, h = img.height
        var out = SceneGeometry(subject: nil, subjectIndex: subject, nThings: segs.filter(\.thing).count)
        for g in segs {
            if g.label == "sky" { out.sky += g.area }
            if waterNames.contains(g.label) { out.water += g.area }
        }
        if let sky = segs.first(where: { $0.label == "sky" }), sky.area >= 0.04 {
            var lows: [Float] = []
            for x in stride(from: 0, to: w, by: 2) {
                var low = -1
                for y in 0..<h where sky.mask[y * w + x] { low = y }
                if low >= 0 { lows.append(Float(low)) }
            }
            if Double(lows.count) >= 0.6 * Double((w + 1) / 2) { out.horizonY = Imaging.median(lows) / Double(h) }
        }
        guard let si = subject else { return out }
        let g = segs[si]
        // subject vs its surroundings at quarter resolution
        let sw = max(64, w / 4), sh = max(64, h / 4)
        let small = ImagePrep.resizeArea(img, width: sw, height: sh)
        var ms = [Bool](repeating: false, count: sw * sh)
        for y in 0..<sh { for x in 0..<sw { ms[y * sw + x] = g.mask[min(h - 1, y * h / sh) * w + min(w - 1, x * w / sw)] } }
        var sep: Double?
        let cnt = ms.filter { $0 }.count
        if cnt >= 12 {
            let r = Float(max(3, (0.35 * Double(cnt).squareRoot()).rounded()))
            let dist = distance(toTrue: ms, w: sw, h: sh)
            var inS = SIMD3<Double>(0, 0, 0), inR = SIMD3<Double>(0, 0, 0), nS = 0.0, nR = 0.0
            for i in 0..<(sw * sh) {
                let (L, a, b) = Imaging.lab(small.data[i * 3], small.data[i * 3 + 1], small.data[i * 3 + 2])
                let v = SIMD3<Double>(Double(L), Double(a), Double(b))
                if ms[i] { inS += v; nS += 1 } else if dist[i] <= r { inR += v; nR += 1 }
            }
            if nR >= 12 { sep = simd_length(inS / nS - inR / nR) }
        }
        let d = Imaging.resize(detail, w: w, h: h)
        var inside: [Float] = [], outside: [Float] = []
        for i in stride(from: 0, to: w * h, by: 7) { if g.mask[i] { inside.append(d.d[i]) } else { outside.append(d.d[i]) } }
        let sharp = inside.isEmpty || outside.isEmpty ? nil : Imaging.median(inside) / (Imaging.median(outside) + 1e-6)
        out.subject = SubjectInfo(label: g.label, area: g.area, cx: g.cx, cy: g.cy, whereText: whereText(g),
                                  thirdsDist: thirdsDist(g.cx, g.cy), centerDist: hypot(g.cx - 0.5, g.cy - 0.5),
                                  cut: cutOff(g), separation: sep, sharpRatio: sharp)
        return out
    }

    // MARK: - distance transform (chamfer 3-4, in pixels)

    /// Distance from every pixel to the nearest `true` pixel.
    static func distance(toTrue m: [Bool], w: Int, h: Int) -> [Float] {
        let inf: Float = 1e9
        var d = m.map { $0 ? 0 : inf }
        let a: Float = 1, b: Float = 1.4142
        for y in 0..<h {
            for x in 0..<w {
                var v = d[y * w + x]
                if x > 0 { v = min(v, d[y * w + x - 1] + a) }
                if y > 0 {
                    v = min(v, d[(y - 1) * w + x] + a)
                    if x > 0 { v = min(v, d[(y - 1) * w + x - 1] + b) }
                    if x < w - 1 { v = min(v, d[(y - 1) * w + x + 1] + b) }
                }
                d[y * w + x] = v
            }
        }
        for y in stride(from: h - 1, through: 0, by: -1) {
            for x in stride(from: w - 1, through: 0, by: -1) {
                var v = d[y * w + x]
                if x < w - 1 { v = min(v, d[y * w + x + 1] + a) }
                if y < h - 1 {
                    v = min(v, d[(y + 1) * w + x] + a)
                    if x < w - 1 { v = min(v, d[(y + 1) * w + x + 1] + b) }
                    if x > 0 { v = min(v, d[(y + 1) * w + x - 1] + b) }
                }
                d[y * w + x] = v
            }
        }
        return d
    }

    /// Point deepest inside the mask (normalised), so a number badge sits on the object.
    static func deepestPoint(_ m: [Bool], w: Int, h: Int) -> (Double, Double) {
        let s = 4
        let sw = max(1, w / s), sh = max(1, h / s)
        var small = [Bool](repeating: false, count: sw * sh)
        for y in 0..<sh { for x in 0..<sw { small[y * sw + x] = m[min(h - 1, y * s) * w + min(w - 1, x * s)] } }
        // distance to the nearest outside pixel, with the frame border counting as outside
        var outside = small.map { !$0 }
        for x in 0..<sw { outside[x] = true; outside[(sh - 1) * sw + x] = true }
        for y in 0..<sh { outside[y * sw] = true; outside[y * sw + sw - 1] = true }
        let d = distance(toTrue: outside, w: sw, h: sh)
        var best = 0
        for i in d.indices where small[i] && (d[i] > d[best] || !small[best]) { best = i }
        return ((Double(best % sw) + 0.5) / Double(sw), (Double(best / sw) + 0.5) / Double(sh))
    }

    // MARK: - painting an object out

    /// Push-pull hole filling: average known pixels down a pyramid, then fill holes from coarser levels.
    static func inpaint(_ img: RGBImage, hole: [Bool]) -> RGBImage {
        struct Level { var w: Int; var h: Int; var c: [SIMD4<Float>] }      // rgb premultiplied by weight, weight in .w
        var levels: [Level] = []
        var base = [SIMD4<Float>](repeating: .zero, count: img.width * img.height)
        for i in 0..<(img.width * img.height) where !hole[i] {
            base[i] = SIMD4(Float(img.data[i * 3]), Float(img.data[i * 3 + 1]), Float(img.data[i * 3 + 2]), 1)
        }
        levels.append(Level(w: img.width, h: img.height, c: base))
        while levels.last!.w > 1 || levels.last!.h > 1 {
            let p = levels.last!
            let w = max(1, (p.w + 1) / 2), h = max(1, (p.h + 1) / 2)
            var c = [SIMD4<Float>](repeating: .zero, count: w * h)
            for y in 0..<p.h { for x in 0..<p.w { c[(y / 2) * w + x / 2] += p.c[y * p.w + x] } }
            levels.append(Level(w: w, h: h, c: c))
        }
        // push: resolve the coarsest level, then each finer level from its parent
        var parent = levels.last!.c.map { $0.w > 0 ? $0 / $0.w : SIMD4<Float>(128, 128, 128, 1) }
        for li in stride(from: levels.count - 2, through: 0, by: -1) {
            let L = levels[li], P = levels[li + 1]
            var cur = [SIMD4<Float>](repeating: .zero, count: L.w * L.h)
            for y in 0..<L.h {
                for x in 0..<L.w {
                    // bilinear sample of the parent at this pixel's centre
                    let fx = max(0, min(Float(P.w - 1), (Float(x) + 0.5) / 2 - 0.5))
                    let fy = max(0, min(Float(P.h - 1), (Float(y) + 0.5) / 2 - 0.5))
                    let x0 = Int(fx), y0 = Int(fy), x1 = min(P.w - 1, x0 + 1), y1 = min(P.h - 1, y0 + 1)
                    let tx = fx - Float(x0), ty = fy - Float(y0)
                    let up = (parent[y0 * P.w + x0] * (1 - tx) + parent[y0 * P.w + x1] * tx) * (1 - ty)
                        + (parent[y1 * P.w + x0] * (1 - tx) + parent[y1 * P.w + x1] * tx) * ty
                    let v = L.c[y * L.w + x]
                    let a = min(1, v.w)
                    let own = v.w > 0 ? v / v.w : up
                    cur[y * L.w + x] = own * a + up * (1 - a)
                }
            }
            parent = cur
        }
        var out = img.data
        for i in 0..<(img.width * img.height) where hole[i] {
            out[i * 3] = UInt8(max(0, min(255, parent[i].x.rounded())))
            out[i * 3 + 1] = UInt8(max(0, min(255, parent[i].y.rounded())))
            out[i * 3 + 2] = UInt8(max(0, min(255, parent[i].z.rounded())))
        }
        return RGBImage(width: img.width, height: img.height, data: out)
    }

    /// The photo without this region.  Compact objects are painted out from their surroundings;
    /// large regions cannot be invented, so they are replaced by a heavy blur of themselves.
    static func remove(_ img: RGBImage, _ g: Segment) -> RGBImage {
        let w = img.width, h = img.height
        let mx = Double(max(w, h))
        var fill: RGBImage
        var alpha = Plane(w: w, h: h)
        if g.area <= 0.12 {
            let s = 384 / mx
            let sw = max(1, Int((Double(w) * s).rounded())), sh = max(1, Int((Double(h) * s).rounded()))
            let sm = ImagePrep.resizeArea(img, width: sw, height: sh)
            var hole = [Bool](repeating: false, count: sw * sh)
            for y in 0..<sh { for x in 0..<sw { hole[y * sw + x] = g.mask[min(h - 1, y * h / sh) * w + min(w - 1, x * w / sw)] } }
            let dist = distance(toTrue: hole, w: sw, h: sh)
            let r = Float(max(3, Int((0.012 * 384).rounded())))
            hole = dist.map { $0 <= r }
            fill = Imaging.resizeBilinear(inpaint(sm, hole: hole), w: w, h: h)
            let a = Plane(w: sw, h: sh, d: hole.map { $0 ? 1 : 0 })
            alpha = Imaging.gaussianBlur(Imaging.resize(a, w: w, h: h), sigma: 0.006 * mx)
        } else {
            let q = ImagePrep.resizeArea(img, width: max(8, w / 6), height: max(8, h / 6))
            fill = Imaging.resizeBilinear(Imaging.blurRGB(q, sigma: 0.01 * mx), w: w, h: h)
            alpha = Imaging.gaussianBlur(Plane(w: w, h: h, d: g.mask.map { $0 ? 1 : 0 }), sigma: 0.008 * mx)
        }
        var out = img.data
        for i in 0..<(w * h) {
            let a = max(0, min(1, alpha.d[i]))
            guard a > 0.002 else { continue }
            for k in 0..<3 {
                out[i * 3 + k] = UInt8(max(0, min(255, (Float(img.data[i * 3 + k]) * (1 - a) + Float(fill.data[i * 3 + k]) * a).rounded())))
            }
        }
        return RGBImage(width: w, height: h, data: out)
    }
}

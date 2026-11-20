import CoreML
import UIKit

/// One analysed photo: raw attribute scores from the shared tower and where they sit among reference photos.
struct ScoredPhoto {
    let name: String
    let image: UIImage               // the standard image (orientation applied), used for display and overlays
    let std: RGBImage
    let pixelSize: CGSize            // original file size
    let raw: [Double]                // one per head
    let percentile: [Double]         // 0...100
    let z: [Double]                  // standard deviations from the reference mean
    let m: Measurement
    let exif: ExifInfo
    let g: [Float]                   // whole-frame embedding
    let l: [[Float]]                 // four detail-crop embeddings
    let overall: Double

    var signals: [String: Double] { m.s }
    var globalEmbedHead: [Double] { g.prefix(8).map(Double.init) }
    var aspect: Double { Double(std.width) / Double(std.height) }
}

enum Lens: String, CaseIterable, Identifiable {
    case photo, clipping, focus, composition, objects, attention, framing
    var id: String { rawValue }

    var title: String {
        switch self {
        case .photo: return "Photo"
        case .clipping: return "Clipping"
        case .focus: return "Focus"
        case .composition: return "Composition"
        case .objects: return "Objects"
        case .attention: return "AI attention"
        case .framing: return "Framing"
        }
    }

    var symbol: String {
        switch self {
        case .photo: return "photo"
        case .clipping: return "sun.max.trianglebadge.exclamationmark"
        case .focus: return "scope"
        case .composition: return "squareshape.split.3x3"
        case .objects: return "viewfinder.rectangular"
        case .attention: return "eye"
        case .framing: return "crop"
        }
    }

    var legend: String {
        switch self {
        case .photo: return "The photo as analysed."
        case .clipping: return "Red: blown to pure white. Blue: crushed to pure black. Neither holds any detail."
        case .focus: return "Brighter = crisper fine detail. Both photos use the same scale, so they compare directly."
        case .composition: return "Thirds grid. Yellow ring: strongest visual pull. Cyan cross: centre of visual weight. Line: detected horizon."
        case .objects: return "Yellow: the subject. Red: the score rises with it painted out. Green: the score leans on it."
        case .attention: return "The network's view. Green regions carry the score more than the rest; red ones less, or they hurt it."
        case .framing: return "Same scene matched point-to-point. The box shows where the other photo's frame falls in this one."
        }
    }

    /// Coaching items name overlays the desktop way.
    init?(overlay: String?) {
        switch overlay {
        case "clip": self = .clipping
        case "focus": self = .focus
        case "comp": self = .composition
        case "objects": self = .objects
        case "impact": self = .attention
        case "frame": self = .framing
        default: return nil
        }
    }
}

struct PhotoExtras {
    var authenticity: Authenticity?
    var objects: [Segment] = []
    var geometry: SceneGeometry?
    var layers: [Lens: UIImage] = [:]
    var otherFrame: Quad?            // the other photo's frame drawn inside this one
}

struct AttributeRow: Identifiable {
    let id: String
    let label: String
    let blurb: String
    let group: String
    let reliable: Bool
    let a: Double
    let b: Double
}

struct CoachItem: Identifiable {
    let id: String
    let title: String
    let heads: [String]
    let gap: Double
    let priority: Double
    let evidence: [String]
    let shoot: [String]
    let edit: [String]
    let overlay: String?
    let fix: String
}

struct Reason: Identifiable {
    let head: String
    let label: String
    let gap: Double
    let text: String
    let winnerPct: Double
    let loserPct: Double
    var id: String { head }
}

struct Preview: Identifiable {
    let kind: String                 // "edit" or "crop_edit"
    let image: UIImage
    let overall: Double
    let aesthetic: Double
    let technical: Double
    let closure: Double              // share of the score gap this edit closes
    var id: String { kind }
    var title: String { kind == "edit" ? "Tone & colour" : "Crop + tone" }
}

struct Comparison {
    let a: ScoredPhoto
    let b: ScoredPhoto
    let extras: [PhotoExtras]        // [a, b]
    /// Model's estimate of how often viewers would prefer A.
    let pA: Double
    let winnerIsA: Bool
    let headline: String
    let summary: String
    let closeCall: Bool
    let reasons: [Reason]
    let loserLeads: [Reason]
    let coaching: [CoachItem]
    let attributes: [AttributeRow]
    let relation: String             // aligned / similar / different
    let relationText: String
    let similarity: Double
    let frame: Framing?
    let recipe: LookRecipe?
    let previews: [Preview]
    let seconds: Double

    var winner: ScoredPhoto { winnerIsA ? a : b }
    var loser: ScoredPhoto { winnerIsA ? b : a }
    var winnerIndex: Int { winnerIsA ? 0 : 1 }
    var loserIndex: Int { winnerIsA ? 1 : 0 }
    var confidence: Double { max(pA, 1 - pA) }
    func photo(_ i: Int) -> ScoredPhoto { i == 0 ? a : b }
    var bestPreview: Preview? { previews.max { $0.overall < $1.overall } }
}

/// The free tier's result: a winner and nothing else.
struct QuickVerdict {
    let images: [UIImage]
    let aspects: [Double]
    let pA: Double
    let seconds: Double
    var authenticity: [Authenticity] = []

    var winnerIndex: Int { pA >= 0.5 ? 0 : 1 }
    var confidence: Double { max(pA, 1 - pA) }
    /// A word, not a number: the free tier shows no metrics.
    var strength: (title: String, line: String) {
        switch confidence {
        case ..<0.56: return ("Close call", "It's tight - most people would struggle to pick, but this one edges it.")
        case ..<0.70: return ("Solid edge", "Most viewers would pick this one.")
        default: return ("Clear winner", "Viewers would pick this one by a wide margin.")
        }
    }
}

enum Stage: Int, CaseIterable, Identifiable {
    case reading, measuring, scoring, authenticity, matching, objects, testing, attention, editing, coaching
    var id: Int { rawValue }

    var title: String {
        switch self {
        case .reading: return "Reading both photos"
        case .measuring: return "Measuring light, colour and focus"
        case .scoring: return "Scoring with the twin network"
        case .authenticity: return "Checking for AI generation"
        case .matching: return "Matching the two scenes"
        case .objects: return "Finding what's in each frame"
        case .testing: return "Painting out each object to test it"
        case .attention: return "Mapping what the network looks at"
        case .editing: return "Building an edit toward the winner"
        case .coaching: return "Writing your coaching"
        }
    }

    var symbol: String {
        switch self {
        case .reading: return "photo.on.rectangle"
        case .measuring: return "chart.bar.xaxis"
        case .scoring: return "brain"
        case .authenticity: return "sparkle.magnifyingglass"
        case .matching: return "square.on.square.dashed"
        case .objects: return "viewfinder"
        case .testing: return "wand.and.rays"
        case .attention: return "eye"
        case .editing: return "slider.horizontal.3"
        case .coaching: return "text.bubble"
        }
    }
}

/// The twin network on device.  Both photos go through the same encoder and the same tower;
/// the verdict is a function of the score difference, so swapping the photos flips it exactly.
actor PhotoJudge {
    static let shared = PhotoJudge()

    /// Global-embedding cosine similarity above which two photos count as the same kind of scene.
    static let similarScene = 0.80

    private var info: JudgeInfo?
    private var encoder: MLModel?
    private var tower: MLModel?
    private var detector: MLModel?

    func loadIfNeeded() async throws -> JudgeInfo {
        if let info { return info }
        let i = try JudgeInfo.load()
        #if targetEnvironment(simulator)
        encoder = try await Self.loadModel("Encoder", units: .cpuOnly)   // the simulator's GPU path returns zeros
        #else
        encoder = try await Self.loadModel("Encoder", units: .all)
        #endif
        tower = try await Self.loadModel("Tower", units: .cpuOnly)       // tiny MLP: CPU keeps full float32 precision
        info = i
        return i
    }

    /// Xcode normally compiles .mlpackage into .mlmodelc at build time; if the package was bundled
    /// as a plain folder instead, compile it on first launch.
    private static func loadModel(_ name: String, units: MLComputeUnits) async throws -> MLModel {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return try MLModel(contentsOf: url, configuration: cfg)
        }
        guard let pkg = Bundle.main.url(forResource: name, withExtension: "mlpackage") else {
            throw JudgeError.missingResource("\(name).mlpackage")
        }
        let compiled = try await MLModel.compileModel(at: pkg)
        return try MLModel(contentsOf: compiled, configuration: cfg)
    }

    // MARK: - network calls

    private var encoderOnCPU = false

    /// Run the encoder; if the accelerated path ever hands back all-zero embeddings (seen on the
    /// simulator), switch the encoder to the CPU once and redo the batch rather than score garbage.
    private func embed(_ views: [RGBImage]) throws -> [[Float]] {
        let e = try embedOnce(views)
        let dead = e.allSatisfy { v in v.allSatisfy { $0 == 0 } }
        guard dead, !encoderOnCPU, let url = Bundle.main.url(forResource: "Encoder", withExtension: "mlmodelc") else { return e }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuOnly
        encoder = try MLModel(contentsOf: url, configuration: cfg)
        encoderOnCPU = true
        return try embedOnce(views)
    }

    private func embedOnce(_ views: [RGBImage]) throws -> [[Float]] {
        guard let encoder else { throw JudgeError.model("encoder not loaded") }
        let inputs: [MLFeatureProvider] = try views.map {
            try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: try ImagePrep.pixelBuffer($0))])
        }
        let out = try encoder.predictions(fromBatch: MLArrayBatchProvider(array: inputs))
        return try (0..<out.count).map { i in
            guard let arr = out.features(at: i).featureValue(for: "embedding")?.multiArrayValue else {
                throw JudgeError.model("encoder returned no embedding")
            }
            return (0..<arr.count).map { arr[$0].floatValue }
        }
    }

    private static func array(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        for (i, v) in values.enumerated() { arr[i] = NSNumber(value: v) }
        return arr
    }

    /// The tower only reads the signals it was trained with; the rest are zeroed.
    private func towerSignals(_ s: [String: Double], _ info: JudgeInfo) -> [Float] {
        let used = Set(info.signalsUsed)
        return info.signalKeys.map { used.contains($0) ? Float(s[$0] ?? 0) : 0 }
    }

    private func towerScores(g: [Float], l: [[Float]], h: [Float]) throws -> [Double] {
        guard let tower else { throw JudgeError.model("tower not loaded") }
        let dim = g.count
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "g": MLFeatureValue(multiArray: try Self.array(g, shape: [1, dim])),
            "l": MLFeatureValue(multiArray: try Self.array(Array(l.joined()), shape: [1, 4, dim])),
            "h": MLFeatureValue(multiArray: try Self.array(h, shape: [1, h.count])),
        ])
        let out = try tower.prediction(from: input)
        guard let s = out.featureValue(for: "scores")?.multiArrayValue else { throw JudgeError.model("tower returned no scores") }
        return (0..<s.count).map { s[$0].doubleValue }
    }

    private func overall(_ pct: [Double], _ info: JudgeInfo) -> Double {
        let ia = info.index(of: "aesthetic") ?? 0, it = info.index(of: "technical") ?? 1
        return info.weights.aesthetic * pct[ia] + info.weights.technical * pct[it]
    }

    // MARK: - scoring one photo

    private func featurize(std: RGBImage, m: Measurement, name: String, image: UIImage, pixelSize: CGSize, exif: ExifInfo,
                           info: JudgeInfo) throws -> ScoredPhoto {
        let views = ImagePrep.views(std: std, size: info.inputSize)
        let e = try embed(views)
        let raw = try towerScores(g: e[0], l: Array(e[1...]), h: towerSignals(m.s, info))
        guard raw.count == info.heads.count else { throw JudgeError.model("expected \(info.heads.count) scores, got \(raw.count)") }
        let pct = raw.enumerated().map { info.percentile($1, head: $0) }
        let z = raw.enumerated().map { ($1 - info.heads[$0].refMean) / max(info.heads[$0].refStd, 1e-9) }
        return ScoredPhoto(name: name, image: image, std: std, pixelSize: pixelSize, raw: raw, percentile: pct, z: z, m: m,
                           exif: exif, g: e[0], l: Array(e[1...]), overall: overall(pct, info))
    }

    func score(data: Data, name: String) async throws -> ScoredPhoto {
        let info = try await loadIfNeeded()
        let (std, ui) = try ImagePrep.standardImage(from: data, longSide: info.stdLongSide)
        return try featurize(std: std, m: Signals.measure(std), name: name, image: Imaging.uiImage(std),
                             pixelSize: CGSize(width: ui.size.width * ui.scale, height: ui.size.height * ui.scale),
                             exif: ExifInfo.read(data), info: info)
    }

    /// Score an edited version of a photo (tone/colour signals re-measured, as the desktop app does).
    private func scoreEdited(_ img: RGBImage, info: JudgeInfo) throws -> (pct: [Double], overall: Double) {
        var s = Signals.tone(img)
        s.merge(Signals.color(img)) { $1 }
        s["log_long_side"] = log(Double(max(img.width, img.height)) / 1024.0 + 1e-6)
        s["aspect"] = Double(img.width) / Double(img.height)
        let e = try embed(ImagePrep.views(std: img, size: info.inputSize))
        let raw = try towerScores(g: e[0], l: Array(e[1...]), h: towerSignals(s, info))
        let pct = raw.enumerated().map { info.percentile($1, head: $0) }
        return (pct, overall(pct, info))
    }

    // MARK: - what the score leans on

    /// Slide a window that replaces each region with a heavy blur; + = the score drops without it.
    private func occlusion(_ p: ScoredPhoto, info: JudgeInfo, n: Int = 5, win: Double = 0.36) throws -> Plane {
        let gv = ImagePrep.resizeArea(p.std, width: info.inputSize, height: info.inputSize)
        let H = gv.height, W = gv.width
        let blur = Imaging.blurRGB(gv, sigma: 0.09 * Double(max(H, W)))
        let wh = max(14, Int((win * Double(H)).rounded())), ww = max(14, Int((win * Double(W)).rounded()))
        let ys = (0..<n).map { Int((Double(H - wh) * Double($0) / Double(n - 1)).rounded()) }
        let xs = (0..<n).map { Int((Double(W - ww) * Double($0) / Double(n - 1)).rounded()) }
        var batch: [RGBImage] = [], boxes: [(Int, Int)] = []
        for y in ys {
            for x in xs {
                var im = gv
                for yy in y..<(y + wh) {
                    for xx in x..<(x + ww) {
                        let o = (yy * W + xx) * 3
                        im.data[o] = blur.data[o]; im.data[o + 1] = blur.data[o + 1]; im.data[o + 2] = blur.data[o + 2]
                    }
                }
                batch.append(im); boxes.append((y, x))
            }
        }
        let g = try embed(batch)
        let hv = towerSignals(p.m.s, info)
        let ia = info.index(of: "aesthetic") ?? 0
        var heat = Plane(w: W, h: H), cnt = Plane(w: W, h: H)
        for (k, (y, x)) in boxes.enumerated() {
            let s = try towerScores(g: g[k], l: p.l, h: hv)[ia]
            let d = Float((p.raw[ia] - s) / info.heads[ia].refStd)
            for yy in y..<(y + wh) { for xx in x..<(x + ww) { heat[xx, yy] += d; cnt[xx, yy] += 1 } }
        }
        for i in heat.d.indices { heat.d[i] /= max(cnt.d[i], 1) }
        return Imaging.gaussianBlur(heat, sigma: 0.04 * Double(max(H, W)))
    }

    /// Re-score the photo with each object painted out.  delta = change in the aesthetic percentile
    /// (+ = it scores higher without it, i.e. it distracts); lean = how much the score depends on it.
    private func objectEffects(_ p: ScoredPhoto, info: JudgeInfo, maxObjects: Int = 10) throws -> [Segment] {
        // compact things and sizeable background regions, things first (objects.py / Engine.object_effects)
        var segs = Objects.detect(p.std).filter { $0.thing ? $0.area >= 0.002 : $0.area >= 0.015 }
        segs = Array(segs.sorted { $0.area * ($0.thing ? 3 : 1) > $1.area * ($1.thing ? 3 : 1) }.prefix(maxObjects))
        segs.sort { $0.area > $1.area }
        let ia = info.index(of: "aesthetic") ?? 0
        let hv = towerSignals(p.m.s, info)
        for i in segs.indices {
            let mod = Objects.remove(p.std, segs[i])
            let e = try embed(ImagePrep.views(std: mod, size: info.inputSize))
            let r = try towerScores(g: e[0], l: Array(e[1...]), h: hv)
            segs[i].delta = info.percentile(r[ia], head: ia) - p.percentile[ia]
            segs[i].lean = (p.raw[ia] - r[ia]) / info.heads[ia].refStd
        }
        for i in segs.indices { segs[i].n = i + 1 }
        return segs
    }

    // MARK: - AI-generation check

    /// Labels in the file first; the detector network for everything (its reading is shown either way).
    /// Both shots' AI checks, or `known` when reopening a saved comparison: the saved copies are re-encoded
    /// and carry no C2PA/XMP labels, so checking them again would be less reliable than the original result.
    private func authenticity(_ dA: Data, _ uA: UIImage, _ dB: Data, _ uB: UIImage, known: [Authenticity]?) async -> [Authenticity] {
        if let known, known.count == 2 { return known }
        return [await authenticity(data: dA, image: uA), await authenticity(data: dB, image: uB)]
    }

    func authenticity(data: Data, image: UIImage) async -> Authenticity {
        let label = Provenance.read(data)
        let camera = ExifInfo.read(data).camera
        var p: Double?
        do {
            if detector == nil {
                #if targetEnvironment(simulator)
                detector = try await Self.loadModel("AIDetector", units: .cpuOnly)
                #else
                detector = try await Self.loadModel("AIDetector", units: .all)
                #endif
            }
            if let detector {
                let x = try DetectorInput.make(image)
                let out = try await detector.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: x)]))
                p = out.featureValue(for: "ai")?.multiArrayValue.map { $0[0].doubleValue }
            }
        } catch {
            p = nil
        }
        return Authenticity.decide(label: label, p: p, camera: camera)
    }

    // MARK: - the free tier: verdict only

    /// Just the twin network's call: which shot wins and how decisively.  Skips scene matching, objects,
    /// attention maps, edits and coaching, so it's a fraction of the full pipeline's time.
    func quickVerdict(dataA: Data, dataB: Data, known: [Authenticity]? = nil, progress: @escaping @Sendable (Stage) -> Void = { _ in }) async throws -> QuickVerdict {
        let t0 = Date()
        progress(.reading)
        let info = try await loadIfNeeded()
        let (stdA, uiA) = try ImagePrep.standardImage(from: dataA, longSide: info.stdLongSide)
        let (stdB, uiB) = try ImagePrep.standardImage(from: dataB, longSide: info.stdLongSide)
        try Task.checkCancellation()
        progress(.measuring)
        let mA = Signals.measure(stdA), mB = Signals.measure(stdB)
        try Task.checkCancellation()
        progress(.scoring)
        func size(_ u: UIImage) -> CGSize { CGSize(width: u.size.width * u.scale, height: u.size.height * u.scale) }
        let a = try featurize(std: stdA, m: mA, name: "Shot A", image: Imaging.uiImage(stdA), pixelSize: size(uiA), exif: ExifInfo(), info: info)
        let b = try featurize(std: stdB, m: mB, name: "Shot B", image: Imaging.uiImage(stdB), pixelSize: size(uiB), exif: ExifInfo(), info: info)
        let ia = info.index(of: "aesthetic") ?? 0, it = info.index(of: "technical") ?? 1
        func logit(_ i: Int) -> Double { (a.raw[i] - b.raw[i]) * info.heads[i].scale }
        let pA = 1 / (1 + exp(-(info.weights.aesthetic * logit(ia) + info.weights.technical * logit(it))))
        progress(.authenticity)
        let au = await authenticity(dataA, uiA, dataB, uiB, known: known)
        return QuickVerdict(images: [a.image, b.image], aspects: [a.aspect, b.aspect], pA: pA, seconds: Date().timeIntervalSince(t0),
                            authenticity: au)
    }

    // MARK: - the twin comparison

    func compare(dataA: Data, dataB: Data, nameA: String, nameB: String, known: [Authenticity]? = nil,
                 progress: @escaping @Sendable (Stage) -> Void = { _ in }) async throws -> Comparison {
        let t0 = Date()
        progress(.reading)
        let info = try await loadIfNeeded()
        let (stdA, uiA) = try ImagePrep.standardImage(from: dataA, longSide: info.stdLongSide)
        let (stdB, uiB) = try ImagePrep.standardImage(from: dataB, longSide: info.stdLongSide)
        let exA = ExifInfo.read(dataA), exB = ExifInfo.read(dataB)

        progress(.measuring)
        let mA = Signals.measure(stdA), mB = Signals.measure(stdB)

        progress(.scoring)
        func size(_ u: UIImage) -> CGSize { CGSize(width: u.size.width * u.scale, height: u.size.height * u.scale) }
        let a = try featurize(std: stdA, m: mA, name: nameA, image: Imaging.uiImage(stdA), pixelSize: size(uiA), exif: exA, info: info)
        let b = try featurize(std: stdB, m: mB, name: nameB, image: Imaging.uiImage(stdB), pixelSize: size(uiB), exif: exB, info: info)
        let P = [a, b]

        // verdict
        let ia = info.index(of: "aesthetic") ?? 0, it = info.index(of: "technical") ?? 1
        func logit(_ i: Int) -> Double { (a.raw[i] - b.raw[i]) * info.heads[i].scale }
        let pA = 1 / (1 + exp(-(info.weights.aesthetic * logit(ia) + info.weights.technical * logit(it))))
        let W = pA >= 0.5 ? 0 : 1, L = 1 - W

        progress(.authenticity)
        let au = await authenticity(dataA, uiA, dataB, uiB, known: known)

        progress(.matching)
        let sim = Self.cosine(a.g, b.g)
        let al = SceneMatch.align(stdA, stdB)
        let relation = al.aligned ? "aligned" : (sim >= Self.similarScene ? "similar" : "different")
        var frame: Framing?
        var masks: [[Bool]?] = [nil, nil]
        var extras = [PhotoExtras(authenticity: au[0]), PhotoExtras(authenticity: au[1])]
        if al.aligned, let Hinv = al.H.inverse {
            masks = [SceneMatch.overlapMask(w: stdA.width, h: stdA.height, meToOther: al.H),
                     SceneMatch.overlapMask(w: stdB.width, h: stdB.height, meToOther: Hinv)]
            let quad = L == 0 ? al.quadBinA : al.quadAinB          // the winner's frame inside the loser's
            frame = SceneMatch.framing(quad, myAspect: P[L].aspect)
            extras[0].otherFrame = al.quadBinA
            extras[1].otherFrame = al.quadAinB
        }

        progress(.objects)
        try Task.checkCancellation()
        var objs: [[Segment]] = []
        progress(.testing)
        for p in P { objs.append(try objectEffects(p, info: info)) }
        for k in 0..<2 {
            let s = P[k].m
            var segs = objs[k]
            let si = Objects.pickSubject(segs, salXY: (s["subject_x"], s["subject_y"]))
            Objects.assignRoles(&segs, subject: si)
            extras[k].objects = segs
            extras[k].geometry = Objects.geometry(P[k].std, segs: segs, subject: si, detail: s.detail)
        }

        progress(.attention)
        try Task.checkCancellation()
        let heat = try P.map { try occlusion($0, info: info) }

        progress(.editing)
        let labs = [LabImage(stdA), LabImage(stdB)]
        let recipe = SceneMatch.lookRecipe(src: labs[L], ref: labs[W], srcMask: masks[L], refMask: masks[W], colorOK: relation != "different")
        let oW = P[W].overall, oL = P[L].overall
        func closure(_ o: Double) -> Double { max(-1, min(1.5, (o - oL) / max(oW - oL, 1e-6))) }
        var previews: [Preview] = []
        let edited = SceneMatch.applyLook(labs[L], recipe)
        let pe = try scoreEdited(edited, info: info)
        previews.append(Preview(kind: "edit", image: Imaging.uiImage(edited), overall: pe.overall, aesthetic: pe.pct[ia],
                                technical: pe.pct[it], closure: closure(pe.overall)))
        if let fr = frame, fr.zoom >= 1.08, fr.inside >= 0.85 {
            let Hlw = L == 0 ? al.H : al.H.inverse
            if let Hlw, let crop = SceneMatch.cropToOther(P[L].std, meToOther: Hlw, otherW: P[W].std.width, otherH: P[W].std.height) {
                let rc = SceneMatch.lookRecipe(src: LabImage(crop), ref: labs[W], colorOK: true)
                let ce = SceneMatch.applyLook(LabImage(crop), rc)
                let pc = try scoreEdited(ce, info: info)
                previews.append(Preview(kind: "crop_edit", image: Imaging.uiImage(ce), overall: pc.overall, aesthetic: pc.pct[ia],
                                        technical: pc.pct[it], closure: closure(pc.overall)))
            }
        }

        progress(.coaching)
        let scale = max(mA.detail.max, mB.detail.max)
        for k in 0..<2 {
            extras[k].layers[.clipping] = Overlays.clipping(P[k].std)
            extras[k].layers[.focus] = Overlays.focus(P[k].std, detail: P[k].m.detail, scale: scale)
            extras[k].layers[.attention] = Overlays.impact(P[k].std, heat: heat[k])
            if !extras[k].objects.isEmpty { extras[k].layers[.objects] = Overlays.objects(P[k].std, segs: extras[k].objects) }
        }
        let ctx = Coach.Input(info: info, photos: P, W: W, L: L, recipe: recipe, frame: frame, relation: relation,
                              objects: extras.map { ($0.objects, $0.geometry) })
        return Coach.build(ctx, pA: pA, extras: extras, similarity: sim, previews: previews, seconds: Date().timeIntervalSince(t0))
    }

    static func cosine(_ x: [Float], _ y: [Float]) -> Double {
        var d = 0.0, nx = 0.0, ny = 0.0
        for i in 0..<min(x.count, y.count) {
            d += Double(x[i] * y[i]); nx += Double(x[i] * x[i]); ny += Double(y[i] * y[i])
        }
        return d / (nx.squareRoot() * ny.squareRoot() + 1e-12)
    }
}

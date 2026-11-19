import CoreML
import UIKit

/// One analysed photo: raw attribute scores from the shared tower and where they sit among reference photos.
struct ScoredPhoto {
    let name: String
    let image: UIImage
    let pixelSize: CGSize
    let raw: [Double]            // one per head
    let percentile: [Double]     // 0...100
    let z: [Double]              // standard deviations from the reference mean
    let signals: [String: Double]
    let globalEmbedHead: [Double]   // first values of the whole-frame embedding (used by the self-test)
    let overall: Double
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
    let fix: String
    let evidence: String
    let shoot: [String]
    let edit: [String]
}

struct Comparison {
    let a: ScoredPhoto
    let b: ScoredPhoto
    /// Model's estimate of how often viewers would prefer A.
    let pA: Double
    let winnerIsA: Bool
    let headline: String
    let summary: String
    let closeCall: Bool
    let reasons: [String]
    let loserLeads: [String]
    let coaching: [CoachItem]
    let attributes: [AttributeRow]
    let seconds: Double

    var winner: ScoredPhoto { winnerIsA ? a : b }
    var loser: ScoredPhoto { winnerIsA ? b : a }
}

/// The twin network on device.  Both photos go through the same encoder and the same tower;
/// the verdict is a function of the score difference, so swapping the photos flips it exactly.
actor PhotoJudge {
    static let shared = PhotoJudge()

    private var info: JudgeInfo?
    private var encoder: MLModel?
    private var tower: MLModel?

    func loadIfNeeded() async throws -> JudgeInfo {
        if let info { return info }
        let i = try JudgeInfo.load()
        encoder = try await Self.loadModel("Encoder", units: .all)
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

    // MARK: - scoring one photo

    private func embed(_ view: RGBImage) throws -> [Float] {
        guard let encoder else { throw JudgeError.model("encoder not loaded") }
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: try ImagePrep.pixelBuffer(view))])
        let out = try encoder.prediction(from: input)
        guard let arr = out.featureValue(for: "embedding")?.multiArrayValue else { throw JudgeError.model("encoder returned no embedding") }
        return (0..<arr.count).map { arr[$0].floatValue }
    }

    private static func array(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        for (i, v) in values.enumerated() { arr[i] = NSNumber(value: v) }
        return arr
    }

    func score(data: Data, name: String) async throws -> ScoredPhoto {
        let info = try await loadIfNeeded()
        guard let tower else { throw JudgeError.model("tower not loaded") }
        let (std, ui) = try ImagePrep.standardImage(from: data, longSide: info.stdLongSide)
        let views = try ImagePrep.views(std: std, original: ui, size: info.inputSize)
        let embeds = try views.map { try embed($0) }
        let sig = Signals.measure(std)
        let h = info.signalKeys.map { Float(sig[$0] ?? 0) }
        let dim = embeds[0].count
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "g": MLFeatureValue(multiArray: try Self.array(embeds[0], shape: [1, dim])),
            "l": MLFeatureValue(multiArray: try Self.array(Array(embeds[1...].joined()), shape: [1, 4, dim])),
            "h": MLFeatureValue(multiArray: try Self.array(h, shape: [1, h.count])),
        ])
        let out = try tower.prediction(from: input)
        guard let s = out.featureValue(for: "scores")?.multiArrayValue else { throw JudgeError.model("tower returned no scores") }
        let raw = (0..<s.count).map { s[$0].doubleValue }
        guard raw.count == info.heads.count else { throw JudgeError.model("expected \(info.heads.count) scores, got \(raw.count)") }
        let pct = raw.enumerated().map { info.percentile($1, head: $0) }
        let z = raw.enumerated().map { ($1 - info.heads[$0].refMean) / max(info.heads[$0].refStd, 1e-9) }
        let ia = info.index(of: "aesthetic") ?? 0
        let it = info.index(of: "technical") ?? 1
        let overall = info.weights.aesthetic * pct[ia] + info.weights.technical * pct[it]
        return ScoredPhoto(name: name, image: ui, pixelSize: CGSize(width: ui.size.width * ui.scale, height: ui.size.height * ui.scale),
                           raw: raw, percentile: pct, z: z, signals: sig,
                           globalEmbedHead: embeds[0].prefix(8).map(Double.init), overall: overall)
    }

    // MARK: - the twin comparison

    func compare(dataA: Data, dataB: Data, nameA: String, nameB: String) async throws -> Comparison {
        let t0 = Date()
        let info = try await loadIfNeeded()
        let a = try await score(data: dataA, name: nameA)
        let b = try await score(data: dataB, name: nameB)
        return Coach.build(info: info, a: a, b: b, seconds: Date().timeIntervalSince(t0))
    }
}

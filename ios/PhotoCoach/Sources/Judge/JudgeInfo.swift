import Foundation

/// Everything the app needs besides network weights (written by `python -m mobile.export_coreml`).
struct JudgeInfo: Decodable {
    struct Head: Decodable {
        let key: String
        let label: String
        let blurb: String
        let group: String
        /// False when the head predicts human ratings only weakly on held-out photos:
        /// it is shown, marked, and never used for advice.
        let reliable: Bool
        let validity: Double?
        let metric: String?
        /// Sharpness of the pairwise comparison for this attribute.
        let scale: Double
        let refMean: Double
        let refStd: Double
        /// Raw-score quantiles (0...100) over the pool of human-rated reference photos.
        let quantiles: [Double]
    }

    struct Advice: Decodable {
        let title: String
        let fix: String
        let shoot: [String]
        let edit: [String]
    }

    struct Weights: Decodable {
        let aesthetic: Double
        let technical: Double
    }

    let encoder: String
    let inputSize: Int
    let stdLongSide: Int
    let weights: Weights
    let groups: [String]
    let signalKeys: [String]
    let signalsUsed: [String]
    let heads: [Head]
    let poolSize: Int?
    let trained: String?
    let advice: [String: Advice]

    static func load() throws -> JudgeInfo {
        guard let url = Bundle.main.url(forResource: "judge", withExtension: "json") else {
            throw JudgeError.missingResource("judge.json")
        }
        return try JSONDecoder().decode(JudgeInfo.self, from: Data(contentsOf: url))
    }

    func index(of key: String) -> Int? { heads.firstIndex { $0.key == key } }

    /// Raw score -> 0...100 standing among the reference pool (linear interpolation in the quantile table).
    func percentile(_ raw: Double, head i: Int) -> Double {
        let q = heads[i].quantiles
        guard let first = q.first, let last = q.last else { return 50 }
        if raw <= first { return 0 }
        if raw >= last { return Double(q.count - 1) }
        var lo = 0
        var hi = q.count - 1
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if q[mid] <= raw { lo = mid } else { hi = mid }
        }
        let span = q[hi] - q[lo]
        return Double(lo) + (span > 0 ? (raw - q[lo]) / span : 0)
    }
}

enum JudgeError: LocalizedError {
    case missingResource(String)
    case badImage
    case model(String)

    var errorDescription: String? {
        switch self {
        case .missingResource(let n): return "Missing app resource: \(n)"
        case .badImage: return "That photo could not be read."
        case .model(let m): return "Model error: \(m)"
        }
    }
}

import SwiftUI

/// Runs the bundled test photos through the on-device pipeline and compares the results with what
/// the PyTorch model produced for the same files (Resources/parity.json).  This is the check that
/// the Core ML conversion and the Swift preprocessing match training.
struct SelfTestView: View {
    struct Expected: Decodable {
        let file: String
        let stdSize: [Int]
        let globalEmbedHead: [Double]
        let signals: [String: Double]
        let scores: [String: Double]
        let percentiles: [String: Double]
    }

    struct Line: Identifiable {
        let id = UUID()
        let text: String
        let ok: Bool?
    }

    @State private var lines: [Line] = []
    @State private var running = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("Compares this phone's results for two bundled photos with the reference PyTorch outputs. Percentiles within about 3 points mean the port is faithful; small differences come from image resizing and 16-bit model weights.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                ForEach(lines) { l in
                    HStack(alignment: .firstTextBaseline) {
                        if let ok = l.ok {
                            Image(systemName: ok ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                .foregroundStyle(ok ? .green : .orange)
                        }
                        Text(l.text).font(.system(.footnote, design: .monospaced))
                    }
                }
                if running { ProgressView() }
            }
            .navigationTitle("Self-test")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() } } }
            .task { await run() }
        }
    }

    private func add(_ text: String, _ ok: Bool? = nil) { lines.append(Line(text: text, ok: ok)) }

    private func run() async {
        guard !running, lines.isEmpty else { return }
        running = true
        defer { running = false }
        do {
            guard let url = Bundle.main.url(forResource: "parity", withExtension: "json") else {
                throw JudgeError.missingResource("parity.json")
            }
            let expected = try JSONDecoder().decode([Expected].self, from: Data(contentsOf: url))
            let info = try await PhotoJudge.shared.loadIfNeeded()
            for e in expected {
                let name = (e.file as NSString).deletingPathExtension
                let ext = (e.file as NSString).pathExtension
                guard let f = Bundle.main.url(forResource: name, withExtension: ext) else {
                    throw JudgeError.missingResource(e.file)
                }
                let t0 = Date()
                let p = try await PhotoJudge.shared.score(data: Data(contentsOf: f), name: e.file)
                add("\(e.file)  (\(String(format: "%.2f", Date().timeIntervalSince(t0))) s)")

                let embDiff = zip(p.globalEmbedHead, e.globalEmbedHead).map { abs($0 - $1) }.max() ?? 0
                let embScale = e.globalEmbedHead.map { abs($0) }.max() ?? 1
                add(String(format: "  encoder output: max diff %.3f (values up to %.2f)", embDiff, embScale), embDiff <= 0.15 * max(embScale, 1e-6))

                var worstSig = ("", 0.0)
                for k in info.signalsUsed {
                    guard let want = e.signals[k], let got = p.signals[k] else { continue }
                    let d = abs(got - want) / max(abs(want), 0.05)
                    if d > worstSig.1 { worstSig = (k, d) }
                }
                if !info.signalsUsed.isEmpty {
                    add(String(format: "  measured signals: worst relative diff %.0f%% (%@)", worstSig.1 * 100, worstSig.0), worstSig.1 <= 0.15)
                }

                var worst = ("", 0.0)
                for (i, h) in info.heads.enumerated() {
                    guard let want = e.percentiles[h.key] else { continue }
                    let d = abs(p.percentile[i] - want)
                    if d > worst.1 { worst = (h.label, d) }
                }
                add(String(format: "  attribute percentiles: worst diff %.1f points (%@)", worst.1, worst.0), worst.1 <= 3)
                for key in ["aesthetic", "technical"] {
                    if let i = info.index(of: key), let want = e.percentiles[key] {
                        add(String(format: "  %@: %.1f here, %.1f reference", key, p.percentile[i], want), abs(p.percentile[i] - want) <= 3)
                    }
                }
            }
        } catch {
            add("Failed: \(error.localizedDescription)", false)
        }
    }
}

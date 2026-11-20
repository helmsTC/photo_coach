import SwiftUI

/// Runs two bundled photos through the on-device pipeline and compares the results with what the
/// PyTorch model produced for the same files (Resources/parity.json).
struct SelfTestView: View {
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
                    Text("Compares this phone's results for two bundled photos with the reference PyTorch outputs. Overall scores within about 3 points mean the port is faithful; small differences come from image resizing and 16-bit model weights.")
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
            .task {
                guard !running, lines.isEmpty else { return }
                running = true
                await ParityCheck.run { text, ok in
                    Task { @MainActor in lines.append(Line(text: text, ok: ok)) }
                }
                running = false
            }
        }
        .preferredColorScheme(.dark)
    }
}

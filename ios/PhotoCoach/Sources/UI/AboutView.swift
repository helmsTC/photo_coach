import SwiftUI

/// The app icon's artwork: the splash's iris, half open, on the app's dark background.
/// Also rendered to `Assets.xcassets/AppIcon` with the DEBUG `-exporticon <path>` launch flag.
struct AppIconArt: View {
    var size: CGFloat = 1024

    var body: some View {
        let s = size / 200
        ZStack {
            Theme.bg
            RadialGradient(colors: [Theme.violet.opacity(0.55), .clear], center: .center, startRadius: 0, endRadius: 120 * s)
            Circle()
                .stroke(LinearGradient(colors: [.white.opacity(0.9), .white.opacity(0.25)], startPoint: .top, endPoint: .bottom), lineWidth: 6 * s)
                .frame(width: 150 * s, height: 150 * s)
            IrisShape(open: 0.42)
                .fill(Theme.cta)
                .overlay(IrisShape(open: 0.42).stroke(Color.black.opacity(0.55), lineWidth: 2 * s))
                .frame(width: 134 * s, height: 134 * s)
                .rotationEffect(.degrees(-18))
            Circle().fill(.white).frame(width: 22 * s, height: 22 * s).blur(radius: 7 * s).opacity(0.85)
        }
        .frame(width: size, height: size)
    }
}

/// Version, privacy, purchase restore and third-party credits.
struct AboutView: View {
    @EnvironmentObject private var store: Store
    @Environment(\.dismiss) private var dismiss

    private var version: String {
        let i = Bundle.main.infoDictionary
        return "\(i?["CFBundleShortVersionString"] as? String ?? "?") (\(i?["CFBundleVersion"] as? String ?? "?"))"
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 22) {
                    VStack(spacing: 10) {
                        AppIconArt(size: 96)
                            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
                        Wordmark(size: 28)
                        Text("Version \(version)").font(.caption).foregroundStyle(Theme.dim)
                    }
                    .padding(.top, 8)

                    section("Privacy", "lock.fill") {
                        Text("Every photo is analysed on this iPhone. Nothing is uploaded, there are no accounts, and the app collects no data. Your recent comparisons are kept as small copies in the app's own storage, without location, and are deleted with the app.")
                    }

                    section("Purchases", "bag.fill") {
                        VStack(alignment: .leading, spacing: 12) {
                            Text(store.ownsPro ? "Pro is unlocked on this Apple ID." : "Bought Pro on another device? Restore it here.")
                            Button {
                                Task { await store.restore() }
                            } label: {
                                HStack {
                                    if store.busy { ProgressView().tint(.white) }
                                    Text("Restore purchases").font(.subheadline.weight(.semibold))
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 11)
                                .background(.white.opacity(0.1), in: Capsule())
                            }
                            .buttonStyle(.plain)
                            .disabled(store.busy)
                            if let m = store.message { Text(m).font(.footnote).foregroundStyle(Theme.dim) }
                        }
                    }

                    section("Acknowledgements", "heart.text.square.fill") {
                        VStack(alignment: .leading, spacing: 18) {
                            Text("ApertureIQ's photo judge is its own twin network: both photos go through the same model and are scored side by side. Its scoring layers were trained for ApertureIQ on five photo-rating datasets. It and the other features below build on these models.")
                                .font(.footnote).foregroundStyle(Theme.dim)
                            ForEach(Credit.all, id: \.name) { c in
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(c.name).font(.subheadline.weight(.semibold))
                                    Text(c.detail).font(.footnote).foregroundStyle(Theme.dim)
                                    if let l = c.license {
                                        Text(l).font(.system(size: 10, design: .monospaced)).foregroundStyle(Theme.dim)
                                    }
                                }
                            }
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 30)
            }
            .background(Theme.bg.ignoresSafeArea())
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .onAppear { store.message = nil }
    }

    private func section<C: View>(_ title: String, _ symbol: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title.uppercased(), systemImage: symbol)
                .font(.caption.weight(.heavy)).kerning(1.2)
                .foregroundStyle(Theme.dim)
            content().font(.subheadline)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassCard(padding: 18)
    }
}

private struct Credit {
    let name: String
    let detail: String
    let license: String?

    static let all: [Credit] = [
        Credit(name: "Community Forensics",
               detail: "The AI-image detector is the Community Forensics ViT-S/16-384 model by Jeongsoo Park and Andrew Owens (CVPR 2025), converted to Core ML.",
               license: mit("2025 Jeongsoo Park")),
        Credit(name: "DETR semantic segmentation",
               detail: "Object detection uses Apple's Core ML DETR-ResNet50 semantic segmentation model, licensed under the Apache License 2.0.",
               license: nil),
        Credit(name: "CLIP",
               detail: "The twin network uses OpenAI's CLIP ViT-B/32 image encoder (Radford et al., 2021) to turn each photo into features, which ApertureIQ's own trained layers then score.",
               license: mit("2021 OpenAI")),
    ]

    static func mit(_ holder: String) -> String { """
    MIT License

    Copyright (c) \(holder)

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """ }
}

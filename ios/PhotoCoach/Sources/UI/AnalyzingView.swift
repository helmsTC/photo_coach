import SwiftUI

struct AnalyzingView: View {
    @ObservedObject var model: CompareModel

    private var current: Int { model.stage?.rawValue ?? -1 }
    private var progress: Double { Double(current + 1) / Double(model.stages.count) }

    var body: some View {
        ZStack {
            AmbientBackground()
            VStack(spacing: 26) {
                Spacer(minLength: 10)
                HStack(spacing: 14) {
                    ScanningPhoto(image: model.thumbs[0], index: 0)
                    ScanningPhoto(image: model.thumbs[1], index: 1)
                }
                .padding(.horizontal, 24)

                VStack(spacing: 6) {
                    Text("Analysing")
                        .font(.system(size: 28, weight: .heavy, design: .rounded))
                    Text(model.stage?.title ?? "Warming up the network…")
                        .font(.subheadline)
                        .foregroundStyle(Theme.dim)
                        .contentTransition(.opacity)
                        .id(model.stage?.rawValue ?? -1)
                }

                ProgressView(value: progress)
                    .tint(.white)
                    .padding(.horizontal, 40)
                    .animation(.easeInOut, value: progress)

                VStack(alignment: .leading, spacing: 11) {
                    ForEach(model.stages) { s in
                        StageRow(stage: s, state: s.rawValue < current ? .done : (s.rawValue == current ? .active : .pending))
                    }
                }
                .glassCard(padding: 18)
                .padding(.horizontal, 20)

                Spacer(minLength: 0)
                Button("Cancel", role: .cancel) { model.cancel() }
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.dim)
                    .padding(.bottom, 14)
            }
        }
    }
}

private struct StageRow: View {
    enum State { case pending, active, done }
    let stage: Stage
    let state: State

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                switch state {
                case .done:
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.good)
                case .active:
                    ProgressView().controlSize(.small).tint(.white)
                case .pending:
                    Image(systemName: stage.symbol).foregroundStyle(Color.white.opacity(0.3))
                }
            }
            .frame(width: 22)
            Text(stage.title)
                .font(.subheadline.weight(state == .active ? .semibold : .regular))
                .foregroundStyle(state == .pending ? Color.white.opacity(0.35) : .white)
            Spacer()
        }
        .animation(.easeInOut(duration: 0.25), value: state == .done)
    }
}

/// Photo with a sweeping scan line.
private struct ScanningPhoto: View {
    let image: UIImage?
    let index: Int

    var body: some View {
        TimelineView(.animation) { tl in
            let t = tl.date.timeIntervalSinceReferenceDate
            let phase = (sin(t * 1.6 + Double(index) * 1.3) + 1) / 2
            GeometryReader { geo in
                ZStack(alignment: .topLeading) {
                    if let image {
                        Image(uiImage: image).resizable().scaledToFill()
                            .frame(width: geo.size.width, height: geo.size.height)
                            .clipped()
                    } else {
                        Theme.card
                    }
                    // scan beam
                    LinearGradient(colors: [.clear, Theme.tint(index).opacity(0.55), .white.opacity(0.9), Theme.tint(index).opacity(0.55), .clear],
                                   startPoint: .top, endPoint: .bottom)
                        .frame(height: 46)
                        .offset(y: CGFloat(phase) * (geo.size.height - 46))
                        .blendMode(.screen)
                    // grid
                    Path { p in
                        for f in [1.0 / 3, 2.0 / 3] {
                            p.move(to: CGPoint(x: geo.size.width * f, y: 0)); p.addLine(to: CGPoint(x: geo.size.width * f, y: geo.size.height))
                            p.move(to: CGPoint(x: 0, y: geo.size.height * f)); p.addLine(to: CGPoint(x: geo.size.width, y: geo.size.height * f))
                        }
                    }
                    .stroke(Color.white.opacity(0.25), lineWidth: 0.7)
                    ShotTag(index: index, small: true).padding(8)
                }
            }
        }
        .aspectRatio(3 / 4, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Theme.tint(index).opacity(0.8), lineWidth: 1.5))
        .shadow(color: Theme.tint(index).opacity(0.4), radius: 18)
    }
}

import SwiftUI

/// Photo viewer with analysis lenses: pixel overlays (clipping, focus, AI attention, objects) plus
/// live vector annotations (thirds grid, subject, horizon, the other photo's frame, object numbers).
struct Inspector: View {
    let r: Comparison
    @Binding var photo: Int
    @Binding var lens: Lens

    private var p: ScoredPhoto { r.photo(photo) }
    private var ex: PhotoExtras { r.extras[photo] }

    private func lenses(for i: Int) -> [Lens] {
        Lens.allCases.filter { l in
            switch l {
            case .objects: return !r.extras[i].objects.isEmpty
            case .framing: return r.extras[i].otherFrame != nil
            case .clipping, .focus, .attention: return r.extras[i].layers[l] != nil
            default: return true
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            shotPicker
            viewer
            lensBar
            Text(lens.legend)
                .font(.footnote)
                .foregroundStyle(Theme.dim)
                .fixedSize(horizontal: false, vertical: true)
                .id(lens)
                .transition(.opacity)
            if lens == .objects { ObjectList(segs: ex.objects) }
            if lens == .composition { CompositionFacts(m: p.m) }
        }
        .onChange(of: photo) { _, new in
            if !lenses(for: new).contains(lens) { lens = .photo }
        }
    }

    private var shotPicker: some View {
        HStack(spacing: 8) {
            ForEach(0..<2, id: \.self) { i in
                Button {
                    withAnimation(.snappy) { photo = i }
                    Theme.tick()
                } label: {
                    HStack(spacing: 8) {
                        ShotTag(index: i, small: true)
                        Text(r.photo(i).name).font(.subheadline.weight(.semibold))
                        if i == r.winnerIndex { Image(systemName: "crown.fill").font(.caption).foregroundStyle(Theme.gold) }
                        Spacer(minLength: 0)
                        Text("\(Int(r.photo(i).overall.rounded()))").font(.subheadline.weight(.bold).monospacedDigit())
                            .foregroundStyle(Theme.dim)
                    }
                    .padding(.horizontal, 12).padding(.vertical, 9)
                    .background(photo == i ? Theme.tint(i).opacity(0.22) : Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(photo == i ? Theme.tint(i) : Theme.stroke))
                }
                .buttonStyle(.plain)
            }
        }
    }

    /// The framing lens shows the wider of the two shots with the tighter one's frame drawn inside it,
    /// so the box is always visible: on the tighter shot the other frame would fall outside the picture.
    private var framingFlipped: Bool {
        guard lens == .framing, let q = ex.otherFrame else { return false }
        let inside = q.filter { $0.0 >= -0.05 && $0.0 <= 1.05 && $0.1 >= -0.05 && $0.1 <= 1.05 }.count
        return inside < 2 && r.extras[1 - photo].otherFrame != nil
    }

    private var viewer: some View {
        let flipped = framingFlipped
        let shown = flipped ? 1 - photo : photo
        let sp = r.photo(shown)
        let base: UIImage = r.extras[shown].layers[lens] ?? sp.image
        return ZStack {
            Image(uiImage: base)
                .resizable()
                .scaledToFit()
                .id("\(shown)-\(lens.rawValue)")
                .transition(.opacity)
            GeometryReader { geo in
                switch lens {
                case .composition: CompositionOverlay(m: sp.m, size: geo.size)
                case .framing:
                    if flipped, let q = r.extras[shown].otherFrame {
                        FrameOverlay(quad: q, size: geo.size, color: Theme.tint(photo), label: "\(p.name)'s frame inside \(sp.name)")
                    } else if let q = ex.otherFrame {
                        FrameOverlay(quad: q, size: geo.size, color: Theme.tint(1 - photo), label: "\(r.photo(1 - photo).name)'s frame")
                    }
                case .objects: ObjectBadges(segs: r.extras[shown].objects, size: geo.size)
                default: EmptyView()
                }
            }
            .allowsHitTesting(false)
        }
        .aspectRatio(sp.aspect, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(Theme.tint(shown).opacity(0.6), lineWidth: 1.5))
        .animation(.easeInOut(duration: 0.3), value: lens)
        .animation(.easeInOut(duration: 0.3), value: photo)
    }

    private var lensBar: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                ForEach(lenses(for: photo)) { l in
                    Button {
                        withAnimation(.snappy) { lens = l }
                        Theme.tick()
                    } label: {
                        Label(l.title, systemImage: l.symbol)
                            .font(.footnote.weight(.semibold))
                            .padding(.horizontal, 12).padding(.vertical, 8)
                            .foregroundStyle(lens == l ? .black : .white)
                            .background(lens == l ? Color.white : Color.white.opacity(0.07), in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .scrollIndicators(.hidden)
    }
}

// MARK: - overlays

struct CompositionOverlay: View {
    let m: Measurement
    let size: CGSize

    var body: some View {
        let w = size.width, h = size.height
        ZStack {
            Path { p in
                for f in [1.0 / 3, 2.0 / 3] {
                    p.move(to: CGPoint(x: w * f, y: 0)); p.addLine(to: CGPoint(x: w * f, y: h))
                    p.move(to: CGPoint(x: 0, y: h * f)); p.addLine(to: CGPoint(x: w, y: h * f))
                }
            }
            .stroke(Color.white.opacity(0.75), lineWidth: 1)
            .shadow(color: .black.opacity(0.6), radius: 2)

            if m["tilt_conf"] >= 0.3 && m["horizon_y"] >= 0 {
                let t = tan(m["tilt_deg"] * .pi / 180)
                let yc = m["horizon_y"] * h
                Path { p in
                    p.move(to: CGPoint(x: 0, y: yc - t * w / 2))
                    p.addLine(to: CGPoint(x: w, y: yc + t * w / 2))
                }
                .stroke(abs(m["tilt_deg"]) >= 1 ? Theme.bad : Theme.good, style: StrokeStyle(lineWidth: 2.5, dash: [8, 5]))
                .shadow(color: .black, radius: 2)
            }

            let r = max(14, min(w, h) * 0.06)
            Circle()
                .stroke(Theme.gold, lineWidth: 3)
                .frame(width: r * 2, height: r * 2)
                .shadow(color: .black, radius: 3)
                .position(x: m["subject_x"] * w, y: m["subject_y"] * h)
            Image(systemName: "plus")
                .font(.system(size: r * 1.1, weight: .bold))
                .foregroundStyle(Color.cyan)
                .shadow(color: .black, radius: 2)
                .position(x: m["weight_cx"] * w, y: m["weight_cy"] * h)
        }
    }
}

struct CompositionFacts: View {
    let m: Measurement
    var body: some View {
        let tiltKnown = m["tilt_conf"] >= 0.3 && m["horizon_y"] >= 0
        HStack(spacing: 8) {
            Pill(text: String(format: "%.0f%% weight at edges", 100 * m["edge_weight"]), color: m["edge_weight"] > 0.25 ? Theme.b : .white, symbol: "square.dashed")
            if tiltKnown {
                Pill(text: String(format: "Horizon %+.1f°", m["tilt_deg"]), color: abs(m["tilt_deg"]) >= 1 ? Theme.bad : Theme.good, symbol: "level")
            }
        }
    }
}

struct FrameOverlay: View {
    let quad: Quad
    let size: CGSize
    let color: Color
    let label: String

    var body: some View {
        let pts = quad.map { CGPoint(x: $0.0 * size.width, y: $0.1 * size.height) }
        ZStack(alignment: .topLeading) {
            Path { p in
                p.addRect(CGRect(origin: .zero, size: size))
                p.addLines(pts); p.closeSubpath()
            }
            .fill(Color.black.opacity(0.55), style: FillStyle(eoFill: true))
            Path { p in p.addLines(pts); p.closeSubpath() }
                .stroke(color, style: StrokeStyle(lineWidth: 3, lineJoin: .round))
                .shadow(color: .black, radius: 3)
            if let tl = pts.first {
                Text(label)
                    .font(.caption2.weight(.bold))
                    .padding(.horizontal, 7).padding(.vertical, 3)
                    .background(color, in: Capsule())
                    .foregroundStyle(.black)
                    .position(x: min(max(tl.x + 50, 50), size.width - 50), y: min(max(tl.y + 14, 12), size.height - 12))
            }
        }
    }
}

extension Segment {
    var roleColor: Color {
        switch role {
        case "subject": return Theme.gold
        case "distraction": return Theme.bad
        case "helps": return Theme.good
        default: return Color.white.opacity(0.85)
        }
    }

    var roleText: String {
        switch role {
        case "subject": return "Subject"
        case "distraction": return "Distracts"
        case "helps": return "Carries the score"
        default: return "Neutral"
        }
    }
}

struct ObjectBadges: View {
    let segs: [Segment]
    let size: CGSize
    var body: some View {
        ForEach(segs) { g in
            Text("\(g.n)")
                .font(.system(size: 13, weight: .heavy, design: .rounded))
                .foregroundStyle(.black)
                .frame(width: 24, height: 24)
                .background(g.roleColor, in: Circle())
                .overlay(Circle().stroke(.black.opacity(0.6), lineWidth: 1))
                .shadow(color: .black.opacity(0.6), radius: 3)
                .position(x: g.labelPoint.0 * size.width, y: g.labelPoint.1 * size.height)
        }
    }
}

struct ObjectList: View {
    let segs: [Segment]
    var body: some View {
        VStack(spacing: 8) {
            ForEach(segs) { g in
                HStack(spacing: 10) {
                    Text("\(g.n)")
                        .font(.system(size: 12, weight: .heavy, design: .rounded))
                        .foregroundStyle(.black)
                        .frame(width: 22, height: 22)
                        .background(g.roleColor, in: Circle())
                    VStack(alignment: .leading, spacing: 1) {
                        Text(g.label.capitalized).font(.subheadline.weight(.semibold))
                        Text("\(Objects.whereText(g)) · \(String(format: "%.1f", g.area * 100))% of frame")
                            .font(.caption).foregroundStyle(Theme.dim)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 1) {
                        Text(g.roleText).font(.caption.weight(.semibold)).foregroundStyle(g.roleColor)
                        Text(String(format: "%+.0f without it", g.delta))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(Theme.dim)
                    }
                }
                .padding(10)
                .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
            Text("Each object was painted out and the photo re-scored. \"+5 without it\" means the network likes the photo 5 points more without that object.")
                .font(.caption2)
                .foregroundStyle(Theme.dim)
        }
    }
}

import SwiftUI

struct ResultView: View {
    let r: Comparison

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            verdict
            photoCards
            if !r.reasons.isEmpty || !r.loserLeads.isEmpty { reasons }
            coaching
            AttributeChart(rows: r.attributes, nameA: r.a.name, nameB: r.b.name)
            Text("Scores are percentiles: 50 is a typical photo, 90 beats nine in ten of the human-rated reference photos. The network judges general appeal, not your intent, and scores within about 10 points of each other are not meaningfully different. Analysed on this device in \(String(format: "%.1f", r.seconds)) s; your photos never leave it.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var verdict: some View {
        let pa = Int((r.pA * 100).rounded())
        return VStack(alignment: .leading, spacing: 8) {
            Text(r.headline).font(.title2.bold())
            Text(r.summary + (r.closeCall ? " This is a close call - treat the verdict as a coin-flip and read the attribute breakdown instead." : ""))
                .font(.subheadline)
                .foregroundStyle(.secondary)
            GeometryReader { geo in
                HStack(spacing: 2) {
                    Rectangle().fill(Palette.a).frame(width: max(0, geo.size.width * r.pA - 1))
                    Rectangle().fill(Palette.b)
                }
                .clipShape(Capsule())
            }
            .frame(height: 8)
            .accessibilityLabel("\(r.a.name) preferred \(pa) percent, \(r.b.name) \(100 - pa) percent")
            HStack(spacing: 6) {
                Dot(color: Palette.a); Text("\(r.a.name) preferred \(pa)%")
                Text("·").foregroundStyle(.secondary)
                Dot(color: Palette.b); Text("\(r.b.name) \(100 - pa)%")
            }
            .font(.caption)
            Text("The model's estimate of how often viewers would pick each.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private var photoCards: some View {
        HStack(alignment: .top, spacing: 12) {
            PhotoCard(p: r.a, tint: Palette.a, isWinner: r.winnerIsA)
            PhotoCard(p: r.b, tint: Palette.b, isWinner: !r.winnerIsA)
        }
    }

    private var reasons: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !r.reasons.isEmpty {
                Card(title: "Why \(r.winner.name) won") { Bullets(items: r.reasons) }
            }
            Card(title: "Where \(r.loser.name) is ahead") {
                if r.loserLeads.isEmpty {
                    Text("Nowhere by a meaningful margin.").font(.subheadline).foregroundStyle(.secondary)
                } else {
                    Bullets(items: r.loserLeads)
                }
            }
        }
    }

    private var coaching: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("How to get \(r.loser.name) closer to \(r.winner.name)").font(.headline)
            if r.coaching.isEmpty {
                Card(title: nil) {
                    Text("No single weakness stands out - the photos differ by small margins across the board. Look at the attribute breakdown below.")
                        .font(.subheadline)
                }
            }
            ForEach(Array(r.coaching.enumerated()), id: \.element.id) { n, item in
                Card(title: "\(n + 1). \(item.title)") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(Self.fixLabel(item.fix))
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 7).padding(.vertical, 2)
                            .overlay(RoundedRectangle(cornerRadius: 4).stroke(Palette.line))
                            .foregroundStyle(.secondary)
                        Text(item.evidence).font(.subheadline)
                        if !item.shoot.isEmpty {
                            Text("AT THE SHOOT").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                            Bullets(items: item.shoot)
                        }
                        if !item.edit.isEmpty {
                            Text("IN THE EDIT").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                            Bullets(items: item.edit)
                        }
                    }
                }
            }
        }
    }

    static func fixLabel(_ fix: String) -> String {
        switch fix {
        case "edit": return "FIXABLE IN EDITING"
        case "reshoot": return "NEEDS A RESHOOT"
        default: return "SHOOT + EDIT"
        }
    }
}

struct Dot: View {
    let color: Color
    var body: some View { Circle().fill(color).frame(width: 8, height: 8) }
}

struct Card<Content: View>: View {
    let title: String?
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let title { Text(title).font(.subheadline.weight(.semibold)) }
            content
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Palette.card)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Palette.line))
    }
}

struct Bullets: View {
    let items: [String]
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, t in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text("•").foregroundStyle(.secondary)
                    Text(t).font(.subheadline).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

struct PhotoCard: View {
    let p: ScoredPhoto
    let tint: Color
    let isWinner: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(uiImage: p.image)
                .resizable()
                .scaledToFill()
                .frame(height: 120)
                .frame(maxWidth: .infinity)
                .clipped()
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Dot(color: tint)
                    Text(p.name).font(.footnote.weight(.semibold))
                    if isWinner {
                        Text("WINNER").font(.system(size: 9, weight: .bold))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(.white).foregroundStyle(.black).clipShape(Capsule())
                    }
                }
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Stat(value: p.overall, label: "Overall", big: true)
                    Stat(value: p.percentile[0], label: "Aesthetic", big: false)
                    Stat(value: p.percentile[1], label: "Technical", big: false)
                }
            }
            .padding(.horizontal, 10)
            .padding(.bottom, 10)
        }
        .background(Palette.card)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(isWinner ? Color.white : Palette.line, lineWidth: isWinner ? 1.5 : 1))
    }
}

struct Stat: View {
    let value: Double
    let label: String
    let big: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("\(Int(value.rounded()))").font(big ? .title.bold() : .headline).monospacedDigit()
            Text(label.uppercased()).font(.system(size: 9)).foregroundStyle(.secondary)
        }
    }
}

/// Dumbbell chart: where each photo stands among reference photos, attribute by attribute.
struct AttributeChart: View {
    let rows: [AttributeRow]
    let nameA: String
    let nameB: String

    private var groups: [(String, [AttributeRow])] {
        var order: [String] = []
        var by: [String: [AttributeRow]] = [:]
        for r in rows {
            if by[r.group] == nil { order.append(r.group) }
            by[r.group, default: []].append(r)
        }
        return order.map { ($0, by[$0] ?? []) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Attribute by attribute").font(.headline)
            HStack(spacing: 6) {
                Dot(color: Palette.a); Text(nameA)
                Dot(color: Palette.b); Text(nameB)
                Spacer()
                Text("0 – 100, 50 = typical").foregroundStyle(.secondary)
            }
            .font(.caption)
            Card(title: nil) {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(groups, id: \.0) { g, items in
                        Text(g.uppercased()).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                        ForEach(items) { row in DumbbellRow(row: row) }
                    }
                }
            }
            if rows.contains(where: { !$0.reliable }) {
                Text("* The network predicts this attribute only weakly on held-out photos; it is shown but never used for coaching.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

struct DumbbellRow: View {
    let row: AttributeRow

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(row.label + (row.reliable ? "" : " *")).font(.footnote)
                    .foregroundStyle(row.reliable ? .primary : .secondary)
                Spacer()
                Text("\(Int(row.a.rounded()))").font(.footnote.monospacedDigit()).frame(width: 26, alignment: .trailing)
                Text("\(Int(row.b.rounded()))").font(.footnote.monospacedDigit()).frame(width: 26, alignment: .trailing)
            }
            GeometryReader { geo in
                let w = geo.size.width
                let xa = w * row.a / 100
                let xb = w * row.b / 100
                ZStack(alignment: .leading) {
                    Rectangle().fill(Palette.line).frame(height: 1)
                    Rectangle().fill(Color(white: 0.4)).frame(width: 1, height: 8).offset(x: w / 2)
                    Rectangle().fill(Color(white: 0.45)).frame(width: abs(xa - xb), height: 2).offset(x: min(xa, xb))
                    Circle().fill(Palette.a).frame(width: 10, height: 10)
                        .overlay(Circle().stroke(Palette.card, lineWidth: 2)).offset(x: xa - 5)
                    Circle().fill(Palette.b).frame(width: 10, height: 10)
                        .overlay(Circle().stroke(Palette.card, lineWidth: 2)).offset(x: xb - 5)
                }
                .frame(height: 12)
            }
            .frame(height: 12)
        }
        .opacity(row.reliable ? 1 : 0.6)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(row.label): \(Int(row.a.rounded())) versus \(Int(row.b.rounded()))")
    }
}

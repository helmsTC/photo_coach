import SwiftUI

extension Authenticity {
    var title: String {
        switch verdict {
        case .labelledAI: return "AI-generated"
        case .aiEdited: return "Edited with AI"
        case .likelyAI: return "Likely AI"
        case .unclear: return "Can't tell"
        case .noSigns: return "No AI signs"
        }
    }

    var symbol: String {
        switch verdict {
        case .labelledAI, .likelyAI: return "sparkles"
        case .aiEdited: return "wand.and.stars"
        case .unclear: return "questionmark.circle"
        case .noSigns: return "checkmark.seal"
        }
    }

    var color: Color {
        switch verdict {
        case .labelledAI, .likelyAI: return Theme.bad
        case .aiEdited: return Theme.gold
        case .unclear: return .white.opacity(0.8)
        case .noSigns: return Theme.good
        }
    }

    /// One line on how the call was made.
    var basis: String {
        switch verdict {
        case .labelledAI, .aiEdited:
            return "Labelled in the file\(label?.tool.map { " by \($0)" } ?? "")"
        case .likelyAI: return "The detector found strong AI traces"
        case .unclear: return "The detector's reading is mixed"
        case .noSigns: return "No labels, and the detector found no AI traces"
        }
    }
}

/// Compact pill for overlays: "Likely AI", "No AI signs", ...
struct AuthBadge: View {
    let a: Authenticity
    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: a.symbol).font(.caption2.weight(.bold))
            Text(a.title).font(.caption2.weight(.bold))
        }
        .padding(.horizontal, 8).padding(.vertical, 5)
        .foregroundStyle(a.color)
        .background(.ultraThinMaterial, in: Capsule())
        .background(Color.black.opacity(0.35), in: Capsule())
        .overlay(Capsule().stroke(a.color.opacity(0.45)))
    }
}

/// Pro: both shots' AI check, as three labelled facts per shot rather than paragraphs.
struct AuthenticityCard: View {
    let r: Comparison

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionHeader(title: "AI check", subtitle: "Was either shot made by an image generator?", symbol: "sparkle.magnifyingglass")
            ForEach(0..<2, id: \.self) { i in
                if let a = r.extras[i].authenticity { panel(i, a) }
            }
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "info.circle").foregroundStyle(Theme.dim)
                Text("A label written by the app that made the image is decisive. The detector is a trained guess: it can miss new generators and heavily compressed images, so it only ever says \"likely\".")
                    .fixedSize(horizontal: false, vertical: true)
            }
            .font(.caption2)
            .foregroundStyle(Theme.dim)
        }
    }

    private func panel(_ i: Int, _ a: Authenticity) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Image(uiImage: r.photo(i).image).resizable().scaledToFill()
                    .frame(width: 56, height: 56)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(a.color.opacity(0.7), lineWidth: 1.5))
                    .overlay(alignment: .topLeading) { ShotTag(index: i, small: true).offset(x: -6, y: -6) }
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Image(systemName: a.symbol)
                        Text(a.title)
                    }
                    .font(.headline)
                    .foregroundStyle(a.color)
                    Text(a.basis).font(.caption).foregroundStyle(Theme.dim).fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            VStack(spacing: 0) {
                FactRow(symbol: "doc.text.magnifyingglass", name: "File labels", value: labelText(a), color: labelColor(a))
                Divider().overlay(Theme.stroke)
                FactRow(symbol: "cpu", name: "Detector", value: a.aiProbability.map { "\(Int(($0 * 100).rounded()))% AI-like" } ?? "Didn't run",
                        color: detectorColor(a)) {
                    if let p = a.aiProbability {
                        GeometryReader { g in
                            ZStack(alignment: .leading) {
                                Capsule().fill(Color.white.opacity(0.08))
                                Capsule().fill(detectorColor(a)).frame(width: max(4, g.size.width * p))
                                // the "likely AI" threshold
                                Rectangle().fill(Color.white.opacity(0.5)).frame(width: 1.5)
                                    .offset(x: g.size.width * Authenticity.likelyAI)
                            }
                        }
                        .frame(height: 5)
                    }
                }
                Divider().overlay(Theme.stroke)
                FactRow(symbol: "camera", name: "Camera data", value: a.camera ?? "None", color: .white)
            }
            .background(Color.white.opacity(0.03), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .glassCard(padding: 14)
    }

    private func labelText(_ a: Authenticity) -> String {
        guard let l = a.label else { return "None found" }
        switch l.kind {
        case .generated: return "Generated · \(l.tool ?? "AI tool")"
        case .edited: return "AI-edited\(l.tool.map { " · \($0)" } ?? "")"
        case .capture: return "Camera capture"
        }
    }

    private func labelColor(_ a: Authenticity) -> Color {
        switch a.label?.kind {
        case .generated: return Theme.bad
        case .edited: return Theme.gold
        case .capture: return Theme.good
        case nil: return Theme.dim
        }
    }

    private func detectorColor(_ a: Authenticity) -> Color {
        guard let p = a.aiProbability else { return Theme.dim }
        return p >= Authenticity.likelyAI ? Theme.bad : (p <= Authenticity.noSigns ? Theme.good : Theme.gold)
    }
}

/// "File labels ........ None found", with an optional visual under the value.
struct FactRow<Extra: View>: View {
    let symbol: String
    let name: String
    let value: String
    let color: Color
    @ViewBuilder var extra: () -> Extra

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 10) {
                Image(systemName: symbol).font(.footnote).foregroundStyle(Theme.dim).frame(width: 20)
                Text(name).font(.footnote).foregroundStyle(Theme.dim)
                Spacer(minLength: 8)
                Text(value).font(.footnote.weight(.semibold)).foregroundStyle(color).lineLimit(1).minimumScaleFactor(0.8)
            }
            extra().padding(.leading, 30)
        }
        .padding(.horizontal, 12).padding(.vertical, 10)
    }
}

extension FactRow where Extra == EmptyView {
    init(symbol: String, name: String, value: String, color: Color) {
        self.init(symbol: symbol, name: name, value: value, color: color) { EmptyView() }
    }
}


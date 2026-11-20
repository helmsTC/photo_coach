import SwiftUI

/// The free tier's result: both shots, a crown on the winner, one line on how decisive it was.
/// No scores or metrics; the "why" is the Pro upsell underneath.
struct VerdictView: View {
    let v: QuickVerdict
    let onNew: () -> Void
    let onUpgrade: () -> Void

    @State private var revealed = false

    private var w: Int { v.winnerIndex }

    var body: some View {
        ZStack {
            AmbientBackground(tintA: Theme.tint(w), tintB: Theme.violet)
            ScrollView {
                VStack(spacing: 26) {
                    Text("THE VERDICT")
                        .font(.caption.weight(.heavy)).kerning(2)
                        .foregroundStyle(Theme.dim)
                        .padding(.top, 8)
                    photos
                    headline
                    upsell
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 40)
            }
            .scrollIndicators(.hidden)
        }
        .safeAreaInset(edge: .top) { topBar }
        .task {
            try? await Task.sleep(for: .seconds(0.35))
            withAnimation(.spring(response: 0.5, dampingFraction: 0.62)) { revealed = true }
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        }
    }

    private var topBar: some View {
        HStack {
            Button(action: onNew) {
                Label("New comparison", systemImage: "chevron.left")
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 14).padding(.vertical, 9)
                    .background(.ultraThinMaterial, in: Capsule())
            }
            .buttonStyle(.plain)
            Spacer()
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 6)
    }

    private var photos: some View {
        HStack(spacing: 12) {
            ForEach(0..<2, id: \.self) { i in
                let winner = i == w
                ZStack(alignment: .topLeading) {
                    Color.clear
                        .aspectRatio(3 / 4, contentMode: .fit)
                        .overlay(Image(uiImage: v.images[i]).resizable().scaledToFill())
                        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous)
                            .stroke(winner && revealed ? Theme.gold : Theme.tint(i).opacity(0.6), lineWidth: winner && revealed ? 3 : 1.5))
                        .shadow(color: (winner && revealed ? Theme.gold : .clear).opacity(0.5), radius: 22, y: 6)
                    ShotTag(index: i).padding(10)
                }
                .overlay(alignment: .top) {
                    if winner {
                        Image(systemName: "crown.fill")
                            .font(.system(size: 34))
                            .foregroundStyle(Theme.gold)
                            .shadow(color: .black.opacity(0.6), radius: 6)
                            .offset(y: revealed ? -26 : -90)
                            .opacity(revealed ? 1 : 0)
                    }
                }
                .scaleEffect(revealed ? (winner ? 1.04 : 0.9) : 1)
                .saturation(revealed && !winner ? 0.35 : 1)
                .opacity(revealed && !winner ? 0.6 : 1)
                // the AI badge stays fully legible on the dimmed loser
                .overlay(alignment: .bottom) {
                    if v.authenticity.indices.contains(i) {
                        AuthBadge(a: v.authenticity[i])
                            .padding(.bottom, revealed && !winner ? 22 : 10)
                            .opacity(revealed ? 1 : 0)
                    }
                }
                .zIndex(winner ? 1 : 0)
            }
        }
        .padding(.top, 26)
    }

    private var headline: some View {
        VStack(spacing: 10) {
            Text("Shot \(w == 0 ? "A" : "B") wins")
                .font(.system(size: 34, weight: .heavy, design: .rounded))
            Pill(text: v.strength.title, color: v.confidence < 0.56 ? Theme.gold : Theme.good,
                 symbol: v.confidence < 0.56 ? "scalemass.fill" : "checkmark.circle.fill")
            Text(v.strength.line)
                .font(.subheadline)
                .foregroundStyle(Theme.dim)
                .multilineTextAlignment(.center)
            if let note = aiNote {
                Label(note, systemImage: "sparkles")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(Theme.bad)
                    .multilineTextAlignment(.center)
                    .padding(.top, 4)
            }
        }
        .opacity(revealed ? 1 : 0)
        .offset(y: revealed ? 0 : 14)
    }

    /// Calls out a shot that looks AI-made, since it changes what the verdict means.
    private var aiNote: String? {
        let flagged = v.authenticity.indices.filter { v.authenticity[$0].isAI }
        switch flagged.count {
        case 2: return "Heads up: both shots look AI-generated."
        case 1: return "Heads up: Shot \(flagged[0] == 0 ? "A" : "B") looks AI-generated."
        default: return nil
        }
    }

    private var upsell: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Text("Want to know why?").font(.headline)
                ProBadge()
            }
            VStack(alignment: .leading, spacing: 11) {
                ForEach(ProFeatures.list.prefix(4), id: \.title) { f in
                    HStack(spacing: 12) {
                        Image(systemName: f.symbol).foregroundStyle(Theme.cta).frame(width: 24)
                        Text(f.title).font(.subheadline)
                        Spacer(minLength: 0)
                        Image(systemName: "lock.fill").font(.caption).foregroundStyle(Theme.dim)
                    }
                }
            }
            Button(action: onUpgrade) {
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                    Text("See the full breakdown").font(.subheadline.weight(.bold))
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Theme.cta, in: Capsule())
                .foregroundStyle(.white)
            }
            .buttonStyle(.plain)
        }
        .glassCard(padding: 18)
        .opacity(revealed ? 1 : 0)
    }
}

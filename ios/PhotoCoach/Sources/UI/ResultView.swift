import SwiftUI

/// The Pro result, split into tabs so no single screen is a wall of text.
enum ResultTab: String, CaseIterable, Identifiable {
    case overview, inspect, coach, scores, ai
    var id: String { rawValue }
    var title: String {
        switch self {
        case .overview: return "Overview"
        case .inspect: return "Inspect"
        case .coach: return "Coach"
        case .scores: return "Scores"
        case .ai: return "AI check"
        }
    }
    var symbol: String {
        switch self {
        case .overview: return "crown.fill"
        case .inspect: return "eye.fill"
        case .coach: return "graduationcap.fill"
        case .scores: return "chart.bar.xaxis"
        case .ai: return "sparkle.magnifyingglass"
        }
    }
}

struct ResultScreen: View {
    let r: Comparison
    let onNew: () -> Void

    @State private var tab: ResultTab = .overview
    @State private var inspectPhoto = 0
    @State private var lens: Lens = .photo
    @State private var detailsPhoto = 0
    @State private var expanded: Set<String> = []
    @Namespace private var tabNS

    var body: some View {
        ZStack {
            AmbientBackground(tintA: Theme.tint(r.winnerIndex), tintB: Theme.violet)
            ScrollViewReader { proxy in
                ScrollView {
                    Color.clear.frame(height: 0).id("top")
                    Group {
                        switch tab {
                        case .overview: OverviewTab(r: r, go: go)
                        case .inspect:
                            VStack(alignment: .leading, spacing: 14) {
                                SectionHeader(title: "Inspect", subtitle: "Switch lenses to see what the analysis sees.", symbol: "eye.fill")
                                Inspector(r: r, photo: $inspectPhoto, lens: $lens)
                            }
                        case .coach:
                            VStack(alignment: .leading, spacing: 30) {
                                coaching(proxy)
                                if let pv = r.bestPreview { EditTowardWinner(r: r, initial: pv).id("edit") }
                            }
                        case .scores:
                            VStack(alignment: .leading, spacing: 30) {
                                WhyItWon(r: r)
                                AttributeBreakdown(r: r)
                                PhotoDetails(r: r, photo: $detailsPhoto).id("details")
                                Text("Scores are percentiles: 50 is a typical photo, 90 beats nine in ten human-rated reference photos. Gaps under about 10 points aren't meaningful.")
                                    .font(.caption).foregroundStyle(Theme.dim)
                            }
                        case .ai: AuthenticityCard(r: r)
                        }
                    }
                    .id(tab)
                    .transition(.opacity)
                    .padding(.horizontal, 18)
                    .padding(.top, 8)
                    .padding(.bottom, 40)
                }
                .scrollIndicators(.hidden)
                .onChange(of: tab) { _, _ in proxy.scrollTo("top", anchor: .top) }
                .task { await debugJump(proxy) }
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            VStack(spacing: 8) {
                topBar
                tabBar
            }
            .padding(.bottom, 8)
            .background(LinearGradient(colors: [Theme.bg, Theme.bg.opacity(0.92), Theme.bg.opacity(0)], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea(edges: .top))
        }
        .onAppear {
            inspectPhoto = r.loserIndex
            detailsPhoto = r.loserIndex
        }
    }

    private func go(_ t: ResultTab) {
        Theme.tick()
        withAnimation(.snappy) { tab = t }
    }

    /// Launch arguments for screenshots: `-tab <tab>`, `-scrollto <section>`, `-lens <lens>`, `-photo <0|1>`, `-expandall`.
    private func debugJump(_ proxy: ScrollViewProxy) async {
        let args = ProcessInfo.processInfo.arguments
        func arg(_ k: String) -> String? { args.firstIndex(of: k).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
        try? await Task.sleep(for: .seconds(0.3))
        if let v = arg("-tab"), let t = ResultTab(rawValue: v) { tab = t }
        if let v = arg("-photo"), let i = Int(v) { inspectPhoto = i; detailsPhoto = i }
        if let v = arg("-lens"), let l = Lens(rawValue: v) { lens = l; tab = .inspect }
        if args.contains("-expandall") { expanded = Set(r.coaching.map(\.id)) }
        if let id = arg("-scrollto") {
            try? await Task.sleep(for: .seconds(0.8))
            proxy.scrollTo(id, anchor: .top)
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
            Text(String(format: "%.1fs on device", r.seconds))
                .font(.caption.weight(.medium))
                .foregroundStyle(Theme.dim)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
        }
        .padding(.horizontal, 18)
        .padding(.top, 6)
    }

    private var tabBar: some View {
        HStack(spacing: 2) {
            ForEach(ResultTab.allCases) { t in
                Button { go(t) } label: {
                    VStack(spacing: 3) {
                        Image(systemName: t.symbol).font(.system(size: 14, weight: .semibold))
                        Text(t.title).font(.system(size: 10, weight: .bold)).lineLimit(1).minimumScaleFactor(0.8)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 7)
                    .foregroundStyle(tab == t ? .white : Theme.dim)
                    .background {
                        if tab == t {
                            Capsule().fill(Theme.cta).opacity(0.9)
                                .matchedGeometryEffect(id: "tab", in: tabNS)
                        }
                    }
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .overlay(alignment: .topTrailing) {
                    if t == .ai, r.extras.contains(where: { $0.authenticity?.isAI == true }) {
                        Circle().fill(Theme.bad).frame(width: 7, height: 7).offset(x: -10, y: 4)
                    }
                }
            }
        }
        .padding(4)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(Capsule().stroke(Theme.stroke))
        .padding(.horizontal, 18)
    }

    // MARK: coaching

    private func coaching(_ proxy: ScrollViewProxy) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(title: "Your coaching plan",
                          subtitle: "How to bring \(r.loser.name) up to \(r.winner.name), most important first.",
                          symbol: "graduationcap.fill")
            if r.coaching.isEmpty {
                Text("No single weakness stands out - the photos differ by small margins across the board. The Scores tab shows where.")
                    .font(.subheadline)
                    .glassCard()
            }
            ForEach(Array(r.coaching.enumerated()), id: \.element.id) { n, item in
                CoachCard(n: n + 1, item: item, expanded: expanded.contains(item.id),
                          toggle: {
                              withAnimation(.spring(response: 0.4, dampingFraction: 0.85)) {
                                  if expanded.contains(item.id) { expanded.remove(item.id) } else { expanded.insert(item.id) }
                              }
                          },
                          showMe: showMeAction(item, proxy))
            }
        }
    }

    /// "Show me" jumps to the tab and lens that illustrate the point.
    private func showMeAction(_ item: CoachItem, _ proxy: ScrollViewProxy) -> (() -> Void)? {
        if let l = Lens(overlay: item.overlay) {
            if l == .objects && r.extras[r.loserIndex].objects.isEmpty { return nil }
            if l == .framing && r.extras[r.loserIndex].otherFrame == nil { return nil }
            return {
                inspectPhoto = r.loserIndex
                lens = l
                go(.inspect)
            }
        }
        if item.overlay == "hist" || item.overlay == "palette" {
            return {
                detailsPhoto = r.loserIndex
                go(.scores)
                Task {
                    try? await Task.sleep(for: .seconds(0.35))
                    withAnimation(.easeInOut(duration: 0.5)) { proxy.scrollTo("details", anchor: .top) }
                }
            }
        }
        return nil
    }
}

// MARK: - overview

/// The headline result at a glance, with tiles that lead into the other tabs.
struct OverviewTab: View {
    let r: Comparison
    let go: (ResultTab) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            hero
            Scoreboard(r: r)
            if !r.reasons.isEmpty { reasons.id("reasons") }
            tiles
            Label("Analysed entirely on this iPhone.", systemImage: "lock.fill")
                .font(.caption).foregroundStyle(Theme.dim)
                .frame(maxWidth: .infinity)
        }
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: 12) {
            ZStack(alignment: .bottomLeading) {
                Image(uiImage: r.winner.image)
                    .resizable()
                    .scaledToFit()
                    .overlay(LinearGradient(colors: [.clear, .black.opacity(0.75)], startPoint: .center, endPoint: .bottom))
                HStack(spacing: 8) {
                    Image(systemName: "crown.fill").foregroundStyle(Theme.gold)
                    Text("WINNER").font(.caption.weight(.heavy)).kerning(1.5)
                    ShotTag(index: r.winnerIndex, small: true)
                }
                .padding(.horizontal, 12).padding(.vertical, 7)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(14)
                if let a = r.extras[r.winnerIndex].authenticity, a.isAI || a.verdict == .aiEdited {
                    AuthBadge(a: a).padding(14).frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 26, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 26, style: .continuous).stroke(Theme.tint(r.winnerIndex).opacity(0.8), lineWidth: 2))
            .shadow(color: Theme.tint(r.winnerIndex).opacity(0.45), radius: 24, y: 8)

            HStack(alignment: .firstTextBaseline) {
                Text(r.headline).font(.system(size: 28, weight: .heavy, design: .rounded))
                Spacer(minLength: 0)
                if r.closeCall { Pill(text: "Close call", color: Theme.gold, symbol: "scalemass.fill") }
            }
            Text(r.summary)
                .font(.subheadline)
                .foregroundStyle(Theme.dim)
                .lineLimit(3)
        }
    }

    /// The top reasons as compact chips: the full list lives in the Scores tab.
    private var reasons: some View {
        VStack(alignment: .leading, spacing: 10) {
            tileHeader("Where \(r.winner.name) pulls ahead", action: "All scores") { go(.scores) }
            HStack(spacing: 8) {
                ForEach(r.reasons.prefix(3)) { reason in
                    VStack(spacing: 4) {
                        Text(String(format: "+%.0f", reason.gap))
                            .font(.system(size: 22, weight: .heavy, design: .rounded).monospacedDigit())
                            .foregroundStyle(Theme.tint(r.winnerIndex))
                        Text(reason.label)
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(Theme.dim)
                            .lineLimit(2)
                            .multilineTextAlignment(.center)
                            .minimumScaleFactor(0.85)
                    }
                    .frame(maxWidth: .infinity, minHeight: 74)
                    .background(Theme.card, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Theme.stroke))
                }
            }
        }
    }

    private var tiles: some View {
        LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 10) {
            if let first = r.coaching.first {
                let fix = Theme.fixStyle(first.fix)
                Tile(symbol: Theme.itemSymbol(first.id), color: fix.color, kicker: "TOP FIX",
                     title: first.title, footer: "\(r.coaching.count) step\(r.coaching.count == 1 ? "" : "s") in your plan") { go(.coach) }
            }
            if let pv = r.bestPreview {
                let d = Int((pv.overall - r.loser.overall).rounded())
                Tile(symbol: "wand.and.stars", color: d > 0 ? Theme.good : (d < 0 ? Theme.bad : Theme.dim), kicker: "ONE-TAP EDIT",
                     title: d == 0 ? "No change" : "\(d > 0 ? "+" : "")\(d) point\(abs(d) == 1 ? "" : "s")",
                     footer: "\(r.loser.name) with the winner's look") { go(.coach) }
            }
            aiTile
            Tile(symbol: relationSymbol, color: Theme.violet, kicker: "THE SCENES", title: relationTitle,
                 footer: r.relation == "aligned" ? "Framing compared directly" : "Judged on their own merits") { go(.inspect) }
        }
    }

    private var aiTile: some View {
        let au = r.extras.compactMap(\.authenticity)
        let flagged = au.indices.filter { au[$0].isAI }
        let title = flagged.isEmpty ? (au.allSatisfy { $0.verdict == .noSigns } ? "No AI signs" : "Mostly clear")
            : (flagged.count == 2 ? "Both look AI" : "Shot \(flagged[0] == 0 ? "A" : "B") looks AI")
        return Tile(symbol: "sparkle.magnifyingglass", color: flagged.isEmpty ? Theme.good : Theme.bad, kicker: "AI CHECK",
                    title: title, footer: Set(au.map(\.title)).count <= 1 ? "Both shots checked"
                        : au.indices.map { "\($0 == 0 ? "A" : "B"): \(au[$0].title)" }.joined(separator: " · ")) { go(.ai) }
    }

    private var relationSymbol: String {
        r.relation == "aligned" ? "square.on.square.intersection.dashed" : (r.relation == "similar" ? "square.on.square" : "rectangle.split.2x1")
    }
    private var relationTitle: String {
        r.relation == "aligned" ? "Same scene" : (r.relation == "similar" ? "Similar subject" : "Different scenes")
    }

    private func tileHeader(_ title: String, action: String, _ tap: @escaping () -> Void) -> some View {
        HStack {
            Text(title).font(.subheadline.weight(.bold))
            Spacer()
            Button(action: tap) {
                HStack(spacing: 3) { Text(action); Image(systemName: "chevron.right") }
                    .font(.caption.weight(.semibold)).foregroundStyle(Theme.dim)
            }
            .buttonStyle(.plain)
        }
    }
}

/// A tappable summary tile that leads into another tab.
struct Tile: View {
    let symbol: String
    let color: Color
    let kicker: String
    let title: String
    let footer: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: symbol)
                        .font(.system(size: 14, weight: .bold))
                        .foregroundStyle(color)
                        .frame(width: 30, height: 30)
                        .background(color.opacity(0.16), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                    Spacer()
                    Image(systemName: "arrow.up.right").font(.caption2.weight(.bold)).foregroundStyle(Theme.dim)
                }
                Text(kicker).font(.system(size: 10, weight: .heavy)).kerning(0.8).foregroundStyle(Theme.dim)
                Text(title).font(.subheadline.weight(.bold)).lineLimit(2).multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
                Text(footer).font(.caption2).foregroundStyle(Theme.dim).lineLimit(2).multilineTextAlignment(.leading)
            }
            .frame(maxWidth: .infinity, minHeight: 138, alignment: .topLeading)
            .padding(13)
            .background(.ultraThinMaterial.opacity(0.6), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .background(Theme.card, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Theme.stroke))
        }
        .buttonStyle(.plain)
    }
}

// MARK: - scoreboard

struct Scoreboard: View {
    let r: Comparison
    @State private var bar = 0.5

    var body: some View {
        VStack(spacing: 18) {
            HStack(alignment: .top) {
                column(0)
                Spacer()
                Text("VS").font(.system(size: 15, weight: .black, design: .rounded)).foregroundStyle(Theme.dim).padding(.top, 44)
                Spacer()
                column(1)
            }
            VStack(spacing: 8) {
                GeometryReader { geo in
                    HStack(spacing: 3) {
                        Capsule().fill(Theme.a).frame(width: max(6, geo.size.width * bar - 1.5))
                        Capsule().fill(Theme.b)
                    }
                }
                .frame(height: 10)
                HStack {
                    Text("\(Int((r.pA * 100).rounded()))%").foregroundStyle(Theme.a)
                    Spacer()
                    Text("would pick each").foregroundStyle(Theme.dim)
                    Spacer()
                    Text("\(100 - Int((r.pA * 100).rounded()))%").foregroundStyle(Theme.b)
                }
                .font(.caption.weight(.bold).monospacedDigit())
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Viewers would prefer shot A \(Int((r.pA * 100).rounded())) percent of the time")
        }
        .glassCard(padding: 18)
        .onAppear { withAnimation(.spring(response: 1.2, dampingFraction: 0.8).delay(0.4)) { bar = r.pA } }
    }

    private func column(_ i: Int) -> some View {
        let p = r.photo(i)
        return VStack(spacing: 10) {
            ZStack(alignment: .topTrailing) {
                ScoreRing(value: p.overall, color: Theme.tint(i), label: "OVERALL")
                if i == r.winnerIndex {
                    Image(systemName: "crown.fill").font(.callout).foregroundStyle(Theme.gold)
                        .shadow(color: Theme.gold.opacity(0.7), radius: 6)
                        .offset(x: 4, y: -6)
                }
            }
            HStack(spacing: 5) { ShotTag(index: i, small: true); Text(p.name).font(.footnote.weight(.semibold)) }
            VStack(alignment: .leading, spacing: 5) {
                mini("Aesthetic", p.percentile[0], i)
                mini("Technical", p.percentile[1], i)
            }
            .frame(width: 118)
        }
    }

    private func mini(_ label: String, _ v: Double, _ i: Int) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(label).font(.caption2).foregroundStyle(Theme.dim)
                Spacer()
                Text("\(Int(v.rounded()))").font(.caption2.weight(.bold).monospacedDigit())
            }
            Meter(value: v, color: Theme.tint(i), height: 4)
        }
    }
}

// MARK: - why it won

struct WhyItWon: View {
    let r: Comparison

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(title: "Why \(r.winner.name) won", symbol: "trophy.fill")
            VStack(spacing: 10) {
                if r.reasons.isEmpty {
                    Text("No single attribute separates them by a meaningful margin.").font(.subheadline).foregroundStyle(Theme.dim)
                }
                ForEach(r.reasons) { reason in
                    ReasonRow(reason: reason, lead: Theme.tint(r.winnerIndex), trail: Theme.tint(r.loserIndex))
                }
            }
            .glassCard()
            if !r.loserLeads.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 6) {
                        Image(systemName: "hand.thumbsup.fill").foregroundStyle(Theme.tint(r.loserIndex))
                        Text("Where \(r.loser.name) is ahead").font(.subheadline.weight(.bold))
                    }
                    ForEach(r.loserLeads) { reason in
                        ReasonRow(reason: reason, lead: Theme.tint(r.loserIndex), trail: Theme.tint(r.winnerIndex), flipped: true)
                    }
                }
                .glassCard()
            }
        }
    }
}

struct ReasonRow: View {
    let reason: Reason
    let lead: Color
    let trail: Color
    var flipped = false

    var body: some View {
        let hi = flipped ? reason.loserPct : reason.winnerPct
        let lo = flipped ? reason.winnerPct : reason.loserPct
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(reason.label).font(.subheadline.weight(.semibold))
                Spacer()
                Text(String(format: "+%.0f", reason.gap))
                    .font(.caption.weight(.heavy).monospacedDigit())
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(lead.opacity(0.2), in: Capsule())
                    .foregroundStyle(lead)
            }
            Dumbbell(a: hi, b: lo, colorA: lead, colorB: trail)
        }
    }
}

// MARK: - coaching card

struct CoachCard: View {
    let n: Int
    let item: CoachItem
    let expanded: Bool
    let toggle: () -> Void
    let showMe: (() -> Void)?

    var body: some View {
        let fix = Theme.fixStyle(item.fix)
        VStack(alignment: .leading, spacing: 12) {
            Button(action: toggle) {
                HStack(alignment: .top, spacing: 12) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 12, style: .continuous).fill(fix.color.opacity(0.18))
                        Image(systemName: Theme.itemSymbol(item.id)).foregroundStyle(fix.color)
                    }
                    .frame(width: 42, height: 42)
                    VStack(alignment: .leading, spacing: 5) {
                        Text("\(n). \(item.title)").font(.headline).multilineTextAlignment(.leading)
                        Pill(text: fix.text, color: fix.color, symbol: fix.symbol)
                    }
                    Spacer(minLength: 0)
                    Image(systemName: "chevron.down")
                        .font(.footnote.weight(.bold))
                        .foregroundStyle(Theme.dim)
                        .rotationEffect(.degrees(expanded ? 180 : 0))
                        .padding(.top, 4)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if expanded {
                VStack(alignment: .leading, spacing: 14) {
                    if !item.evidence.isEmpty {
                        VStack(alignment: .leading, spacing: 7) {
                            label("THE EVIDENCE", "chart.bar.doc.horizontal")
                            ForEach(item.evidence, id: \.self) { Bullet(text: $0) }
                        }
                    }
                    if !item.shoot.isEmpty {
                        VStack(alignment: .leading, spacing: 7) {
                            label("AT THE SHOOT", "camera.fill")
                            ForEach(item.shoot, id: \.self) { Bullet(text: $0, color: Theme.b) }
                        }
                    }
                    if !item.edit.isEmpty {
                        VStack(alignment: .leading, spacing: 7) {
                            label("IN THE EDIT", "slider.horizontal.3")
                            ForEach(item.edit, id: \.self) { Bullet(text: $0, color: Theme.good) }
                        }
                    }
                    if let showMe {
                        Button(action: showMe) {
                            Label("Show me", systemImage: "eye.fill")
                                .font(.subheadline.weight(.semibold))
                                .padding(.horizontal, 14).padding(.vertical, 8)
                                .background(Color.white.opacity(0.1), in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .glassCard()
    }

    private func label(_ t: String, _ s: String) -> some View {
        Label(t, systemImage: s).font(.caption2.weight(.heavy)).kerning(0.8).foregroundStyle(Theme.dim)
    }
}

// MARK: - edit toward the winner

struct EditTowardWinner: View {
    let r: Comparison
    let initial: Preview
    @State private var kind = ""

    private var pv: Preview { r.previews.first { $0.kind == kind } ?? initial }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionHeader(title: "Edit toward the winner",
                          subtitle: "\(r.loser.name) with the winner's tone\(r.recipe?.colorOK == true ? " and colour" : "") applied, re-scored by the same network.",
                          symbol: "wand.and.stars")
            if r.previews.count > 1 {
                Picker("Preview", selection: $kind) {
                    ForEach(r.previews) { Text($0.title).tag($0.kind) }
                }
                .pickerStyle(.segmented)
            }
            BeforeAfter(before: r.loser.image, after: pv.image)
                .id(pv.kind)
            scoreDelta
            if let rec = r.recipe, !rec.lines.isEmpty, pv.kind == "edit" { RecipeList(lines: rec.lines) }
        }
        .onAppear { if kind.isEmpty { kind = initial.kind } }
    }

    private var scoreDelta: some View {
        let before = r.loser.overall, after = pv.overall
        let d = after - before
        return HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 2) {
                Text("SCORE").font(.caption2.weight(.heavy)).foregroundStyle(Theme.dim)
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(Int(before.rounded()))").font(.title2.weight(.bold).monospacedDigit()).foregroundStyle(Theme.dim)
                    Image(systemName: "arrow.right").font(.footnote.weight(.bold)).foregroundStyle(Theme.dim)
                    Text("\(Int(after.rounded()))").font(.title.weight(.heavy).monospacedDigit())
                    let di = Int(d.rounded())
                    Text(di == 0 ? "±0" : String(format: "%+d", di)).font(.subheadline.weight(.bold)).foregroundStyle(di > 0 ? Theme.good : (di < 0 ? Theme.bad : Theme.dim))
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text(pv.closure >= 0 ? "Closes \(Int((min(pv.closure, 1) * 100).rounded()))% of the gap" : "Doesn't help")
                    .font(.caption.weight(.semibold))
                Meter(value: max(0, min(1, pv.closure)) * 100, color: Theme.good, height: 6).frame(width: 130)
            }
        }
        .glassCard()
    }
}

struct BeforeAfter: View {
    let before: UIImage
    let after: UIImage
    @State private var split: CGFloat = 0.5
    @State private var hinted = false

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            ZStack(alignment: .leading) {
                Image(uiImage: after).resizable().scaledToFill().frame(width: w, height: geo.size.height).clipped()
                Image(uiImage: before).resizable().scaledToFill().frame(width: w, height: geo.size.height).clipped()
                    .mask(alignment: .leading) { Rectangle().frame(width: w * split) }
                Rectangle().fill(.white).frame(width: 2).offset(x: w * split - 1).shadow(color: .black.opacity(0.5), radius: 3)
                Circle().fill(.white).frame(width: 34, height: 34)
                    .overlay(Image(systemName: "arrow.left.and.right").font(.caption.weight(.bold)).foregroundStyle(.black))
                    .shadow(radius: 4)
                    .offset(x: w * split - 17)
                VStack {
                    HStack {
                        tag("BEFORE")
                        Spacer()
                        tag("AFTER")
                    }
                    Spacer()
                }
                .padding(10)
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { v in split = min(1, max(0, v.location.x / w)) })
        }
        .aspectRatio(after.size.width / max(after.size.height, 1), contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(Theme.stroke))
        .onAppear {
            guard !hinted else { return }
            hinted = true
            withAnimation(.easeInOut(duration: 0.7).delay(0.4)) { split = 0.3 }
            withAnimation(.easeInOut(duration: 0.8).delay(1.1)) { split = 0.5 }
        }
    }

    private func tag(_ t: String) -> some View {
        Text(t).font(.caption2.weight(.heavy)).kerning(1)
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(.black.opacity(0.55), in: Capsule())
    }
}

struct RecipeList: View {
    let lines: [LookRecipe.Line]
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("THE RECIPE", systemImage: "slider.horizontal.3").font(.caption2.weight(.heavy)).kerning(0.8).foregroundStyle(Theme.dim)
            ForEach(lines) { l in
                HStack(spacing: 12) {
                    Text(l.name).font(.subheadline).frame(width: 104, alignment: .leading)
                    GeometryReader { geo in
                        let mid = geo.size.width / 2
                        let len = mid * CGFloat(min(1, abs(l.amount) / 100))
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color.white.opacity(0.08)).frame(height: 4)
                            Rectangle().fill(Color.white.opacity(0.3)).frame(width: 1, height: 10).offset(x: mid)
                            Capsule().fill(l.amount >= 0 ? Theme.good : Theme.b)
                                .frame(width: max(4, len), height: 4)
                                .offset(x: l.amount >= 0 ? mid : mid - len)
                        }
                        .frame(height: geo.size.height)
                    }
                    .frame(height: 14)
                    Text(l.value).font(.caption.weight(.bold).monospacedDigit()).frame(width: 84, alignment: .trailing)
                }
            }
        }
        .glassCard()
    }
}

// MARK: - attributes

struct Dumbbell: View {
    let a: Double
    let b: Double
    let colorA: Color
    let colorB: Color
    @State private var t = 0.0

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let xa = w * (50 + (a - 50) * t) / 100, xb = w * (50 + (b - 50) * t) / 100
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.07)).frame(height: 3)
                Rectangle().fill(Color.white.opacity(0.25)).frame(width: 1, height: 9).offset(x: w / 2)
                Capsule().fill(LinearGradient(colors: [colorB.opacity(0.6), colorA.opacity(0.6)], startPoint: xb < xa ? .leading : .trailing, endPoint: xb < xa ? .trailing : .leading))
                    .frame(width: max(0, abs(xa - xb)), height: 3)
                    .offset(x: min(xa, xb))
                Circle().fill(colorB).frame(width: 11, height: 11).offset(x: xb - 5.5)
                Circle().fill(colorA).frame(width: 13, height: 13).overlay(Circle().stroke(.white.opacity(0.8), lineWidth: 1.5)).offset(x: xa - 6.5)
            }
            .frame(height: 14)
        }
        .frame(height: 14)
        .onAppear { withAnimation(.spring(response: 1.0, dampingFraction: 0.8).delay(0.2)) { t = 1 } }
    }
}

struct AttributeBreakdown: View {
    let r: Comparison

    private var groups: [(String, [AttributeRow])] {
        var order: [String] = [], by: [String: [AttributeRow]] = [:]
        for row in r.attributes {
            if by[row.group] == nil { order.append(row.group) }
            by[row.group, default: []].append(row)
        }
        return order.map { ($0, by[$0] ?? []) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(title: "Attribute by attribute", subtitle: "Percentile among human-rated photos; 50 is typical.", symbol: "chart.bar.xaxis")
            VStack(alignment: .leading, spacing: 18) {
                HStack(spacing: 14) {
                    HStack(spacing: 5) { ShotTag(index: 0, small: true); Text(r.a.name) }
                    HStack(spacing: 5) { ShotTag(index: 1, small: true); Text(r.b.name) }
                }
                .font(.caption.weight(.semibold))
                ForEach(groups, id: \.0) { g, rows in
                    VStack(alignment: .leading, spacing: 10) {
                        Text(g.uppercased()).font(.caption2.weight(.heavy)).kerning(0.8).foregroundStyle(Theme.dim)
                        ForEach(rows) { row in
                            VStack(alignment: .leading, spacing: 5) {
                                HStack {
                                    Text(row.label + (row.reliable ? "" : " *")).font(.footnote)
                                        .foregroundStyle(row.reliable ? .white : Theme.dim)
                                    Spacer()
                                    Text("\(Int(row.a.rounded()))").font(.footnote.weight(.bold).monospacedDigit()).foregroundStyle(Theme.a).frame(width: 26, alignment: .trailing)
                                    Text("\(Int(row.b.rounded()))").font(.footnote.weight(.bold).monospacedDigit()).foregroundStyle(Theme.b).frame(width: 26, alignment: .trailing)
                                }
                                Dumbbell(a: row.a, b: row.b, colorA: Theme.a, colorB: Theme.b)
                            }
                            .opacity(row.reliable ? 1 : 0.55)
                            .accessibilityElement(children: .ignore)
                            .accessibilityLabel("\(row.label): A \(Int(row.a.rounded())), B \(Int(row.b.rounded()))")
                        }
                    }
                }
                if r.attributes.contains(where: { !$0.reliable }) {
                    Text("* The network predicts this attribute only weakly on held-out photos; it is shown but never used for coaching.")
                        .font(.caption2).foregroundStyle(Theme.dim)
                }
            }
            .glassCard()
        }
    }
}

// MARK: - details

struct PhotoDetails: View {
    let r: Comparison
    @Binding var photo: Int

    var body: some View {
        let p = r.photo(photo)
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(title: "Under the hood", subtitle: "Measured straight off the pixels and the file.", symbol: "cpu")
            Picker("Shot", selection: $photo) {
                Text(r.a.name).tag(0)
                Text(r.b.name).tag(1)
            }
            .pickerStyle(.segmented)
            VStack(alignment: .leading, spacing: 16) {
                if !p.exif.chips.isEmpty || p.exif.camera != nil {
                    VStack(alignment: .leading, spacing: 8) {
                        if let cam = p.exif.camera {
                            Label(cam + (p.exif.lens.map { " · \($0)" } ?? ""), systemImage: "camera").font(.footnote).foregroundStyle(Theme.dim)
                        }
                        FlowChips(items: p.exif.chips)
                    }
                }
                VStack(alignment: .leading, spacing: 6) {
                    caption("HISTOGRAM")
                    HistogramView(h: p.m.histogram).frame(height: 90)
                }
                VStack(alignment: .leading, spacing: 6) {
                    caption("PALETTE")
                    PaletteStrip(swatches: p.m.palette)
                }
                VStack(alignment: .leading, spacing: 8) {
                    caption("MEASUREMENTS")
                    MeasureGrid(m: p.m)
                }
            }
            .glassCard()
        }
    }

    private func caption(_ t: String) -> some View {
        Text(t).font(.caption2.weight(.heavy)).kerning(0.8).foregroundStyle(Theme.dim)
    }
}

struct FlowChips: View {
    let items: [String]
    var body: some View {
        HStack(spacing: 6) {
            ForEach(items, id: \.self) { c in
                Text(c).font(.caption.weight(.semibold).monospacedDigit())
                    .padding(.horizontal, 9).padding(.vertical, 5)
                    .background(Color.white.opacity(0.08), in: Capsule())
            }
        }
    }
}

struct HistogramView: View {
    let h: Histogram
    var body: some View {
        Canvas { ctx, size in
            let peak = max(h.r.max() ?? 0, h.g.max() ?? 0, h.b.max() ?? 0, h.l.max() ?? 0, 1e-9)
            func path(_ v: [Double]) -> Path {
                var p = Path()
                p.move(to: CGPoint(x: 0, y: size.height))
                for (i, x) in v.enumerated() {
                    p.addLine(to: CGPoint(x: size.width * CGFloat(i) / CGFloat(v.count - 1), y: size.height * (1 - CGFloat(min(1, x / peak * 1.1)))))
                }
                p.addLine(to: CGPoint(x: size.width, y: size.height))
                p.closeSubpath()
                return p
            }
            ctx.blendMode = .screen
            ctx.fill(path(h.r), with: .color(.red.opacity(0.45)))
            ctx.fill(path(h.g), with: .color(.green.opacity(0.45)))
            ctx.fill(path(h.b), with: .color(.blue.opacity(0.5)))
            ctx.blendMode = .normal
            ctx.stroke(path(h.l), with: .color(.white.opacity(0.85)), lineWidth: 1.2)
        }
        .background(Color.black.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

struct PaletteStrip: View {
    let swatches: [Swatch]
    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 2) {
                ForEach(swatches, id: \.self) { s in
                    Rectangle().fill(Color(uiColor: s.color)).frame(width: max(4, geo.size.width * s.share - 2))
                }
            }
        }
        .frame(height: 30)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct MeasureGrid: View {
    let m: Measurement
    var body: some View {
        let cells: [(String, String, Bool)] = [
            ("Brightness", String(format: "%.2f", m["luma_median"]), m["luma_median"] < 0.25 || m["luma_median"] > 0.7),
            ("Blown highlights", String(format: "%.1f%%", 100 * m["clip_hi"]), m["clip_hi"] >= 0.01),
            ("Crushed shadows", String(format: "%.1f%%", 100 * m["clip_lo"]), m["clip_lo"] >= 0.03),
            ("Contrast", String(format: "%.2f", m["contrast_rms"]), m["contrast_rms"] < 0.12),
            ("Colourfulness", String(format: "%.2f", m["colorfulness"]), false),
            ("Saturation", String(format: "%.0f%%", 100 * m["sat_mean"]), m["sat_mean"] > 0.62),
            ("Edge crispness", String(format: "%.2f", m["edge_crispness"]), false),
            ("Noise", String(format: "%.1f", 1000 * m["noise_luma"]), m["noise_luma"] > 0.006),
        ]
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
            ForEach(cells, id: \.0) { c in
                VStack(alignment: .leading, spacing: 2) {
                    Text(c.0).font(.caption2).foregroundStyle(Theme.dim)
                    Text(c.1).font(.subheadline.weight(.bold).monospacedDigit()).foregroundStyle(c.2 ? Theme.b : .white)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }
}

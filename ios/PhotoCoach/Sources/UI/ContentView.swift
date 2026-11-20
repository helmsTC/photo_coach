import PhotosUI
import SwiftUI

@MainActor
final class CompareModel: ObservableObject {
    enum Phase { case picking, analyzing, verdict, result }

    @Published var data: [Data?] = [nil, nil]
    @Published var thumbs: [UIImage?] = [nil, nil]
    @Published var phase: Phase = .picking
    @Published var stage: Stage?
    @Published var error: String?
    @Published var result: Comparison?
    @Published var quick: QuickVerdict?
    var tier: Tier = .pro
    var history: History?
    var store: Store?
    /// The saved entry being shown, so a re-run updates it instead of adding a duplicate.
    private var entryID: UUID?
    /// AI checks saved with a reopened comparison, reused instead of checking the re-encoded copies.
    private var known: [Authenticity]?
    /// Samples aren't saved to history.
    private var fromSample = false
    private var task: Task<Void, Never>?

    var ready: Bool { data[0] != nil && data[1] != nil }
    /// The free tier only runs the first four stages: read, measure, score, AI check.
    var stages: [Stage] { tier == .free ? [.reading, .measuring, .scoring, .authenticity] : Stage.allCases }

    func set(_ d: Data?, at i: Int) {
        data[i] = d
        entryID = nil
        known = nil
        fromSample = false
        thumbs[i] = d.flatMap { UIImage(data: $0) }.map { Self.thumbnail($0) }
        error = nil
    }

    func swap() {
        data.swapAt(0, 1)
        entryID = nil
        known?.swapAt(0, 1)
        thumbs.swapAt(0, 1)
        Theme.tick()
    }

    func load(_ pair: Samples.Pair) {
        set(Samples.data(pair.a), at: 0)
        set(Samples.data(pair.b), at: 1)
        fromSample = true
        Theme.tick()
    }

    /// Reopens a saved comparison: same photos, analysed again in the current tier.
    func open(_ e: HistoryEntry) {
        guard let (a, b) = history?.photos(e) else { history?.delete(e); return }
        set(a, at: 0)
        set(b, at: 1)
        entryID = e.id
        known = e.authenticity
        Theme.tick()
        run()
    }

    private func remember(a: Data, b: Data, winner: Int, confidence: Double, authenticity: [Authenticity]) async {
        var skip = fromSample
        #if DEBUG
        if ProcessInfo.processInfo.arguments.contains("-savesamples") { skip = false }
        #endif
        guard !skip, let history else { return }
        entryID = await history.record(id: entryID, a: a, b: b, winner: winner, confidence: confidence,
                                       pro: tier == .pro, ownsPro: store?.ownsPro ?? false, authenticity: authenticity)
    }

    func run() {
        guard let a = data[0], let b = data[1], phase != .analyzing else { return }
        error = nil
        stage = nil
        withAnimation(.spring(response: 0.5, dampingFraction: 0.9)) { phase = .analyzing }
        task = Task {
            do {
                let progress: @Sendable (Stage) -> Void = { s in
                    Task { @MainActor in
                        withAnimation(.easeInOut(duration: 0.25)) { self.stage = s }
                    }
                }
                if tier == .free {
                    let v = try await PhotoJudge.shared.quickVerdict(dataA: a, dataB: b, known: known, progress: progress)
                    try Task.checkCancellation()
                    quick = v
                    await remember(a: a, b: b, winner: v.winnerIndex, confidence: v.confidence, authenticity: v.authenticity)
                    Theme.haptic(.success)
                    withAnimation(.spring(response: 0.6, dampingFraction: 0.88)) { phase = .verdict }
                } else {
                    let r = try await PhotoJudge.shared.compare(dataA: a, dataB: b, nameA: "Shot A", nameB: "Shot B", known: known, progress: progress)
                    try Task.checkCancellation()
                    result = r
                    await remember(a: a, b: b, winner: r.winnerIndex, confidence: r.confidence, authenticity: r.extras.map { $0.authenticity ?? Authenticity.decide(label: nil, p: nil, camera: nil) })
                    Theme.haptic(.success)
                    withAnimation(.spring(response: 0.6, dampingFraction: 0.88)) { phase = .result }
                }
            } catch is CancellationError {
                withAnimation { phase = .picking }
            } catch {
                self.error = error.localizedDescription
                Theme.haptic(.error)
                withAnimation { phase = .picking }
            }
        }
    }

    func cancel() {
        task?.cancel()
        withAnimation { phase = .picking }
    }

    func newComparison() {
        withAnimation(.spring(response: 0.5, dampingFraction: 0.9)) {
            phase = .picking
            result = nil
            quick = nil
        }
    }

    static func thumbnail(_ img: UIImage, maxSide: CGFloat = 900) -> UIImage {
        let s = min(1, maxSide / max(img.size.width, img.size.height))
        guard s < 1 else { return img }
        let size = CGSize(width: img.size.width * s, height: img.size.height * s)
        return UIGraphicsImageRenderer(size: size).image { _ in img.draw(in: CGRect(origin: .zero, size: size)) }
    }
}

struct ContentView: View {
    @Binding var tier: Tier
    @StateObject private var model = CompareModel()
    @EnvironmentObject private var history: History
    @EnvironmentObject private var store: Store
    @State private var paywall = false

    var body: some View {
        ZStack {
            switch model.phase {
            case .picking:
                HomeView(model: model, tier: tier) { paywall = true }.transition(.opacity)
            case .analyzing:
                AnalyzingView(model: model).transition(.opacity.combined(with: .scale(scale: 1.04)))
            case .result:
                if let r = model.result {
                    ResultScreen(r: r) { model.newComparison() }
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            case .verdict:
                if let v = model.quick {
                    VerdictView(v: v, onNew: { model.newComparison() }, onUpgrade: { paywall = true })
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
        }
        .sheet(isPresented: $paywall) {
            PaywallView {
                // upgraded mid-session: switch to Pro, and if a verdict is showing, rerun it as the full analysis
                tier = .pro
                if model.phase == .verdict { model.run() }
            }
            .environmentObject(store)
        }
        .onChange(of: tier) { _, t in model.tier = t }
        .onAppear {
            model.tier = tier
            model.history = history
            model.store = store
            // `-autosample N` jumps straight to a sample comparison (used for screenshots)
            let args = ProcessInfo.processInfo.arguments
            #if DEBUG
            // `-pairfiles A B` compares two files from disk (simulator testing)
            if let i = args.firstIndex(of: "-pairfiles"), i + 2 < args.count,
               let a = try? Data(contentsOf: URL(fileURLWithPath: args[i + 1])), let b = try? Data(contentsOf: URL(fileURLWithPath: args[i + 2])) {
                model.set(a, at: 0)
                model.set(b, at: 1)
                model.run()
            }
            // `-openrecent N` reopens saved comparison N, as if its card were tapped
            if let i = args.firstIndex(of: "-openrecent"), i + 1 < args.count, let n = Int(args[i + 1]),
               history.entries.indices.contains(n) {
                model.open(history.entries[n])
            }
            #endif
            if let i = args.firstIndex(of: "-autosample"), i + 1 < args.count, let n = Int(args[i + 1]), Samples.pairs.indices.contains(n) {
                model.load(Samples.pairs[n])
                model.run()
            }
        }
    }
}

// MARK: - home

struct HomeView: View {
    @ObservedObject var model: CompareModel
    @EnvironmentObject private var history: History
    @EnvironmentObject private var store: Store
    let tier: Tier
    let onUpgrade: () -> Void
    @State private var showSelfTest = false
    @State private var showAbout = false
    @State private var glow = false

    var body: some View {
        ZStack {
            AmbientBackground()
            ScrollView {
                VStack(alignment: .leading, spacing: 26) {
                    header
                    slots
                    compareButton
                    if let e = model.error {
                        Label(e, systemImage: "exclamationmark.triangle.fill")
                            .font(.footnote).foregroundStyle(Theme.bad)
                    }
                    recent
                    samples
                    footer
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 30)
            }
            .scrollIndicators(.hidden)
        }
        .sheet(isPresented: $showSelfTest) { SelfTestView() }
        .sheet(isPresented: $showAbout) { AboutView() }
        #if DEBUG
        .onAppear { if ProcessInfo.processInfo.arguments.contains("-about") { showAbout = true } }
        #endif
    }

    private var header: some View {
        HStack(alignment: .center) {
            HStack(spacing: 8) {
                Wordmark(size: 38)
                if tier == .pro { ProBadge() }
            }
            Spacer()
            if tier == .free {
                Button(action: onUpgrade) {
                    Label("Go Pro", systemImage: "sparkles")
                        .font(.caption.weight(.bold))
                        .padding(.horizontal, 12).padding(.vertical, 9)
                        .background(Theme.cta, in: Capsule())
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
            }
            Menu {
                Button { showAbout = true } label: { Label("About ApertureIQ", systemImage: "info.circle") }
                #if DEBUG
                Button { showSelfTest = true } label: { Label("Run self-test", systemImage: "checkmark.seal") }
                #endif
            } label: {
                Image(systemName: "ellipsis")
                    .font(.headline)
                    .frame(width: 38, height: 38)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .foregroundStyle(.white)
        }
    }

    private var slots: some View {
        ZStack {
            HStack(spacing: 14) {
                PhotoSlot(index: 0, image: model.thumbs[0]) { model.set($0, at: 0) }
                    .rotationEffect(.degrees(-2.5))
                PhotoSlot(index: 1, image: model.thumbs[1]) { model.set($0, at: 1) }
                    .rotationEffect(.degrees(2.5))
            }
            Button(action: model.swap) {
                ZStack {
                    Circle().fill(Theme.bg).frame(width: 56, height: 56)
                    Circle().stroke(Theme.cta, lineWidth: 2.5).frame(width: 56, height: 56)
                    VStack(spacing: 0) {
                        Text("VS").font(.system(size: 16, weight: .black, design: .rounded))
                        Image(systemName: "arrow.left.arrow.right").font(.system(size: 9, weight: .bold)).foregroundStyle(Theme.dim)
                    }
                }
                .shadow(color: .black.opacity(0.6), radius: 10)
            }
            .buttonStyle(.plain)
            .disabled(model.data[0] == nil && model.data[1] == nil)
            .accessibilityLabel("Swap shots")
        }
    }

    private var compareButton: some View {
        Button(action: model.run) {
            HStack(spacing: 10) {
                Image(systemName: "sparkles")
                Text("Compare").font(.headline.weight(.bold))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 17)
            .background(Theme.cta, in: Capsule())
            .shadow(color: Color(red: 0.95, green: 0.33, blue: 0.62).opacity(model.ready ? (glow ? 0.7 : 0.35) : 0), radius: glow ? 22 : 12)
            .opacity(model.ready ? 1 : 0.35)
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .disabled(!model.ready)
        .onAppear { withAnimation(.easeInOut(duration: 1.6).repeatForever()) { glow = true } }
    }

    private var recent: some View {
        let shown = history.visible(ownsPro: store.ownsPro)
        return VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Recent").font(.subheadline.weight(.semibold)).foregroundStyle(Theme.dim)
                    Spacer()
                    if store.ownsPro {
                        Text("\(shown.count) of 10").font(.caption).foregroundStyle(Theme.dim)
                    } else {
                        Button(action: onUpgrade) {
                            (Text("\(shown.count) of 3 · ") + Text("Pro keeps 10").bold())
                                .font(.caption).foregroundStyle(Theme.dim)
                        }
                        .buttonStyle(.plain)
                    }
                }
                if shown.isEmpty {
                    HStack(spacing: 12) {
                        Image(systemName: "clock.arrow.circlepath").font(.title3).foregroundStyle(Theme.dim)
                        Text("Your comparisons show up here so you can reopen them later. Pick two of your own photos to start (the sample pairs aren't saved).")
                            .font(.footnote).foregroundStyle(Theme.dim)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .glassCard(padding: 14)
                } else {
                ScrollView(.horizontal) {
                    HStack(spacing: 12) {
                        ForEach(shown) { e in
                            Button { model.open(e) } label: { HistoryCard(entry: e, thumbs: history.thumbs(e)) }
                                .buttonStyle(.plain)
                                .contextMenu {
                                    Button(role: .destructive) { withAnimation { history.delete(e) } } label: {
                                        Label("Delete", systemImage: "trash")
                                    }
                                }
                        }
                    }
                    .padding(.horizontal, 20)
                }
                .scrollIndicators(.hidden)
                .padding(.horizontal, -20)
                }
            }
    }

    private var samples: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("No photos handy? Try a pair")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Theme.dim)
            ScrollView(.horizontal) {
                HStack(spacing: 12) {
                    ForEach(Samples.pairs) { pair in
                        Button { model.load(pair) } label: { SampleCard(pair: pair) }
                            .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 20)
            }
            .scrollIndicators(.hidden)
            .padding(.horizontal, -20)
        }
    }

    private var footer: some View {
        Label("Everything runs on this iPhone. Your photos never leave it.", systemImage: "lock.fill")
            .font(.caption)
            .foregroundStyle(Theme.dim)
            .frame(maxWidth: .infinity)
    }
}

struct PhotoSlot: View {
    let index: Int
    let image: UIImage?
    let onPick: (Data?) -> Void
    @State private var item: PhotosPickerItem?

    var body: some View {
        PhotosPicker(selection: $item, matching: .images) {
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(Theme.card)
                    .overlay {
                        if let image {
                            Image(uiImage: image).resizable().scaledToFill()
                        } else {
                            VStack(spacing: 10) {
                                Image(systemName: "plus.viewfinder").font(.system(size: 30, weight: .light))
                                Text(index == 0 ? "Your shot" : "Compare with").font(.footnote.weight(.semibold))
                                Text("Tap to choose").font(.caption2).foregroundStyle(Theme.dim)
                            }
                            .foregroundStyle(Theme.tint(index))
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .stroke(Theme.tint(index).opacity(image == nil ? 0.5 : 0.9), style: StrokeStyle(lineWidth: 2, dash: image == nil ? [7, 6] : [])))
                    .shadow(color: Theme.tint(index).opacity(image == nil ? 0 : 0.35), radius: 16, y: 6)
                ShotTag(index: index).padding(10)
            }
            .aspectRatio(3 / 4, contentMode: .fit)
        }
        .buttonStyle(.plain)
        .onChange(of: item) { _, new in
            guard let new else { return }
            Task {
                let data = try? await new.loadTransferable(type: Data.self)
                onPick(data)
                item = nil
            }
        }
        .accessibilityLabel(index == 0 ? "Choose shot A" : "Choose shot B")
    }
}

struct SampleCard: View {
    let pair: Samples.Pair
    @State private var images: [UIImage] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack {
                ForEach(Array(images.enumerated()), id: \.offset) { i, img in
                    Image(uiImage: img).resizable().scaledToFill()
                        .frame(width: 92, height: 70)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Theme.tint(i), lineWidth: 1.5))
                        .rotationEffect(.degrees(i == 0 ? -6 : 6))
                        .offset(x: i == 0 ? -22 : 22)
                }
            }
            .frame(width: 150, height: 84)
            Text(pair.title).font(.footnote.weight(.semibold))
            Text(pair.subtitle).font(.caption2).foregroundStyle(Theme.dim)
        }
        .padding(12)
        .frame(width: 174, alignment: .leading)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(Theme.stroke))
        .task {
            if images.isEmpty {
                images = [pair.a, pair.b].compactMap { Samples.data($0).flatMap(UIImage.init(data:)).map { CompareModel.thumbnail($0, maxSide: 240) } }
            }
        }
    }
}

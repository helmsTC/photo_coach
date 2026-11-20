import SwiftUI

/// The app's wordmark: "Aperture" in white, "IQ" in the brand gradient.
struct Wordmark: View {
    var size: CGFloat = 38

    var body: some View {
        HStack(spacing: 0) {
            Text("Aperture")
                .foregroundStyle(LinearGradient(colors: [.white, .white.opacity(0.78)], startPoint: .top, endPoint: .bottom))
            Text("IQ")
                .foregroundStyle(Theme.cta)
        }
        .font(.system(size: size, weight: .heavy, design: .rounded))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("ApertureIQ")
    }
}

/// Six shutter blades around a lens opening; `open` 0 = closed iris, 1 = wide open.
struct IrisShape: Shape {
    var open: CGFloat
    var blades = 6

    var animatableData: CGFloat {
        get { open }
        set { open = newValue }
    }

    func path(in rect: CGRect) -> Path {
        let c = CGPoint(x: rect.midX, y: rect.midY)
        let R = min(rect.width, rect.height) / 2
        let hole = R * (0.08 + 0.5 * open)                       // radius of the opening
        var p = Path()
        for i in 0..<blades {
            let a0 = Double(i) * 2 * .pi / Double(blades)
            let a1 = a0 + 2 * .pi / Double(blades)
            let twist = Double(1 - open) * 0.9 + 0.35                // blades swing in as the iris closes
            // each blade: from the opening's edge out to the rim, swept sideways by `twist`
            let inner0 = CGPoint(x: c.x + hole * CGFloat(cos(a0 + twist)), y: c.y + hole * CGFloat(sin(a0 + twist)))
            let inner1 = CGPoint(x: c.x + hole * CGFloat(cos(a1 + twist)), y: c.y + hole * CGFloat(sin(a1 + twist)))
            let outer0 = CGPoint(x: c.x + R * CGFloat(cos(a0)), y: c.y + R * CGFloat(sin(a0)))
            var b = Path()
            b.move(to: inner0)
            b.addLine(to: outer0)
            b.addArc(center: c, radius: R, startAngle: .radians(a0), endAngle: .radians(a1), clockwise: false)
            b.addLine(to: inner1)
            b.closeSubpath()
            p.addPath(b)
        }
        return p
    }
}

/// Startup animation: the iris spins shut, snaps open, the wordmark lands, then Free / Pro cards rise in.
struct SplashView: View {
    let onChoose: (Tier) -> Void

    @State private var open: CGFloat = 1
    @State private var spin = -140.0
    @State private var markScale = 0.4
    @State private var markOpacity = 0.0
    @State private var flash = 0.0
    @State private var ring = 0.0
    @State private var titleShown = false
    @State private var taglineShown = false
    @State private var chooserShown = false

    var body: some View {
        ZStack {
            AmbientBackground(tintA: Theme.violet, tintB: Theme.b)

            // shockwave ring when the iris snaps open
            Circle()
                .stroke(Theme.cta, lineWidth: 3)
                .frame(width: 150, height: 150)
                .scaleEffect(1 + ring * 2.6)
                .opacity((1 - ring) * 0.8)
                .offset(y: -90)

            VStack(spacing: 0) {
                Spacer(minLength: 0)
                VStack(spacing: 28) {
                    ZStack {
                        Circle()
                            .fill(RadialGradient(colors: [Theme.violet.opacity(0.55), .clear], center: .center, startRadius: 10, endRadius: 140))
                            .frame(width: 280, height: 280)
                            .opacity(markOpacity)
                        // lens barrel
                        Circle()
                            .stroke(LinearGradient(colors: [.white.opacity(0.9), .white.opacity(0.25)], startPoint: .top, endPoint: .bottom), lineWidth: 5)
                            .frame(width: 168, height: 168)
                        // the blades: brand gradient with dark seams between them
                        IrisShape(open: open)
                            .fill(Theme.cta)
                            .overlay(IrisShape(open: open).stroke(Color.black.opacity(0.55), lineWidth: 2))
                            .frame(width: 152, height: 152)
                            .rotationEffect(.degrees(spin))
                        // glint of light through the opening
                        Circle()
                            .fill(.white)
                            .frame(width: 26, height: 26)
                            .blur(radius: 8)
                            .opacity(flash)
                    }
                    .frame(width: 200, height: 200)
                    .scaleEffect(markScale * (chooserShown ? 0.62 : 1))
                    .frame(height: chooserShown ? 130 : 200)
                    .opacity(markOpacity)
                    .shadow(color: Color(red: 0.95, green: 0.33, blue: 0.62).opacity(0.5), radius: 30)

                    VStack(spacing: 10) {
                        Wordmark(size: 44)
                            .opacity(titleShown ? 1 : 0)
                            .offset(y: titleShown ? 0 : 18)
                            .scaleEffect(titleShown ? 1 : 0.92)
                        Text("Smarter shots, one comparison at a time")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(Theme.dim)
                            .opacity(taglineShown ? 1 : 0)
                            .offset(y: taglineShown ? 0 : 10)
                    }
                }
                Spacer(minLength: 0)
                if chooserShown {
                    TierChooser(onChoose: onChoose)
                        .padding(.horizontal, 20)
                        .padding(.bottom, 12)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
        }
        .task { await play() }
    }

    private func play() async {
        // 1. lens arrives, iris spins closed
        withAnimation(.spring(response: 0.7, dampingFraction: 0.75)) { markScale = 1; markOpacity = 1 }
        withAnimation(.easeInOut(duration: 0.75)) { open = 0; spin = 0 }
        try? await Task.sleep(for: .seconds(0.8))
        // 2. "click": iris snaps open with a flash and a shockwave
        UIImpactFeedbackGenerator(style: .rigid).impactOccurred()
        withAnimation(.spring(response: 0.35, dampingFraction: 0.6)) { open = 0.72; spin = 60 }
        withAnimation(.easeOut(duration: 0.15)) { flash = 1 }
        withAnimation(.easeOut(duration: 0.9)) { ring = 1 }
        try? await Task.sleep(for: .seconds(0.15))
        withAnimation(.easeOut(duration: 0.6)) { flash = 0.35 }
        // 3. wordmark and tagline
        withAnimation(.spring(response: 0.55, dampingFraction: 0.8)) { titleShown = true }
        try? await Task.sleep(for: .seconds(0.25))
        withAnimation(.easeOut(duration: 0.5)) { taglineShown = true }
        try? await Task.sleep(for: .seconds(0.9))
        // 4. the logo lifts and the tier cards rise in
        withAnimation(.spring(response: 0.6, dampingFraction: 0.85)) { chooserShown = true }
    }
}

/// Free or Pro: the two ways into the app.
struct TierChooser: View {
    let onChoose: (Tier) -> Void
    @EnvironmentObject var store: Store

    var body: some View {
        VStack(spacing: 12) {
            // Free
            Button {
                Theme.tick()
                onChoose(.free)
            } label: {
                HStack(spacing: 14) {
                    Image(systemName: "crown")
                        .font(.title3).foregroundStyle(Theme.gold)
                        .frame(width: 40, height: 40)
                        .background(Color.white.opacity(0.07), in: Circle())
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Free").font(.headline)
                        Text("Pick two shots, see which one wins.").font(.footnote).foregroundStyle(Theme.dim)
                    }
                    Spacer(minLength: 0)
                    Image(systemName: "chevron.right").font(.subheadline.weight(.bold)).foregroundStyle(Theme.dim)
                }
                .glassCard()
            }
            .buttonStyle(.plain)

            // Pro
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    Text("Pro").font(.headline)
                    ProBadge()
                    Spacer()
                    if store.ownsPro {
                        Label("Unlocked", systemImage: "checkmark.seal.fill").font(.caption.weight(.bold)).foregroundStyle(Theme.good)
                    }
                }
                Text("The full coach: why it won, analysis lenses, object detection, a coaching plan and an edit toward the winner.")
                    .font(.footnote)
                    .foregroundStyle(Theme.dim)
                    .fixedSize(horizontal: false, vertical: true)
                PurchaseButton { unlocked in if unlocked { onChoose(.pro) } }
            }
            .padding(16)
            .background(.ultraThinMaterial.opacity(0.6), in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .background(Theme.violet.opacity(0.12), in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Theme.cta, lineWidth: 1.5))
        }
    }
}

/// Shows the splash over the app until a tier is chosen, then opens onto the home screen.
struct RootView: View {
    @StateObject private var store = Store()
    @StateObject private var history = History()
    @State private var tier: Tier = .free
    @State private var splash = true

    init() {
        #if DEBUG
        // screenshot/diagnostic launches skip the splash; `-free` picks the free tier, otherwise Pro
        let args = ProcessInfo.processInfo.arguments
        if args.contains("-autosample") || args.contains("-diag") || args.contains("-nosplash") {
            _splash = State(initialValue: false)
            _tier = State(initialValue: args.contains("-free") ? .free : .pro)
        }
        // `-exporticon <path>` writes the 1024px app icon PNG and quits
        if let i = args.firstIndex(of: "-exporticon"), i + 1 < args.count {
            let r = ImageRenderer(content: AppIconArt(size: 1024).environment(\.colorScheme, .dark))
            r.scale = 1
            r.isOpaque = true
            if let img = r.uiImage, let png = img.pngData() { try? png.write(to: URL(fileURLWithPath: args[i + 1])) }
            exit(0)
        }
        #endif
    }

    var body: some View {
        ZStack {
            ContentView(tier: $tier)
            if splash {
                SplashView { chosen in
                    tier = chosen
                    withAnimation(.easeInOut(duration: 0.55)) { splash = false }
                }
                .transition(.asymmetric(insertion: .identity, removal: .opacity.combined(with: .scale(scale: 1.25))))
                .zIndex(1)
            }
        }
        .environmentObject(store)
        .environmentObject(history)
    }
}

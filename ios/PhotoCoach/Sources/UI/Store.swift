import StoreKit
import SwiftUI

/// Which experience the user is in this session.
enum Tier { case free, pro }

/// ApertureIQ Pro: a one-time, non-consumable unlock sold through StoreKit 2.
/// Ownership comes from Apple's signed transactions, so it survives reinstalls and syncs across the
/// user's devices; "Restore purchases" just forces a re-sync with the App Store.
@MainActor
final class Store: ObservableObject {
    static let proID = "com.mosaiclens.apertureiq.pro"

    @Published private(set) var product: Product?
    @Published private(set) var ownsPro = false
    @Published private(set) var busy = false
    @Published private(set) var loaded = false
    @Published var message: String?

    private var updates: Task<Void, Never>?

    init() {
        #if DEBUG
        // `-pro` / `-free` force the entitlement for screenshots and testing
        let args = ProcessInfo.processInfo.arguments
        if args.contains("-pro") { ownsPro = true }
        #endif
        updates = Task { [weak self] in
            for await result in Transaction.updates {
                if case .verified(let t) = result { await t.finish() }
                await self?.refresh()
            }
        }
        Task { await load() }
    }

    deinit { updates?.cancel() }

    /// "Unlock Pro · $4.99", or just "Unlock Pro" until the price has loaded.
    var unlockTitle: String { product.map { "Unlock Pro · \($0.displayPrice)" } ?? "Unlock Pro" }

    func load() async {
        do {
            product = try await Product.products(for: [Self.proID]).first
        } catch {
            product = nil
        }
        await refresh()
        loaded = true
    }

    /// Re-reads the user's entitlements from the signed transaction history.
    func refresh() async {
        var owns = false
        for await result in Transaction.currentEntitlements {
            if case .verified(let t) = result, t.productID == Self.proID, t.revocationDate == nil { owns = true }
        }
        #if DEBUG
        if ProcessInfo.processInfo.arguments.contains("-pro") { owns = true }
        #endif
        ownsPro = owns
    }

    /// Returns true once Pro is owned.
    @discardableResult
    func purchase() async -> Bool {
        guard let product else {
            message = "The App Store isn't reachable right now. Try again in a moment."
            return false
        }
        busy = true
        defer { busy = false }
        do {
            switch try await product.purchase() {
            case .success(let result):
                guard case .verified(let t) = result else {
                    message = "Apple couldn't verify that purchase."
                    return false
                }
                await t.finish()
                await refresh()
                Theme.haptic(.success)
            case .pending:
                message = "Purchase pending approval (Ask to Buy). Pro unlocks as soon as it's approved."
            case .userCancelled:
                break
            @unknown default:
                break
            }
        } catch {
            message = error.localizedDescription
        }
        return ownsPro
    }

    func restore() async {
        busy = true
        defer { busy = false }
        do {
            try await AppStore.sync()
            await refresh()
            message = ownsPro ? nil : "No Pro purchase found for this Apple ID."
            if ownsPro { Theme.haptic(.success) }
        } catch {
            message = error.localizedDescription
        }
    }
}

/// What Pro adds, shown on the splash card and the paywall.
enum ProFeatures {
    static let list: [(symbol: String, title: String)] = [
        ("text.magnifyingglass", "Why it won, in plain words"),
        ("sparkle.magnifyingglass", "How the AI check decided, with the evidence"),
        ("eye", "Lenses: clipping, focus, composition, AI attention"),
        ("viewfinder", "Object detection: subject vs. distractions"),
        ("graduationcap.fill", "A step-by-step coaching plan"),
        ("slider.horizontal.3", "A one-tap edit toward the winner"),
        ("chart.bar.xaxis", "Full scores, histograms, palettes and EXIF"),
    ]
}

/// Upgrade sheet, opened from inside the free experience.
struct PaywallView: View {
    @EnvironmentObject var store: Store
    @Environment(\.dismiss) private var dismiss
    var onUnlocked: () -> Void = {}

    var body: some View {
        ZStack {
            AmbientBackground(tintA: Theme.violet, tintB: Theme.b)
            ScrollView {
                VStack(spacing: 24) {
                    ZStack {
                        Circle().stroke(.white.opacity(0.8), lineWidth: 3).frame(width: 84, height: 84)
                        IrisShape(open: 0.55).fill(Theme.cta).frame(width: 76, height: 76)
                    }
                    .padding(.top, 36)
                    VStack(spacing: 8) {
                        HStack(spacing: 8) {
                            Wordmark(size: 32)
                            ProBadge()
                        }
                        Text("Know exactly why one shot beats the other - and how to fix the other one.")
                            .font(.subheadline)
                            .foregroundStyle(Theme.dim)
                            .multilineTextAlignment(.center)
                    }
                    VStack(alignment: .leading, spacing: 14) {
                        ForEach(ProFeatures.list, id: \.title) { f in
                            HStack(spacing: 12) {
                                Image(systemName: f.symbol).foregroundStyle(Theme.cta).frame(width: 26)
                                Text(f.title).font(.subheadline.weight(.medium))
                            }
                        }
                    }
                    .glassCard(padding: 20)
                    PurchaseButton { unlocked in
                        if unlocked { dismiss(); onUnlocked() }
                    }
                    Text("One-time purchase. No subscription. Everything still runs on your iPhone.")
                        .font(.caption)
                        .foregroundStyle(Theme.dim)
                        .multilineTextAlignment(.center)
                }
                .padding(.horizontal, 22)
                .padding(.bottom, 30)
            }
            .scrollIndicators(.hidden)
        }
        .overlay(alignment: .topTrailing) {
            Button { dismiss() } label: {
                Image(systemName: "xmark").font(.subheadline.weight(.bold))
                    .frame(width: 34, height: 34).background(.ultraThinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
            .padding(16)
        }
        .presentationDragIndicator(.visible)
    }
}

/// "Unlock Pro · $4.99" plus restore and any store message.  Calls back with whether Pro is now owned.
struct PurchaseButton: View {
    @EnvironmentObject var store: Store
    let done: (Bool) -> Void

    var body: some View {
        VStack(spacing: 12) {
            Button {
                if store.ownsPro { done(true); return }
                Task { done(await store.purchase()) }
            } label: {
                HStack(spacing: 10) {
                    if store.busy { ProgressView().tint(.white) } else { Image(systemName: store.ownsPro ? "checkmark.seal.fill" : "lock.open.fill") }
                    Text(store.ownsPro ? "Continue with Pro" : store.unlockTitle).font(.headline.weight(.bold))
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(Theme.cta, in: Capsule())
                .foregroundStyle(.white)
                .opacity(store.ownsPro || store.product != nil ? 1 : 0.45)
            }
            .buttonStyle(.plain)
            .disabled(store.busy || (!store.ownsPro && store.product == nil && store.loaded))
            if !store.ownsPro {
                Button("Restore purchases") { Task { await store.restore(); if store.ownsPro { done(true) } } }
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(Theme.dim)
                    .disabled(store.busy)
                if store.loaded && store.product == nil {
                    Text("The App Store isn't available right now.").font(.caption).foregroundStyle(Theme.dim)
                }
            }
            if let m = store.message {
                Text(m).font(.caption).foregroundStyle(Theme.gold).multilineTextAlignment(.center)
            }
        }
    }
}

struct ProBadge: View {
    var body: some View {
        Text("PRO")
            .font(.system(size: 11, weight: .black, design: .rounded))
            .kerning(1)
            .padding(.horizontal, 7).padding(.vertical, 3)
            .background(Theme.cta, in: Capsule())
            .foregroundStyle(.white)
    }
}

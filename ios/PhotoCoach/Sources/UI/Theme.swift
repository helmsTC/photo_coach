import SwiftUI
import UIKit

enum Theme {
    static let a = Color(red: 0.25, green: 0.58, blue: 1.0)          // shot A: electric blue
    static let b = Color(red: 1.0, green: 0.48, blue: 0.22)          // shot B: ember orange
    static let gold = Color(red: 1.0, green: 0.80, blue: 0.25)
    static let good = Color(red: 0.25, green: 0.86, blue: 0.62)
    static let bad = Color(red: 1.0, green: 0.36, blue: 0.36)
    static let violet = Color(red: 0.58, green: 0.42, blue: 1.0)
    static let bg = Color(red: 0.035, green: 0.04, blue: 0.07)
    static let card = Color.white.opacity(0.055)
    static let stroke = Color.white.opacity(0.09)
    static let dim = Color.white.opacity(0.55)

    static let cta = LinearGradient(colors: [Color(red: 0.45, green: 0.36, blue: 1.0), Color(red: 0.95, green: 0.33, blue: 0.62), Color(red: 1.0, green: 0.55, blue: 0.25)],
                                    startPoint: .leading, endPoint: .trailing)

    static func tint(_ i: Int) -> Color { i == 0 ? a : b }

    static func fixStyle(_ fix: String) -> (text: String, color: Color, symbol: String) {
        switch fix {
        case "edit": return ("Fix in the edit", good, "slider.horizontal.3")
        case "reshoot": return ("Needs a reshoot", b, "camera.fill")
        default: return ("Shoot + edit", violet, "sparkles")
        }
    }

    static func itemSymbol(_ id: String) -> String {
        switch id {
        case "highlights": return "sun.max.fill"
        case "shadows": return "moon.fill"
        case "brightness": return "plusminus.circle.fill"
        case "contrast": return "circle.lefthalf.filled"
        case "lighting": return "sun.horizon.fill"
        case "muted", "garish", "palette": return "paintpalette.fill"
        case "cast": return "thermometer.medium"
        case "sharpness": return "scope"
        case "noise": return "aqi.medium"
        case "dof": return "camera.aperture"
        case "level": return "level.fill"
        case "distraction": return "exclamationmark.triangle.fill"
        case "framing": return "crop"
        case "composition": return "squareshape.split.3x3"
        case "content": return "star.fill"
        default: return "lightbulb.fill"
        }
    }

    static func haptic(_ type: UINotificationFeedbackGenerator.FeedbackType) {
        UINotificationFeedbackGenerator().notificationOccurred(type)
    }

    static func tick() { UISelectionFeedbackGenerator().selectionChanged() }
}

/// Slow-drifting coloured glows behind everything.
struct AmbientBackground: View {
    var tintA: Color = Theme.a
    var tintB: Color = Theme.b

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30)) { tl in
            let t = tl.date.timeIntervalSinceReferenceDate
            GeometryReader { geo in
                let w = geo.size.width, h = geo.size.height
                ZStack {
                    Theme.bg
                    Circle().fill(tintA.opacity(0.32))
                        .frame(width: w * 1.1).blur(radius: 90)
                        .offset(x: -w * 0.35 + CGFloat(sin(t * 0.21)) * 40, y: -h * 0.3 + CGFloat(cos(t * 0.17)) * 50)
                    Circle().fill(tintB.opacity(0.26))
                        .frame(width: w * 1.0).blur(radius: 90)
                        .offset(x: w * 0.4 + CGFloat(cos(t * 0.19)) * 40, y: h * 0.05 + CGFloat(sin(t * 0.23)) * 60)
                    Circle().fill(Theme.violet.opacity(0.18))
                        .frame(width: w * 0.9).blur(radius: 100)
                        .offset(x: CGFloat(sin(t * 0.13)) * 60, y: h * 0.45)
                }
            }
        }
        .ignoresSafeArea()
    }
}

struct GlassCard: ViewModifier {
    var padding: CGFloat = 16
    func body(content: Content) -> some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.ultraThinMaterial.opacity(0.6), in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .background(Theme.card, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Theme.stroke))
    }
}

extension View {
    func glassCard(padding: CGFloat = 16) -> some View { modifier(GlassCard(padding: padding)) }
}

struct SectionHeader: View {
    let title: String
    var subtitle: String?
    var symbol: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                if let symbol { Image(systemName: symbol).foregroundStyle(Theme.cta) }
                Text(title).font(.title3.weight(.bold))
            }
            if let subtitle { Text(subtitle).font(.subheadline).foregroundStyle(Theme.dim) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct ShotTag: View {
    let index: Int
    var small = false
    var body: some View {
        Text(index == 0 ? "A" : "B")
            .font(.system(size: small ? 11 : 13, weight: .heavy, design: .rounded))
            .foregroundStyle(.white)
            .frame(width: small ? 20 : 26, height: small ? 20 : 26)
            .background(Theme.tint(index), in: Circle())
            .shadow(color: Theme.tint(index).opacity(0.6), radius: 6)
    }
}

struct Pill: View {
    let text: String
    var color: Color = .white
    var symbol: String?
    var body: some View {
        HStack(spacing: 5) {
            if let symbol { Image(systemName: symbol).font(.caption2.weight(.bold)) }
            Text(text).font(.caption.weight(.semibold))
        }
        .padding(.horizontal, 10).padding(.vertical, 5)
        .foregroundStyle(color)
        .background(color.opacity(0.14), in: Capsule())
        .overlay(Capsule().stroke(color.opacity(0.3)))
    }
}

/// Animated ring that fills to a 0...100 score, with the number counting up.
struct ScoreRing: View {
    let value: Double
    let color: Color
    var size: CGFloat = 104
    var line: CGFloat = 10
    var label: String?
    @State private var shown = 0.0

    var body: some View {
        ZStack {
            Circle().stroke(Color.white.opacity(0.08), lineWidth: line)
            Circle()
                .trim(from: 0, to: shown / 100)
                .stroke(AngularGradient(colors: [color.opacity(0.5), color], center: .center, startAngle: .degrees(0), endAngle: .degrees(360 * max(shown, 1) / 100)),
                        style: StrokeStyle(lineWidth: line, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .shadow(color: color.opacity(0.55), radius: 8)
            VStack(spacing: 0) {
                Text("\(Int(shown.rounded()))")
                    .font(.system(size: size * 0.32, weight: .heavy, design: .rounded))
                    .contentTransition(.numericText(value: shown))
                    .monospacedDigit()
                if let label { Text(label).font(.system(size: size * 0.1, weight: .semibold)).foregroundStyle(Theme.dim) }
            }
        }
        .frame(width: size, height: size)
        .onAppear {
            withAnimation(.spring(response: 1.4, dampingFraction: 0.85).delay(0.15)) { shown = value }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(label ?? "Score") \(Int(value.rounded())) out of 100")
    }
}

/// Thin horizontal meter, 0...100.
struct Meter: View {
    let value: Double
    let color: Color
    var height: CGFloat = 6
    @State private var shown = 0.0

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.08))
                Capsule().fill(color).frame(width: max(height, geo.size.width * shown / 100))
            }
        }
        .frame(height: height)
        .onAppear { withAnimation(.easeOut(duration: 1.0).delay(0.3)) { shown = max(0, min(100, value)) } }
    }
}

struct Bullet: View {
    let text: String
    var color: Color = Theme.dim
    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 9) {
            Circle().fill(color).frame(width: 5, height: 5).offset(y: -2)
            Text(text).font(.subheadline).fixedSize(horizontal: false, vertical: true)
        }
    }
}

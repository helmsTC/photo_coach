import CryptoKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

/// One saved comparison: two compact photo copies on disk plus the verdict.
struct HistoryEntry: Codable, Identifiable, Equatable {
    let id: UUID
    var date: Date
    var winner: Int
    var confidence: Double
    /// Last analysed with the full Pro pipeline.
    var pro: Bool
    /// Fingerprint of the two original files (in order), so comparing the same pair again updates its entry.
    var key: String?
    /// Each shot's AI check from the original files (the saved copies lose their labels).
    var authenticity: [Authenticity]?

    static func key(_ a: Data, _ b: Data) -> String {
        var h = SHA256()
        h.update(data: a)
        h.update(data: b)
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// Recent comparisons, kept on this device only (Application Support/History).
/// Each entry keeps downscaled copies of both photos with their EXIF and colour profile, so reopening one
/// simply re-runs the analysis; nothing heavier than the photos and the verdict is stored.
@MainActor
final class History: ObservableObject {
    static func limit(ownsPro: Bool) -> Int { ownsPro ? 10 : 3 }
    /// The network analyses at 1024px; 2048 leaves headroom for the Pro viewer.  HEIC at 0.8 is about 40% smaller
    /// than the JPEG q0.95 copies earlier builds kept, with no larger change to the verdict.
    nonisolated static let storedLongSide = 2048
    nonisolated static let storedQuality = 0.8

    @Published private(set) var entries: [HistoryEntry] = []
    private var thumbCache: [UUID: [UIImage]] = [:]
    private let root: URL

    init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        root = base.appendingPathComponent("History", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        // personal photos that only serve as a short "recent" list: keep them out of iCloud/computer backups
        var url = root
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
        if let d = try? Data(contentsOf: indexURL), let e = try? JSONDecoder().decode([HistoryEntry].self, from: d) {
            entries = e.filter { photoURL($0.id, "a") != nil }
        }
    }

    private var indexURL: URL { root.appendingPathComponent("index.json") }
    private func dir(_ id: UUID) -> URL { root.appendingPathComponent(id.uuidString, isDirectory: true) }
    /// a.heic / b.heic, or .jpg in entries saved by earlier builds.
    private func photoURL(_ id: UUID, _ name: String) -> URL? {
        ["heic", "jpg"].map { dir(id).appendingPathComponent("\(name).\($0)") }.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    func visible(ownsPro: Bool) -> [HistoryEntry] { Array(entries.prefix(Self.limit(ownsPro: ownsPro))) }

    /// Saves a finished comparison, or refreshes `id` if it is one being reopened.  Returns the entry's id.
    func record(id: UUID?, a: Data, b: Data, winner: Int, confidence: Double, pro: Bool, ownsPro: Bool, authenticity: [Authenticity]) async -> UUID? {
        let key = HistoryEntry.key(a, b)
        if let i = entries.firstIndex(where: { $0.id == id || $0.key == key }) {
            let id = entries[i].id
            var e = entries.remove(at: i)
            e.date = Date(); e.winner = winner; e.confidence = confidence; e.pro = e.pro || pro
            if e.authenticity == nil { e.authenticity = authenticity }
            entries.insert(e, at: 0)
            save()
            return id
        }
        let id = UUID()
        let folder = dir(id)
        let ok = await Task.detached(priority: .utility) { () -> Bool in
            func stored(_ d: Data) -> (Data, String)? {
                if let h = Self.compact(d, maxSide: Self.storedLongSide, type: .heic, quality: Self.storedQuality) { return (h, "heic") }
                return Self.compact(d, maxSide: Self.storedLongSide).map { ($0, "jpg") }
            }
            guard let (ca, ea) = stored(a), let (cb, eb) = stored(b),
                  let ta = Self.compact(a, maxSide: 300, keepMetadata: false), let tb = Self.compact(b, maxSide: 300, keepMetadata: false)
            else { return false }
            do {
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                for (name, d) in [("a.\(ea)", ca), ("b.\(eb)", cb), ("ta.jpg", ta), ("tb.jpg", tb)] {
                    try d.write(to: folder.appendingPathComponent(name), options: [.atomic, .completeFileProtection])
                }
                return true
            } catch {
                try? FileManager.default.removeItem(at: folder)
                return false
            }
        }.value
        guard ok else { return nil }
        entries.insert(HistoryEntry(id: id, date: Date(), winner: winner, confidence: confidence, pro: pro, key: key, authenticity: authenticity), at: 0)
        // drop the oldest beyond this user's allowance
        let keep = Self.limit(ownsPro: ownsPro)
        while entries.count > keep { remove(entries.removeLast().id) }
        save()
        return id
    }

    func photos(_ e: HistoryEntry) -> (Data, Data)? {
        guard let ua = photoURL(e.id, "a"), let ub = photoURL(e.id, "b"),
              let a = try? Data(contentsOf: ua), let b = try? Data(contentsOf: ub) else { return nil }
        return (a, b)
    }

    func thumbs(_ e: HistoryEntry) -> [UIImage] {
        if let t = thumbCache[e.id] { return t }
        let t = ["ta.jpg", "tb.jpg"].compactMap { UIImage(contentsOfFile: dir(e.id).appendingPathComponent($0).path) }
        if t.count == 2 { thumbCache[e.id] = t }
        return t
    }

    func delete(_ e: HistoryEntry) {
        entries.removeAll { $0.id == e.id }
        remove(e.id)
        save()
    }

    private func remove(_ id: UUID) {
        thumbCache[id] = nil
        try? FileManager.default.removeItem(at: dir(id))
    }

    private func save() {
        if let d = try? JSONEncoder().encode(entries) { try? d.write(to: indexURL, options: .atomic) }
    }

    /// Downscaled JPEG copy with orientation baked in.  The pixels stay in the file's own colour space
    /// (its ICC profile is carried over) so the re-analysis sees the same values as the original;
    /// EXIF is kept for the Pro details panel.
    nonisolated static func compact(_ data: Data, maxSide: Int, keepMetadata: Bool = true, type: UTType = .jpeg, quality: Double? = nil) -> Data? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxSide,
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }
        var props: [CFString: Any] = [:]
        if keepMetadata, let p = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any] {
            for k in [kCGImagePropertyExifDictionary, kCGImagePropertyTIFFDictionary, kCGImagePropertyGPSDictionary] {
                if let v = p[k] { props[k] = v }
            }
            if var tiff = props[kCGImagePropertyTIFFDictionary] as? [CFString: Any] {
                tiff[kCGImagePropertyTIFFOrientation] = 1
                props[kCGImagePropertyTIFFDictionary] = tiff
            }
            props[kCGImagePropertyGPSDictionary] = nil       // no location in stored copies
        }
        props[kCGImagePropertyOrientation] = 1
        props[kCGImageDestinationLossyCompressionQuality] = quality ?? (keepMetadata ? 0.95 : 0.8)
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, type.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, props as CFDictionary)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    /// A word for how decisive a verdict was (the free tier shows no numbers).
    static func strength(_ confidence: Double) -> String {
        confidence < 0.56 ? "Close call" : (confidence < 0.70 ? "Solid edge" : "Clear winner")
    }
}

/// A saved comparison on the home screen.
struct HistoryCard: View {
    let entry: HistoryEntry
    let thumbs: [UIImage]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack {
                ForEach(Array(thumbs.enumerated()), id: \.offset) { i, img in
                    let won = i == entry.winner
                    Image(uiImage: img).resizable().scaledToFill()
                        .frame(width: 70, height: 88)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(won ? Theme.gold : Theme.tint(i), lineWidth: won ? 2 : 1))
                        .saturation(won ? 1 : 0.4)
                        .overlay(alignment: .top) {
                            if won {
                                Image(systemName: "crown.fill").font(.system(size: 15)).foregroundStyle(Theme.gold)
                                    .shadow(color: .black.opacity(0.7), radius: 3).offset(y: -10)
                            }
                        }
                        .rotationEffect(.degrees(i == 0 ? -5 : 5))
                        .offset(x: i == 0 ? -20 : 20)
                        .zIndex(won ? 1 : 0)
                }
            }
            .frame(width: 130, height: 100)
            HStack(spacing: 6) {
                Text("Shot \(entry.winner == 0 ? "A" : "B") won").font(.footnote.weight(.semibold))
                if entry.pro { ProBadge().scaleEffect(0.8) }
            }
            Text("\(History.strength(entry.confidence)) · \(entry.date.formatted(.relative(presentation: .named, unitsStyle: .abbreviated)))")
                .font(.caption2).foregroundStyle(Theme.dim).lineLimit(1)
        }
        .padding(12)
        .frame(width: 154, alignment: .leading)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(Theme.stroke))
    }
}

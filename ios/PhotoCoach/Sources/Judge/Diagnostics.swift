import CoreML
import UIKit
import Vision
import UniformTypeIdentifiers

/// Built-in sample pairs (bundled demo photos).
enum Samples {
    struct Pair: Identifiable {
        let title: String
        let subtitle: String
        let a: String
        let b: String
        var id: String { title }
    }

    static let pairs: [Pair] = [
        Pair(title: "Lakeside château", subtitle: "Same scene, two framings", a: "sample_louise_12", b: "sample_louise_13"),
        Pair(title: "Boathouse", subtitle: "Same subject, different light", a: "sample_louise_07", b: "sample_louise_09"),
        Pair(title: "Moraine lookout", subtitle: "Same view, two exposures", a: "sample_moraine_03", b: "sample_moraine_04"),
        Pair(title: "Busy table", subtitle: "Clutter vs a clean scene", a: "sample_zclutter_table", b: "parity_louise_11"),
    ]

    static func data(_ name: String) -> Data? {
        Bundle.main.url(forResource: name, withExtension: "jpg").flatMap { try? Data(contentsOf: $0) }
    }
}

/// Compares this device's results for two bundled photos with the reference PyTorch outputs
/// (Resources/parity.json): the check that Core ML conversion and Swift preprocessing match training.
enum ParityCheck {
    struct Expected: Decodable {
        let file: String
        let stdSize: [Int]
        let globalEmbedHead: [Double]
        let signals: [String: Double]
        let scores: [String: Double]
        let percentiles: [String: Double]
    }

    /// Calls `log(text, ok)` for each line; ok == nil is informational.
    static func run(_ log: @escaping (String, Bool?) -> Void) async {
        do {
            guard let url = Bundle.main.url(forResource: "parity", withExtension: "json") else {
                throw JudgeError.missingResource("parity.json")
            }
            let expected = try JSONDecoder().decode([Expected].self, from: Data(contentsOf: url))
            let info = try await PhotoJudge.shared.loadIfNeeded()
            for e in expected {
                let name = (e.file as NSString).deletingPathExtension
                let ext = (e.file as NSString).pathExtension
                guard let f = Bundle.main.url(forResource: name, withExtension: ext) else { throw JudgeError.missingResource(e.file) }
                let t0 = Date()
                let p = try await PhotoJudge.shared.score(data: Data(contentsOf: f), name: e.file)
                log("\(e.file)  (\(String(format: "%.2f", Date().timeIntervalSince(t0))) s)", nil)

                let embDiff = zip(p.globalEmbedHead, e.globalEmbedHead).map { abs($0 - $1) }.max() ?? 0
                let embScale = e.globalEmbedHead.map { abs($0) }.max() ?? 1
                log(String(format: "  encoder output: max diff %.3f (values up to %.2f)", embDiff, embScale), embDiff <= 0.15 * max(embScale, 1e-6))

                func worst(_ keys: [String]) -> (String, Double) {
                    var w = ("", 0.0)
                    for k in keys {
                        guard let want = e.signals[k], let got = p.signals[k] else { continue }
                        let d = abs(got - want) / max(abs(want), 0.05)
                        if d > w.1 { w = (k, d) }
                    }
                    return w
                }
                let used = worst(info.signalsUsed)
                log(String(format: "  model-input signals: worst relative diff %.0f%% (%@)", used.1 * 100, used.0), used.1 <= 0.15)
                let extraKeys = e.signals.keys.filter { !info.signalsUsed.contains($0) }.sorted()
                let extra = worst(extraKeys)
                log(String(format: "  coaching signals: worst relative diff %.0f%% (%@)", extra.1 * 100, extra.0), extra.1 <= 0.25)
                for k in extraKeys {
                    if let want = e.signals[k], let got = p.signals[k] {
                        log(String(format: "    %@: %.4f here, %.4f reference", k, got, want), nil)
                    }
                }

                var w = ("", 0.0)
                for (i, h) in info.heads.enumerated() {
                    guard let want = e.percentiles[h.key] else { continue }
                    let d = abs(p.percentile[i] - want)
                    if d > w.1 { w = (h.label, d) }
                }
                log(String(format: "  attribute percentiles: worst diff %.1f points (%@)", w.1, w.0), w.1 <= 3)
                for key in ["aesthetic", "technical"] {
                    if let i = info.index(of: key), let want = e.percentiles[key] {
                        log(String(format: "  %@: %.1f here, %.1f reference", key, p.percentile[i], want), abs(p.percentile[i] - want) <= 3)
                    }
                }
            }
        } catch {
            log("Failed: \(error.localizedDescription)", false)
        }
    }
}

/// Headless checks: launch with `-diag` to print results to the console and exit.
enum Diagnostics {
    static var requested: Bool { ProcessInfo.processInfo.arguments.contains("-diag") }

    static func say(_ s: String) { print("DIAG: " + s); fflush(stdout) }

    static func probe() async {
        guard let d = Samples.data("parity_louise_03"), let std = try? ImagePrep.standardImage(from: d, longSide: 1024).0 else { return }
        let v = ImagePrep.views(std: std, size: 256)
        for units in [MLComputeUnits.all, .cpuOnly] {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = units
            guard let url = Bundle.main.url(forResource: "Encoder", withExtension: "mlmodelc"),
                  let m = try? MLModel(contentsOf: url, configuration: cfg) else { say("probe: no encoder"); continue }
            say("probe: encoder inputs \(m.modelDescription.inputDescriptionsByName.mapValues { "\($0.type.rawValue) \($0.imageConstraint.map { "\($0.pixelsWide)x\($0.pixelsHigh) fmt \($0.pixelFormatType)" } ?? "")" })")
            do {
                let pb = try ImagePrep.pixelBuffer(v[0])
                let out = try await m.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pb)]))
                let arr = out.featureValue(for: "embedding")!.multiArrayValue!
                let vals = (0..<arr.count).map { arr[$0].floatValue }
                say("probe units=\(units.rawValue) single: count \(arr.count) type \(arr.dataType.rawValue) shape \(arr.shape) first \(vals.prefix(4)) norm \(sqrt(vals.map { $0 * $0 }.reduce(0, +)))")
            } catch { say("probe units=\(units.rawValue) error \(error)") }
        }
        for name in ["parity_louise_03", "parity_louise_11", "sample_moraine_03", "sample_louise_09", "sample_louise_12"] {
            if let dd = Samples.data(name), let im = try? ImagePrep.standardImage(from: dd, longSide: 1024) {
                let t0 = Signals.horizon(im.0)
                let r = Diagnostics.rotated(im.1, degrees: 4)
                let rs = (try? ImagePrep.standardImage(from: r.jpegData(compressionQuality: 0.9)!, longSide: 1024).0).map { Signals.horizon($0) } ?? [:]
                say(String(format: "probe horizon %@: %.2f deg conf %.2f y %.2f | rotated +4 cw: %.2f deg conf %.2f", name,
                           t0["tilt_deg"]!, t0["tilt_conf"]!, t0["horizon_y"]!, rs["tilt_deg"] ?? -99, rs["tilt_conf"] ?? -1))
            }
        }
        if let dd = Samples.data("sample_louise_12"), let im = try? ImagePrep.standardImage(from: dd, longSide: 1024), let cg = Imaging.cgImage(im.0) {
            let h = VNImageRequestHandler(cgImage: cg, options: [:])
            let ob = VNGenerateObjectnessBasedSaliencyImageRequest().cpuOnSimulator()
            do { try h.perform([ob]); say("probe objectness: \(ob.results?.first?.salientObjects?.map { $0.boundingBox } ?? [])") } catch { say("probe objectness error \(error)") }
            let pp = VNGeneratePersonInstanceMaskRequest().cpuOnSimulator()
            do { try h.perform([pp]); say("probe person: instances \(pp.results?.first?.allInstances.count ?? -1)") } catch { say("probe person error \(error)") }
            let segs = Objects.detect(im.0)
            say("probe detect: " + segs.map { "\($0.label) \(String(format: "%.3f", $0.area))" }.joined(separator: ", "))
        }
        if let cg = Imaging.cgImage(std) {
            let h = VNImageRequestHandler(cgImage: cg, options: [:])
            let reqs: [(String, VNRequest)] = [("horizon", VNDetectHorizonRequest().cpuOnSimulator()), ("saliency", VNGenerateAttentionBasedSaliencyImageRequest().cpuOnSimulator()),
                                               ("foreground", VNGenerateForegroundInstanceMaskRequest().cpuOnSimulator()), ("classify", VNClassifyImageRequest().cpuOnSimulator())]
            for (n, r) in reqs {
                do { try h.perform([r]); say("probe vision \(n): \(r.results?.count ?? -1) results \(String(describing: r.results?.first).prefix(160))") }
                catch { say("probe vision \(n) error: \(error)") }
            }
        }
    }

    static func run() async {
        if ProcessInfo.processInfo.arguments.contains("-probe") { await probe() }
        if ProcessInfo.processInfo.arguments.contains("-probe") { say("done"); exit(0) }
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-aicheck"), i + 1 < args.count {
            // run the app's AI check on every image in a folder: "p  verdict  label  file"
            let dir = URL(fileURLWithPath: args[i + 1])
            let files = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            let t0 = Date()
            for f in files {
                guard let d = try? Data(contentsOf: f), let ui = UIImage(data: d) else { continue }
                let a = await PhotoJudge.shared.authenticity(data: d, image: ui)
                let lab = a.label.map { "\($0.kind):\($0.tool ?? "-")@\($0.source)" } ?? "none"
                var saved = ""
                if args.contains("-saved") {
                    // the same check on the copies History keeps
                    for (t, q) in [(UTType.jpeg, 0.95), (UTType.heic, 0.8)] {
                        if let c = History.compact(d, maxSide: History.storedLongSide, type: t, quality: q), let cu = UIImage(data: c) {
                            let ca = await PhotoJudge.shared.authenticity(data: c, image: cu)
                            saved += String(format: " %6.3f %@", ca.aiProbability ?? -1, "\(ca.verdict)")
                        }
                    }
                }
                say(String(format: "%6.3f  %@  %@  %@ |%@", a.aiProbability ?? -1, "\(a.verdict)", lab, f.lastPathComponent, saved))
            }
            say(String(format: "aicheck %d files in %.1fs", files.count, Date().timeIntervalSince(t0)))
            say("done"); exit(0)
        }
        if ProcessInfo.processInfo.arguments.contains("-historycheck") {
            // saved copies must reach the network with the same verdict as the originals
            for pair in Samples.pairs {
                guard let a = Samples.data(pair.a), let b = Samples.data(pair.b),
                      let v0 = try? await PhotoJudge.shared.quickVerdict(dataA: a, dataB: b) else { continue }
                func ai(_ v: QuickVerdict) -> String { v.authenticity.map { String(format: "%.3f", $0.aiProbability ?? -1) }.joined(separator: "/") }
                say(String(format: "history %@: original pA %.4f ai %@ | %d+%d KB", pair.title, v0.pA, ai(v0), a.count / 1024, b.count / 1024))
                for (name, t, q) in [("jpeg95", UTType.jpeg, 0.95), ("heic90", UTType.heic, 0.9), ("heic80", UTType.heic, 0.8)] {
                    guard let ca = History.compact(a, maxSide: History.storedLongSide, type: t, quality: q),
                          let cb = History.compact(b, maxSide: History.storedLongSide, type: t, quality: q),
                          let v1 = try? await PhotoJudge.shared.quickVerdict(dataA: ca, dataB: cb) else { say("  \(name) failed"); continue }
                    let exif = ExifInfo.read(ca).isEmpty == ExifInfo.read(a).isEmpty
                    let cs = UIImage(data: ca)?.cgImage?.colorSpace?.name as String? ?? "?"
                    say(String(format: "  %@: pA %.4f (d %.4f) ai %@ | %d+%d KB | %@ | exif %@", name, v1.pA, v1.pA - v0.pA, ai(v1), ca.count / 1024, cb.count / 1024, cs, exif ? "kept" : "LOST"))
                }
            }
            // a big photo is downscaled to 2048 on the long side
            if let d = Samples.data("parity_louise_11"), let ui = UIImage(data: d) {
                let big = UIGraphicsImageRenderer(size: CGSize(width: 4000, height: 3000)).jpegData(withCompressionQuality: 0.9) { _ in ui.draw(in: CGRect(x: 0, y: 0, width: 4000, height: 3000)) }
                if let c = History.compact(big, maxSide: History.storedLongSide), let u = UIImage(data: c) {
                    say("history downscale 4000x3000 -> \(Int(u.size.width))x\(Int(u.size.height)), \(c.count / 1024) KB")
                }
            }
            say("done"); exit(0)
        }
        say("parity")
        await ParityCheck.run { text, ok in say((ok == nil ? "   " : (ok! ? "OK " : "BAD")) + " " + text) }

        // horizon direction: rotate a photo clockwise and the tilt should grow positive
        if let d = Samples.data("parity_louise_03"), let ui = UIImage(data: d), let std = try? ImagePrep.standardImage(from: d, longSide: 1024).0 {
            let base = Signals.horizon(std)
            let rot = rotated(ui, degrees: 5)
            if let rd = rot.jpegData(compressionQuality: 0.9), let rs = try? ImagePrep.standardImage(from: rd, longSide: 1024).0 {
                let t = Signals.horizon(rs)
                say(String(format: "horizon: original %.2f deg (conf %.2f), rotated 5 deg clockwise -> %.2f deg (conf %.2f)",
                           base["tilt_deg"] ?? 0, base["tilt_conf"] ?? 0, t["tilt_deg"] ?? 0, t["tilt_conf"] ?? 0))
            }
            // alignment against a known crop: the central 70% should come back as ~1.43x zoom
            let cw = Int(Double(std.width) * 0.7), ch = Int(Double(std.height) * 0.7)
            var crop = [UInt8](repeating: 0, count: cw * ch * 3)
            let x0 = (std.width - cw) / 2, y0 = (std.height - ch) / 2
            for y in 0..<ch { for x in 0..<cw { for k in 0..<3 { crop[(y * cw + x) * 3 + k] = std.data[((y + y0) * std.width + x + x0) * 3 + k] } } }
            let c = RGBImage(width: cw, height: ch, data: crop)
            let al = SceneMatch.align(std, c)
            if al.aligned {
                let fr = SceneMatch.framing(al.quadBinA, myAspect: Double(std.width) / Double(std.height))
                say(String(format: "align known crop: aligned match %.2f cover %.2f zoom %.2f (expect 1.43) dx %.3f dy %.3f",
                           al.match, al.coverage, fr.zoom, fr.dx, fr.dy))
            } else {
                say(String(format: "align known crop: NOT aligned (match %.2f cover %.2f)", al.match, al.coverage))
            }
        }

        for pair in Samples.pairs {
            guard let da = Samples.data(pair.a), let db = Samples.data(pair.b) else { say("missing sample \(pair.title)"); continue }
            let t0 = Date()
            let stamps = StageClock()
            do {
                let r = try await PhotoJudge.shared.compare(dataA: da, dataB: db, nameA: "Shot A", nameB: "Shot B") { s in stamps.mark(s) }
                say("=== \(pair.title)  (\(String(format: "%.1f", Date().timeIntervalSince(t0))) s)  stages: \(stamps.summary())")
                say(String(format: "  %@  pA=%.2f  relation=%@ sim=%.3f", r.headline, r.pA, r.relation, r.similarity))
                if let f = r.frame { say(String(format: "  frame zoom %.2f area %.2f dx %.2f dy %.2f rot %.1f inside %.2f", f.zoom, f.areaFrac, f.dx, f.dy, f.rotation, f.inside)) }
                for k in 0..<2 {
                    let p = r.photo(k)
                    say(String(format: "  %@: overall %.0f aes %.0f tech %.0f tilt %.1f/%.2f subj(%.2f,%.2f) edgeW %.2f",
                               p.name, p.overall, p.percentile[0], p.percentile[1], p.m["tilt_deg"], p.m["tilt_conf"], p.m["subject_x"], p.m["subject_y"], p.m["edge_weight"]))
                    for g in r.extras[k].objects {
                        say(String(format: "     #%d %@ area %.3f delta %+.1f lean %+.2f role %@ at %@", g.n, g.label, g.area, g.delta, g.lean, g.role, Objects.whereText(g)))
                    }
                    if let s = r.extras[k].geometry?.subject {
                        say("     subject: \(s.label) \(s.whereText) cut=\(s.cut) sep=\(s.separation.map { String(format: "%.1f", $0) } ?? "-")")
                    }
                }
                for pv in r.previews { say(String(format: "  preview %@: overall %.1f closure %.2f", pv.kind, pv.overall, pv.closure)) }
                if let rec = r.recipe { say("  recipe: " + rec.lines.map { "\($0.name) \($0.value)" }.joined(separator: ", ")) }
                for it in r.coaching {
                    say("  [\(it.fix)] \(it.title)  (gap \(Int(it.gap)), overlay \(it.overlay ?? "-"))")
                    for e in it.evidence { say("      - " + e) }
                }
                say("  reasons: " + r.reasons.map(\.label).joined(separator: ", ") + " | loser leads: " + r.loserLeads.map(\.label).joined(separator: ", "))
            } catch {
                say("FAILED \(pair.title): \(error)")
            }
        }
        say("done")
        exit(0)
    }

    static func rotated(_ img: UIImage, degrees: Double) -> UIImage {
        let r = UIGraphicsImageRenderer(size: img.size)
        return r.image { ctx in
            let c = ctx.cgContext
            c.translateBy(x: img.size.width / 2, y: img.size.height / 2)
            c.rotate(by: degrees * .pi / 180)                 // UIKit (y down): positive = clockwise on screen
            let s = 1.12                                      // zoom so no empty corners show
            img.draw(in: CGRect(x: -img.size.width * s / 2, y: -img.size.height * s / 2, width: img.size.width * s, height: img.size.height * s))
        }
    }
}

/// Records when each pipeline stage starts.
final class StageClock: @unchecked Sendable {
    private var marks: [(Stage, Date)] = []
    private let lock = NSLock()

    func mark(_ s: Stage) { lock.lock(); marks.append((s, Date())); lock.unlock() }

    func summary() -> String {
        lock.lock(); defer { lock.unlock() }
        var parts: [String] = []
        for (i, m) in marks.enumerated() {
            let end = i + 1 < marks.count ? marks[i + 1].1 : Date()
            parts.append("\(m.0)=\(String(format: "%.1f", end.timeIntervalSince(m.1)))")
        }
        return parts.joined(separator: " ")
    }
}

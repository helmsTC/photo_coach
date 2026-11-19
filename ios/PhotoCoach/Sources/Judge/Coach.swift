import Foundation

/// Turns the two sets of attribute scores into a verdict, reasons and advice.
/// Mirrors the desktop app's rules (photocoach/engine.py, coach.py) for the parts that only need
/// network scores; measured evidence and object analysis are layered on in later steps.
enum Coach {

    static func ordinal(_ p: Double) -> String {
        let n = max(1, min(99, Int(p.rounded())))
        let suffix: String
        if (10...20).contains(n % 100) { suffix = "th" } else {
            switch n % 10 { case 1: suffix = "st"; case 2: suffix = "nd"; case 3: suffix = "rd"; default: suffix = "th" }
        }
        return "\(n)\(suffix)"
    }

    static func build(info: JudgeInfo, a: ScoredPhoto, b: ScoredPhoto, seconds: Double) -> Comparison {
        let ia = info.index(of: "aesthetic") ?? 0
        let it = info.index(of: "technical") ?? 1

        // P(A beats B) = sigmoid( scale * (score_A - score_B) ), combined over the two overall heads
        func logit(_ i: Int) -> Double { (a.raw[i] - b.raw[i]) * info.heads[i].scale }
        let v = info.weights.aesthetic * logit(ia) + info.weights.technical * logit(it)
        let pA = 1 / (1 + exp(-v))
        let aWins = pA >= 0.5
        let w = aWins ? a : b
        let l = aWins ? b : a
        let conf = max(pA, 1 - pA)

        func gap(_ i: Int) -> Double { w.percentile[i] - l.percentile[i] }
        func zgap(_ i: Int) -> Double { w.z[i] - l.z[i] }

        let dzA = zgap(ia), dzT = zgap(it)
        let summary: String
        if dzA > 0.25 && dzT > 0.25 {
            summary = "It is ahead on both aesthetic appeal and technical quality."
        } else if dzA > 0.25 {
            summary = dzT > -0.25 ? "It wins on aesthetic appeal; technically the two are about level."
                                  : "It wins on aesthetic appeal; \(l.name) is actually the cleaner capture technically."
        } else if dzT > 0.25 {
            summary = dzA > -0.25 ? "It wins on technical quality; aesthetically the two are about level."
                                  : "It wins on technical quality; \(l.name) is the more appealing image aesthetically."
        } else {
            summary = "The two are close on both overall scores."
        }
        let strength = conf >= 0.85 ? "clearly" : (conf >= 0.68 ? "fairly confidently" : "narrowly")

        // attribute heads that are judged good/bad (not the two overall heads, not style descriptors)
        let judged = info.heads.indices.filter { info.groups.contains(info.heads[$0].group) }
        let ranked = judged.filter { info.heads[$0].reliable }.sorted { zgap($0) > zgap($1) }

        let reasons = ranked.filter { gap($0) >= 8 }.prefix(5).map { i in
            "\(info.heads[i].label): \(ordinal(w.percentile[i])) percentile vs \(ordinal(l.percentile[i])). \(info.heads[i].blurb)"
        }
        let leads = ranked.reversed().filter { gap($0) <= -8 }.prefix(4).map { i in
            "\(info.heads[i].label): \(ordinal(l.percentile[i])) percentile vs \(ordinal(w.percentile[i]))."
        }

        var coaching: [CoachItem] = []
        for i in ranked where gap(i) >= 12 {
            let h = info.heads[i]
            guard let adv = info.advice[h.key] else { continue }
            coaching.append(CoachItem(
                id: h.key, title: adv.title, fix: adv.fix,
                evidence: "\(h.label): the network puts \(l.name) at the \(ordinal(l.percentile[i])) percentile, \(w.name) at the \(ordinal(w.percentile[i])).",
                shoot: adv.shoot, edit: adv.edit))
            if coaching.count == 6 { break }
        }

        var rows: [AttributeRow] = []
        for g in info.groups {
            for i in info.heads.indices where info.heads[i].group == g {
                let h = info.heads[i]
                rows.append(AttributeRow(id: h.key, label: h.label, blurb: h.blurb, group: g, reliable: h.reliable,
                                         a: a.percentile[i], b: b.percentile[i]))
            }
        }

        return Comparison(a: a, b: b, pA: pA, winnerIsA: aWins,
                          headline: "\(w.name) wins \(strength)", summary: summary, closeCall: conf < 0.56,
                          reasons: Array(reasons), loserLeads: Array(leads), coaching: coaching,
                          attributes: rows, seconds: seconds)
    }
}

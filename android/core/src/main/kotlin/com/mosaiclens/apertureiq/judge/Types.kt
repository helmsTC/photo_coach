package com.mosaiclens.apertureiq.judge

import kotlin.math.max

/** One analysed photo: raw attribute scores from the shared tower and where they sit among reference photos. */
class ScoredPhoto(
    val name: String,
    val std: RGBImage,               // the standard image (orientation applied): analysed, displayed, overlaid
    val pixelW: Int,                 // original file size
    val pixelH: Int,
    val raw: DoubleArray,            // one per head
    val percentile: DoubleArray,     // 0...100
    val z: DoubleArray,              // standard deviations from the reference mean
    val m: Measurement,
    val exif: ExifInfo,
    val g: FloatArray,               // whole-frame embedding
    val l: List<FloatArray>,         // four detail-crop embeddings
    val overall: Double,
) {
    val signals: Map<String, Double> get() = m.s
    val globalEmbedHead: List<Double> get() = g.take(8).map { it.toDouble() }
    val aspect: Double get() = std.width.toDouble() / std.height
}

enum class Lens(val title: String, val legend: String) {
    PHOTO("Photo", "The photo as analysed."),
    CLIPPING("Clipping", "Red: blown to pure white. Blue: crushed to pure black. Neither holds any detail."),
    FOCUS("Focus", "Brighter = crisper fine detail. Both photos use the same scale, so they compare directly."),
    COMPOSITION("Composition", "Thirds grid. Yellow ring: strongest visual pull. Cyan cross: centre of visual weight. Line: detected horizon."),
    OBJECTS("Objects", "Yellow: the subject. Red: the score rises with it painted out. Green: the score leans on it."),
    ATTENTION("AI attention", "The network's view. Green regions carry the score more than the rest; red ones less, or they hurt it."),
    FRAMING("Framing", "Same scene matched point-to-point. The box shows where the other photo's frame falls in this one.");

    companion object {
        /** Coaching items name overlays the desktop way. */
        fun fromOverlay(overlay: String?): Lens? = when (overlay) {
            "clip" -> CLIPPING
            "focus" -> FOCUS
            "comp" -> COMPOSITION
            "objects" -> OBJECTS
            "impact" -> ATTENTION
            "frame" -> FRAMING
            else -> null
        }
    }
}

class PhotoExtras(
    var authenticity: Authenticity? = null,
    var objects: List<Segment> = emptyList(),
    var geometry: SceneGeometry? = null,
    val layers: MutableMap<Lens, RGBImage> = HashMap(),
    var otherFrame: Quad? = null,    // the other photo's frame drawn inside this one
)

class AttributeRow(
    val id: String,
    val label: String,
    val blurb: String,
    val group: String,
    val reliable: Boolean,
    val a: Double,
    val b: Double,
)

class CoachItem(
    val id: String,
    val title: String,
    val heads: List<String>,
    val gap: Double,
    val priority: Double,
    val evidence: List<String>,
    val shoot: List<String>,
    val edit: List<String>,
    val overlay: String?,
    val fix: String,
)

class Reason(
    val head: String,
    val label: String,
    val gap: Double,
    val text: String,
    val winnerPct: Double,
    val loserPct: Double,
)

class Preview(
    val kind: String,                // "edit" or "crop_edit"
    val image: RGBImage,
    val overall: Double,
    val aesthetic: Double,
    val technical: Double,
    val closure: Double,             // share of the score gap this edit closes
) {
    val title: String get() = if (kind == "edit") "Tone & colour" else "Crop + tone"
}

class Comparison(
    val a: ScoredPhoto,
    val b: ScoredPhoto,
    val extras: List<PhotoExtras>,   // [a, b]
    /** Model's estimate of how often viewers would prefer A. */
    val pA: Double,
    val winnerIsA: Boolean,
    val headline: String,
    val summary: String,
    val closeCall: Boolean,
    val reasons: List<Reason>,
    val loserLeads: List<Reason>,
    val coaching: List<CoachItem>,
    val attributes: List<AttributeRow>,
    val relation: String,            // aligned / similar / different
    val relationText: String,
    val similarity: Double,
    val frame: Framing?,
    val recipe: LookRecipe?,
    val previews: List<Preview>,
    val seconds: Double,
) {
    val winner: ScoredPhoto get() = if (winnerIsA) a else b
    val loser: ScoredPhoto get() = if (winnerIsA) b else a
    val winnerIndex: Int get() = if (winnerIsA) 0 else 1
    val loserIndex: Int get() = if (winnerIsA) 1 else 0
    val confidence: Double get() = max(pA, 1 - pA)
    fun photo(i: Int): ScoredPhoto = if (i == 0) a else b
    val bestPreview: Preview? get() = previews.maxByOrNull { it.overall }
}

/** The free tier's result: a winner and nothing else. */
class QuickVerdict(
    val images: List<RGBImage>,
    val aspects: List<Double>,
    val pA: Double,
    val seconds: Double,
    val authenticity: List<Authenticity> = emptyList(),
) {
    val winnerIndex: Int get() = if (pA >= 0.5) 0 else 1
    val confidence: Double get() = max(pA, 1 - pA)

    /** A word, not a number: the free tier shows no metrics. */
    val strength: Pair<String, String>
        get() = when {
            confidence < 0.56 -> Pair("Close call", "It's tight - most people would struggle to pick, but this one edges it.")
            confidence < 0.70 -> Pair("Solid edge", "Most viewers would pick this one.")
            else -> Pair("Clear winner", "Viewers would pick this one by a wide margin.")
        }
}

enum class Stage(val title: String) {
    READING("Reading both photos"),
    MEASURING("Measuring light, colour and focus"),
    SCORING("Scoring with the twin network"),
    AUTHENTICITY("Checking for AI generation"),
    MATCHING("Matching the two scenes"),
    OBJECTS("Finding what's in each frame"),
    TESTING("Painting out each object to test it"),
    ATTENTION("Mapping what the network looks at"),
    EDITING("Building an edit toward the winner"),
    COACHING("Writing your coaching"),
}

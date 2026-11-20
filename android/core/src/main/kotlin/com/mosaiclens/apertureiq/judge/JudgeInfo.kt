package com.mosaiclens.apertureiq.judge

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** Everything the app needs besides network weights (written by `python -m mobile.export_onnx`). */
@Serializable
data class JudgeInfo(
    val encoder: String,
    val inputSize: Int,
    val stdLongSide: Int,
    val weights: Weights,
    val groups: List<String>,
    val signalKeys: List<String>,
    val signalsUsed: List<String>,
    val heads: List<Head>,
    val poolSize: Int? = null,
    val trained: String? = null,
    val advice: Map<String, Advice>,
) {
    @Serializable
    data class Head(
        val key: String,
        val label: String,
        val blurb: String,
        val group: String,
        /** False when the head predicts human ratings only weakly on held-out photos: shown, marked, never used for advice. */
        val reliable: Boolean,
        val validity: Double? = null,
        val metric: String? = null,
        /** Sharpness of the pairwise comparison for this attribute. */
        val scale: Double,
        val refMean: Double,
        val refStd: Double,
        /** Raw-score quantiles (0...100) over the pool of human-rated reference photos. */
        val quantiles: List<Double>,
    )

    @Serializable
    data class Advice(val title: String, val fix: String, val shoot: List<String>, val edit: List<String>)

    @Serializable
    data class Weights(val aesthetic: Double, val technical: Double)

    fun index(key: String): Int? = heads.indexOfFirst { it.key == key }.takeIf { it >= 0 }

    val ia: Int get() = index("aesthetic") ?: 0
    val itech: Int get() = index("technical") ?: 1

    /** Raw score -> 0...100 standing among the reference pool (linear interpolation in the quantile table). */
    fun percentile(raw: Double, head: Int): Double {
        val q = heads[head].quantiles
        if (q.isEmpty()) return 50.0
        if (raw <= q.first()) return 0.0
        if (raw >= q.last()) return (q.size - 1).toDouble()
        var lo = 0
        var hi = q.size - 1
        while (hi - lo > 1) {
            val mid = (lo + hi) / 2
            if (q[mid] <= raw) lo = mid else hi = mid
        }
        val span = q[hi] - q[lo]
        return lo + (if (span > 0) (raw - q[lo]) / span else 0.0)
    }

    companion object {
        private val json = Json { ignoreUnknownKeys = true }
        fun parse(text: String): JudgeInfo = json.decodeFromString(serializer(), text)
    }
}

sealed class JudgeError(message: String) : Exception(message) {
    class MissingResource(name: String) : JudgeError("Missing app resource: $name")
    class BadImage : JudgeError("That photo could not be read.")
    class Model(what: String) : JudgeError("Model error: $what")
}

/** Capture settings read from the file (scene.exif in the desktop app). */
data class ExifInfo(
    val camera: String? = null,
    val lens: String? = null,
    val aperture: Double? = null,
    val shutter: Double? = null,
    val iso: Int? = null,
    val focal: Double? = null,
    val focal35: Int? = null,
    val evComp: Double? = null,
    val taken: String? = null,
    val flash: Boolean? = null,
) {
    val isEmpty: Boolean get() = camera == null && aperture == null && shutter == null && iso == null && focal == null

    /** Short chips for the UI: "ƒ/1.8", "1/120s", "ISO 400", "26mm". */
    val chips: List<String>
        get() {
            val c = ArrayList<String>()
            if (focal35 != null) c.add("${focal35}mm") else if (focal != null) c.add("${fmt(focal)}mm")
            aperture?.let { c.add("ƒ/${fmt(it)}") }
            shutter?.let { c.add(shutterString(it)) }
            iso?.let { c.add("ISO $it") }
            evComp?.let { if (it != 0.0) c.add(String.format("%+.1f EV", it)) }
            if (flash == true) c.add("Flash")
            return c
        }

    companion object {
        fun shutterString(t: Double): String = if (t < 0.4) "1/${rndInt(1 / t)}s" else String.format("%.1fs", t)

        /** "Apple iPhone 15 Pro" rather than "Apple Apple iPhone…" when the model repeats the make. */
        fun cameraName(make: String?, model: String?): String? {
            val cam = listOfNotNull(make?.trim()?.ifEmpty { null }, model?.trim()?.ifEmpty { null })
            if (cam.isEmpty()) return null
            return if (cam.size == 2 && cam[1].startsWith(cam[0])) cam[1] else cam.joinToString(" ")
        }
    }
}

/** "%g"-style: 1.8 -> "1.8", 26.0 -> "26". */
fun fmt(v: Double): String = if (v == rnd(v)) v.toInt().toString() else String.format("%.1f", v)

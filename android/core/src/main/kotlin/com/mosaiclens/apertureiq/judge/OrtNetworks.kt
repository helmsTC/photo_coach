package com.mosaiclens.apertureiq.judge

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.nio.FloatBuffer

/** Class names of the segmenter (written next to the model by `python -m mobile.export_onnx`). */
@Serializable
data class SegmenterLabels(val size: Int, val labels: List<String>, val things: List<Int>) {
    companion object {
        fun parse(text: String): SegmenterLabels = Json { ignoreUnknownKeys = true }.decodeFromString(serializer(), text)
    }
}

/**
 * The four networks through ONNX Runtime (the same Java API on Android and on a desktop JVM, which
 * is what lets the unit tests check this port against the PyTorch reference).
 *
 * `source` maps a model name ("encoder", "tower", "segmenter", "ai_detector") to a readable .onnx file
 * path (String) or a direct ByteBuffer holding the model (e.g. memory-mapped from an APK asset), or null
 * when that model is not installed.  The encoder and tower load up front; the segmenter and
 * the detector load on first use.
 */
class OrtNetworks(
    private val source: (String) -> Any?,
    private val segLabels: SegmenterLabels?,
    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
    /** Applied to the options of the three big models (e.g. to add NNAPI / XNNPACK); the tower stays on plain CPU. */
    private val configure: (OrtSession.SessionOptions) -> Unit = {},
) : Networks, AutoCloseable {

    /** Time spent inside each network since the last [resetStats] (for profiling on a device).
     *  Declared before the sessions: loading them is timed too. */
    private val nanos = java.util.concurrent.ConcurrentHashMap<String, Long>()
    private val calls = java.util.concurrent.ConcurrentHashMap<String, Int>()
    private inline fun <T> timed(name: String, n: Int, body: () -> T): T {
        val t0 = System.nanoTime()
        try { return body() } finally { nanos.merge(name, System.nanoTime() - t0, Long::plus); calls.merge(name, n, Int::plus) }
    }
    fun resetStats() { nanos.clear(); calls.clear() }
    fun stats(): String = nanos.keys.sorted().joinToString(", ") { "$it ${"%.1f".format(java.util.Locale.US, nanos[it]!! / 1e9)} s / ${calls[it]}" }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val encoder: OrtSession = open("encoder", true) ?: throw JudgeError.MissingResource("encoder.onnx")
    private val towerNet: OrtSession = open("tower", false) ?: throw JudgeError.MissingResource("tower.onnx")
    private var segNet: OrtSession? = null
    private var detNet: OrtSession? = null
    private var segTried = false
    private var detTried = false

    private fun open(name: String, accelerated: Boolean): OrtSession? = timed("load $name", 1) { open0(name, accelerated) }

    private fun open0(name: String, accelerated: Boolean): OrtSession? {
        val src = source(name) ?: return null
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(threads)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        if (accelerated) configure(opts)
        return when (src) {
            is String -> env.createSession(src, opts)
            is java.nio.ByteBuffer -> env.createSession(src, opts)
            is ByteArray -> env.createSession(src, opts)
            else -> throw JudgeError.Model("unsupported model source for $name")
        }
    }

    /** Images as one float tensor [n, 3, h, w], values 0...255 (the models normalise internally). */
    private fun nchw(views: List<RGBImage>): OnnxTensor {
        val w = views[0].width; val h = views[0].height
        val plane = w * h
        val buf = FloatArray(views.size * 3 * plane)
        for ((n, v) in views.withIndex()) {
            val base = n * 3 * plane
            val d = v.data
            for (i in 0 until plane) {
                buf[base + i] = d.u(i * 3).toFloat()
                buf[base + plane + i] = d.u(i * 3 + 1).toFloat()
                buf[base + 2 * plane + i] = d.u(i * 3 + 2).toFloat()
            }
        }
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(buf), longArrayOf(views.size.toLong(), 3, h.toLong(), w.toLong()))
    }

    @Synchronized
    override fun embed(views: List<RGBImage>): List<FloatArray> {
        if (views.isEmpty()) return emptyList()
        val out = ArrayList<FloatArray>(views.size)
        // modest batches keep peak memory flat on phones
        for (chunk in views.chunked(8)) {
            nchw(chunk).use { x ->
                timed("encoder", chunk.size) { encoder.run(mapOf("image" to x)) }.use { res ->
                    val fb = (res[0] as OnnxTensor).floatBuffer
                    val dim = fb.remaining() / chunk.size
                    for (i in chunk.indices) { val e = FloatArray(dim); fb.get(e); out.add(e) }
                }
            }
        }
        return out
    }

    @Synchronized
    override fun tower(g: FloatArray, l: List<FloatArray>, h: FloatArray): DoubleArray {
        val dim = g.size
        val lf = FloatArray(4 * dim)
        for (i in 0 until 4) System.arraycopy(l[i], 0, lf, i * dim, dim)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(g), longArrayOf(1, dim.toLong())).use { tg ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(lf), longArrayOf(1, 4, dim.toLong())).use { tl ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(h), longArrayOf(1, h.size.toLong())).use { th ->
                    timed("tower", 1) { towerNet.run(mapOf("g" to tg, "l" to tl, "h" to th)) }.use { res ->
                        val fb = (res[0] as OnnxTensor).floatBuffer
                        return DoubleArray(fb.remaining()) { fb.get().toDouble() }
                    }
                }
            }
        }
    }

    override val segmenter: SemanticSegmenter?
        get() {
            val lab = segLabels ?: return null
            synchronized(this) {
                if (!segTried) {
                    segTried = true
                    segNet = try { open("segmenter", true) } catch (e: Exception) { null }
                }
            }
            val net = segNet ?: return null
            val things = lab.things.toSet()
            return object : SemanticSegmenter {
                override val size = lab.size
                override val labels = lab.labels
                override fun isThing(id: Int) = things.contains(id)
                override fun classMap(square: RGBImage): IntArray? = synchronized(this@OrtNetworks) {
                    try {
                        nchw(listOf(square)).use { x ->
                            timed("segmenter", 1) { net.run(mapOf("image" to x)) }.use { res ->
                                val ib = (res[0] as OnnxTensor).intBuffer
                                IntArray(ib.remaining()).also { ib.get(it) }
                            }
                        }
                    } catch (e: Exception) {
                        null
                    }
                }
            }
        }

    @Synchronized
    override fun aiProbability(chw: FloatArray): Double? {
        if (!detTried) {
            detTried = true
            detNet = try { open("ai_detector", true) } catch (e: Exception) { null }
        }
        val net = detNet ?: return null
        val s = DetectorInput.crop.toLong()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), longArrayOf(1, 3, s, s)).use { x ->
            timed("detector", 1) { net.run(mapOf("image" to x)) }.use { res ->
                return (res[0] as OnnxTensor).floatBuffer.get(0).toDouble()
            }
        }
    }

    @Synchronized
    override fun close() {
        encoder.close(); towerNet.close(); segNet?.close(); detNet?.close()
    }
}

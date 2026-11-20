package com.mosaiclens.apertureiq.judge

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Desktop stand-in for the app's Bitmap decoder (test photos are upright, so no EXIF rotation).
 * JPEGs are read as raw YCbCr and converted with the plain JFIF matrix: ImageIO's own conversion
 * goes through the embedded ICC profile, and the pipeline (like PIL in training) must see the
 * file's values untouched.
 */
object JvmDecoder : Decoder {
    override fun decode(bytes: ByteArray): Decoded {
        val isJpeg = bytes.size > 2 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xD8.toByte()
        if (isJpeg) {
            val reader = ImageIO.getImageReadersByFormatName("jpeg").next()
            try {
                ImageIO.createImageInputStream(ByteArrayInputStream(bytes)).use { input ->
                    reader.input = input
                    val r = reader.readRaster(0, null)
                    if (r.numBands == 3) {
                        val w = r.width; val h = r.height
                        val s = r.getPixels(0, 0, w, h, null as IntArray?)
                        val out = ByteArray(w * h * 3)
                        for (i in 0 until w * h) {
                            val y = s[i * 3].toFloat(); val cb = s[i * 3 + 1] - 128f; val cr = s[i * 3 + 2] - 128f
                            out[i * 3] = clampByte(y + 1.402f * cr)
                            out[i * 3 + 1] = clampByte(y - 0.344136f * cb - 0.714136f * cr)
                            out[i * 3 + 2] = clampByte(y + 1.772f * cb)
                        }
                        return Decoded(RGBImage(w, h, out), w, h, ExifInfo())
                    }
                }
            } finally {
                reader.dispose()
            }
        }
        val im: BufferedImage = ImageIO.read(ByteArrayInputStream(bytes)) ?: throw JudgeError.BadImage()
        val w = im.width; val h = im.height
        return Decoded(RGBImage.fromArgb(im.getRGB(0, 0, w, h, null, 0, w), w, h), w, h, ExifInfo())
    }
}

fun RGBImage.save(f: File) {
    val im = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    im.setRGB(0, 0, width, height, toArgb(), 0, width)
    f.parentFile.mkdirs()
    ImageIO.write(im, "png", f)
}

object Fixtures {
    val assets = File(System.getProperty("assets"))
    val demo = File(System.getProperty("demo"))
    val out = File(System.getProperty("out"))
    val info: JudgeInfo by lazy { JudgeInfo.parse(File(assets, "judge.json").readText()) }
    val nets: OrtNetworks by lazy {
        val seg = File(assets, "segmenter.json").takeIf { it.exists() }?.let { SegmenterLabels.parse(it.readText()) }
        OrtNetworks({ n -> File(assets, "$n.onnx").takeIf { it.exists() }?.absolutePath }, seg)
    }
    val judge: PhotoJudge by lazy { PhotoJudge(info, nets, JvmDecoder) }
}

/**
 * The check that the ONNX conversion and the Kotlin preprocessing match training: the bundled test
 * photos must score what the PyTorch model scored for the same files (assets/parity.json).
 */
class ParityTest {
    @Serializable
    data class Expected(
        val file: String,
        val stdSize: List<Int>,
        val globalEmbedHead: List<Double>,
        val signals: Map<String, Double>,
        val scores: Map<String, Double>,
        val percentiles: Map<String, Double>,
    )

    @Test
    fun scoresMatchThePyTorchReference() {
        val expected = Json { ignoreUnknownKeys = true }.decodeFromString<List<Expected>>(File(Fixtures.assets, "parity.json").readText())
        val info = Fixtures.info
        for (e in expected) {
            val t0 = System.nanoTime()
            val p = Fixtures.judge.score(File(Fixtures.assets, e.file).readBytes(), e.file)
            println("${e.file}  (${"%.2f".format((System.nanoTime() - t0) / 1e9)} s)")
            assertEquals(e.stdSize, listOf(p.std.width, p.std.height), "standard size")

            val embDiff = p.globalEmbedHead.zip(e.globalEmbedHead).maxOf { abs(it.first - it.second) }
            val embScale = e.globalEmbedHead.maxOf { abs(it) }
            println("  encoder output: max diff %.4f (values up to %.2f)".format(embDiff, embScale))
            assertTrue(embDiff <= 0.15 * embScale, "encoder output differs")

            var worstSig = Pair("", 0.0)
            for ((k, want) in e.signals) {
                val got = p.signals[k] ?: continue
                val d = abs(got - want) / max(abs(want), 0.05)
                if (d > worstSig.second) worstSig = Pair(k, d)
                if (info.signalsUsed.contains(k)) assertTrue(d <= 0.15, "tower signal $k: $got vs $want")
            }
            println("  measured signals: worst relative diff %.1f%% (%s)".format(worstSig.second * 100, worstSig.first))

            var worst = Pair("", 0.0)
            for ((i, h) in info.heads.withIndex()) {
                val want = e.percentiles[h.key] ?: continue
                val d = abs(p.percentile[i] - want)
                if (d > worst.second) worst = Pair(h.label, d)
            }
            println("  attribute percentiles: worst diff %.2f points (%s)".format(worst.second, worst.first))
            for (key in listOf("aesthetic", "technical")) {
                val i = info.index(key)!!
                println("  %s: %.1f here, %.1f reference".format(key, p.percentile[i], e.percentiles[key]))
            }
            assertTrue(worst.second <= 3.0, "percentiles differ by ${worst.second} (${worst.first})")
        }
    }
}

/** The whole comparison on bundled sample pairs: it must run, agree with itself when swapped, and find what is known to be there. */
class EndToEndTest {
    private fun report(c: Comparison, tag: String) {
        println("== $tag: ${c.headline}  (P(A) %.3f, %s, similarity %.2f, %.1f s)".format(c.pA, c.relation, c.similarity, c.seconds))
        println("   ${c.summary}")
        for (k in 0 until 2) {
            val p = c.photo(k); val x = c.extras[k]
            println("   ${p.name}: overall %.1f  AI check ${x.authenticity?.verdict} (p=${x.authenticity?.aiProbability?.let { "%.3f".format(it) }})".format(p.overall))
            val subj = x.geometry?.subject?.let { it.label + " at " + it.whereText + ", " + Coach.pct(it.area) + " of frame" } ?: "none"
            println("      subject: " + subj + "; sky " + Coach.pct(x.geometry?.sky ?: 0.0, 0) + ", water " + Coach.pct(x.geometry?.water ?: 0.0, 0))
            for (g in x.objects) println("      #${g.n} ${g.label} (${g.role}) area " + Coach.pct(g.area) + " delta %+.1f lean %+.2f".format(g.delta, g.lean))
            for ((lens, im) in x.layers) im.save(File(Fixtures.out, "${tag}_${k}_${lens.name.lowercase()}.png"))
        }
        c.frame?.let { println("   framing: zoom %.2f dx %+.2f dy %+.2f inside %.2f".format(it.zoom, it.dx, it.dy, it.inside)) }
        c.recipe?.let { r -> println("   recipe: " + r.lines.joinToString(", ") { "${it.name} ${it.value}" }) }
        for (p in c.previews) { println("   preview ${p.kind}: overall %.1f, closes ".format(p.overall) + Coach.pct(p.closure, 0) + " of the gap"); p.image.save(File(Fixtures.out, "${tag}_${p.kind}.png")) }
        for (r in c.reasons) println("   + ${r.text.take(90)}")
        for (i in c.coaching) { println("   * ${i.title} [${i.fix}, ${i.overlay}]"); i.evidence.forEach { println("       - ${it.take(150)}") } }
    }

    @Test
    fun sameSceneIsAlignedAndSwappingFlipsTheVerdict() {
        val a = File(Fixtures.assets, "sample_louise_12.jpg").readBytes()
        val b = File(Fixtures.assets, "sample_louise_13.jpg").readBytes()
        val c = Fixtures.judge.compare(a, b, "Shot A", "Shot B")
        report(c, "louise")
        val r = Fixtures.judge.compare(b, a, "Shot A", "Shot B")
        assertEquals(c.pA, 1 - r.pA, 1e-6, "swapping the photos must flip the verdict exactly")
        assertEquals(c.relation, r.relation)
        val q = Fixtures.judge.quickVerdict(a, b)
        assertEquals(c.pA, q.pA, 1e-9, "free and full tiers must agree on the verdict")
    }

    @Test
    fun aKnownDistractionIsFlagged() {
        val a = File(Fixtures.demo, "zclutter_stage.jpg").readBytes()
        val b = File(Fixtures.demo, "louise_11.jpg").readBytes()
        val c = Fixtures.judge.compare(a, b, "Shot A", "Shot B")
        report(c, "clutter")
        assertTrue(!c.winnerIsA, "the cluttered frame should lose")
        assertTrue(c.extras[0].objects.isNotEmpty(), "objects should be found")
    }

    @Test
    fun differentScenes() {
        val c = Fixtures.judge.compare(File(Fixtures.assets, "sample_zclutter_table.jpg").readBytes(),
            File(Fixtures.assets, "sample_moraine_04.jpg").readBytes(), "Shot A", "Shot B")
        report(c, "table")
    }
}

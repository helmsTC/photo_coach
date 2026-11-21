package com.mosaiclens.apertureiq.judge

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** A decoded file: upright RGB pixels (possibly already reduced, never below 2x the standard size when the file allows). */
class Decoded(val full: RGBImage, val pixelW: Int, val pixelH: Int, val exif: ExifInfo)

/** Platform image decoding (EXIF orientation applied, embedded colour profiles NOT applied, as in PIL). */
fun interface Decoder {
    fun decode(bytes: ByteArray): Decoded
}

/** The neural networks, wherever they run. */
interface Networks {
    /** Square RGB views (JudgeInfo.inputSize) -> one embedding each. */
    fun embed(views: List<RGBImage>): List<FloatArray>
    /** Whole-frame embedding, four crop embeddings, measured signals -> one raw score per head. */
    fun tower(g: FloatArray, l: List<FloatArray>, h: FloatArray): DoubleArray
    val segmenter: SemanticSegmenter?
    /** P(AI-generated) for a DetectorInput tensor, or null if the detector is unavailable. */
    fun aiProbability(chw: FloatArray): Double?
}

/**
 * The twin network on device.  Both photos go through the same encoder and the same tower;
 * the verdict is a function of the score difference, so swapping the photos flips it exactly.
 */
class PhotoJudge(val info: JudgeInfo, private val nets: Networks, private val decoder: Decoder) {

    companion object {
        /** Global-embedding cosine similarity above which two photos count as the same kind of scene. */
        const val similarScene = 0.80

        /** Run two independent pieces of CPU work side by side. */
        internal fun <T> both(a: () -> T, b: () -> T): Pair<T, T> {
            val fb = java.util.concurrent.CompletableFuture.supplyAsync { b() }
            val ra = a()
            return try {
                Pair(ra, fb.get())
            } catch (e: java.util.concurrent.ExecutionException) {
                throw e.cause ?: e
            }
        }

        fun cosine(x: FloatArray, y: FloatArray): Double {
            var d = 0.0; var nx = 0.0; var ny = 0.0
            for (i in 0 until min(x.size, y.size)) {
                d += (x[i] * y[i]).toDouble(); nx += (x[i] * x[i]).toDouble(); ny += (y[i] * y[i]).toDouble()
            }
            return d / (sqrt(nx) * sqrt(ny) + 1e-12)
        }
    }

    // ---- network calls

    /** The tower only reads the signals it was trained with; the rest are zeroed. */
    private fun towerSignals(s: Map<String, Double>): FloatArray {
        val used = info.signalsUsed.toSet()
        return FloatArray(info.signalKeys.size) { val k = info.signalKeys[it]; if (used.contains(k)) (s[k] ?: 0.0).toFloat() else 0f }
    }

    private fun overall(pct: DoubleArray): Double = info.weights.aesthetic * pct[info.ia] + info.weights.technical * pct[info.itech]

    private fun percentiles(raw: DoubleArray) = DoubleArray(raw.size) { info.percentile(raw[it], it) }

    // ---- scoring one photo

    private fun featurize(std: RGBImage, m: Measurement, name: String, d: Decoded, exif: ExifInfo): ScoredPhoto {
        val e = nets.embed(ImagePrep.views(std, info.inputSize))
        val raw = nets.tower(e[0], e.subList(1, e.size), towerSignals(m.s))
        if (raw.size != info.heads.size) throw JudgeError.Model("expected ${info.heads.size} scores, got ${raw.size}")
        val pct = percentiles(raw)
        val z = DoubleArray(raw.size) { (raw[it] - info.heads[it].refMean) / max(info.heads[it].refStd, 1e-9) }
        return ScoredPhoto(name, std, d.pixelW, d.pixelH, raw, pct, z, m, exif, e[0], e.subList(1, e.size).toList(), overall(pct))
    }

    fun score(bytes: ByteArray, name: String): ScoredPhoto {
        val d = decoder.decode(bytes)
        val std = ImagePrep.standardImage(d.full, info.stdLongSide)
        return featurize(std, Signals.measure(std), name, d, d.exif)
    }

    /** Score an edited version of a photo (tone/colour signals re-measured, as the desktop app does). */
    private fun scoreEdited(img: RGBImage): Pair<DoubleArray, Double> {
        val e = nets.embed(ImagePrep.views(img, info.inputSize))
        val pct = percentiles(nets.tower(e[0], e.subList(1, e.size), towerSignals(Signals.toneColor(img))))
        return Pair(pct, overall(pct))
    }

    // ---- what the score leans on

    /** Slide a window that replaces each region with a heavy blur; + = the score drops without it. */
    private fun occlusion(p: ScoredPhoto, n: Int = 5, win: Double = 0.36): Plane {
        val gv = ImagePrep.resizeArea(p.std, info.inputSize, info.inputSize)
        val hh = gv.height; val ww = gv.width
        val blur = Imaging.blurRGB(gv, 0.09 * max(hh, ww))
        val wh = max(14, rndInt(win * hh)); val wwin = max(14, rndInt(win * ww))
        val ys = IntArray(n) { rndInt((hh - wh).toDouble() * it / (n - 1)) }
        val xs = IntArray(n) { rndInt((ww - wwin).toDouble() * it / (n - 1)) }
        val batch = ArrayList<RGBImage>(); val boxes = ArrayList<IntArray>()
        for (y in ys) {
            for (x in xs) {
                val im = gv.copy()
                for (yy in y until y + wh) {
                    System.arraycopy(blur.data, (yy * ww + x) * 3, im.data, (yy * ww + x) * 3, wwin * 3)
                }
                batch.add(im); boxes.add(intArrayOf(y, x))
            }
        }
        val g = nets.embed(batch)
        val hv = towerSignals(p.m.s)
        val ia = info.ia
        val heat = Plane(ww, hh); val cnt = Plane(ww, hh)
        for ((k, box) in boxes.withIndex()) {
            val s = nets.tower(g[k], p.l, hv)[ia]
            val d = ((p.raw[ia] - s) / info.heads[ia].refStd).toFloat()
            for (yy in box[0] until box[0] + wh) for (xx in box[1] until box[1] + wwin) { heat.d[yy * ww + xx] += d; cnt.d[yy * ww + xx] += 1f }
        }
        for (i in heat.d.indices) heat.d[i] /= max(cnt.d[i], 1f)
        return Imaging.gaussianBlur(heat, 0.04 * max(hh, ww))
    }

    /**
     * Re-score the photo with each object painted out.  delta = change in the aesthetic percentile
     * (+ = it scores higher without it, i.e. it distracts); lean = how much the score depends on it.
     */
    private fun objectEffects(p: ScoredPhoto, check: () -> Unit, maxObjects: Int = 10): List<Segment> {
        // compact things and sizeable background regions, things first (objects.py / Engine.object_effects)
        var segs = Objects.detect(p.std, nets.segmenter).filter { if (it.thing) it.area >= 0.002 else it.area >= 0.015 }
        segs = segs.sortedByDescending { it.area * (if (it.thing) 3 else 1) }.take(maxObjects).sortedByDescending { it.area }
        val ia = info.ia
        val hv = towerSignals(p.m.s)
        check()
        // painting out is plain CPU work, so the objects are prepared side by side, then embedded in batches
        val views = segs.parallelStream().map { ImagePrep.views(Objects.remove(p.std, it), info.inputSize) }.collect(java.util.stream.Collectors.toList())
        check()
        // A detail crop the object does not reach is pixel-for-pixel the original crop, whose embedding
        // we already have: only the views that actually changed go through the encoder.
        val orig = ImagePrep.views(p.std, info.inputSize)
        val todo = ArrayList<RGBImage>()
        val slot = views.map { v -> IntArray(5) { i -> if (i > 0 && v[i].data.contentEquals(orig[i].data)) -1 else { todo.add(v[i]); todo.size - 1 } } }
        val e = nets.embed(todo)
        for ((k, s) in segs.withIndex()) {
            val r = nets.tower(e[slot[k][0]], (1..4).map { i -> if (slot[k][i] < 0) p.l[i - 1] else e[slot[k][i]] }, hv)
            s.delta = info.percentile(r[ia], ia) - p.percentile[ia]
            s.lean = (p.raw[ia] - r[ia]) / info.heads[ia].refStd
        }
        for ((i, s) in segs.withIndex()) s.n = i + 1
        return segs
    }

    // ---- AI-generation check

    /** Labels in the file first; the detector network for everything (its reading is shown either way). */
    fun authenticity(bytes: ByteArray, d: Decoded): Authenticity {
        val label = Provenance.read(bytes)
        val p = try { nets.aiProbability(DetectorInput.make(d.full)) } catch (e: Exception) { null }
        return Authenticity.decide(label, p, d.exif.camera)
    }

    /**
     * Both shots' AI checks, or `known` when reopening a saved comparison: the saved copies are re-encoded
     * and carry no C2PA/XMP labels, so checking them again would be less reliable than the original result.
     */
    private fun authenticity(bA: ByteArray, dA: Decoded, bB: ByteArray, dB: Decoded, known: List<Authenticity>?): List<Authenticity> {
        if (known != null && known.size == 2) return known
        return listOf(authenticity(bA, dA), authenticity(bB, dB))
    }

    private fun pA(a: ScoredPhoto, b: ScoredPhoto): Double {
        fun logit(i: Int) = (a.raw[i] - b.raw[i]) * info.heads[i].scale
        return 1 / (1 + exp(-(info.weights.aesthetic * logit(info.ia) + info.weights.technical * logit(info.itech))))
    }

    // ---- the free tier: verdict only

    /**
     * Just the twin network's call: which shot wins and how decisively.  Skips scene matching, objects,
     * attention maps, edits and coaching, so it's a fraction of the full pipeline's time.
     * `check` is called between stages and may throw to cancel.
     */
    fun quickVerdict(
        bytesA: ByteArray, bytesB: ByteArray, known: List<Authenticity>? = null,
        progress: (Stage) -> Unit = {}, check: () -> Unit = {},
    ): QuickVerdict {
        val t0 = System.nanoTime()
        progress(Stage.READING)
        val (dA, dB) = both({ decoder.decode(bytesA) }, { decoder.decode(bytesB) })
        val stdA = ImagePrep.standardImage(dA.full, info.stdLongSide); val stdB = ImagePrep.standardImage(dB.full, info.stdLongSide)
        check()
        progress(Stage.MEASURING)
        val (mA, mB) = both({ Signals.measure(stdA) }, { Signals.measure(stdB) })
        check()
        progress(Stage.SCORING)
        val a = featurize(stdA, mA, "Shot A", dA, ExifInfo())
        val b = featurize(stdB, mB, "Shot B", dB, ExifInfo())
        check()
        progress(Stage.AUTHENTICITY)
        val au = authenticity(bytesA, dA, bytesB, dB, known)
        return QuickVerdict(listOf(a.std, b.std), listOf(a.aspect, b.aspect), pA(a, b), (System.nanoTime() - t0) / 1e9, au)
    }

    // ---- the twin comparison

    fun compare(
        bytesA: ByteArray, bytesB: ByteArray, nameA: String, nameB: String, known: List<Authenticity>? = null,
        progress: (Stage) -> Unit = {}, check: () -> Unit = {},
        /** Receives the verdict as soon as it is known (scores, attributes, AI check), before the slow
         *  parts - objects, attention, edit previews and the coaching built on them - are worked out. */
        early: (Comparison) -> Unit = {},
    ): Comparison {
        val t0 = System.nanoTime()
        progress(Stage.READING)
        val (dA, dB) = both({ decoder.decode(bytesA) }, { decoder.decode(bytesB) })
        val stdA = ImagePrep.standardImage(dA.full, info.stdLongSide); val stdB = ImagePrep.standardImage(dB.full, info.stdLongSide)
        check()

        progress(Stage.MEASURING)
        val (mA, mB) = both({ Signals.measure(stdA) }, { Signals.measure(stdB) })
        check()

        progress(Stage.SCORING)
        val a = featurize(stdA, mA, nameA, dA, dA.exif)
        val b = featurize(stdB, mB, nameB, dB, dB.exif)
        val photos = listOf(a, b)

        // verdict
        val ia = info.ia; val it = info.itech
        val pa = pA(a, b)
        val w = if (pa >= 0.5) 0 else 1
        val l = 1 - w
        check()

        progress(Stage.AUTHENTICITY)
        val au = authenticity(bytesA, dA, bytesB, dB, known)
        check()

        val sim = cosine(a.g, b.g)
        early(Coach.build(
            Coach.Input(info, photos, w, l, null, null, if (sim >= similarScene) "similar" else "different", listOf(Pair(emptyList(), null), Pair(emptyList(), null))),
            pa, listOf(PhotoExtras(authenticity = au[0]), PhotoExtras(authenticity = au[1])), sim, emptyList(), (System.nanoTime() - t0) / 1e9,
        ))

        progress(Stage.MATCHING)
        val al = SceneMatch.align(stdA, stdB)
        val relation = if (al.aligned) "aligned" else if (sim >= similarScene) "similar" else "different"
        var frame: Framing? = null
        val masks = arrayOfNulls<BooleanArray>(2)
        val extras = listOf(PhotoExtras(authenticity = au[0]), PhotoExtras(authenticity = au[1]))
        val hinv = if (al.aligned) al.H.inverse else null
        if (al.aligned && hinv != null) {
            masks[0] = SceneMatch.overlapMask(stdA.width, stdA.height, al.H)
            masks[1] = SceneMatch.overlapMask(stdB.width, stdB.height, hinv)
            val quad = if (l == 0) al.quadBinA else al.quadAinB          // the winner's frame inside the loser's
            frame = SceneMatch.framing(quad, photos[l].aspect)
            extras[0].otherFrame = al.quadBinA
            extras[1].otherFrame = al.quadAinB
        }
        check()

        progress(Stage.OBJECTS)
        progress(Stage.TESTING)
        for (k in 0 until 2) {
            val segs = objectEffects(photos[k], check)
            val s = photos[k].m
            val si = Objects.pickSubject(segs, s["subject_x"], s["subject_y"])
            Objects.assignRoles(segs, si)
            extras[k].objects = segs
            extras[k].geometry = Objects.geometry(photos[k].std, segs, si, s.detail)
        }

        progress(Stage.ATTENTION)
        check()
        val heat = photos.map { occlusion(it) }
        check()

        progress(Stage.EDITING)
        val labs = listOf(LabImage(stdA), LabImage(stdB))
        val recipe = SceneMatch.lookRecipe(labs[l], labs[w], masks[l], masks[w], relation != "different")
        val oW = photos[w].overall; val oL = photos[l].overall
        fun closure(o: Double) = max(-1.0, min(1.5, (o - oL) / max(oW - oL, 1e-6)))
        val previews = ArrayList<Preview>()
        val edited = SceneMatch.applyLook(labs[l], recipe)
        val pe = scoreEdited(edited)
        previews.add(Preview("edit", edited, pe.second, pe.first[ia], pe.first[it], closure(pe.second)))
        val fr = frame
        if (fr != null && fr.zoom >= 1.08 && fr.inside >= 0.85) {
            val hlw = if (l == 0) al.H else al.H.inverse
            val crop = hlw?.let { SceneMatch.cropToOther(photos[l].std, it, photos[w].std.width, photos[w].std.height) }
            if (crop != null) {
                val cropLab = LabImage(crop)
                val ce = SceneMatch.applyLook(cropLab, SceneMatch.lookRecipe(cropLab, labs[w], colorOK = true))
                val pc = scoreEdited(ce)
                previews.add(Preview("crop_edit", ce, pc.second, pc.first[ia], pc.first[it], closure(pc.second)))
            }
        }
        check()

        progress(Stage.COACHING)
        val scale = max(mA.detail.max, mB.detail.max)
        for (k in 0 until 2) {
            extras[k].layers[Lens.CLIPPING] = Overlays.clipping(photos[k].std)
            extras[k].layers[Lens.FOCUS] = Overlays.focus(photos[k].std, photos[k].m.detail, scale)
            extras[k].layers[Lens.ATTENTION] = Overlays.impact(photos[k].std, heat[k])
            if (extras[k].objects.isNotEmpty()) extras[k].layers[Lens.OBJECTS] = Overlays.objects(photos[k].std, extras[k].objects)
        }
        val ctx = Coach.Input(info, photos, w, l, recipe, frame, relation, extras.map { Pair(it.objects, it.geometry) })
        return Coach.build(ctx, pa, extras, sim, previews, (System.nanoTime() - t0) / 1e9)
    }
}

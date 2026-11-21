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
        /** Thread priority of the plain-CPU stages (see cpuPool). */
        @Volatile var cpuPriority: Int = Thread.MIN_PRIORITY

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

    private fun embedViews(std: RGBImage): List<FloatArray> = nets.embed(Prof.t("views") { ImagePrep.views(std, info.inputSize) })

    private fun featurize(std: RGBImage, m: Measurement, name: String, d: Decoded, exif: ExifInfo, e: List<FloatArray> = embedViews(std)): ScoredPhoto {
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
        val e = nets.embed(Prof.t("views") { ImagePrep.views(img, info.inputSize) })
        val pct = percentiles(nets.tower(e[0], e.subList(1, e.size), towerSignals(Prof.t("toneColor") { Signals.toneColor(img) })))
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
        var segs = Prof.t("detect (incl. segmenter)") { Objects.detect(p.std, nets.segmenter) }.filter { if (it.thing) it.area >= 0.002 else it.area >= 0.015 }
        segs = segs.sortedByDescending { it.area * (if (it.thing) 3 else 1) }.take(maxObjects).sortedByDescending { it.area }
        val ia = info.ia
        val hv = towerSignals(p.m.s)
        check()
        // painting out is plain CPU work, so the objects are prepared side by side, then embedded in batches
        val views = Prof.t("paint-out wall") {
            segs.map { g -> cpuPool.submit(java.util.concurrent.Callable { check(); ImagePrep.views(Prof.t("remove") { Objects.remove(p.std, g) }, info.inputSize) }) }.map { it.await() }
        }
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

    // ---- running stages side by side

    /**
     * Independent stages run at the same time: the networks keep the fast cores busy while measuring,
     * scene matching, painting and rendering use the others.  Every stage computes exactly what it
     * would alone, so results do not depend on the timing.
     */
    private val pool = java.util.concurrent.Executors.newCachedThreadPool { r -> Thread(r, "judge").apply { isDaemon = true } }

    /**
     * Plain CPU work (measuring, matching, painting, rendering) runs at low priority, so on phones with
     * a few fast cores and several slow ones it yields the fast cores to the networks.
     */
    private val cpuPool = java.util.concurrent.Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors().coerceIn(2, 8)) { r ->
        Thread(r, "judge-cpu").apply { isDaemon = true; priority = cpuPriority }
    }

    private class Run(val check: () -> Unit) {
        @Volatile var stopped = false
        /** Called inside stages: stops work that is no longer wanted. */
        fun ok() { if (stopped) throw java.util.concurrent.CancellationException(); check() }
    }

    private fun <T> Run.go(body: () -> T): java.util.concurrent.Future<T> = pool.submit(java.util.concurrent.Callable { ok(); body() })

    private fun <T> Run.cpu(body: () -> T): java.util.concurrent.Future<T> = cpuPool.submit(java.util.concurrent.Callable { ok(); body() })

    private fun <T> java.util.concurrent.Future<T>.await(): T = try { get() } catch (e: java.util.concurrent.ExecutionException) { throw e.cause ?: e }

    private inline fun <T> running(noinline check: () -> Unit, body: (Run) -> T): T {
        val run = Run(check)
        try { return body(run) } catch (e: Throwable) { run.stopped = true; throw e }
    }

    private class First(val dA: Decoded, val dB: Decoded, val a: ScoredPhoto, val b: ScoredPhoto, val au: List<Authenticity>)

    /** Read, measure, score and AI-check both photos: everything the verdict needs. */
    private fun first(
        run: Run, bytesA: ByteArray, bytesB: ByteArray, nameA: String, nameB: String, exif: Boolean,
        known: List<Authenticity>?, progress: (Stage) -> Unit,
    ): First {
        progress(Stage.READING)
        val (dA, dB) = Prof.t("decode") { both({ decoder.decode(bytesA) }, { decoder.decode(bytesB) }) }
        val (stdA, stdB) = Prof.t("standard image") { both({ ImagePrep.standardImage(dA.full, info.stdLongSide) }, { ImagePrep.standardImage(dB.full, info.stdLongSide) }) }
        run.ok()

        progress(Stage.MEASURING)
        val fmA = run.cpu { Prof.t("measure") { Signals.measure(stdA) } }
        val fmB = run.cpu { Prof.t("measure") { Signals.measure(stdB) } }
        val fe = run.go { Pair(embedViews(stdA), embedViews(stdB)) }
        val fau = run.go { Prof.t("authenticity") { authenticity(bytesA, dA, bytesB, dB, known) } }
        val mA = fmA.await(); val mB = fmB.await()
        run.ok()

        progress(Stage.SCORING)
        val (eA, eB) = fe.await()
        val a = featurize(stdA, mA, nameA, dA, if (exif) dA.exif else ExifInfo(), eA)
        val b = featurize(stdB, mB, nameB, dB, if (exif) dB.exif else ExifInfo(), eB)
        run.ok()

        progress(Stage.AUTHENTICITY)
        return First(dA, dB, a, b, fau.await())
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
    ): QuickVerdict = running(check) { run ->
        val t0 = System.nanoTime()
        val f = first(run, bytesA, bytesB, "Shot A", "Shot B", false, known, progress)
        QuickVerdict(listOf(f.a.std, f.b.std), listOf(f.a.aspect, f.b.aspect), pA(f.a, f.b), (System.nanoTime() - t0) / 1e9, f.au)
    }

    // ---- the twin comparison

    private class Scene(val al: Alignment, val relation: String, val frame: Framing?, val masks: Array<BooleanArray?>)

    private class Found(val objects: List<Segment>, val geometry: SceneGeometry?, val overlay: RGBImage?)

    fun compare(
        bytesA: ByteArray, bytesB: ByteArray, nameA: String, nameB: String, known: List<Authenticity>? = null,
        progress: (Stage) -> Unit = {}, check: () -> Unit = {},
        /** Receives the verdict as soon as it is known (scores, attributes, AI check), before the slow
         *  parts - objects, attention, edit previews and the coaching built on them - are worked out. */
        early: (Comparison) -> Unit = {},
    ): Comparison = running(check) { run ->
        val t0 = System.nanoTime()
        val f = first(run, bytesA, bytesB, nameA, nameB, true, known, progress)
        val a = f.a; val b = f.b; val au = f.au
        val stdA = a.std; val stdB = b.std
        val photos = listOf(a, b)

        // verdict
        val ia = info.ia; val it = info.itech
        val pa = pA(a, b)
        val w = if (pa >= 0.5) 0 else 1
        val l = 1 - w
        val sim = cosine(a.g, b.g)
        early(Coach.build(
            Coach.Input(info, photos, w, l, null, null, if (sim >= similarScene) "similar" else "different", listOf(Pair(emptyList(), null), Pair(emptyList(), null))),
            pa, listOf(PhotoExtras(authenticity = au[0]), PhotoExtras(authenticity = au[1])), sim, emptyList(), (System.nanoTime() - t0) / 1e9,
        ))
        run.ok()

        // everything below starts now and is collected in the order the stages are shown
        progress(Stage.MATCHING)
        val fscene = run.cpu {
            val al = Prof.t("align") { SceneMatch.align(stdA, stdB) }
            val relation = if (al.aligned) "aligned" else if (sim >= similarScene) "similar" else "different"
            var frame: Framing? = null
            val masks = arrayOfNulls<BooleanArray>(2)
            val hinv = if (al.aligned) al.H.inverse else null
            if (al.aligned && hinv != null) {
                masks[0] = SceneMatch.overlapMask(stdA.width, stdA.height, al.H)
                masks[1] = SceneMatch.overlapMask(stdB.width, stdB.height, hinv)
                val quad = if (l == 0) al.quadBinA else al.quadAinB      // the winner's frame inside the loser's
                frame = SceneMatch.framing(quad, photos[l].aspect)
            }
            Scene(al, relation, frame, masks)
        }
        val ffound = photos.map { p ->
            run.go {
                val segs = objectEffects(p, run::ok)
                val si = Objects.pickSubject(segs, p.m["subject_x"], p.m["subject_y"])
                Objects.assignRoles(segs, si)
                Found(
                    segs, Prof.t("geometry") { Objects.geometry(p.std, segs, si, p.m.detail) },
                    if (segs.isNotEmpty()) Prof.t("overlay objects") { Overlays.objects(p.std, segs) } else null,
                )
            }
        }
        val fheat = run.go {
            Prof.t("occlusion total") { photos.map { p -> run.ok(); val h = occlusion(p); Prof.t("overlay attention") { Overlays.impact(p.std, h) } } }
        }
        val scale = max(a.m.detail.max, b.m.detail.max)
        val fbase = photos.map { p ->
            run.cpu {
                Pair(Prof.t("overlay clipping") { Overlays.clipping(p.std) }, Prof.t("overlay focus") { Overlays.focus(p.std, p.m.detail, scale) })
            }
        }
        val flabs = run.cpu { Prof.t("lab") { listOf(LabImage(stdA), LabImage(stdB)) } }
        val fedit = run.go {
            val labs = flabs.await()
            val sc = fscene.await()
            run.ok()
            val recipe = Prof.t("lookRecipe") { SceneMatch.lookRecipe(labs[l], labs[w], sc.masks[l], sc.masks[w], sc.relation != "different") }
            val oW = photos[w].overall; val oL = photos[l].overall
            fun closure(o: Double) = max(-1.0, min(1.5, (o - oL) / max(oW - oL, 1e-6)))
            val previews = ArrayList<Preview>()
            val edited = Prof.t("applyLook") { SceneMatch.applyLook(labs[l], recipe) }
            val pe = scoreEdited(edited)
            previews.add(Preview("edit", edited, pe.second, pe.first[ia], pe.first[it], closure(pe.second)))
            val fr = sc.frame
            if (fr != null && fr.zoom >= 1.08 && fr.inside >= 0.85) {
                val hlw = if (l == 0) sc.al.H else sc.al.H.inverse
                val crop = hlw?.let { SceneMatch.cropToOther(photos[l].std, it, photos[w].std.width, photos[w].std.height) }
                if (crop != null) {
                    val cropLab = LabImage(crop)
                    val ce = SceneMatch.applyLook(cropLab, SceneMatch.lookRecipe(cropLab, labs[w], colorOK = true))
                    val pc = scoreEdited(ce)
                    previews.add(Preview("crop_edit", ce, pc.second, pc.first[ia], pc.first[it], closure(pc.second)))
                }
            }
            Pair(recipe, previews)
        }

        val sc = fscene.await()
        val extras = listOf(PhotoExtras(authenticity = au[0]), PhotoExtras(authenticity = au[1]))
        if (sc.frame != null) {
            extras[0].otherFrame = sc.al.quadBinA
            extras[1].otherFrame = sc.al.quadAinB
        }
        run.ok()

        progress(Stage.OBJECTS)
        progress(Stage.TESTING)
        for (k in 0 until 2) {
            val found = ffound[k].await()
            extras[k].objects = found.objects
            extras[k].geometry = found.geometry
            found.overlay?.let { extras[k].layers[Lens.OBJECTS] = it }
            run.ok()
        }

        progress(Stage.ATTENTION)
        val heat = fheat.await()
        run.ok()

        progress(Stage.EDITING)
        val (recipe, previews) = fedit.await()
        run.ok()

        progress(Stage.COACHING)
        for (k in 0 until 2) {
            val base = fbase[k].await()
            extras[k].layers[Lens.CLIPPING] = base.first
            extras[k].layers[Lens.FOCUS] = base.second
            extras[k].layers[Lens.ATTENTION] = heat[k]
        }
        val ctx = Coach.Input(info, photos, w, l, recipe, sc.frame, sc.relation, extras.map { Pair(it.objects, it.geometry) })
        Prof.t("coach build") { Coach.build(ctx, pa, extras, sim, previews, (System.nanoTime() - t0) / 1e9) }
    }
}

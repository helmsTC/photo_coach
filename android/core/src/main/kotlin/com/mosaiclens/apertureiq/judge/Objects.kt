package com.mosaiclens.apertureiq.judge

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** One thing in the frame: its mask, where it sits, and (after re-scoring) what it does to the score. */
class Segment(
    val label: String,
    val score: Double,
    val mask: BooleanArray,              // std resolution
    val w: Int,
    val h: Int,
    val area: Double,
    val cx: Double,
    val cy: Double,
    val box: DoubleArray,                // x0, y0, x1, y1 (0...1)
    val edges: Map<String, Double>,      // share of each frame edge the object lies against
    val thing: Boolean,                  // countable object (vs background "stuff": sky, water, trees...)
) {
    var n = 0
    var delta = 0.0                      // aesthetic percentile change with it painted out (+ = it distracts)
    var lean = 0.0                       // how much the score leans on it, in reference standard deviations
    var role = "other"                   // subject / distraction / helps / other
    var labelPoint = Pair(0.5, 0.5)      // deepest point inside the mask, for the number badge

    val compact: Boolean get() = area <= 0.12
}

data class SubjectInfo(
    val label: String,
    val area: Double,
    val cx: Double,
    val cy: Double,
    val whereText: String,
    val thirdsDist: Double,
    val centerDist: Double,
    val cut: List<String>,
    val separation: Double?,
    val sharpRatio: Double?,
)

class SceneGeometry(
    var subject: SubjectInfo? = null,
    var subjectIndex: Int? = null,
    var nThings: Int = 0,
    var sky: Double = 0.0,               // share of the frame
    var water: Double = 0.0,
    var horizonY: Double? = null,        // where sky meets land, as a fraction of height
)

/** The COCO-panoptic segmenter: a class id per pixel of a square input. */
interface SemanticSegmenter {
    val size: Int
    /** Class names by id; "--" marks unused ids. */
    val labels: List<String>
    fun isThing(id: Int): Boolean
    /** Class id per pixel on a size x size grid (the whole frame, squashed), or null if unavailable. */
    fun classMap(square: RGBImage): IntArray?
}

/**
 * What is in the frame: a COCO-panoptic segmenter (DETR-ResNet50, the same 133 classes as the
 * desktop's Mask2Former: 80 countable things + background stuff such as sky, water and trees),
 * then the subject / distraction logic of photocoach/objects.py.  The segmenter labels pixels by
 * class and does not separate instances, so objects are connected regions of a class.
 */
object Objects {

    /** Desktop naming: "sky (other)" -> "sky", "tree" -> "trees", "river" -> "water" ... */
    private val rename = mapOf(
        "sky" to "sky", "water" to "water", "river" to "water", "tree" to "trees", "rock" to "rocks", "dirt" to "ground",
        "potted plant" to "plant", "floor" to "floor", "wall" to "wall", "window" to "window", "food" to "food",
        "building" to "building", "playingfield" to "playing field", "tv" to "TV",
    )

    fun displayName(raw: String): String {
        var n = raw
        val p = n.indexOf(" (")
        if (p >= 0) n = n.substring(0, p)
        return rename[n] ?: n
    }

    /** 8-connected components of `cls` in the label grid; returns one id map (0 = none) and the count. */
    private fun components(map: IntArray, cls: Int, n: Int): Pair<IntArray, Int> {
        val comp = IntArray(n * n)
        var next = 0
        val stack = IntArray(n * n)
        for (start in 0 until n * n) {
            if (map[start] != cls || comp[start] != 0) continue
            next++
            comp[start] = next
            var sp = 0
            stack[sp++] = start
            while (sp > 0) {
                val i = stack[--sp]
                val x = i % n; val y = i / n
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue
                        val xx = x + dx; val yy = y + dy
                        if (xx < 0 || yy < 0 || xx >= n || yy >= n) continue
                        val j = yy * n + xx
                        if (map[j] == cls && comp[j] == 0) { comp[j] = next; stack[sp++] = j }
                    }
                }
            }
        }
        return Pair(comp, next)
    }

    /** Upsample a grid selection to a std-resolution mask. */
    private inline fun upsample(n: Int, w: Int, h: Int, sel: (Int) -> Boolean): BooleanArray {
        val m = BooleanArray(w * h)
        for (y in 0 until h) {
            val row = min(n - 1, y * n / h) * n
            for (x in 0 until w) if (sel(row + min(n - 1, x * n / w))) m[y * w + x] = true
        }
        return m
    }

    fun detect(img: RGBImage, seg: SemanticSegmenter?): List<Segment> {
        if (seg == null) return emptyList()
        val w = img.width; val h = img.height; val n = seg.size
        val map = seg.classMap(ImagePrep.resizeArea(img, n, n)) ?: return emptyList()
        if (map.size != n * n) return emptyList()
        val minPix = (0.002 * n * n).toInt()

        val counts = java.util.TreeMap<Int, Int>()
        for (c in map) counts[c] = (counts[c] ?: 0) + 1
        val segs = ArrayList<Segment>()
        val stuff = LinkedHashMap<String, ArrayList<Int>>()           // fused background classes by display name
        for ((cls, cnt) in counts) {
            if (cls <= 0 || cls >= seg.labels.size) continue
            val raw = seg.labels[cls]
            if (raw == "--") continue
            if (seg.isThing(cls)) {
                if (cnt < minPix) continue
                val (comp, k) = components(map, cls, n)
                val sizes = IntArray(k + 1)
                for (c in comp) if (c > 0) sizes[c]++
                for (id in 1..k) {
                    if (sizes[id] < minPix) continue
                    val m = upsample(n, w, h) { comp[it] == id }
                    describe(m, w, h, displayName(raw), 1.0, true)?.let { segs.add(it) }
                }
            } else {
                stuff.getOrPut(displayName(raw)) { ArrayList() }.add(cls)
            }
        }
        for ((name, classes) in stuff) {
            // fuse the classes, then drop specks (the desktop overlay ignores contours under 0.04% of the frame)
            val first = classes[0]
            val fused = IntArray(map.size) { if (classes.contains(map[it])) first else -1 }
            val (comp, k) = components(fused, first, n)
            val sizes = IntArray(k + 1)
            for (c in comp) if (c > 0) sizes[c]++
            val speck = (0.0004 * n * n).toInt()
            val m = upsample(n, w, h) { comp[it] > 0 && sizes[comp[it]] >= speck }
            val s = describe(m, w, h, name, 1.0, false)
            if (s != null && s.area >= 0.0008) segs.add(s)
        }
        // a region covering nearly the whole frame is not something to analyse
        return segs.filter { it.area <= 0.95 }.sortedByDescending { it.area }
    }

    // ---- geometry of one mask

    fun describe(m: BooleanArray, w: Int, h: Int, label: String, score: Double, thing: Boolean = true): Segment? {
        var n = 0; var sx = 0.0; var sy = 0.0
        var x0 = w; var x1 = 0; var y0 = h; var y1 = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                if (!m[y * w + x]) continue
                n++; sx += x; sy += y
                x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
            }
        }
        if (n == 0) return null
        val bw = max(2, rndInt(0.012 * w)); val bh = max(2, rndInt(0.012 * h))
        var left = 0; var right = 0; var top = 0; var bottom = 0
        for (y in 0 until h) {
            var l = false; var r = false
            for (x in 0 until bw) if (m[y * w + x]) { l = true; break }
            for (x in w - bw until w) if (m[y * w + x]) { r = true; break }
            if (l) left++
            if (r) right++
        }
        for (x in 0 until w) {
            var t = false; var b = false
            for (y in 0 until bh) if (m[y * w + x]) { t = true; break }
            for (y in h - bh until h) if (m[y * w + x]) { b = true; break }
            if (t) top++
            if (b) bottom++
        }
        val s = Segment(
            label, score, m, w, h, n.toDouble() / (w * h),
            sx / n / w, sy / n / h,
            doubleArrayOf(x0.toDouble() / w, y0.toDouble() / h, (x1 + 1).toDouble() / w, (y1 + 1).toDouble() / h),
            mapOf("left" to left.toDouble() / h, "right" to right.toDouble() / h, "top" to top.toDouble() / w, "bottom" to bottom.toDouble() / w),
            thing,
        )
        s.labelPoint = deepestPoint(m, w, h)
        return s
    }

    /** "lower left", "centre", "right edge" … */
    fun whereText(g: Segment): String {
        val bw = g.box[2] - g.box[0]; val bh = g.box[3] - g.box[1]
        if (g.area < 0.25) {
            for (side in listOf("left", "right", "top", "bottom")) {
                val horizontalSide = side == "left" || side == "right"
                if ((g.edges[side] ?: 0.0) > 0.02 && (bw < 0.5 || !horizontalSide) && (bh < 0.5 || horizontalSide)) return "$side edge"
            }
        }
        val v = if (g.cy < 0.38) "upper" else if (g.cy > 0.62) "lower" else ""
        val hz = if (g.cx < 0.38) "left" else if (g.cx > 0.62) "right" else ""
        val s = "$v $hz".trim()
        return s.ifEmpty { "centre" }
    }

    /** Background classes that can still be a subject (a cabin, a bridge) rather than plain surroundings. */
    val stuffSubjects = setOf("house", "bridge", "tent", "flower", "tower", "building", "lighthouse")
    val waterNames = setOf("water", "sea")

    /** Frame edges that slice through the object. */
    fun cutOff(g: Segment): List<String> {
        if (!(g.thing || stuffSubjects.contains(g.label))) return emptyList()
        val bw = g.box[2] - g.box[0]; val bh = g.box[3] - g.box[1]
        val out = ArrayList<String>()
        for ((side, span) in listOf("left" to bh, "right" to bh, "top" to bw, "bottom" to bw)) {
            val extent = g.edges[side] ?: 0.0
            if (extent >= 0.25 * max(span, 1e-6) && extent >= 0.03) out.add(side)
        }
        return out
    }

    private val prior = mapOf(
        "person" to 1.6, "bird" to 1.4, "cat" to 1.4, "dog" to 1.4, "horse" to 1.4, "sheep" to 1.3, "cow" to 1.3, "elephant" to 1.4,
        "bear" to 1.4, "zebra" to 1.4, "giraffe" to 1.4, "boat" to 1.2, "house" to 1.1, "bridge" to 1.1, "tent" to 1.1,
        "flower" to 1.1, "airplane" to 1.2, "train" to 1.2, "bicycle" to 1.1, "motorcycle" to 1.1, "car" to 1.0,
        "building" to 1.1, "lighthouse" to 1.2, "canoe" to 1.2, "kayak" to 1.2,
        "dining table" to 0.45, "table" to 0.45, "bed" to 0.5, "sofa" to 0.55, "couch" to 0.55, "chair" to 0.6, "bench" to 0.6, "TV" to 0.7,
    )

    /** Index of the main subject, or null for a scene with no single object. */
    fun pickSubject(segs: List<Segment>, salX: Double, salY: Double): Int? {
        var best: Int? = null
        var bestS = 0.0
        for ((i, g) in segs.withIndex()) {
            if (!(g.thing || stuffSubjects.contains(g.label)) || g.area < 0.008 || g.area > 0.7) continue
            val central = 1 - min(1.0, hypot(g.cx - 0.5, g.cy - 0.5) / 0.6)
            var s = sqrt(g.area) * (0.45 + central) * (prior[g.label] ?: (if (g.thing) 0.9 else 0.8))
            if (g.box[0] <= salX && salX <= g.box[2] && g.box[1] <= salY && salY <= g.box[3]) s *= 1.3
            s *= 1 + 2 * max(0.0, g.lean)
            if (s > bestS) { best = i; bestS = s }
        }
        return if (bestS >= 0.06) best else null
    }

    fun assignRoles(segs: List<Segment>, subject: Int?) {
        for ((i, g) in segs.withIndex()) {
            g.role = when {
                i == subject -> "subject"
                // landmarks (a cabin, a bridge) are never "distractions": you cannot step around them
                g.area <= 0.12 && g.delta >= 3 && g.lean <= -0.10 && !stuffSubjects.contains(g.label) -> "distraction"
                g.lean >= 0.30 -> "helps"
                else -> "other"
            }
        }
    }

    private fun thirdsDist(x: Double, y: Double): Double {
        var best = Double.POSITIVE_INFINITY
        for (px in doubleArrayOf(1.0 / 3, 2.0 / 3)) for (py in doubleArrayOf(1.0 / 3, 2.0 / 3)) best = min(best, hypot(x - px, y - py))
        return best
    }

    fun geometry(img: RGBImage, segs: List<Segment>, subject: Int?, detail: Plane): SceneGeometry {
        val w = img.width; val h = img.height
        val out = SceneGeometry(subjectIndex = subject, nThings = segs.count { it.thing })
        for (g in segs) {
            if (g.label == "sky") out.sky += g.area
            if (waterNames.contains(g.label)) out.water += g.area
        }
        val sky = segs.firstOrNull { it.label == "sky" }
        if (sky != null && sky.area >= 0.04) {
            val lows = ArrayList<Float>()
            var x = 0
            while (x < w) {
                var low = -1
                for (y in 0 until h) if (sky.mask[y * w + x]) low = y
                if (low >= 0) lows.add(low.toFloat())
                x += 2
            }
            if (lows.size >= 0.6 * ((w + 1) / 2)) out.horizonY = Imaging.median(lows.toFloatArray()) / h
        }
        val si = subject ?: return out
        val g = segs[si]
        // subject vs its surroundings at quarter resolution
        val sw = max(64, w / 4); val sh = max(64, h / 4)
        val small = ImagePrep.resizeArea(img, sw, sh)
        val ms = BooleanArray(sw * sh)
        var cnt = 0
        for (y in 0 until sh) {
            for (x in 0 until sw) {
                val v = g.mask[min(h - 1, y * h / sh) * w + min(w - 1, x * w / sw)]
                ms[y * sw + x] = v
                if (v) cnt++
            }
        }
        var sep: Double? = null
        if (cnt >= 12) {
            val r = max(3.0, rnd(0.35 * sqrt(cnt.toDouble()))).toFloat()
            val dist = distanceToTrue(ms, sw, sh)
            val inS = DoubleArray(3); val inR = DoubleArray(3)
            var nS = 0.0; var nR = 0.0
            val t = FloatArray(3)
            for (i in 0 until sw * sh) {
                Imaging.lab(small.data.u(i * 3), small.data.u(i * 3 + 1), small.data.u(i * 3 + 2), t)
                if (ms[i]) { inS[0] += t[0]; inS[1] += t[1]; inS[2] += t[2]; nS += 1 }
                else if (dist[i] <= r) { inR[0] += t[0]; inR[1] += t[1]; inR[2] += t[2]; nR += 1 }
            }
            if (nR >= 12) {
                var d2 = 0.0
                for (k in 0 until 3) { val d = inS[k] / nS - inR[k] / nR; d2 += d * d }
                sep = sqrt(d2)
            }
        }
        val d = Imaging.resize(detail, w, h)
        val inside = ArrayList<Float>(); val outside = ArrayList<Float>()
        var i = 0
        while (i < w * h) { if (g.mask[i]) inside.add(d.d[i]) else outside.add(d.d[i]); i += 7 }
        val sharp = if (inside.isEmpty() || outside.isEmpty()) null
        else Imaging.median(inside.toFloatArray()) / (Imaging.median(outside.toFloatArray()) + 1e-6)
        out.subject = SubjectInfo(
            g.label, g.area, g.cx, g.cy, whereText(g), thirdsDist(g.cx, g.cy), hypot(g.cx - 0.5, g.cy - 0.5),
            cutOff(g), sep, sharp,
        )
        return out
    }

    // ---- distance transform (chamfer, in pixels)

    /** Distance from every pixel to the nearest `true` pixel. */
    fun distanceToTrue(m: BooleanArray, w: Int, h: Int): FloatArray {
        val inf = 1e9f
        val d = FloatArray(w * h) { if (m[it]) 0f else inf }
        val a = 1f; val b = 1.4142f
        for (y in 0 until h) {
            for (x in 0 until w) {
                var v = d[y * w + x]
                if (x > 0) v = min(v, d[y * w + x - 1] + a)
                if (y > 0) {
                    v = min(v, d[(y - 1) * w + x] + a)
                    if (x > 0) v = min(v, d[(y - 1) * w + x - 1] + b)
                    if (x < w - 1) v = min(v, d[(y - 1) * w + x + 1] + b)
                }
                d[y * w + x] = v
            }
        }
        for (y in h - 1 downTo 0) {
            for (x in w - 1 downTo 0) {
                var v = d[y * w + x]
                if (x < w - 1) v = min(v, d[y * w + x + 1] + a)
                if (y < h - 1) {
                    v = min(v, d[(y + 1) * w + x] + a)
                    if (x < w - 1) v = min(v, d[(y + 1) * w + x + 1] + b)
                    if (x > 0) v = min(v, d[(y + 1) * w + x - 1] + b)
                }
                d[y * w + x] = v
            }
        }
        return d
    }

    /** Point deepest inside the mask (normalised), so a number badge sits on the object. */
    fun deepestPoint(m: BooleanArray, w: Int, h: Int): Pair<Double, Double> {
        val s = 4
        val sw = max(1, w / s); val sh = max(1, h / s)
        val small = BooleanArray(sw * sh)
        for (y in 0 until sh) for (x in 0 until sw) small[y * sw + x] = m[min(h - 1, y * s) * w + min(w - 1, x * s)]
        // distance to the nearest outside pixel, with the frame border counting as outside
        val outside = BooleanArray(sw * sh) { !small[it] }
        for (x in 0 until sw) { outside[x] = true; outside[(sh - 1) * sw + x] = true }
        for (y in 0 until sh) { outside[y * sw] = true; outside[y * sw + sw - 1] = true }
        val d = distanceToTrue(outside, sw, sh)
        var best = 0
        for (i in d.indices) if (small[i] && (d[i] > d[best] || !small[best])) best = i
        return Pair((best % sw + 0.5) / sw, (best / sw + 0.5) / sh)
    }

    // ---- painting an object out

    /** Push-pull hole filling: average known pixels down a pyramid, then fill holes from coarser levels. */
    fun inpaint(img: RGBImage, hole: BooleanArray): RGBImage {
        // each level: rgb premultiplied by weight, weight in the 4th slot
        val ws = ArrayList<Int>(); val hs = ArrayList<Int>(); val cs = ArrayList<FloatArray>()
        val base = FloatArray(img.width * img.height * 4)
        for (i in 0 until img.width * img.height) {
            if (hole[i]) continue
            base[i * 4] = img.data.u(i * 3).toFloat(); base[i * 4 + 1] = img.data.u(i * 3 + 1).toFloat()
            base[i * 4 + 2] = img.data.u(i * 3 + 2).toFloat(); base[i * 4 + 3] = 1f
        }
        ws.add(img.width); hs.add(img.height); cs.add(base)
        while (ws.last() > 1 || hs.last() > 1) {
            val pw = ws.last(); val ph = hs.last(); val pc = cs.last()
            val w = max(1, (pw + 1) / 2); val h = max(1, (ph + 1) / 2)
            val c = FloatArray(w * h * 4)
            for (y in 0 until ph) {
                for (x in 0 until pw) {
                    val o = ((y / 2) * w + x / 2) * 4; val s = (y * pw + x) * 4
                    c[o] += pc[s]; c[o + 1] += pc[s + 1]; c[o + 2] += pc[s + 2]; c[o + 3] += pc[s + 3]
                }
            }
            ws.add(w); hs.add(h); cs.add(c)
        }
        // push: resolve the coarsest level, then each finer level from its parent
        val top = cs.last()
        var parent = FloatArray(top.size)
        for (i in 0 until top.size / 4) {
            val wt = top[i * 4 + 3]
            if (wt > 0) {
                parent[i * 4] = top[i * 4] / wt; parent[i * 4 + 1] = top[i * 4 + 1] / wt; parent[i * 4 + 2] = top[i * 4 + 2] / wt
            } else {
                parent[i * 4] = 128f; parent[i * 4 + 1] = 128f; parent[i * 4 + 2] = 128f
            }
        }
        for (li in cs.size - 2 downTo 0) {
            val lw = ws[li]; val lh = hs[li]; val lc = cs[li]
            val pw = ws[li + 1]; val ph = hs[li + 1]
            val cur = FloatArray(lw * lh * 4)
            for (y in 0 until lh) {
                val fy = max(0f, min((ph - 1).toFloat(), (y + 0.5f) / 2 - 0.5f))
                val y0 = fy.toInt(); val y1 = min(ph - 1, y0 + 1); val ty = fy - y0
                for (x in 0 until lw) {
                    // bilinear sample of the parent at this pixel's centre
                    val fx = max(0f, min((pw - 1).toFloat(), (x + 0.5f) / 2 - 0.5f))
                    val x0 = fx.toInt(); val x1 = min(pw - 1, x0 + 1); val tx = fx - x0
                    val o = (y * lw + x) * 4
                    val wt = lc[o + 3]
                    val a = min(1f, wt)
                    for (k in 0 until 3) {
                        val up = (parent[(y0 * pw + x0) * 4 + k] * (1 - tx) + parent[(y0 * pw + x1) * 4 + k] * tx) * (1 - ty) +
                            (parent[(y1 * pw + x0) * 4 + k] * (1 - tx) + parent[(y1 * pw + x1) * 4 + k] * tx) * ty
                        val own = if (wt > 0) lc[o + k] / wt else up
                        cur[o + k] = own * a + up * (1 - a)
                    }
                }
            }
            parent = cur
        }
        val out = img.data.copyOf()
        for (i in 0 until img.width * img.height) {
            if (!hole[i]) continue
            out[i * 3] = clampByte(parent[i * 4]); out[i * 3 + 1] = clampByte(parent[i * 4 + 1]); out[i * 3 + 2] = clampByte(parent[i * 4 + 2])
        }
        return RGBImage(img.width, img.height, out)
    }

    /**
     * The photo without this region.  Compact objects are painted out from their surroundings;
     * large regions cannot be invented, so they are replaced by a heavy blur of themselves.
     */
    fun remove(img: RGBImage, g: Segment): RGBImage {
        val w = img.width; val h = img.height
        val mx = max(w, h).toDouble()
        val fill: RGBImage
        val alpha: Plane
        if (g.area <= 0.12) {
            val s = 384 / mx
            val sw = max(1, rndInt(w * s)); val sh = max(1, rndInt(h * s))
            val sm = ImagePrep.resizeArea(img, sw, sh)
            var hole = BooleanArray(sw * sh)
            for (y in 0 until sh) for (x in 0 until sw) hole[y * sw + x] = g.mask[min(h - 1, y * h / sh) * w + min(w - 1, x * w / sw)]
            val dist = distanceToTrue(hole, sw, sh)
            val r = max(3, rndInt(0.012 * 384)).toFloat()
            hole = BooleanArray(sw * sh) { dist[it] <= r }
            fill = Imaging.resizeBilinear(inpaint(sm, hole), w, h)
            val a = Plane(sw, sh, FloatArray(sw * sh) { if (hole[it]) 1f else 0f })
            alpha = Imaging.gaussianBlur(Imaging.resize(a, w, h), 0.006 * mx)
        } else {
            val q = ImagePrep.resizeArea(img, max(8, w / 6), max(8, h / 6))
            fill = Imaging.resizeBilinear(Imaging.blurRGB(q, 0.01 * mx), w, h)
            alpha = Imaging.gaussianBlur(Plane(w, h, FloatArray(w * h) { if (g.mask[it]) 1f else 0f }), 0.008 * mx)
        }
        val out = img.data.copyOf()
        for (i in 0 until w * h) {
            val a = max(0f, min(1f, alpha.d[i]))
            if (a <= 0.002f) continue
            for (k in 0 until 3) {
                out[i * 3 + k] = clampByte(img.data.u(i * 3 + k) * (1 - a) + fill.data.u(i * 3 + k) * a)
            }
        }
        return RGBImage(w, h, out)
    }
}

package com.mosaiclens.apertureiq.judge

import java.util.Locale
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/** String.format in a fixed locale, so decimals never turn into commas. */
fun usf(format: String, vararg args: Any?): String = String.format(Locale.US, format, *args)

class Pt(val x: Double, val y: Double)

typealias Quad = List<Pt>   // TL, TR, BR, BL

/** 3x3 homography on normalised (0...1) image coordinates, row-major. */
class H3(val m: DoubleArray) {

    fun project(pts: List<Pt>): List<Pt> = pts.map {
        val w = m[6] * it.x + m[7] * it.y + m[8]
        Pt((m[0] * it.x + m[1] * it.y + m[2]) / w, (m[3] * it.x + m[4] * it.y + m[5]) / w)
    }

    operator fun times(b: H3): H3 {
        val r = DoubleArray(9)
        for (i in 0 until 3) for (j in 0 until 3) for (k in 0 until 3) r[i * 3 + j] += m[i * 3 + k] * b.m[k * 3 + j]
        return H3(r)
    }

    val inverse: H3?
        get() {
            val a = m
            val det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6])
            if (abs(det) <= 1e-12) return null
            val inv = doubleArrayOf(
                a[4] * a[8] - a[5] * a[7], a[2] * a[7] - a[1] * a[8], a[1] * a[5] - a[2] * a[4],
                a[5] * a[6] - a[3] * a[8], a[0] * a[8] - a[2] * a[6], a[2] * a[3] - a[0] * a[5],
                a[3] * a[7] - a[4] * a[6], a[1] * a[6] - a[0] * a[7], a[0] * a[4] - a[1] * a[3],
            )
            return H3(DoubleArray(9) { inv[it] / det })
        }

    companion object {
        val identity get() = H3(doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    }
}

class Alignment(
    var aligned: Boolean = false,
    var H: H3 = H3.identity,              // A (0..1) -> B (0..1)
    var quadBinA: Quad = emptyList(),
    var quadAinB: Quad = emptyList(),
    var match: Double = 0.0,              // structure correlation of the matched frames
    var coverage: Double = 0.0,
)

/** Where the OTHER photo's frame sits inside MY frame (scene.framing). */
class Framing(
    val areaFrac: Double,
    val zoom: Double,
    val dx: Double,
    val dy: Double,
    val rotation: Double,
    val inside: Double,
    val quad: Quad,
)

/** Editing-slider style numbers that move one photo's tone / colour toward another's. */
class LookRecipe(
    val exposureEV: Double,
    val contrastPct: Double,
    val highlights: Double,
    val shadows: Double,
    val blacks: Double,
    val whites: Double,
    val saturationPct: Double,
    val temp: Double,
    val tint: Double,
    val qs: DoubleArray,
    val qr: DoubleArray,
    val sat: Double,
    val colorOK: Boolean,
) {
    /** `amount` is signed, roughly -100...100, for drawing a slider. */
    class Line(val name: String, val value: String, val amount: Double)

    /** coach.recipe_lines */
    val lines: List<Line>
        get() {
            val out = ArrayList<Line>()
            fun line(name: String, v: Double, unit: String, thr: Double, f: String, scale: Double) {
                if (abs(v) >= thr) out.add(Line(name, usf(f, v) + unit, max(-100.0, min(100.0, v * scale))))
            }
            line("Exposure", exposureEV, " EV", 0.15, "%+.1f", 50.0)
            line("Contrast", contrastPct * 0.6, "", 6.0, "%+.0f", 1.0)
            line("Highlights", highlights * 3, "", 8.0, "%+.0f", 1.0)
            line("Shadows", shadows * 3, "", 8.0, "%+.0f", 1.0)
            line("Whites", whites * 2.5, "", 8.0, "%+.0f", 1.0)
            line("Blacks", blacks * 2.5, "", 8.0, "%+.0f", 1.0)
            line("Vibrance", saturationPct * 0.8, "", 6.0, "%+.0f", 1.0)
            if (colorOK) {
                if (abs(temp) >= 2) out.add(Line("Temperature", (if (temp > 0) "warmer " else "cooler ") + usf("%.0f", abs(temp) * 2.2), max(-100.0, min(100.0, temp * 2.2))))
                if (abs(tint) >= 2) out.add(Line("Tint", (if (tint > 0) "magenta " else "green ") + usf("%.0f", abs(tint) * 2.2), max(-100.0, min(100.0, tint * 2.2))))
            }
            return out
        }
}

/** Per-pixel CIE Lab of a photo (cached: several analyses need it). */
class LabImage(img: RGBImage) {
    val w = img.width
    val h = img.height
    val L = FloatArray(w * h)
    val a = FloatArray(w * h)
    val b = FloatArray(w * h)

    init {
        val t = FloatArray(3)
        val p = img.data
        for (i in 0 until w * h) {
            Imaging.lab(p.u(i * 3), p.u(i * 3 + 1), p.u(i * 3 + 2), t)
            L[i] = t[0]; a[i] = t[1]; b[i] = t[2]
        }
    }
}

/** Pair-specific analysis: same scene?  how do the framings differ?  what edit moves one toward the other? */
object SceneMatch {

    // ---- alignment

    /**
     * Structure image for matching: grey at 128 px, high-passed so only edges / texture count,
     * which makes the correlation immune to exposure and colour differences.
     */
    private fun structure(img: RGBImage, size: Int = 128): Plane {
        val g = Imaging.gray(ImagePrep.resizeArea(img, size, size))
        val a = Imaging.gaussianBlur(g, 1.0)
        val b = Imaging.gaussianBlur(g, 6.0)
        return Plane(size, size, FloatArray(size * size) { a.d[it] - b.d[it] })
    }

    private fun sample(p: Plane, u: Double, v: Double): Float {
        val fx = max(0.0, min((p.w - 1).toDouble(), u * p.w - 0.5))
        val fy = max(0.0, min((p.h - 1).toDouble(), v * p.h - 0.5))
        val x0 = fx.toInt(); val y0 = fy.toInt()
        val x1 = min(p.w - 1, x0 + 1); val y1 = min(p.h - 1, y0 + 1)
        val tx = (fx - x0).toFloat(); val ty = (fy - y0).toFloat()
        return (p.d[y0 * p.w + x0] * (1 - tx) + p.d[y0 * p.w + x1] * tx) * (1 - ty) +
            (p.d[y1 * p.w + x0] * (1 - tx) + p.d[y1 * p.w + x1] * tx) * ty
    }

    /** Scratch buffers and the last coverage of `correlation`, so the search does not allocate. */
    private class Corr(n: Int) {
        val xs = FloatArray(n * n)
        val ys = FloatArray(n * n)
        var coverage = 0.0
    }

    /** Normalised cross-correlation of A's structure with B's structure seen through H (A -> B). */
    private fun correlation(sa: Plane, sb: Plane, m: DoubleArray, n: Int, c: Corr): Double {
        var k = 0
        for (j in 0 until n) {
            val v = (j + 0.5) / n
            for (i in 0 until n) {
                val u = (i + 0.5) / n
                val w = m[6] * u + m[7] * v + m[8]
                if (w <= 0) continue
                val x = (m[0] * u + m[1] * v + m[2]) / w
                val y = (m[3] * u + m[4] * v + m[5]) / w
                if (x < 0 || x > 1 || y < 0 || y > 1) continue
                c.xs[k] = sample(sa, u, v); c.ys[k] = sample(sb, x, y); k++
            }
        }
        c.coverage = k.toDouble() / (n * n)
        if (k <= n * n / 5) return 0.0
        var mx = 0f; var my = 0f
        for (i in 0 until k) { mx += c.xs[i]; my += c.ys[i] }
        mx /= k; my /= k
        var sxy = 0f; var sxx = 0f; var syy = 0f
        for (i in 0 until k) {
            val dx = c.xs[i] - mx; val dy = c.ys[i] - my
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy
        }
        return (sxy / (sqrt(sxx * syy) + 1e-9f)).toDouble()
    }

    private fun saneQuad(q: Quad, aspect: Double): Boolean {
        if (q.size != 4 || !q.all { it.x.isFinite() && it.y.isFinite() }) return false
        val px = DoubleArray(4) { q[it].x * aspect }
        val py = DoubleArray(4) { q[it].y }
        var pos = 0; var neg = 0
        for (i in 0 until 4) {
            val a = i; val b = (i + 1) % 4; val c = (i + 2) % 4
            val s = (px[b] - px[a]) * (py[c] - py[b]) - (py[b] - py[a]) * (px[c] - px[b])
            if (s > 0) pos++ else if (s < 0) neg++
        }
        if (pos != 4 && neg != 4) return false
        var area = 0.0
        for (i in 0 until 4) area += px[i] * py[(i + 1) % 4] - px[(i + 1) % 4] * py[i]
        area = abs(area) / 2 / aspect
        val sides = DoubleArray(4) { hypot(px[it] - px[(it + 1) % 4], py[it] - py[(it + 1) % 4]) }
        return area > 0.04 && area < 25 && sides.max() / (sides.min() + 1e-9) < 6
    }

    val corners: Quad = listOf(Pt(0.0, 0.0), Pt(1.0, 0.0), Pt(1.0, 1.0), Pt(0.0, 1.0))

    /** Structure correlation above which two frames count as the same scene, point for point. */
    const val alignThreshold = 0.45

    private class Seed(val c: Double, val s: Double, val x: Double, val y: Double)

    /**
     * Find the mapping A -> B.  Two photos of the same scene mostly differ by zoom and aim, so a direct
     * search over B's frame as a rectangle inside A (scale + offset), scored by structure correlation,
     * finds it robustly.  It only counts as "aligned" when that correlation is strong.
     */
    fun align(a: RGBImage, b: RGBImage): Alignment {
        val res = Alignment()
        val aspA = a.width.toDouble() / a.height; val aspB = b.width.toDouble() / b.height
        val sa64 = structure(a, 64); val sb64 = structure(b, 64)
        val sa = structure(a); val sb = structure(b)
        val corr = Corr(48)

        fun rect(s: Double, x0: Double, y0: Double): DoubleArray {
            val sw = s; val sh = s * aspA / aspB
            return doubleArrayOf(1 / sw, 0.0, -x0 / sw, 0.0, 1 / sh, -y0 / sh, 0.0, 0.0, 1.0)
        }
        // coarse: B's frame = [x0, x0+s] x [y0, y0+s*aspA/aspB] in A's normalised coordinates
        val coarse = ArrayList<Seed>()
        for (k in 0 until 28) {
            val s = exp(ln(0.45) + (ln(2.2) - ln(0.45)) * k / 27)
            val sw = s; val sh = s * aspA / aspB
            if (!(sh > 0.3 && sh < 3)) continue
            var ix = 0
            while (true) {
                val x0 = -sw + 0.3 + ix * 0.03
                if (x0 > 0.7) break
                ix++
                var iy = 0
                while (true) {
                    val y0 = -sh + 0.3 + iy * 0.03
                    if (y0 > 0.7) break
                    iy++
                    val ow = max(0.0, min(1.0, x0 + sw) - max(0.0, x0)); val oh = max(0.0, min(1.0, y0 + sh) - max(0.0, y0))
                    if (ow * oh < 0.3 || ow * oh / (sw * sh) < 0.25) continue
                    coarse.add(Seed(correlation(sa64, sb64, rect(s, x0, y0), 24, corr), s, x0, y0))
                }
            }
        }
        coarse.sortByDescending { it.c }
        val seeds = ArrayList<Seed>()
        for (c in coarse) {
            if (seeds.any { abs(ln(it.s / c.s)) < 0.08 && abs(it.x - c.x) < 0.06 && abs(it.y - c.y) < 0.06 }) continue
            seeds.add(c)
            if (seeds.size == 4) break
        }
        // refine each seed at 128 px
        val cands = ArrayList<DoubleArray>()
        for (sd in seeds) {
            var bestC = -2.0
            var bestH = rect(sd.s, sd.x, sd.y)
            for (i in 0..8) {
                val ds = -0.04 + i * 0.01
                for (j in 0..10) {
                    val dx = -0.03 + j * 0.006
                    for (l in 0..10) {
                        val dy = -0.03 + l * 0.006
                        val h = rect(sd.s * (1 + ds), sd.x + dx, sd.y + dy)
                        val c = correlation(sa, sb, h, 32, corr)
                        if (corr.coverage >= 0.25 && c > bestC) { bestC = c; bestH = h }
                    }
                }
            }
            cands.add(bestH)
        }

        var best: DoubleArray? = null
        var bestC = 0.0; var bestCov = 0.0
        for (h in cands) {
            val c = correlation(sa, sb, h, 48, corr)
            if (corr.coverage >= 0.25 && (best == null || c > bestC)) { best = h; bestC = c; bestCov = corr.coverage }
        }
        val h = H3(best ?: return res)
        val hinv = h.inverse ?: return res
        res.match = bestC
        res.coverage = bestCov
        val qBinA = hinv.project(corners); val qAinB = h.project(corners)
        if (bestC >= alignThreshold && saneQuad(qBinA, aspA) && saneQuad(qAinB, aspB)) {
            res.aligned = true
            res.H = h
            res.quadBinA = qBinA
            res.quadAinB = qAinB
        }
        return res
    }

    fun framing(q: Quad, myAspect: Double): Framing {
        var area = 0.0
        for (i in 0 until 4) area += q[i].x * q[(i + 1) % 4].y - q[(i + 1) % 4].x * q[i].y
        area = abs(area) / 2
        val cx = q.sumOf { it.x } / 4; val cy = q.sumOf { it.y } / 4
        val rot = atan2(q[1].y - q[0].y, (q[1].x - q[0].x) * myAspect) * 180 / PI
        var ins = 0
        for (p in q) {
            if (p.x >= -0.03 && p.x <= 1.03) ins++
            if (p.y >= -0.03 && p.y <= 1.03) ins++
        }
        return Framing(area, 1 / sqrt(max(area, 1e-6)), cx - 0.5, cy - 0.5, rot, ins / 8.0, q)
    }

    /** Pixels of "me" that the other photo also shows. */
    fun overlapMask(w: Int, h: Int, meToOther: H3): BooleanArray {
        val m = BooleanArray(w * h)
        val hm = meToOther.m
        val step = 4
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val u = (x + 0.5) / w; val v = (y + 0.5) / h
                val ww = hm[6] * u + hm[7] * v + hm[8]
                val px = (hm[0] * u + hm[1] * v + hm[2]) / ww; val py = (hm[3] * u + hm[4] * v + hm[5]) / ww
                if (ww > 0 && px >= 0 && px <= 1 && py >= 0 && py <= 1) {
                    for (yy in y until min(h, y + step)) for (xx in x until min(w, x + step)) m[yy * w + xx] = true
                }
                x += step
            }
            y += step
        }
        return m
    }

    // ---- match the look

    val qLevels = doubleArrayOf(0.5, 2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 98.0, 99.5)

    private fun relY(L: Double): Double = if (L > 8) ((L + 16) / 116).pow(3) else L / 903.3

    private class Stats(val q: DoubleArray, val chroma: Double, val a: Double, val b: Double)

    private fun stats(lab: LabImage, mask: BooleanArray?): Stats {
        val n = lab.L.size
        val ls = FloatArray(n)
        var k = 0
        var chroma = 0.0; var sa = 0.0; var sb = 0.0
        for (i in 0 until n) {
            if (mask != null && !mask[i]) continue
            ls[k++] = lab.L[i]
            chroma += sqrt(lab.a[i] * lab.a[i] + lab.b[i] * lab.b[i])
            sa += lab.a[i]; sb += lab.b[i]
        }
        val cnt = max(k, 1).toDouble()
        val sorted = ls.copyOf(k).also { it.sort() }
        return Stats(DoubleArray(qLevels.size) { Imaging.percentileSorted(sorted, qLevels[it]) }, chroma / cnt, sa / cnt, sb / cnt)
    }

    fun lookRecipe(src: LabImage, ref: LabImage, srcMask: BooleanArray? = null, refMask: BooleanArray? = null, colorOK: Boolean = true): LookRecipe {
        fun usable(m: BooleanArray?, n: Int): BooleanArray? {
            if (m == null || m.size != n) return null
            return if (m.count { it }.toDouble() / n > 0.1) m else null
        }
        val s = stats(src, usable(srcMask, src.L.size))
        val r = stats(ref, usable(refMask, ref.L.size))
        val qs = s.q; val qr = r.q
        val sat = max(0.5, min(2.2, (r.chroma + 2) / (s.chroma + 2)))
        val ev = max(-3.0, min(3.0, ln((relY(qr[5]) + 0.01) / (relY(qs[5]) + 0.01)) / ln(2.0)))
        val iqrS = qs[6] - qs[4]; val iqrR = qr[6] - qr[4]
        return LookRecipe(
            exposureEV = ev,
            contrastPct = max(-0.6, min(1.5, (iqrR + 4) / (iqrS + 4) - 1)) * 100,
            highlights = (qr[8] - qr[5]) - (qs[8] - qs[5]),
            shadows = (qs[5] - qs[2]) - (qr[5] - qr[2]),
            blacks = qr[0] - qs[0],
            whites = qr[10] - qs[10],
            saturationPct = (sat - 1) * 100,
            temp = if (colorOK) r.b - s.b else 0.0,
            tint = if (colorOK) r.a - s.a else 0.0,
            qs = qs, qr = qr, sat = sat, colorOK = colorOK,
        )
    }

    /** numpy.interp for increasing xs. */
    private fun interp(x: Double, xs: DoubleArray, ys: DoubleArray): Double {
        if (x <= xs[0]) return ys[0]
        if (x >= xs[xs.size - 1]) return ys[ys.size - 1]
        var lo = 0; var hi = xs.size - 1
        while (hi - lo > 1) { val mid = (lo + hi) / 2; if (xs[mid] <= x) lo = mid else hi = mid }
        val span = xs[hi] - xs[lo]
        return if (span > 0) ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / span else ys[lo]
    }

    /** The recipe as a smooth global tone curve + white-balance shift + saturation scale. */
    fun applyLook(lab: LabImage, rec: LookRecipe, strength: Double = 0.85): RGBImage {
        val s = strength
        val qs = rec.qs; val qr = rec.qr
        val tgt = DoubleArray(qs.size) { qs[it] + s * (qr[it] - qs[it]) }
        val xs0 = doubleArrayOf(0.0) + qs + doubleArrayOf(100.0)
        val ys0 = doubleArrayOf(max(0.0, tgt[0] - qs[0]) * 0.5) + tgt + doubleArrayOf(min(100.0, 100 + (tgt[tgt.size - 1] - qs[qs.size - 1]) * 0.5))
        // np.unique(return_index): sorted unique xs, first occurrence
        val order = xs0.indices.sortedWith(compareBy<Int> { xs0[it] }.thenBy { it })
        val ux = ArrayList<Double>(); val uy = ArrayList<Double>()
        for (i in order) if (ux.isEmpty() || ux.last() != xs0[i]) { ux.add(xs0[i]); uy.add(ys0[i]) }
        for (i in 1 until uy.size) uy[i] = max(uy[i], uy[i - 1])
        val xs = ux.toDoubleArray(); val ys = uy.toDoubleArray()
        val grid = DoubleArray(256) { it * 100.0 / 255 }
        val lut0 = DoubleArray(256) { interp(grid[it], xs, ys) }
        val padded = DoubleArray(272) { if (it < 8) lut0[0] else if (it >= 264) lut0[255] else lut0[it - 8] }
        val lut = DoubleArray(256) { i -> var t = 0.0; for (j in i until i + 17) t += padded[j]; t / 17 }
        val sat = (1 + s * (rec.sat - 1)).toFloat()
        val da = (s * rec.tint).toFloat(); val db = (s * rec.temp).toFloat()
        val out = ByteArray(lab.w * lab.h * 3)
        for (i in 0 until lab.w * lab.h) {
            // the grid is uniform, so numpy.interp reduces to one lookup
            val t = max(0.0, min(255.0, lab.L[i].toDouble() * 255 / 100))
            val i0 = min(254, t.toInt())
            val L = (lut[i0] + (lut[i0 + 1] - lut[i0]) * (t - i0)).toFloat()
            val p = Imaging.rgb(L, lab.a[i] * sat + da, lab.b[i] * sat + db)
            out[i * 3] = (p shr 16).toByte(); out[i * 3 + 1] = (p shr 8).toByte(); out[i * 3 + 2] = p.toByte()
        }
        return RGBImage(lab.w, lab.h, out)
    }

    /** Re-frame my photo the way the other one is framed (valid when the other frame lies inside mine). */
    fun cropToOther(me: RGBImage, meToOther: H3, otherW: Int, otherH: Int, maxSide: Int = 1024): RGBImage? {
        val inv = (meToOther.inverse ?: return null).m
        val s = min(1.0, maxSide.toDouble() / max(otherW, otherH))
        val ow = max(1, rndInt(otherW * s)); val oh = max(1, rndInt(otherH * s))
        val c = Imaging.planes(me)
        val out = ByteArray(ow * oh * 3)
        for (y in 0 until oh) {
            for (x in 0 until ow) {
                val uu = (x + 0.5) / ow; val vv = (y + 0.5) / oh
                val w = inv[6] * uu + inv[7] * vv + inv[8]
                val u = min(1.0, max(0.0, (inv[0] * uu + inv[1] * vv + inv[2]) / w))
                val v = min(1.0, max(0.0, (inv[3] * uu + inv[4] * vv + inv[5]) / w))
                for (k in 0 until 3) out[(y * ow + x) * 3 + k] = clampByte(sample(c[k], u, v))
            }
        }
        return RGBImage(ow, oh, out)
    }
}

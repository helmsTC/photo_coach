package com.mosaiclens.apertureiq.judge

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/** Dominant colour and its share of the frame. */
data class Swatch(val r: Int, val g: Int, val b: Int, val share: Double)

/** 64-bin histograms, each summing to 1. */
class Histogram(val r: DoubleArray, val g: DoubleArray, val b: DoubleArray, val l: DoubleArray)

/** Everything measured on the pixels of one photo. */
class Measurement(
    /** Scalars, keyed exactly as photocoach/signals.py. */
    val s: Map<String, Double>,
    /** 12x12 maps of fine-edge strength and edge crispness. */
    val detail: Plane,
    val crisp: Plane,
    /** 96x96 visual-weight map, normalised to max 1. */
    val saliency: Plane,
    val palette: List<Swatch>,
    val histogram: Histogram,
) {
    operator fun get(k: String): Double = s[k] ?: 0.0
}

/**
 * Measurements read directly off the pixels of the standard image (long side <= 1024).
 * A port of photocoach/signals.py.  The tone / colour numbers are tower inputs, so their
 * definitions must not drift from the Python; focus / noise / composition numbers feed the
 * coaching evidence and overlays.
 */
object Signals {
    const val tileGrid = 12

    fun measure(img: RGBImage): Measurement {
        // The measurements are independent of each other (saliency only needs focus's detail map),
        // so they run side by side; the result is the same as running them in turn.
        fun <T> go(name: String, body: () -> T) = java.util.concurrent.CompletableFuture.supplyAsync { Prof.t("m.$name") { body() } }
        val fTone = go("tone") { tone(img) }
        val fColor = go("color") { color(img) }
        val fNoise = go("noise") { noise(img) }
        val fHorizon = go("horizon") { horizon(img) }
        val fPalette = go("palette") { palette(img) }
        val fHist = go("histogram") { histogram(img) }
        try {
            val (f, detail, crisp) = Prof.t("m.focus") { focus(img) }
            val (sal, salMap) = Prof.t("m.saliency") { saliency(img, detail) }
            val s = HashMap<String, Double>()
            s.putAll(fTone.get())
            s.putAll(fColor.get())
            s.putAll(f)
            s.putAll(fNoise.get())
            s["log_long_side"] = ln(max(img.width, img.height) / 1024.0 + 1e-6)
            s["aspect"] = img.width.toDouble() / img.height
            s.putAll(fHorizon.get())
            s.putAll(sal)
            return Measurement(s, detail, crisp, salMap, fPalette.get(), fHist.get())
        } catch (e: java.util.concurrent.ExecutionException) {
            throw e.cause ?: e
        }
    }

    /** The tone / colour / size signals of an edited image (what the tower reads). */
    fun toneColor(img: RGBImage): Map<String, Double> {
        val s = HashMap<String, Double>()
        s.putAll(tone(img))
        s.putAll(color(img))
        s["log_long_side"] = ln(max(img.width, img.height) / 1024.0 + 1e-6)
        s["aspect"] = img.width.toDouble() / img.height
        return s
    }

    // ---- tone + colour (tower inputs: keep identical to the Python)

    fun tone(img: RGBImage): Map<String, Double> {
        val n = img.width * img.height
        val out = HashMap<String, Double>()
        if (n <= 0) return out
        val luma = FloatArray(n)
        var sumY = 0.0; var sumY2 = 0.0
        var clipHi = 0; var clipLo = 0; var nearHi = 0; var nearLo = 0
        val p = img.data
        for (i in 0 until n) {
            val r8 = p.u(i * 3); val g8 = p.u(i * 3 + 1); val b8 = p.u(i * 3 + 2)
            val r = r8 / 255f; val g = g8 / 255f; val b = b8 / 255f
            val y = r * 0.2126f + g * 0.7152f + b * 0.0722f
            luma[i] = y
            sumY += y; sumY2 += y.toDouble() * y.toDouble()
            if (max(r8, max(g8, b8)) >= 252) clipHi++
            if (y <= 0.016f) clipLo++
            if (y >= 0.92f) nearHi++
            if (y <= 0.06f) nearLo++
        }
        val nn = n.toDouble()
        val sorted = luma.sortedArray()
        val p01 = Imaging.percentileSorted(sorted, 1.0)
        val p99 = Imaging.percentileSorted(sorted, 99.0)
        val meanY = sumY / nn
        out["luma_mean"] = meanY
        out["luma_median"] = Imaging.percentileSorted(sorted, 50.0)
        out["luma_p01"] = p01
        out["luma_p05"] = Imaging.percentileSorted(sorted, 5.0)
        out["luma_p95"] = Imaging.percentileSorted(sorted, 95.0)
        out["luma_p99"] = p99
        out["clip_hi"] = clipHi / nn
        out["clip_lo"] = clipLo / nn
        out["near_hi"] = nearHi / nn
        out["near_lo"] = nearLo / nn
        out["contrast_rms"] = sqrt(max(0.0, sumY2 / nn - meanY * meanY))
        out["dyn_range"] = p99 - p01

        // local contrast: mean |luma - heavily blurred luma|
        val y = Plane(img.width, img.height, luma)
        val blur = Imaging.gaussianBlur(y, max(2.0, 0.02 * max(img.width, img.height)))
        var lc = 0.0
        for (i in 0 until n) lc += abs(y.d[i] - blur.d[i])
        out["local_contrast"] = lc / nn
        return out
    }

    fun color(img: RGBImage): Map<String, Double> {
        val n = img.width * img.height
        val out = HashMap<String, Double>()
        if (n <= 0) return out
        val chroma = FloatArray(n)
        val labL = FloatArray(n); val labA = FloatArray(n); val labB = FloatArray(n)
        var sumSat = 0.0
        var sumRG = 0.0; var sumRG2 = 0.0; var sumYB = 0.0; var sumYB2 = 0.0
        var sumA = 0.0; var sumB = 0.0
        val p = img.data
        val t = FloatArray(3)
        for (i in 0 until n) {
            val r8 = p.u(i * 3); val g8 = p.u(i * 3 + 1); val b8 = p.u(i * 3 + 2)
            val r = r8 / 255f; val g = g8 / 255f; val b = b8 / 255f
            val mx = max(r, max(g, b)); val mn = min(r, min(g, b))
            sumSat += if (mx > 0) ((mx - mn) / mx).toDouble() else 0.0
            val rg = (r - g).toDouble(); val yb = (0.5f * (r + g) - b).toDouble()
            sumRG += rg; sumRG2 += rg * rg; sumYB += yb; sumYB2 += yb * yb
            Imaging.lab(r8, g8, b8, t)
            labL[i] = t[0]; labA[i] = t[1]; labB[i] = t[2]
            chroma[i] = sqrt(t[1] * t[1] + t[2] * t[2])
            sumA += t[1]; sumB += t[2]
        }
        val nn = n.toDouble()
        val mRG = sumRG / nn; val mYB = sumYB / nn
        val vRG = max(0.0, sumRG2 / nn - mRG * mRG); val vYB = max(0.0, sumYB2 / nn - mYB * mYB)
        out["sat_mean"] = sumSat / nn
        out["colorfulness"] = sqrt(vRG + vYB) + 0.3 * sqrt(mRG * mRG + mYB * mYB)
        out["cast_a"] = sumA / nn / 50
        out["cast_b"] = sumB / nn / 50
        // "should-be-neutral" pixels: the least colourful quarter of the mid-tones; their mean a/b is the cast
        val c25 = Imaging.percentile(chroma, 25.0).toFloat()
        var na = 0.0; var nb = 0.0; var cnt = 0
        for (i in 0 until n) {
            if (chroma[i] < c25 && labL[i] > 20f && labL[i] < 92f) { na += labA[i]; nb += labB[i]; cnt++ }
        }
        if (cnt < 50) { na = sumA; nb = sumB; cnt = n }
        out["neutral_cast_a"] = na / cnt / 50
        out["neutral_cast_b"] = nb / cnt / 50
        return out
    }

    // ---- tiles

    /** Tile edges as numpy.linspace(0, n, grid + 1).astype(int). */
    fun tileEdges(n: Int, grid: Int = tileGrid): IntArray = IntArray(grid + 1) { (n.toDouble() / grid * it).toInt() }

    // ---- focus

    private fun gradientMagnitude(gray: Plane, sigma: Double): Plane {
        val g = Imaging.gaussianBlur(gray, sigma)
        val w = g.w; val h = g.h
        val out = Plane(w, h)
        val p = g.d
        for (y in 0 until h) {
            val ym = Imaging.reflect101(y - 1, h) * w; val y0 = y * w; val yp = Imaging.reflect101(y + 1, h) * w
            for (x in 0 until w) {
                val xm = Imaging.reflect101(x - 1, w); val xp = Imaging.reflect101(x + 1, w)
                val gx = (p[ym + xp] - p[ym + xm]) + 2 * (p[y0 + xp] - p[y0 + xm]) + (p[yp + xp] - p[yp + xm])
                val gy = (p[yp + xm] - p[ym + xm]) + 2 * (p[yp + x] - p[ym + x]) + (p[yp + xp] - p[ym + xp])
                out.d[y0 + x] = sqrt(gx * gx + gy * gy) / 8
            }
        }
        return out
    }

    /** Where is the image sharp, and how sharp?  (signals.focus) */
    fun focus(img: RGBImage): Triple<Map<String, Double>, Plane, Plane> {
        val gray = Imaging.gray(img)
        val g1 = gradientMagnitude(gray, 0.8)
        val g2 = gradientMagnitude(gray, 2.4)
        val n = tileGrid
        val ys = tileEdges(img.height); val xs = tileEdges(img.width)
        val detail = Plane(n, n); val crisp = Plane(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                val cnt = (ys[i + 1] - ys[i]) * (xs[j + 1] - xs[j])
                val t1 = FloatArray(cnt); val t2 = FloatArray(cnt)
                var k = 0
                for (y in ys[i] until ys[i + 1]) {
                    for (x in xs[j] until xs[j + 1]) { t1[k] = g1[x, y]; t2[k] = g2[x, y]; k++ }
                }
                val a = Imaging.percentile(t1, 97.0).toFloat()
                val b = Imaging.percentile(t2, 97.0).toFloat()
                detail[j, i] = a
                crisp[j, i] = if (b > 0.004f) a / (b + 1e-3f) else 0f
            }
        }
        val flat = detail.d.sortedArray()
        val top = flat[(0.9 * flat.size).toInt()].toDouble()
        val med = Imaging.percentileSorted(flat, 50.0)
        val low = flat[(0.2 * flat.size).toInt()].toDouble()
        val topThr = flat[(0.8 * flat.size).toInt()]
        var crispSum = 0.0; var crispN = 0
        var wsum = 0.0; var wx = 0.0; var wy = 0.0; var sharpN = 0
        for (i in 0 until n) {
            for (j in 0 until n) {
                val d = detail[j, i]
                if (d >= topThr) { crispSum += crisp[j, i]; crispN++ }
                if (d.toDouble() >= max(0.5 * top, 1e-4)) {
                    sharpN++
                    wsum += d; wx += d.toDouble() * (j + 0.5); wy += d.toDouble() * (i + 0.5)
                }
            }
        }
        val tot = wsum + 1e-9
        val out = mapOf(
            "sharp_top" to top,
            "sharp_median" to med,
            "edge_crispness" to (if (crispN > 0) crispSum / crispN else 0.0),
            "focus_spread" to ln((top + 1e-3) / (low + 1e-3)),
            "sharp_area" to sharpN.toDouble() / (n * n),
            "sharp_cx" to wx / tot / n,
            "sharp_cy" to wy / tot / n,
        )
        return Triple(out, detail, crisp)
    }

    // ---- noise (Immerkaer, flattest tiles)

    fun noise(img: RGBImage): Map<String, Double> {
        val w = img.width; val h = img.height; val n = w * h
        val yy = FloatArray(n); val cc = FloatArray(n)
        val p = img.data
        for (i in 0 until n) {
            val r = p.u(i * 3).toFloat(); val g = p.u(i * 3 + 1).toFloat(); val b = p.u(i * 3 + 2).toFloat()
            val y = 0.299f * r + 0.587f * g + 0.114f * b
            val cr = max(0f, min(255f, rnd((r - y) * 0.713f + 128f)))
            val cb = max(0f, min(255f, rnd((b - y) * 0.564f + 128f)))
            yy[i] = max(0f, min(255f, rnd(y))) / 255f
            cc[i] = 0.5f * (cr + cb) / 255f
        }
        val ys = tileEdges(h); val xs = tileEdges(w)
        fun sigma(ch: FloatArray): Double {
            val sig = ArrayList<Float>()
            for (i in 0 until tileGrid) {
                for (j in 0 until tileGrid) {
                    var s = 0.0; var c = 0
                    for (y in ys[i] until ys[i + 1]) {
                        val ym = Imaging.reflect101(y - 1, h) * w; val y0 = y * w; val yp = Imaging.reflect101(y + 1, h) * w
                        for (x in xs[j] until xs[j + 1]) {
                            val xm = Imaging.reflect101(x - 1, w); val xp = Imaging.reflect101(x + 1, w)
                            val v = (ch[ym + xm] - 2 * ch[ym + x] + ch[ym + xp]) -
                                2 * (ch[y0 + xm] - 2 * ch[y0 + x] + ch[y0 + xp]) +
                                (ch[yp + xm] - 2 * ch[yp + x] + ch[yp + xp])
                            s += abs(v); c++
                        }
                    }
                    if (c > 0) sig.add((sqrt(PI / 2) / 6 * s / c).toFloat())
                }
            }
            return if (sig.isEmpty()) 0.0 else Imaging.percentile(sig.toFloatArray(), 10.0)
        }
        return mapOf("noise_luma" to sigma(yy), "noise_chroma" to sigma(cc))
    }

    // ---- horizon

    /**
     * Angle of the horizon / main waterline in degrees (positive = clockwise: the right end sits lower).
     * A port of signals.horizon: a candidate must be a long near-horizontal edge that actually separates
     * two different regions (sky|land, land|water) - ripples and rock strata make long lines too, but
     * with the same colour on both sides.  OpenCV's Canny + probabilistic Hough is replaced by a
     * gradient edge map and a Hough vote restricted to +-10 degrees.
     */
    fun horizon(img: RGBImage): Map<String, Double> {
        val out = hashMapOf("tilt_deg" to 0.0, "tilt_conf" to 0.0, "horizon_y" to -1.0)
        val sc = 512.0 / max(img.width, img.height)
        val small = if (sc < 1) ImagePrep.resizeArea(img, max(1, rndInt(img.width * sc)), max(1, rndInt(img.height * sc))) else img
        val w = small.width; val h = small.height
        if (w < 32 || h < 32) return out
        val g0 = Imaging.gray(small)
        for (i in g0.d.indices) g0.d[i] *= 255f
        val gray = Imaging.gaussianBlur(g0, 1.5).d

        // thin, mostly-horizontal edges: vertical-gradient maxima above a Canny-like threshold
        val mag = FloatArray(w * h); val vert = BooleanArray(w * h)
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val i = y * w + x
                val gx = (gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1]) - (gray[i - w - 1] + 2 * gray[i - 1] + gray[i + w - 1])
                val gy = (gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1]) - (gray[i - w - 1] + 2 * gray[i - w] + gray[i - w + 1])
                mag[i] = sqrt(gx * gx + gy * gy)
                vert[i] = abs(gy) >= abs(gx)
            }
        }
        val px = ArrayList<Int>(); val py = ArrayList<Int>()
        for (y in 2 until h - 2) {
            for (x in 1 until w - 1) {
                val i = y * w + x; val m = mag[i]
                if (vert[i] && m >= 70f && m >= mag[i - w] && m > mag[i + w]) { px.add(x); py.add(y) }
            }
        }
        val np = px.size
        if (np < w / 4) return out

        // Hough vote: line y = r + tan(theta) * (x - cx)
        val angles = DoubleArray(81) { -10.0 + 0.25 * it }
        val tans = DoubleArray(81) { tan(angles[it] * PI / 180) }
        val cx = w / 2.0
        val off = (0.2 * h).toInt(); val nR = (1.4 * h).toInt() + 1
        val acc = IntArray(angles.size * nR)
        for (k in 0 until np) {
            val x = px[k]; val y = py[k]
            for (a in angles.indices) {
                val r = rndInt(y - tans[a] * (x - cx)) + off
                if (r in 0 until nR) acc[a * nR + r]++
            }
        }
        val peaks = ArrayList<IntArray>()
        for (a in angles.indices) for (r in 0 until nR) if (acc[a * nR + r] >= w / 4) peaks.add(intArrayOf(acc[a * nR + r], a, r))
        peaks.sortByDescending { it[0] }
        val chosen = ArrayList<IntArray>()
        for (pk in peaks) {
            val a = pk[1]; val r = pk[2]
            if (chosen.any { abs(it[0] - a) <= 4 && abs(it[1] - r) <= 6 }) continue
            chosen.add(intArrayOf(a, r))
            if (chosen.size == 20) break
        }

        // Lab, lightly blurred, with a/b weighted as in 8-bit OpenCV Lab
        var lp = Plane(w, h); var ap = Plane(w, h); var bp = Plane(w, h)
        val t = FloatArray(3)
        for (i in 0 until w * h) {
            Imaging.lab(small.data.u(i * 3), small.data.u(i * 3 + 1), small.data.u(i * 3 + 2), t)
            lp.d[i] = t[0]; ap.d[i] = t[1] / 2.55f; bp.d[i] = t[2] / 2.55f
        }
        lp = Imaging.gaussianBlur(lp, 2.0); ap = Imaging.gaussianBlur(ap, 2.0); bp = Imaging.gaussianBlur(bp, 2.0)
        val band = max(4, (0.035 * h).toInt())
        val maxGap = max(2, (0.04 * w).toInt())
        var bestScore = -1.0; var bestDeg = 0.0; var bestFrac = 0.0; var bestDE = 0.0; var bestY = 0.0
        var found = false
        for (ch in chosen) {
            val a = ch[0]
            val tt = tans[a]; val r0 = (ch[1] - off).toDouble()
            fun lineY(x: Double) = r0 + tt * (x - cx)
            val cols = BooleanArray(w)
            for (k in 0 until np) if (abs(py[k] - lineY(px[k].toDouble())) <= 1.5) cols[px[k]] = true
            // longest run of supported columns, bridging gaps up to maxGap
            var run0 = 0; var run1 = -1; var start = -1; var last = -1
            for (x in 0 until w) {
                if (!cols[x]) continue
                if (start < 0 || x - last > maxGap) start = x
                last = x
                if (last - start > run1 - run0) { run0 = start; run1 = last }
            }
            val len = (run1 - run0).toDouble()
            if (len < 0.25 * w) continue
            val ya = lineY(run0.toDouble()); val yb = lineY(run1.toDouble())
            if (min(ya, yb) - band < 0 || max(ya, yb) + band > h - 1) continue
            val diffs = FloatArray(40)
            for (k in 0 until 40) {
                val x = run0 + len * k / 39
                val y = lineY(x)
                val xi = min(w - 1, max(0, x.toInt()))
                val up = min(h - 1, max(0, (y - band).toInt())) * w + xi
                val dn = min(h - 1, max(0, (y + band).toInt())) * w + xi
                val dl = lp.d[up] - lp.d[dn]; val da = ap.d[up] - ap.d[dn]; val db = bp.d[up] - bp.d[dn]
                diffs[k] = sqrt(dl * dl + da * da + db * db)
            }
            val dE = Imaging.median(diffs)
            if (dE < 9) continue
            val score = len / w * min(dE, 40.0)
            if (!found || score > bestScore) {
                found = true
                bestScore = score; bestDeg = angles[a]; bestFrac = len / w; bestDE = dE; bestY = lineY(cx) / h
            }
        }
        if (!found) return out
        out["tilt_deg"] = bestDeg
        out["tilt_conf"] = min(1.0, bestFrac / 0.55) * min(1.0, bestDE / 18)
        out["horizon_y"] = bestY
        return out
    }

    // ---- saliency / visual weight

    /** 2-D discrete Fourier transform of an n x n complex image, in place (direct sums: n is small). */
    private fun dft2(re: DoubleArray, im: DoubleArray, n: Int, inverse: Boolean) {
        val cs = DoubleArray(n) { cos(2 * PI * it / n) }
        val sn = DoubleArray(n) { (if (inverse) 1 else -1) * sin(2 * PI * it / n) }
        val tr = DoubleArray(n); val ti = DoubleArray(n)
        fun line(off: Int, step: Int) {
            for (k in 0 until n) {
                var a = 0.0; var b = 0.0
                for (j in 0 until n) {
                    val q = (k * j) % n
                    val xr = re[off + j * step]; val xi = im[off + j * step]
                    a += xr * cs[q] - xi * sn[q]
                    b += xr * sn[q] + xi * cs[q]
                }
                tr[k] = a; ti[k] = b
            }
            for (k in 0 until n) { re[off + k * step] = tr[k]; im[off + k * step] = ti[k] }
        }
        for (y in 0 until n) line(y * n, 1)
        for (x in 0 until n) line(x, n)
    }

    /** Spectral-residual saliency on a 96 x 96 grey image, normalised to max 1 (as signals.saliency). */
    private fun spectralResidual(gray: Plane): Plane {
        val n = gray.w
        val re = DoubleArray(n * n) { gray.d[it].toDouble() }
        val im = DoubleArray(n * n)
        dft2(re, im, n, false)
        val la = Plane(n, n, FloatArray(n * n) { ln(sqrt(re[it] * re[it] + im[it] * im[it]) + 1e-8).toFloat() })
        for (y in 0 until n) {
            for (x in 0 until n) {
                var box = 0f
                for (dy in -1..1) for (dx in -1..1) box += la[Imaging.reflect101(x + dx, n), Imaging.reflect101(y + dy, n)]
                val i = y * n + x
                val mag = exp((la.d[i] - box / 9f).toDouble())
                val ang = atan2(im[i], re[i])
                re[i] = mag * cos(ang); im[i] = mag * sin(ang)
            }
        }
        dft2(re, im, n, true)
        val scale = 1.0 / (n.toDouble() * n)
        val sal = Plane(n, n, FloatArray(n * n) {
            val a = re[it] * scale; val b = im[it] * scale
            (a * a + b * b).toFloat()
        })
        val out = Imaging.gaussianBlur(sal, 4.0)
        val mx = out.max + 1e-9f
        for (i in out.d.indices) out.d[i] /= mx
        return out
    }

    /**
     * Spectral-residual saliency blended with colour contrast and the focus map, then the
     * statistics of signals.saliency on a 96x96 grid.
     */
    fun saliency(img: RGBImage, detail: Plane): Pair<Map<String, Double>, Plane> {
        val S = 96
        val small = ImagePrep.resizeArea(img, S, S)
        val sal = spectralResidual(Imaging.gray(small))

        // colour contrast against the frame mean (8-bit Lab scaling, as cv2 on uint8)
        val L = Plane(S, S); val A = Plane(S, S); val B = Plane(S, S)
        val t = FloatArray(3)
        for (i in 0 until S * S) {
            Imaging.lab(small.data.u(i * 3), small.data.u(i * 3 + 1), small.data.u(i * 3 + 2), t)
            L.d[i] = t[0] * 2.55f; A.d[i] = t[1]; B.d[i] = t[2]
        }
        val mL = (L.sum / (S * S)).toFloat(); val mA = (A.sum / (S * S)).toFloat(); val mB = (B.sum / (S * S)).toFloat()
        val bl = Imaging.gaussianBlur(L, 2.0); val ba = Imaging.gaussianBlur(A, 2.0); val bb = Imaging.gaussianBlur(B, 2.0)
        val cc = Plane(S, S)
        for (i in 0 until S * S) {
            val dl = bl.d[i] - mL; val da = ba.d[i] - mA; val db = bb.d[i] - mB
            cc.d[i] = sqrt(dl * dl + da * da + db * db)
        }
        val cmax = cc.max + 1e-9f
        val d = Imaging.resize(detail, S, S)
        val dmax = d.max + 1e-9f
        val m = Plane(S, S)
        for (i in 0 until S * S) {
            m.d[i] = 0.5f * sal.d[i] + 0.3f * cc.d[i] / cmax + 0.35f * max(0f, min(1f, d.d[i] / dmax))
        }
        val p60 = Imaging.percentile(m.d, 60.0).toFloat()
        for (i in m.d.indices) m.d[i] = max(0f, m.d[i] - p60)
        val tot = m.sum.toFloat() + 1e-9f
        for (i in m.d.indices) m.d[i] /= tot

        var cx = 0.0; var cy = 0.0; var right = 0.0; var left = 0.0; var border = 0.0
        for (y in 0 until S) {
            for (x in 0 until S) {
                val v = m[x, y].toDouble()
                cx += v * (x + 0.5); cy += v * (y + 0.5)
                if (x >= 48) right += v else left += v
                if (x < 8 || x >= S - 8 || y < 8 || y >= S - 8) border += v
            }
        }
        val pk = Imaging.gaussianBlur(m, 5.0)
        var best = 0
        for (i in pk.d.indices) if (pk.d[i] > pk.d[best]) best = i
        val px = (best % S + 0.5) / S; val py = (best / S + 0.5) / S
        val t1 = 1.0 / 3; val t2 = 2.0 / 3
        var dThird = Double.MAX_VALUE
        for (tx in doubleArrayOf(t1, t2)) for (ty in doubleArrayOf(t1, t2)) dThird = min(dThird, hypot(px - tx, py - ty))
        val mmax = m.max + 1e-9f
        val norm = m.copy()
        for (i in norm.d.indices) norm.d[i] /= mmax
        return Pair(
            mapOf(
                "weight_cx" to cx / S, "weight_cy" to cy / S,
                "subject_x" to px, "subject_y" to py,
                "thirds_dist" to dThird,
                "center_dist" to hypot(px - 0.5, py - 0.5),
                "lr_balance" to right - left,
                "edge_weight" to border,
            ),
            norm,
        )
    }

    // ---- palette (k-means in Lab on a 64x64 thumbnail)

    fun palette(img: RGBImage, k: Int = 6): List<Swatch> {
        val small = ImagePrep.resizeArea(img, 64, 64)
        val n = 64 * 64
        val pts = FloatArray(n * 3)
        val t = FloatArray(3)
        for (i in 0 until n) {
            Imaging.lab(small.data.u(i * 3), small.data.u(i * 3 + 1), small.data.u(i * 3 + 2), t)
            pts[i * 3] = t[0]; pts[i * 3 + 1] = t[1]; pts[i * 3 + 2] = t[2]
        }
        fun d2(i: Int, c: FloatArray, ci: Int): Float {
            val a = pts[i * 3] - c[ci * 3]; val b = pts[i * 3 + 1] - c[ci * 3 + 1]; val cc = pts[i * 3 + 2] - c[ci * 3 + 2]
            return a * a + b * b + cc * cc
        }
        // deterministic k-means++ seeding
        var rng = 0x9E3779B97F4A7C15uL
        fun rand(): Double {
            rng = rng * 6364136223846793005uL + 1442695040888963407uL
            return (rng shr 11).toDouble() / (1L shl 53).toDouble()
        }
        val centers = FloatArray(k * 3)
        System.arraycopy(pts, (n / 2) * 3, centers, 0, 3)
        val dist = FloatArray(n) { d2(it, centers, 0) }
        var nc = 1
        while (nc < k) {
            var tot = 0f
            for (v in dist) tot += v
            var r = rand().toFloat() * tot
            var pick = 0
            for (i in 0 until n) { r -= dist[i]; if (r <= 0) { pick = i; break } }
            System.arraycopy(pts, pick * 3, centers, nc * 3, 3)
            for (i in 0 until n) dist[i] = min(dist[i], d2(i, centers, nc))
            nc++
        }
        val labels = IntArray(n)
        repeat(20) {
            for (i in 0 until n) {
                var bi = 0; var bd = Float.MAX_VALUE
                for (c in 0 until k) {
                    val d = d2(i, centers, c)
                    if (d < bd) { bd = d; bi = c }
                }
                labels[i] = bi
            }
            val sums = FloatArray(k * 3); val cnt = FloatArray(k)
            for (i in 0 until n) {
                val l = labels[i]
                sums[l * 3] += pts[i * 3]; sums[l * 3 + 1] += pts[i * 3 + 1]; sums[l * 3 + 2] += pts[i * 3 + 2]
                cnt[l] += 1f
            }
            for (c in 0 until k) if (cnt[c] > 0) for (j in 0 until 3) centers[c * 3 + j] = sums[c * 3 + j] / cnt[c]
        }
        val counts = IntArray(k)
        for (l in labels) counts[l]++
        return (0 until k).sortedByDescending { counts[it] }.filter { counts[it] > 0 }.map { c ->
            val p = Imaging.rgb(centers[c * 3], centers[c * 3 + 1], centers[c * 3 + 2])
            Swatch((p shr 16) and 0xFF, (p shr 8) and 0xFF, p and 0xFF, counts[c].toDouble() / n)
        }
    }

    // ---- histogram

    fun histogram(img: RGBImage, bins: Int = 64): Histogram {
        val r = DoubleArray(bins); val g = DoubleArray(bins); val b = DoubleArray(bins); val l = DoubleArray(bins)
        val n = img.width * img.height
        val p = img.data
        for (i in 0 until n) {
            val rr = p.u(i * 3); val gg = p.u(i * 3 + 1); val bb = p.u(i * 3 + 2)
            r[(rr * bins) shr 8] += 1.0; g[(gg * bins) shr 8] += 1.0; b[(bb * bins) shr 8] += 1.0
            val y = rr * 0.2126 + gg * 0.7152 + bb * 0.0722
            l[min(bins - 1, (y * bins / 256).toInt())] += 1.0
        }
        val nn = max(n, 1).toDouble()
        for (i in 0 until bins) { r[i] /= nn; g[i] /= nn; b[i] /= nn; l[i] /= nn }
        return Histogram(r, g, b, l)
    }
}

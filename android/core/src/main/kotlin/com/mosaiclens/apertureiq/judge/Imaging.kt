package com.mosaiclens.apertureiq.judge

import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/** Round half away from zero (Swift's `.rounded()`, C's `round`). Kotlin's own `round` is half-to-even. */
fun rnd(x: Double): Double = if (x >= 0) floor(x + 0.5) else ceil(x - 0.5)
fun rnd(x: Float): Float = if (x >= 0) floor(x + 0.5f) else ceil(x - 0.5f)
fun rndInt(x: Double): Int = rnd(x).toInt()

@Suppress("NOTHING_TO_INLINE")
inline fun ByteArray.u(i: Int): Int = this[i].toInt() and 0xFF

@Suppress("NOTHING_TO_INLINE")
inline fun clampByte(v: Float): Byte = max(0f, min(255f, rnd(v))).toInt().toByte()

/** 8-bit RGB pixels, row-major, top row first, 3 bytes per pixel. */
class RGBImage(val width: Int, val height: Int, val data: ByteArray) {
    init {
        require(data.size == width * height * 3) { "RGBImage data size mismatch" }
    }

    fun copy(): RGBImage = RGBImage(width, height, data.copyOf())

    /** Packed ARGB pixels (alpha 255), for android.graphics.Bitmap / BufferedImage. */
    fun toArgb(): IntArray {
        val out = IntArray(width * height)
        for (i in out.indices) {
            out[i] = (0xFF shl 24) or (data.u(i * 3) shl 16) or (data.u(i * 3 + 1) shl 8) or data.u(i * 3 + 2)
        }
        return out
    }

    companion object {
        /** From packed ARGB pixels; transparent pixels are composited on white. */
        fun fromArgb(px: IntArray, w: Int, h: Int): RGBImage {
            val out = ByteArray(w * h * 3)
            for (i in 0 until w * h) {
                val p = px[i]
                val a = p ushr 24
                var r = (p shr 16) and 0xFF
                var g = (p shr 8) and 0xFF
                var b = p and 0xFF
                if (a != 255) {
                    r = (r * a + 255 * (255 - a) + 127) / 255
                    g = (g * a + 255 * (255 - a) + 127) / 255
                    b = (b * a + 255 * (255 - a) + 127) / 255
                }
                out[i * 3] = r.toByte(); out[i * 3 + 1] = g.toByte(); out[i * 3 + 2] = b.toByte()
            }
            return RGBImage(w, h, out)
        }
    }
}

/** An overlay: packed, non-premultiplied ARGB pixels. */
class Raster(val width: Int, val height: Int, val argb: IntArray)

/** A single-channel float image (row-major, top row first). */
class Plane(val w: Int, val h: Int, val d: FloatArray) {
    constructor(w: Int, h: Int, fill: Float = 0f) : this(w, h, FloatArray(w * h).also { if (fill != 0f) it.fill(fill) })

    init {
        require(d.size == w * h) { "Plane data size mismatch" }
    }

    @Suppress("NOTHING_TO_INLINE")
    inline operator fun get(x: Int, y: Int): Float = d[y * w + x]

    @Suppress("NOTHING_TO_INLINE")
    inline operator fun set(x: Int, y: Int, v: Float) {
        d[y * w + x] = v
    }

    val max: Float get() = if (d.isEmpty()) 0f else d.max()
    val sum: Double get() { var s = 0.0; for (v in d) s += v; return s }
    fun copy(): Plane = Plane(w, h, d.copyOf())
}

/** Small numeric helpers that mirror the OpenCV / numpy calls used by the desktop app. */
object Imaging {

    // ---- percentiles (numpy linear interpolation)

    fun percentileSorted(s: FloatArray, q: Double): Double {
        if (s.isEmpty()) return 0.0
        val pos = q / 100 * (s.size - 1)
        val lo = floor(pos).toInt()
        val hi = min(s.size - 1, lo + 1)
        val f = pos - lo
        return s[lo] * (1 - f) + s[hi] * f
    }

    fun percentile(v: FloatArray, q: Double): Double = percentileSorted(v.sortedArray(), q)

    fun median(v: FloatArray): Double = percentile(v, 50.0)

    // ---- borders

    /** OpenCV BORDER_REFLECT_101 index. */
    fun reflect101(i: Int, n: Int): Int {
        if (n == 1) return 0
        var x = i
        while (x < 0 || x >= n) {
            if (x < 0) x = -x
            if (x >= n) x = 2 * n - 2 - x
        }
        return x
    }

    // ---- Gaussian blur

    /** Same kernel size rule as cv2.GaussianBlur for float images: round(sigma * 8 + 1) | 1. */
    fun gaussianKernel(sigma: Double): FloatArray {
        val k = rndInt(sigma * 8 + 1) or 1
        val c = k / 2
        val w = FloatArray(k) { i ->
            val x = (i - c).toDouble()
            exp(-x * x / (2 * sigma * sigma)).toFloat()
        }
        var s = 0f
        for (v in w) s += v
        for (i in w.indices) w[i] /= s
        return w
    }

    fun gaussianBlur(p: Plane, sigma: Double): Plane {
        if (sigma <= 0 || p.w <= 0 || p.h <= 0) return p
        val kernel = gaussianKernel(sigma)
        val r = kernel.size / 2
        val w = p.w
        val h = p.h
        // horizontal pass
        val tmp = FloatArray(w * h)
        val idx = IntArray(w + 2 * r) { reflect101(it - r, w) }
        val buf = FloatArray(w + 2 * r)
        val src = p.d
        for (y in 0 until h) {
            val row = y * w
            for (i in buf.indices) buf[i] = src[row + idx[i]]
            for (x in 0 until w) {
                var acc = 0f
                for (k in kernel.indices) acc += buf[x + k] * kernel[k]
                tmp[row + x] = acc
            }
        }
        // vertical pass, row by row so memory access stays sequential
        val out = FloatArray(w * h)
        for (y in 0 until h) {
            val o = y * w
            for (k in kernel.indices) {
                val sy = reflect101(y + k - r, h) * w
                val kv = kernel[k]
                for (x in 0 until w) out[o + x] += tmp[sy + x] * kv
            }
        }
        return Plane(w, h, out)
    }

    // ---- resizing float maps

    /** Bilinear resize with pixel-centre alignment (as cv2.resize INTER_LINEAR). */
    fun resize(p: Plane, ow: Int, oh: Int): Plane {
        if (ow == p.w && oh == p.h) return p
        val out = Plane(ow, oh)
        val sx = p.w.toDouble() / ow
        val sy = p.h.toDouble() / oh
        val x0s = IntArray(ow); val x1s = IntArray(ow); val txs = FloatArray(ow)
        for (x in 0 until ow) {
            val fx = max(0.0, min((p.w - 1).toDouble(), (x + 0.5) * sx - 0.5))
            x0s[x] = fx.toInt(); x1s[x] = min(p.w - 1, x0s[x] + 1); txs[x] = (fx - x0s[x]).toFloat()
        }
        for (y in 0 until oh) {
            val fy = max(0.0, min((p.h - 1).toDouble(), (y + 0.5) * sy - 0.5))
            val y0 = fy.toInt()
            val y1 = min(p.h - 1, y0 + 1)
            val ty = (fy - y0).toFloat()
            val r0 = y0 * p.w
            val r1 = y1 * p.w
            for (x in 0 until ow) {
                val tx = txs[x]
                val a = p.d[r0 + x0s[x]] * (1 - tx) + p.d[r0 + x1s[x]] * tx
                val b = p.d[r1 + x0s[x]] * (1 - tx) + p.d[r1 + x1s[x]] * tx
                out.d[y * ow + x] = a * (1 - ty) + b * ty
            }
        }
        return out
    }

    /** Area-average shrink of a float plane (INTER_AREA when shrinking). */
    fun shrink(p: Plane, ow: Int, oh: Int): Plane {
        if (ow >= p.w || oh >= p.h) return resize(p, ow, oh)
        val out = Plane(ow, oh)
        val cnt = Plane(ow, oh)
        for (y in 0 until p.h) {
            val oy = min(oh - 1, y * oh / p.h)
            for (x in 0 until p.w) {
                val ox = min(ow - 1, x * ow / p.w)
                out.d[oy * ow + ox] += p.d[y * p.w + x]
                cnt.d[oy * ow + ox] += 1f
            }
        }
        for (i in out.d.indices) out.d[i] /= max(cnt.d[i], 1f)
        return out
    }

    // ---- colour planes

    /** cv2.cvtColor(RGB2GRAY) on 8-bit: rounded 0.299 R + 0.587 G + 0.114 B, scaled to 0...1. */
    fun gray(img: RGBImage): Plane {
        val n = img.width * img.height
        val out = FloatArray(n)
        val p = img.data
        for (i in 0 until n) {
            val v = 0.299f * p.u(i * 3) + 0.587f * p.u(i * 3 + 1) + 0.114f * p.u(i * 3 + 2)
            out[i] = rnd(v) / 255f
        }
        return Plane(img.width, img.height, out)
    }

    /** Rec.709 luma on 0...1 floats (signals._luma). */
    fun luma(img: RGBImage): Plane {
        val n = img.width * img.height
        val out = FloatArray(n)
        val p = img.data
        for (i in 0 until n) {
            out[i] = (p.u(i * 3) * 0.2126f + p.u(i * 3 + 1) * 0.7152f + p.u(i * 3 + 2) * 0.0722f) / 255f
        }
        return Plane(img.width, img.height, out)
    }

    /** Split into three 0...255 float planes. */
    fun planes(img: RGBImage): List<Plane> {
        val n = img.width * img.height
        val r = FloatArray(n); val g = FloatArray(n); val b = FloatArray(n)
        val p = img.data
        for (i in 0 until n) {
            r[i] = p.u(i * 3).toFloat(); g[i] = p.u(i * 3 + 1).toFloat(); b[i] = p.u(i * 3 + 2).toFloat()
        }
        return listOf(Plane(img.width, img.height, r), Plane(img.width, img.height, g), Plane(img.width, img.height, b))
    }

    fun merge(c: List<Plane>): RGBImage {
        val w = c[0].w
        val h = c[0].h
        val out = ByteArray(w * h * 3)
        for (i in 0 until w * h) {
            for (k in 0 until 3) out[i * 3 + k] = clampByte(c[k].d[i])
        }
        return RGBImage(w, h, out)
    }

    fun blurRGB(img: RGBImage, sigma: Double): RGBImage = merge(planes(img).map { gaussianBlur(it, sigma) })

    /** Resize an RGB image with bilinear sampling (used for enlarging; shrinking uses ImagePrep.resizeArea). */
    fun resizeBilinear(src: RGBImage, ow: Int, oh: Int): RGBImage = merge(planes(src).map { resize(it, ow, oh) })

    // ---- CIE Lab (OpenCV float RGB2LAB conventions: L 0...100, a/b roughly -127...127)

    val srgbToLinear = FloatArray(256) { i ->
        val c = i / 255f
        if (c <= 0.04045f) c / 12.92f else ((c + 0.055f) / 1.055f).toDouble().pow(2.4).toFloat()
    }

    @Suppress("NOTHING_TO_INLINE")
    inline fun labF(t: Float): Float = if (t > 0.008856f) Math.cbrt(t.toDouble()).toFloat() else 7.787f * t + 16.0f / 116.0f

    /** Writes L, a, b into `out[0..2]`. */
    fun lab(r8: Int, g8: Int, b8: Int, out: FloatArray) {
        val lr = srgbToLinear[r8]; val lg = srgbToLinear[g8]; val lb = srgbToLinear[b8]
        val x = (0.412453f * lr + 0.357580f * lg + 0.180423f * lb) / 0.950456f
        val y = 0.212671f * lr + 0.715160f * lg + 0.072169f * lb
        val z = (0.019334f * lr + 0.119193f * lg + 0.950227f * lb) / 1.088754f
        val fx = labF(x); val fy = labF(y); val fz = labF(z)
        out[0] = if (y > 0.008856f) 116f * fy - 16f else 903.3f * y
        out[1] = 500f * (fx - fy)
        out[2] = 200f * (fy - fz)
    }

    private fun finv(t: Float): Float = if (t > 0.206893f) t * t * t else (t - 16.0f / 116.0f) / 7.787f

    private fun gamma(c: Float): Float {
        val v = max(0f, min(1f, c))
        return if (v <= 0.0031308f) 12.92f * v else 1.055f * v.toDouble().pow(1 / 2.4).toFloat() - 0.055f
    }

    /** Lab -> 8-bit sRGB (inverse of `lab`), clipped.  Packed as 0xRRGGBB. */
    fun rgb(L: Float, a: Float, b: Float): Int {
        val fy = (L + 16f) / 116f
        val fx = fy + a / 500f
        val fz = fy - b / 200f
        val y = if (L > 7.9996f) fy * fy * fy else L / 903.3f
        val x = finv(fx) * 0.950456f
        val z = finv(fz) * 1.088754f
        val r = gamma(3.240479f * x - 1.537150f * y - 0.498535f * z)
        val g = gamma(-0.969256f * x + 1.875992f * y + 0.041556f * z)
        val bl = gamma(0.055648f * x - 0.204043f * y + 1.057311f * z)
        return ((r * 255f + 0.5f).toInt() shl 16) or ((g * 255f + 0.5f).toInt() shl 8) or (bl * 255f + 0.5f).toInt()
    }
}

/**
 * Mirrors the Python preprocessing (photocoach/signals.py `to_std`, mobile/extract.py `views`)
 * so the on-device model sees what it was trained on.
 */
object ImagePrep {

    /** The standard size for a w x h file: long side at most `longSide`, never enlarged. */
    fun standardSize(pxW: Int, pxH: Int, longSide: Int): Pair<Int, Int> {
        val s = min(1.0, longSide.toDouble() / max(pxW, pxH))
        return Pair(max(1, rndInt(pxW * s)), max(1, rndInt(pxH * s)))
    }

    /** Shrink a decoded (already upright) image to the standard size. */
    fun standardImage(full: RGBImage, longSide: Int): RGBImage {
        val (w, h) = standardSize(full.width, full.height, longSide)
        return resizeArea(full, w, h)
    }

    private class Spans(val start: IntArray, val idx: IntArray, val wgt: FloatArray)

    /** For each of m outputs: the source indices it covers and their weights (sum 1). */
    private fun spans(n: Int, m: Int): Spans {
        val scale = n.toDouble() / m
        val start = IntArray(m + 1)
        val idx = ArrayList<Int>()
        val wgt = ArrayList<Float>()
        for (o in 0 until m) {
            start[o] = idx.size
            val a = o * scale
            val b = (o + 1) * scale
            var i = floor(a).toInt()
            var any = false
            while (i < b && i < n) {
                val cover = min(b, (i + 1).toDouble()) - max(a, i.toDouble())
                if (cover > 1e-9) { idx.add(i); wgt.add((cover / scale).toFloat()); any = true }
                i += 1
            }
            if (!any) { idx.add(min(n - 1, max(0, a.toInt()))); wgt.add(1f) }
        }
        start[m] = idx.size
        return Spans(start, idx.toIntArray(), wgt.toFloatArray())
    }

    /**
     * Area-averaging resize (what OpenCV's INTER_AREA does when shrinking): every output pixel is the
     * mean of the source rectangle it covers, with fractional coverage at the borders.
     */
    fun resizeArea(src: RGBImage, ow: Int, oh: Int): RGBImage {
        if (ow == src.width && oh == src.height) return src
        val sw = src.width
        val sh = src.height
        val xs = spans(sw, ow)
        val ys = spans(sh, oh)
        val s = src.data
        val tmp = FloatArray(sh * ow * 3)          // horizontal pass
        for (y in 0 until sh) {
            val row = y * sw * 3
            for (x in 0 until ow) {
                var r = 0f; var g = 0f; var b = 0f
                for (k in xs.start[x] until xs.start[x + 1]) {
                    val p = row + xs.idx[k] * 3
                    val wg = xs.wgt[k]
                    r += s.u(p) * wg; g += s.u(p + 1) * wg; b += s.u(p + 2) * wg
                }
                val o = (y * ow + x) * 3
                tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b
            }
        }
        val out = ByteArray(oh * ow * 3)            // vertical pass
        val acc = FloatArray(ow * 3)
        for (y in 0 until oh) {
            acc.fill(0f)
            for (k in ys.start[y] until ys.start[y + 1]) {
                val p = ys.idx[k] * ow * 3
                val wg = ys.wgt[k]
                for (j in 0 until ow * 3) acc[j] += tmp[p + j] * wg
            }
            val o = y * ow * 3
            for (j in 0 until ow * 3) out[o + j] = clampByte(acc[j])
        }
        return RGBImage(ow, oh, out)
    }

    fun crop(src: RGBImage, x0: Int, y0: Int, size: Int): RGBImage {
        val out = ByteArray(size * size * 3)
        for (y in 0 until size) {
            System.arraycopy(src.data, ((y0 + y) * src.width + x0) * 3, out, y * size * 3, size * 3)
        }
        return RGBImage(size, size, out)
    }

    /**
     * The five model inputs: the whole frame squashed to a square, then four native-resolution
     * crops centred on the rule-of-thirds intersections (top-left, top-right, bottom-left, bottom-right).
     */
    fun views(std: RGBImage, size: Int): List<RGBImage> {
        val out = ArrayList<RGBImage>(5)
        out.add(resizeArea(std, size, size))
        var base = std
        if (min(std.width, std.height) < size) {                       // tiny photo: enlarge so a crop fits
            val s = size.toDouble() / min(std.width, std.height)
            base = Imaging.resizeBilinear(std, max(size, rndInt(std.width * s)), max(size, rndInt(std.height * s)))
        }
        for (fy in doubleArrayOf(1.0 / 3.0, 2.0 / 3.0)) {
            for (fx in doubleArrayOf(1.0 / 3.0, 2.0 / 3.0)) {
                val y0 = min(max(fy * base.height - size / 2.0, 0.0), (base.height - size).toDouble()).toInt()
                val x0 = min(max(fx * base.width - size / 2.0, 0.0), (base.width - size).toDouble()).toInt()
                out.add(crop(base, x0, y0, size))
            }
        }
        return out
    }
}

internal fun hypot(a: Double, b: Double): Double = sqrt(a * a + b * b)

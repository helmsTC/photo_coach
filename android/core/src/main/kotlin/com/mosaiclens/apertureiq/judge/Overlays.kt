package com.mosaiclens.apertureiq.judge

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Pixel overlays drawn on the standard image (engine.py _ov_*).  Vector annotations
 * (thirds grid, horizon, framing box, object numbers) are drawn live by the UI instead.
 */
object Overlays {

    /** grey * keep + colour * 0.15: the photo pushed back so the overlay reads. */
    private fun dim(img: RGBImage, keep: Float): FloatArray {
        val n = img.width * img.height
        val out = FloatArray(n * 3)
        for (i in 0 until n) {
            val r = img.data.u(i * 3).toFloat(); val g = img.data.u(i * 3 + 1).toFloat(); val b = img.data.u(i * 3 + 2).toFloat()
            val gray = rnd(0.299f * r + 0.587f * g + 0.114f * b)
            out[i * 3] = gray * keep + r * 0.15f
            out[i * 3 + 1] = gray * keep + g * 0.15f
            out[i * 3 + 2] = gray * keep + b * 0.15f
        }
        return out
    }

    private fun image(f: FloatArray, w: Int, h: Int): RGBImage =
        RGBImage(w, h, ByteArray(f.size) { max(0f, min(255f, f[it])).toInt().toByte() })

    /** Red = any channel blown to white, blue = crushed to black. */
    fun clipping(img: RGBImage): RGBImage {
        val out = dim(img, 0.55f)
        for (i in 0 until img.width * img.height) {
            val r = img.data.u(i * 3); val g = img.data.u(i * 3 + 1); val b = img.data.u(i * 3 + 2)
            val y = (r * 0.2126f + g * 0.7152f + b * 0.0722f) / 255f
            if (max(r, max(g, b)) >= 252) { out[i * 3] = 255f; out[i * 3 + 1] = 45f; out[i * 3 + 2] = 60f }
            if (y <= 0.016f) { out[i * 3] = 40f; out[i * 3 + 1] = 110f; out[i * 3 + 2] = 255f }
        }
        return image(out, img.width, img.height)
    }

    private val infernoT = floatArrayOf(0.00f, 0.13f, 0.25f, 0.38f, 0.50f, 0.63f, 0.75f, 0.88f, 1.00f)
    private val infernoC = arrayOf(
        floatArrayOf(0f, 0f, 4f), floatArrayOf(31f, 12f, 72f), floatArrayOf(85f, 15f, 109f), floatArrayOf(136f, 34f, 106f),
        floatArrayOf(186f, 54f, 85f), floatArrayOf(227f, 89f, 51f), floatArrayOf(249f, 140f, 10f), floatArrayOf(249f, 201f, 50f),
        floatArrayOf(252f, 255f, 164f),
    )

    private fun inferno(t: Float, out: FloatArray) {
        val v = max(0f, min(1f, t))
        for (k in 1 until infernoT.size) {
            if (v <= infernoT[k]) {
                val f = (v - infernoT[k - 1]) / (infernoT[k] - infernoT[k - 1])
                for (j in 0 until 3) out[j] = infernoC[k - 1][j] + (infernoC[k][j] - infernoC[k - 1][j]) * f
                return
            }
        }
        for (j in 0 until 3) out[j] = infernoC.last()[j]
    }

    /** Where the photo is sharp: bright = crisp fine detail.  `scale` is shared by both photos so they compare. */
    fun focus(img: RGBImage, detail: Plane, scale: Float): RGBImage {
        val w = img.width; val h = img.height
        val d = Plane(detail.w, detail.h, FloatArray(detail.d.size) { max(0f, min(1f, detail.d[it] / (scale + 1e-9f))).toDouble().pow(0.6).toFloat() })
        val big = Imaging.gaussianBlur(Imaging.resize(d, w, h), 0.02 * max(w, h))
        val out = dim(img, 0.5f)
        val c = FloatArray(3)
        for (i in 0 until w * h) {
            inferno(big.d[i], c)
            out[i * 3] = out[i * 3] * 0.45f + c[0] * 0.6f
            out[i * 3 + 1] = out[i * 3 + 1] * 0.45f + c[1] * 0.6f
            out[i * 3 + 2] = out[i * 3 + 2] * 0.45f + c[2] * 0.6f
        }
        return image(out, w, h)
    }

    /** Green = the score leans on this region more than on the rest of the frame; red = less (or it hurts). */
    fun impact(img: RGBImage, heat: Plane): RGBImage {
        val w = img.width; val h = img.height
        val med = Imaging.median(heat.d).toFloat()
        val rel = Plane(heat.w, heat.h, FloatArray(heat.d.size) { heat.d[it] - med })
        var mx = 0f
        for (v in rel.d) mx = max(mx, abs(v))
        mx += 1e-9f
        val m = Imaging.resize(rel, w, h)
        val out = dim(img, 0.6f)
        val gc = floatArrayOf(40f, 210f, 150f); val rc = floatArrayOf(255f, 80f, 60f)
        for (i in 0 until w * h) {
            val v = m.d[i] / mx
            val pos = max(0f, min(1f, v)).toDouble().pow(0.8).toFloat(); val neg = max(0f, min(1f, -v)).toDouble().pow(0.8).toFloat()
            for (k in 0 until 3) {
                var o = out[i * 3 + k] * (1 - 0.75f * pos) + gc[k] * 0.75f * pos
                o = o * (1 - 0.75f * neg) + rc[k] * 0.75f * neg
                out[i * 3 + k] = o
            }
        }
        return image(out, w, h)
    }

    /** 0xRRGGBB */
    fun roleColor(role: String): Int = when (role) {
        "subject" -> 0xFFC828
        "distraction" -> 0xFF503C
        "helps" -> 0x28D296
        else -> 0xEBEBEB
    }

    /** Non-mask pixels connected to the frame border. */
    private fun reachableFromBorder(m: BooleanArray, w: Int, h: Int): BooleanArray {
        val out = BooleanArray(w * h)
        val stack = IntArray(w * h)
        var sp = 0
        fun push(i: Int) { if (!m[i] && !out[i]) { out[i] = true; stack[sp++] = i } }
        for (x in 0 until w) { push(x); push((h - 1) * w + x) }
        for (y in 0 until h) { push(y * w); push(y * w + w - 1) }
        while (sp > 0) {
            val i = stack[--sp]
            val x = i % w; val y = i / w
            if (x > 0) push(i - 1)
            if (x < w - 1) push(i + 1)
            if (y > 0) push(i - w)
            if (y < h - 1) push(i + w)
        }
        return out
    }

    /** Tinted masks + outlines; yellow = subject, red = distraction, green = the score leans on it. */
    fun objects(img: RGBImage, segs: List<Segment>): RGBImage {
        val w = img.width; val h = img.height
        val out = FloatArray(w * h * 3)
        for (i in 0 until w * h) {
            val r = img.data.u(i * 3).toFloat(); val g = img.data.u(i * 3 + 1).toFloat(); val b = img.data.u(i * 3 + 2).toFloat()
            val gray = 0.299f * r + 0.587f * g + 0.114f * b
            out[i * 3] = gray * 0.35f + r * 0.4f; out[i * 3 + 1] = gray * 0.35f + g * 0.4f; out[i * 3 + 2] = gray * 0.35f + b * 0.4f
        }
        for (g in segs) {
            if (!(g.role == "distraction" || g.role == "subject" || (g.role == "helps" && g.compact))) continue
            val c = roleColor(g.role)
            val cr = ((c shr 16) and 0xFF) * 0.38f; val cg = ((c shr 8) and 0xFF) * 0.38f; val cb = (c and 0xFF) * 0.38f
            for (i in 0 until w * h) {
                if (!g.mask[i]) continue
                out[i * 3] = out[i * 3] * 0.62f + cr
                out[i * 3 + 1] = out[i * 3 + 1] * 0.62f + cg
                out[i * 3 + 2] = out[i * 3 + 2] * 0.62f + cb
            }
        }
        val t = max(1, rndInt(max(w, h) / 420.0))
        for (g in segs.sortedBy { if (it.role != "other") 1 else 0 }) {
            val c = roleColor(g.role)
            val cr = ((c shr 16) and 0xFF).toFloat(); val cg = ((c shr 8) and 0xFF).toFloat(); val cb = (c and 0xFF).toFloat()
            val thick = if (g.role == "other") t else t + 1
            // outer contour only (cv2.RETR_EXTERNAL): holes inside the mask are not outlined
            val outside = reachableFromBorder(g.mask, w, h)
            for (y in 0 until h) {
                for (x in 0 until w) {
                    if (!g.mask[y * w + x]) continue
                    val edge = x == 0 || y == 0 || x == w - 1 || y == h - 1 ||
                        outside[y * w + x - 1] || outside[y * w + x + 1] || outside[(y - 1) * w + x] || outside[(y + 1) * w + x]
                    if (!edge) continue
                    for (yy in max(0, y - thick / 2)..min(h - 1, y + thick / 2)) {
                        for (xx in max(0, x - thick / 2)..min(w - 1, x + thick / 2)) {
                            val o = (yy * w + xx) * 3
                            out[o] = cr; out[o + 1] = cg; out[o + 2] = cb
                        }
                    }
                }
            }
        }
        return image(out, w, h)
    }
}

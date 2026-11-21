package com.mosaiclens.apertureiq.platform

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.exifinterface.media.ExifInterface
import com.mosaiclens.apertureiq.judge.Decoded
import com.mosaiclens.apertureiq.judge.Decoder
import com.mosaiclens.apertureiq.judge.ExifInfo
import com.mosaiclens.apertureiq.judge.ImagePrep
import com.mosaiclens.apertureiq.judge.JudgeError
import com.mosaiclens.apertureiq.judge.JudgeInfo
import com.mosaiclens.apertureiq.judge.OrtNetworks
import com.mosaiclens.apertureiq.judge.PhotoJudge
import com.mosaiclens.apertureiq.judge.RGBImage
import com.mosaiclens.apertureiq.judge.SegmenterLabels
import com.mosaiclens.apertureiq.judge.rnd
import java.io.ByteArrayInputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max

/** Pixels of a decoded bitmap and the size of the file it came from. */
object AndroidDecoder : Decoder {
    /** Photos are reduced to this long side straight after decoding (the analysis itself runs at 1024). */
    const val workingLongSide = 2048

    override fun decode(bytes: ByteArray): Decoded {
        val (img, w, h) = pixels(bytes, workingLongSide) ?: throw JudgeError.BadImage()
        return Decoded(img, w, h, exif(bytes))
    }

    /**
     * Decode, apply the EXIF orientation, and reduce to at most `longSide`.  Returns the image and the
     * file's own (upright) pixel size.
     *
     * The pixel values must reach the network exactly as stored in the file: the training pipeline (PIL)
     * ignores embedded ICC profiles, so an Adobe RGB / Display P3 photo is NOT converted to sRGB here.
     * Decoding into the file's own colour space and reading the raw buffer does that.
     */
    fun pixels(bytes: ByteArray, longSide: Int): Triple<RGBImage, Int, Int>? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        val fw = bounds.outWidth; val fh = bounds.outHeight
        if (fw <= 0 || fh <= 0) return null
        var sample = 1
        while (max(fw, fh) / (sample * 2) >= longSide) sample *= 2      // power-of-two reduction in the decoder, never below longSide
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inPreferredColorSpace = bounds.outColorSpace                // null = decoder's choice (sRGB)
        }
        var bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts) ?: return null
        if (bmp.config != Bitmap.Config.ARGB_8888) {
            val c = bmp.copy(Bitmap.Config.ARGB_8888, false) ?: return null
            bmp.recycle(); bmp = c
        }
        val w = bmp.width; val h = bmp.height
        val stride = bmp.rowBytes
        val buf = ByteBuffer.allocate(stride * h)
        bmp.copyPixelsToBuffer(buf)                                     // raw RGBA, premultiplied, no colour conversion
        val hasAlpha = bmp.hasAlpha()
        bmp.recycle()
        val raw = buf.array()
        val rgb = ByteArray(w * h * 3)
        for (y in 0 until h) {
            var s = y * stride; var o = y * w * 3
            for (x in 0 until w) {
                if (hasAlpha) {
                    val under = 255 - (raw[s + 3].toInt() and 0xFF)     // composite on white
                    rgb[o] = minOf(255, (raw[s].toInt() and 0xFF) + under).toByte()
                    rgb[o + 1] = minOf(255, (raw[s + 1].toInt() and 0xFF) + under).toByte()
                    rgb[o + 2] = minOf(255, (raw[s + 2].toInt() and 0xFF) + under).toByte()
                } else {
                    rgb[o] = raw[s]; rgb[o + 1] = raw[s + 1]; rgb[o + 2] = raw[s + 2]
                }
                s += 4; o += 3
            }
        }
        var img = RGBImage(w, h, rgb)
        val orientation = try {
            ExifInterface(ByteArrayInputStream(bytes)).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } catch (e: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }
        img = orient(img, orientation)
        val swapped = img.width != w
        if (max(img.width, img.height) > longSide) {
            val (sw, sh) = ImagePrep.standardSize(img.width, img.height, longSide)
            img = ImagePrep.resizeArea(img, sw, sh)
        }
        return Triple(img, if (swapped) fh else fw, if (swapped) fw else fh)
    }

    /** Apply an EXIF orientation (1...8) to upright pixels. */
    fun orient(src: RGBImage, orientation: Int): RGBImage {
        if (orientation <= 1 || orientation > 8) return src
        val w = src.width; val h = src.height
        val swap = orientation >= 5
        val ow = if (swap) h else w; val oh = if (swap) w else h
        val out = ByteArray(ow * oh * 3)
        for (y in 0 until oh) {
            for (x in 0 until ow) {
                val sx: Int; val sy: Int
                when (orientation) {
                    2 -> { sx = w - 1 - x; sy = y }                 // mirrored
                    3 -> { sx = w - 1 - x; sy = h - 1 - y }         // 180
                    4 -> { sx = x; sy = h - 1 - y }                 // flipped
                    5 -> { sx = y; sy = x }                         // transpose
                    6 -> { sx = y; sy = h - 1 - x }                 // rotate 90 clockwise to display
                    7 -> { sx = w - 1 - y; sy = h - 1 - x }         // transverse
                    else -> { sx = w - 1 - y; sy = x }              // 8: rotate 270
                }
                val s = (sy * w + sx) * 3; val o = (y * ow + x) * 3
                out[o] = src.data[s]; out[o + 1] = src.data[s + 1]; out[o + 2] = src.data[s + 2]
            }
        }
        return RGBImage(ow, oh, out)
    }

    fun exif(bytes: ByteArray): ExifInfo {
        val ex = try { ExifInterface(ByteArrayInputStream(bytes)) } catch (e: Exception) { return ExifInfo() }
        fun str(tag: String) = ex.getAttribute(tag)?.trim()?.trim('\u0000')?.ifEmpty { null }
        fun num(tag: String): Double? = ex.getAttributeDouble(tag, Double.NaN).takeIf { !it.isNaN() }
        return ExifInfo(
            camera = ExifInfo.cameraName(str(ExifInterface.TAG_MAKE), str(ExifInterface.TAG_MODEL)),
            lens = str(ExifInterface.TAG_LENS_MODEL),
            aperture = num(ExifInterface.TAG_F_NUMBER)?.takeIf { it > 0 }?.let { rnd(it * 10) / 10 },
            shutter = num(ExifInterface.TAG_EXPOSURE_TIME)?.takeIf { it > 0 },
            iso = ex.getAttributeInt(ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY, 0).takeIf { it > 0 },
            focal = num(ExifInterface.TAG_FOCAL_LENGTH)?.takeIf { it > 0 }?.let { rnd(it * 10) / 10 },
            focal35 = ex.getAttributeInt(ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM, 0).takeIf { it > 0 },
            evComp = num(ExifInterface.TAG_EXPOSURE_BIAS_VALUE)?.let { rnd(it * 10) / 10 },
            taken = str(ExifInterface.TAG_DATETIME_ORIGINAL),
            flash = if (ex.hasAttribute(ExifInterface.TAG_FLASH)) (ex.getAttributeInt(ExifInterface.TAG_FLASH, 0) and 1) == 1 else null,
        )
    }
}

fun RGBImage.toBitmap(): Bitmap = Bitmap.createBitmap(toArgb(), width, height, Bitmap.Config.ARGB_8888)

/** The app's one PhotoJudge: models are memory-mapped out of the APK on first use. */
object Engine {
    @Volatile private var judge: PhotoJudge? = null

    fun asset(ctx: Context, name: String): ByteArray? = try {
        ctx.assets.open(name).use { it.readBytes() }
    } catch (e: Exception) {
        null
    }

    fun hasAsset(ctx: Context, name: String): Boolean = try {
        ctx.assets.open(name).close(); true
    } catch (e: Exception) {
        false
    }

    /** A read-only mapping of an uncompressed asset (see `noCompress` in build.gradle.kts). */
    private fun map(ctx: Context, name: String): ByteBuffer? = try {
        ctx.assets.openFd(name).use { fd ->
            FileInputStream(fd.fileDescriptor).channel.use { ch ->
                ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    } catch (e: Exception) {
        null
    }

    @Volatile var networks: OrtNetworks? = null
        private set

    /** Runtime settings chosen from measurements on a mid-range tablet (see android/README.md). */
    /**
     * Runtime settings, from measurements on a mid-range tablet (MT8781: 2 fast + 6 slow cores); see android/README.md.
     *  - threads: the models only gain from the fast cores (2, 4 and 8 threads took the same time alone,
     *    and 2 was best once other stages run beside them), so use one thread per fast core.
     *  - xnnpack: 1/3 faster segmenter, nothing for the encoder, slower AI detector, and 1.4 GB more memory.
     *  - nnapi: no gain for the encoder, several times slower for the segmenter and the detector.
     *  - arena: ONNX Runtime's memory pool held 2.5 GB after a Pro comparison; without it 0.66 GB at the same speed.
     */
    private object TUNING {
        val threads: Int get() = fastCores()
        const val xnnpack = 0
        const val nnapi = 0
        const val arena = 0
    }

    /** Cores at 70% or more of the strongest core's capacity (2 to 4); 4 when the kernel does not say. */
    private fun fastCores(): Int = runCatching {
        val caps = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull {
            java.io.File("/sys/devices/system/cpu/cpu$it/cpu_capacity").takeIf { f -> f.canRead() }?.readText()?.trim()?.toIntOrNull()
        }
        val top = caps.maxOrNull() ?: return 4
        caps.count { it * 10 >= top * 7 }.coerceIn(2, 4)
    }.getOrDefault(4)

    /** Debug builds: `adb shell setprop debug.aiq.threads 2` etc. to try runtime settings on a device. */
    private fun prop(k: String, d: Int): Int = if (!com.mosaiclens.apertureiq.BuildConfig.DEBUG) d else runCatching {
        Class.forName("android.os.SystemProperties").getMethod("getInt", String::class.java, Int::class.java).invoke(null, "debug.aiq.$k", d) as Int
    }.getOrDefault(d)

    /**
     * Load the models while the user is still choosing photos: the encoder, tower and AI detector
     * for every comparison, and the segmenter too when a Pro comparison is coming.
     */
    fun warm(context: Context, pro: Boolean) {
        if (prop("warm", 1) == 0) return
        val ctx = context.applicationContext
        Thread({
            runCatching {
                judge(ctx)
                networks?.preloadDetector()
                if (pro) networks?.preloadSegmenter()
            }
        }, "warm").apply { isDaemon = true; priority = Thread.NORM_PRIORITY - 1 }.start()
    }

    @Synchronized
    fun judge(context: Context): PhotoJudge {
        judge?.let { return it }
        val ctx = context.applicationContext
        val info = JudgeInfo.parse(asset(ctx, "judge.json")?.decodeToString() ?: throw JudgeError.MissingResource("judge.json"))
        val seg = asset(ctx, "segmenter.json")?.decodeToString()?.let { SegmenterLabels.parse(it) }
        val threads = prop("threads", TUNING.threads)
        val xnn = prop("xnn", TUNING.xnnpack)
        val arena = prop("arena", TUNING.arena)
        android.util.Log.i("ApertureIQ", "TIME runtime: threads $threads, xnnpack mask $xnn, arena $arena")
        PhotoJudge.cpuPriority = prop("cpuprio", Thread.MIN_PRIORITY)
        val nnapi = prop("nnapi", TUNING.nnapi)
        android.util.Log.i("ApertureIQ", "TIME nnapi mask $nnapi (1 encoder, 2 segmenter, 4 detector; same bits for xnnpack)")
        val bit = mapOf("encoder" to 1, "segmenter" to 2, "ai_detector" to 4)
        val nets = OrtNetworks({ name -> map(ctx, "$name.onnx") }, seg, threads) { name, o ->
            val b = bit[name] ?: 0
            if (arena == 0) o.setCPUArenaAllocator(false)
            if (xnn and b != 0) runCatching { o.addXnnpack(mapOf("intra_op_num_threads" to threads.toString())) }
                .onFailure { android.util.Log.i("ApertureIQ", "TIME xnnpack unavailable for $name: $it") }
            if (nnapi and b != 0) runCatching {
                val flags = java.util.EnumSet.noneOf(ai.onnxruntime.providers.NNAPIFlags::class.java)
                if (prop("nnapi_fp16", 1) == 1) flags.add(ai.onnxruntime.providers.NNAPIFlags.USE_FP16)
                if (prop("nnapi_nocpu", 0) == 1) flags.add(ai.onnxruntime.providers.NNAPIFlags.CPU_DISABLED)
                o.addNnapi(flags)
            }.onFailure { android.util.Log.i("ApertureIQ", "TIME nnapi unavailable for $name: $it") }
        }
        networks = nets
        return PhotoJudge(info, nets, AndroidDecoder).also { judge = it }
    }
}

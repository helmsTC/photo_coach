package com.mosaiclens.apertureiq.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.text.format.DateUtils
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.EmojiEvents
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.ColorMatrix
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import androidx.exifinterface.media.ExifInterface
import com.mosaiclens.apertureiq.judge.Authenticity
import com.mosaiclens.apertureiq.platform.AndroidDecoder
import com.mosaiclens.apertureiq.platform.toBitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.io.ByteArrayInputStream
import java.io.File
import java.security.MessageDigest
import java.util.UUID

/** One saved comparison: two compact photo copies on disk plus the verdict. */
@Serializable
data class HistoryEntry(
    val id: String,
    val date: Long,
    val winner: Int,
    val confidence: Double,
    /** Last analysed with the full Pro pipeline. */
    val pro: Boolean,
    /** Fingerprint of the two original files (in order), so comparing the same pair again updates its entry. */
    val key: String? = null,
    /** Each shot's AI check from the original files (the saved copies lose their labels). */
    val authenticity: List<Authenticity>? = null,
) {
    companion object {
        fun key(a: ByteArray, b: ByteArray): String {
            val h = MessageDigest.getInstance("SHA-256")
            h.update(a); h.update(b)
            return h.digest().joinToString("") { "%02x".format(it) }
        }
    }
}

/**
 * Recent comparisons, kept on this device only (app-private files/History, excluded from backup).
 * Each entry keeps downscaled copies of both photos with their EXIF, so reopening one simply re-runs
 * the analysis; nothing heavier than the photos and the verdict is stored.
 */
class History(context: Context) {
    companion object {
        fun limit(ownsPro: Boolean): Int = if (ownsPro) 10 else 3

        /** The network analyses at 1024px; 2048 leaves headroom for the Pro viewer. */
        const val storedLongSide = 2048
        const val storedQuality = 92

        /** A word for how decisive a verdict was (the free tier shows no numbers). */
        fun strength(confidence: Double): String = if (confidence < 0.56) "Close call" else if (confidence < 0.70) "Solid edge" else "Clear winner"

        private val keptTags = listOf(
            ExifInterface.TAG_MAKE, ExifInterface.TAG_MODEL, ExifInterface.TAG_LENS_MODEL, ExifInterface.TAG_F_NUMBER,
            ExifInterface.TAG_EXPOSURE_TIME, ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY, ExifInterface.TAG_FOCAL_LENGTH,
            ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM, ExifInterface.TAG_EXPOSURE_BIAS_VALUE, ExifInterface.TAG_DATETIME_ORIGINAL,
            ExifInterface.TAG_FLASH,
        )

        /**
         * Upright JPEG copy at most `maxSide` long, written to `out`.  The pixel values are the file's own
         * (no colour-profile conversion), which is what the analysis reads, so re-analysing the copy sees
         * the same photo; camera settings are carried over for the Pro details panel, location is not.
         */
        fun compact(bytes: ByteArray, maxSide: Int, out: File, quality: Int, keepMetadata: Boolean): Boolean {
            val (img, _, _) = AndroidDecoder.pixels(bytes, maxSide) ?: return false
            val bmp = img.toBitmap()
            try {
                out.outputStream().use { if (!bmp.compress(Bitmap.CompressFormat.JPEG, quality, it)) return false }
            } finally {
                bmp.recycle()
            }
            if (keepMetadata) {
                try {
                    val src = ExifInterface(ByteArrayInputStream(bytes))
                    val dst = ExifInterface(out)
                    for (t in keptTags) src.getAttribute(t)?.let { dst.setAttribute(t, it) }
                    dst.setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL.toString())
                    dst.saveAttributes()
                } catch (e: Exception) {
                    // the copy is still a valid photo without its camera settings
                }
            }
            return true
        }
    }

    private val root = File(context.applicationContext.filesDir, "History").apply { mkdirs() }
    private val index = File(root, "index.json")
    private val json = Json { ignoreUnknownKeys = true }
    private val thumbCache = HashMap<String, List<ImageBitmap>>()

    var entries by mutableStateOf<List<HistoryEntry>>(emptyList()); private set

    init {
        entries = try {
            json.decodeFromString(ListSerializer(HistoryEntry.serializer()), index.readText()).filter { File(dir(it.id), "a.jpg").exists() }
        } catch (e: Exception) {
            emptyList()
        }
    }

    private fun dir(id: String) = File(root, id)

    fun visible(ownsPro: Boolean): List<HistoryEntry> = entries.take(limit(ownsPro))

    /** Saves a finished comparison, or refreshes `id` if it is one being reopened.  Returns the entry's id. */
    suspend fun record(
        id: String?, a: ByteArray, b: ByteArray, winner: Int, confidence: Double, pro: Boolean, ownsPro: Boolean, authenticity: List<Authenticity>,
    ): String? {
        val key = withContext(Dispatchers.Default) { HistoryEntry.key(a, b) }
        val existing = entries.firstOrNull { it.id == id || it.key == key }
        if (existing != null) {
            val e = existing.copy(
                date = System.currentTimeMillis(), winner = winner, confidence = confidence, pro = existing.pro || pro,
                authenticity = existing.authenticity ?: authenticity,
            )
            entries = listOf(e) + entries.filter { it.id != existing.id }
            save()
            return e.id
        }
        val newId = UUID.randomUUID().toString()
        val folder = dir(newId)
        val ok = withContext(Dispatchers.IO) {
            try {
                folder.mkdirs()
                compact(a, storedLongSide, File(folder, "a.jpg"), storedQuality, true) &&
                    compact(b, storedLongSide, File(folder, "b.jpg"), storedQuality, true) &&
                    compact(a, 300, File(folder, "ta.jpg"), 80, false) &&
                    compact(b, 300, File(folder, "tb.jpg"), 80, false)
            } catch (e: Exception) {
                false
            }
        }
        if (!ok) { folder.deleteRecursively(); return null }
        var list = listOf(HistoryEntry(newId, System.currentTimeMillis(), winner, confidence, pro, key, authenticity)) + entries
        // drop the oldest beyond this user's allowance
        val keep = limit(ownsPro)
        while (list.size > keep) { remove(list.last().id); list = list.dropLast(1) }
        entries = list
        save()
        return newId
    }

    fun photos(e: HistoryEntry): Pair<ByteArray, ByteArray>? = try {
        Pair(File(dir(e.id), "a.jpg").readBytes(), File(dir(e.id), "b.jpg").readBytes())
    } catch (ex: Exception) {
        null
    }

    fun thumbs(e: HistoryEntry): List<ImageBitmap> {
        thumbCache[e.id]?.let { return it }
        val t = listOf("ta.jpg", "tb.jpg").mapNotNull { BitmapFactory.decodeFile(File(dir(e.id), it).path)?.asImageBitmap() }
        if (t.size == 2) thumbCache[e.id] = t
        return t
    }

    fun delete(e: HistoryEntry) {
        entries = entries.filter { it.id != e.id }
        remove(e.id)
        save()
    }

    private fun remove(id: String) {
        thumbCache.remove(id)
        dir(id).deleteRecursively()
    }

    private fun save() {
        try {
            val tmp = File(root, "index.json.tmp")
            tmp.writeText(json.encodeToString(ListSerializer(HistoryEntry.serializer()), entries))
            tmp.renameTo(index)
        } catch (e: Exception) {
            // the list is rebuilt from what is on disk next launch
        }
    }
}

private val desaturate = ColorFilter.colorMatrix(ColorMatrix().apply { setToSaturation(0.4f) })

/** A saved comparison on the home screen. */
@Composable
fun HistoryCard(entry: HistoryEntry, thumbs: List<ImageBitmap>) {
    Column(
        Modifier.width(154.dp).clip(RoundedCornerShape(18.dp)).background(Theme.card).border(1.dp, Theme.stroke, RoundedCornerShape(18.dp)).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(Modifier.size(130.dp, 100.dp), contentAlignment = Alignment.Center) {
            for ((i, img) in thumbs.withIndex()) {
                val won = i == entry.winner
                Box(Modifier.zIndex(if (won) 1f else 0f).offset(x = if (i == 0) (-20).dp else 20.dp).rotate(if (i == 0) -5f else 5f), contentAlignment = Alignment.TopCenter) {
                    Image(
                        img, null, contentScale = ContentScale.Crop, colorFilter = if (won) null else desaturate,
                        modifier = Modifier.size(70.dp, 88.dp).clip(RoundedCornerShape(10.dp))
                            .border(if (won) 2.dp else 1.dp, if (won) Theme.gold else Theme.tint(i), RoundedCornerShape(10.dp)),
                    )
                    if (won) Icon(Icons.Filled.EmojiEvents, null, Modifier.offset(y = (-12).dp).size(18.dp), tint = Theme.gold)
                }
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Shot ${if (entry.winner == 0) "A" else "B"} won", style = Type.footnote.semibold, color = Color.White)
            if (entry.pro) ProBadge(Modifier.scale(0.8f))
        }
        val ago = DateUtils.getRelativeTimeSpanString(entry.date, System.currentTimeMillis(), DateUtils.MINUTE_IN_MILLIS, DateUtils.FORMAT_ABBREV_RELATIVE)
        Text("${History.strength(entry.confidence)} · $ago", style = Type.caption2, color = Theme.dim, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

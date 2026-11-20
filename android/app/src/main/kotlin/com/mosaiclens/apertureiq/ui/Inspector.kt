package com.mosaiclens.apertureiq.ui

import androidx.compose.animation.Crossfade
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.HelpOutline
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.CenterFocusWeak
import androidx.compose.material.icons.filled.Crop
import androidx.compose.material.icons.filled.CropFree
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.EmojiEvents
import androidx.compose.material.icons.filled.FilterCenterFocus
import androidx.compose.material.icons.filled.Grid3x3
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Photo
import androidx.compose.material.icons.filled.Straighten
import androidx.compose.material.icons.filled.Verified
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.PathFillType
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.mosaiclens.apertureiq.judge.Authenticity
import com.mosaiclens.apertureiq.judge.Comparison
import com.mosaiclens.apertureiq.judge.Lens
import com.mosaiclens.apertureiq.judge.Measurement
import com.mosaiclens.apertureiq.judge.Objects
import com.mosaiclens.apertureiq.judge.Provenance
import com.mosaiclens.apertureiq.judge.Quad
import com.mosaiclens.apertureiq.judge.Segment
import com.mosaiclens.apertureiq.judge.usf
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.tan

val Lens.icon: ImageVector
    get() = when (this) {
        Lens.PHOTO -> Icons.Filled.Photo
        Lens.CLIPPING -> Icons.Filled.WbSunny
        Lens.FOCUS -> Icons.Filled.FilterCenterFocus
        Lens.COMPOSITION -> Icons.Filled.Grid3x3
        Lens.OBJECTS -> Icons.Filled.CenterFocusWeak
        Lens.ATTENTION -> Icons.Filled.Visibility
        Lens.FRAMING -> Icons.Filled.Crop
    }

fun lensesFor(r: Comparison, i: Int): List<Lens> = Lens.entries.filter { l ->
    when (l) {
        Lens.OBJECTS -> r.extras[i].objects.isNotEmpty()
        Lens.FRAMING -> r.extras[i].otherFrame != null
        Lens.CLIPPING, Lens.FOCUS, Lens.ATTENTION -> r.extras[i].layers[l] != null
        else -> true
    }
}

/**
 * Photo viewer with analysis lenses: pixel overlays (clipping, focus, AI attention, objects) plus
 * live vector annotations (thirds grid, subject, horizon, the other photo's frame, object numbers).
 */
@Composable
fun Inspector(r: Comparison, photo: Int, lens: Lens, setPhoto: (Int) -> Unit, setLens: (Lens) -> Unit) {
    val view = LocalView.current
    val p = r.photo(photo)
    val ex = r.extras[photo]
    LaunchedEffect(photo) { if (!lensesFor(r, photo).contains(lens)) setLens(Lens.PHOTO) }

    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        // shot picker
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (i in 0 until 2) {
                val sel = photo == i
                val shape = RoundedCornerShape(14.dp)
                Row(
                    Modifier.weight(1f).clip(shape).background(if (sel) Theme.tint(i).copy(alpha = 0.22f) else Color.White.copy(alpha = 0.04f))
                        .border(1.dp, if (sel) Theme.tint(i) else Theme.stroke, shape).clickable { setPhoto(i); Theme.tick(view) }
                        .padding(horizontal = 12.dp, vertical = 9.dp),
                    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    ShotTag(i, small = true)
                    Text(r.photo(i).name, style = Type.subheadline.semibold, color = Color.White)
                    if (i == r.winnerIndex) Icon(Icons.Filled.EmojiEvents, null, Modifier.size(14.dp), tint = Theme.gold)
                    Spacer(Modifier.weight(1f))
                    Text("${r.photo(i).overall.roundToInt()}", style = Type.subheadline.bold, color = Theme.dim)
                }
            }
        }

        // The framing lens shows the wider of the two shots with the tighter one's frame drawn inside it,
        // so the box is always visible: on the tighter shot the other frame would fall outside the picture.
        val flipped = lens == Lens.FRAMING && ex.otherFrame?.let { q ->
            q.count { it.x >= -0.05 && it.x <= 1.05 && it.y >= -0.05 && it.y <= 1.05 } < 2 && r.extras[1 - photo].otherFrame != null
        } == true
        val shown = if (flipped) 1 - photo else photo
        val sp = r.photo(shown)
        val shape = RoundedCornerShape(18.dp)
        BoxWithConstraints(Modifier.fillMaxWidth().aspectRatio(sp.aspect.toFloat()).clip(shape).border(1.5.dp, Theme.tint(shown).copy(alpha = 0.6f), shape)) {
            Crossfade(Pair(shown, lens), label = "lens") { (s, l) ->
                Image(rememberImage(r.extras[s].layers[l] ?: r.photo(s).std), null, Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
            }
            when (lens) {
                Lens.COMPOSITION -> CompositionOverlay(sp.m)
                Lens.FRAMING -> {
                    val q = if (flipped) r.extras[shown].otherFrame else ex.otherFrame
                    if (q != null) {
                        if (flipped) FrameOverlay(q, Theme.tint(photo), "${p.name}'s frame inside ${sp.name}")
                        else FrameOverlay(q, Theme.tint(1 - photo), "${r.photo(1 - photo).name}'s frame")
                    }
                }
                Lens.OBJECTS -> ObjectBadges(r.extras[shown].objects, maxWidth.value, maxHeight.value)
                else -> {}
            }
        }

        // lens bar
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (l in lensesFor(r, photo)) {
                val sel = lens == l
                Row(
                    Modifier.clip(CircleShape).background(if (sel) Color.White else Color.White.copy(alpha = 0.07f)).clickable { setLens(l); Theme.tick(view) }
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Icon(l.icon, null, Modifier.size(15.dp), tint = if (sel) Color.Black else Color.White)
                    Text(l.title, style = Type.footnote.semibold, color = if (sel) Color.Black else Color.White)
                }
            }
        }
        Text(lens.legend, style = Type.footnote, color = Theme.dim)
        if (lens == Lens.OBJECTS) ObjectList(ex.objects)
        if (lens == Lens.COMPOSITION) CompositionFacts(p.m)
    }
}

// ---- overlays

@Composable
fun CompositionOverlay(m: Measurement) {
    Canvas(Modifier.fillMaxSize()) {
        val w = size.width; val h = size.height
        val shadow = Color.Black.copy(alpha = 0.5f)
        for (f in floatArrayOf(1f / 3, 2f / 3)) {
            drawLine(shadow, Offset(w * f, 0f), Offset(w * f, h), 2.5.dp.toPx())
            drawLine(shadow, Offset(0f, h * f), Offset(w, h * f), 2.5.dp.toPx())
            drawLine(Color.White.copy(alpha = 0.75f), Offset(w * f, 0f), Offset(w * f, h), 1.dp.toPx())
            drawLine(Color.White.copy(alpha = 0.75f), Offset(0f, h * f), Offset(w, h * f), 1.dp.toPx())
        }
        if (m["tilt_conf"] >= 0.3 && m["horizon_y"] >= 0) {
            val t = tan(m["tilt_deg"] * PI / 180).toFloat()
            val yc = (m["horizon_y"] * h).toFloat()
            val a = Offset(0f, yc - t * w / 2); val b = Offset(w, yc + t * w / 2)
            val dash = PathEffect.dashPathEffect(floatArrayOf(8.dp.toPx(), 5.dp.toPx()))
            drawLine(Color.Black.copy(alpha = 0.7f), a, b, 4.5.dp.toPx(), pathEffect = dash)
            drawLine(if (abs(m["tilt_deg"]) >= 1) Theme.bad else Theme.good, a, b, 2.5.dp.toPx(), pathEffect = dash)
        }
        val r = max(14.dp.toPx(), min(w, h) * 0.06f)
        val sc = Offset((m["subject_x"] * w).toFloat(), (m["subject_y"] * h).toFloat())
        drawCircle(Color.Black.copy(alpha = 0.6f), r, sc, style = Stroke(5.dp.toPx()))
        drawCircle(Theme.gold, r, sc, style = Stroke(3.dp.toPx()))
        val wc = Offset((m["weight_cx"] * w).toFloat(), (m["weight_cy"] * h).toFloat())
        val arm = r * 0.55f
        for ((c, sw) in listOf(Color.Black.copy(alpha = 0.6f) to 5.dp.toPx(), Color.Cyan to 3.dp.toPx())) {
            drawLine(c, Offset(wc.x - arm, wc.y), Offset(wc.x + arm, wc.y), sw)
            drawLine(c, Offset(wc.x, wc.y - arm), Offset(wc.x, wc.y + arm), sw)
        }
    }
}

@Composable
fun CompositionFacts(m: Measurement) {
    val tiltKnown = m["tilt_conf"] >= 0.3 && m["horizon_y"] >= 0
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Pill(usf("%.0f%% weight at edges", 100 * m["edge_weight"]), if (m["edge_weight"] > 0.25) Theme.b else Color.White, Icons.Filled.CropFree)
        if (tiltKnown) Pill(usf("Horizon %+.1f°", m["tilt_deg"]), if (abs(m["tilt_deg"]) >= 1) Theme.bad else Theme.good, Icons.Filled.Straighten)
    }
}

@Composable
fun FrameOverlay(quad: Quad, color: Color, label: String) {
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val bw = maxWidth; val bh = maxHeight
        Canvas(Modifier.fillMaxSize()) {
            val pts = quad.map { Offset((it.x * size.width).toFloat(), (it.y * size.height).toFloat()) }
            val box = Path().apply {
                moveTo(pts[0].x, pts[0].y)
                for (k in 1 until pts.size) lineTo(pts[k].x, pts[k].y)
                close()
            }
            val shade = Path().apply {
                fillType = PathFillType.EvenOdd
                addRect(androidx.compose.ui.geometry.Rect(Offset.Zero, size))
                addPath(box)
            }
            drawPath(shade, Color.Black.copy(alpha = 0.55f))
            drawPath(box, Color.Black.copy(alpha = 0.6f), style = Stroke(5.dp.toPx(), join = StrokeJoin.Round))
            drawPath(box, color, style = Stroke(3.dp.toPx(), join = StrokeJoin.Round))
        }
        val tl = quad.first()
        val x = (bw.value * tl.x.toFloat() + 6).coerceIn(4f, max(4f, bw.value - 150))
        val y = (bh.value * tl.y.toFloat() + 6).coerceIn(4f, max(4f, bh.value - 24))
        Text(
            label, style = Type.caption2.bold, color = Color.Black, maxLines = 1,
            modifier = Modifier.offset(x.dp, y.dp).background(color, CircleShape).padding(horizontal = 7.dp, vertical = 3.dp),
        )
    }
}

val Segment.roleColor: Color
    get() = when (role) {
        "subject" -> Theme.gold
        "distraction" -> Theme.bad
        "helps" -> Theme.good
        else -> Color.White.copy(alpha = 0.85f)
    }

val Segment.roleText: String
    get() = when (role) {
        "subject" -> "Subject"
        "distraction" -> "Distracts"
        "helps" -> "Carries the score"
        else -> "Neutral"
    }

@Composable
fun ObjectBadges(segs: List<Segment>, widthDp: Float, heightDp: Float) {
    Box(Modifier.fillMaxSize()) {
        for (g in segs) {
            Box(
                Modifier.offset((g.labelPoint.first * widthDp - 12).dp, (g.labelPoint.second * heightDp - 12).dp).size(24.dp)
                    .background(g.roleColor.copy(alpha = 1f), CircleShape).border(1.dp, Color.Black.copy(alpha = 0.6f), CircleShape),
                contentAlignment = Alignment.Center,
            ) { Text("${g.n}", fontSize = 13.sp, fontWeight = FontWeight.ExtraBold, color = Color.Black, lineHeight = 14.sp) }
        }
    }
}

@Composable
fun ObjectList(segs: List<Segment>) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        for (g in segs) {
            Row(
                Modifier.fillMaxWidth().background(Color.White.copy(alpha = 0.04f), RoundedCornerShape(12.dp)).padding(10.dp),
                verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Box(Modifier.size(22.dp).background(g.roleColor.copy(alpha = 1f), CircleShape), contentAlignment = Alignment.Center) {
                    Text("${g.n}", fontSize = 12.sp, fontWeight = FontWeight.ExtraBold, color = Color.Black, lineHeight = 13.sp)
                }
                Column(Modifier.weight(1f)) {
                    Text(g.label.replaceFirstChar { it.uppercase() }, style = Type.subheadline.semibold, color = Color.White)
                    Text("${Objects.whereText(g)} · ${usf("%.1f", g.area * 100)}% of frame", style = Type.caption, color = Theme.dim)
                }
                Column(horizontalAlignment = Alignment.End) {
                    Text(g.roleText, style = Type.caption.semibold, color = g.roleColor)
                    Text(usf("%+.0f without it", g.delta), style = Type.caption2, color = Theme.dim)
                }
            }
        }
        Text(
            "Each object was painted out and the photo re-scored. \"+5 without it\" means the network likes the photo 5 points more without that object.",
            style = Type.caption2, color = Theme.dim,
        )
    }
}

// ---- AI check

val Authenticity.title: String
    get() = when (verdict) {
        Authenticity.Verdict.labelledAI -> "AI-generated"
        Authenticity.Verdict.aiEdited -> "Edited with AI"
        Authenticity.Verdict.likelyAI -> "Likely AI"
        Authenticity.Verdict.unclear -> "Can't tell"
        Authenticity.Verdict.noSigns -> "No AI signs"
    }

val Authenticity.icon: ImageVector
    get() = when (verdict) {
        Authenticity.Verdict.labelledAI, Authenticity.Verdict.likelyAI -> Icons.Filled.AutoAwesome
        Authenticity.Verdict.aiEdited -> Icons.Filled.AutoFixHigh
        Authenticity.Verdict.unclear -> Icons.AutoMirrored.Filled.HelpOutline
        Authenticity.Verdict.noSigns -> Icons.Filled.Verified
    }

val Authenticity.color: Color
    get() = when (verdict) {
        Authenticity.Verdict.labelledAI, Authenticity.Verdict.likelyAI -> Theme.bad
        Authenticity.Verdict.aiEdited -> Theme.gold
        Authenticity.Verdict.unclear -> Color.White.copy(alpha = 0.8f)
        Authenticity.Verdict.noSigns -> Theme.good
    }

/** One line on how the call was made. */
val Authenticity.basis: String
    get() = when (verdict) {
        Authenticity.Verdict.labelledAI, Authenticity.Verdict.aiEdited -> "Labelled in the file${label?.tool?.let { " by $it" } ?: ""}"
        Authenticity.Verdict.likelyAI -> "The detector found strong AI traces"
        Authenticity.Verdict.unclear -> "The detector's reading is mixed"
        Authenticity.Verdict.noSigns -> "No labels, and the detector found no AI traces"
    }

/** Compact pill for overlays: "Likely AI", "No AI signs", ... */
@Composable
fun AuthBadge(a: Authenticity, modifier: Modifier = Modifier) {
    Row(
        modifier.background(Color(0xD9101018), CircleShape).border(1.dp, a.color.copy(alpha = 0.45f), CircleShape).padding(horizontal = 8.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        Icon(a.icon, null, Modifier.size(12.dp), tint = a.color)
        Text(a.title, style = Type.caption2.bold, color = a.color, maxLines = 1)
    }
}

/** Pro: both shots' AI check, as three labelled facts per shot rather than paragraphs. */
@Composable
fun AuthenticityCard(r: Comparison) {
    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        SectionHeader("AI check", "Was either shot made by an image generator?", Icons.Filled.ImageSearch)
        for (i in 0 until 2) r.extras[i].authenticity?.let { AuthPanel(r, i, it) }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Icon(Icons.Filled.Info, null, Modifier.size(14.dp), tint = Theme.dim)
            Text(
                "A label written by the app that made the image is decisive. The detector is a trained guess: it can miss new generators and heavily compressed images, so it only ever says \"likely\".",
                style = Type.caption2, color = Theme.dim,
            )
        }
    }
}

private fun detectorColor(a: Authenticity): Color {
    val p = a.aiProbability ?: return Theme.dim
    return if (p >= Authenticity.likelyAI) Theme.bad else if (p <= Authenticity.noSigns) Theme.good else Theme.gold
}

@Composable
private fun AuthPanel(r: Comparison, i: Int, a: Authenticity) {
    Column(Modifier.glassCard(14.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Box {
                Image(
                    rememberImage(r.photo(i).std), null, contentScale = ContentScale.Crop,
                    modifier = Modifier.padding(start = 6.dp, top = 6.dp).size(56.dp).clip(RoundedCornerShape(12.dp)).border(1.5.dp, a.color.copy(alpha = 0.7f), RoundedCornerShape(12.dp)),
                )
                ShotTag(i, small = true)
            }
            Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Icon(a.icon, null, Modifier.size(18.dp), tint = a.color)
                    Text(a.title, style = Type.headline, color = a.color)
                }
                Text(a.basis, style = Type.caption, color = Theme.dim)
            }
        }
        Column(Modifier.fillMaxWidth().background(Color.White.copy(alpha = 0.03f), RoundedCornerShape(14.dp))) {
            val l = a.label
            val labelText = when (l?.kind) {
                Provenance.Kind.generated -> "Generated · ${l.tool ?: "AI tool"}"
                Provenance.Kind.edited -> "AI-edited${l.tool?.let { " · $it" } ?: ""}"
                Provenance.Kind.capture -> "Camera capture"
                null -> "None found"
            }
            val labelColor = when (l?.kind) {
                Provenance.Kind.generated -> Theme.bad
                Provenance.Kind.edited -> Theme.gold
                Provenance.Kind.capture -> Theme.good
                null -> Theme.dim
            }
            FactRow(Icons.Filled.Description, "File labels", labelText, labelColor)
            Box(Modifier.fillMaxWidth().height(1.dp).background(Theme.stroke))
            val dc = detectorColor(a)
            FactRow(Icons.Filled.Memory, "Detector", a.aiProbability?.let { "${(it * 100).roundToInt()}% AI-like" } ?: "Didn't run", dc) {
                a.aiProbability?.let { p ->
                    Canvas(Modifier.fillMaxWidth().height(5.dp)) {
                        val rr = CornerRadius(size.height / 2)
                        drawRoundRect(Color.White.copy(alpha = 0.08f), cornerRadius = rr)
                        drawRoundRect(dc, size = Size(max(4.dp.toPx(), size.width * p.toFloat()), size.height), cornerRadius = rr)
                        // the "likely AI" threshold
                        val x = size.width * Authenticity.likelyAI.toFloat()
                        drawLine(Color.White.copy(alpha = 0.5f), Offset(x, 0f), Offset(x, size.height), 1.5.dp.toPx())
                    }
                }
            }
            Box(Modifier.fillMaxWidth().height(1.dp).background(Theme.stroke))
            FactRow(Icons.Filled.CameraAlt, "Camera data", a.camera ?: "None", Color.White)
        }
    }
}

/** "File labels ........ None found", with an optional visual under the value. */
@Composable
fun FactRow(icon: ImageVector, name: String, value: String, color: Color, extra: (@Composable () -> Unit)? = null) {
    Column(Modifier.padding(horizontal = 12.dp, vertical = 10.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Icon(icon, null, Modifier.width(20.dp).size(15.dp), tint = Theme.dim)
            Text(name, style = Type.footnote, color = Theme.dim)
            Spacer(Modifier.weight(1f))
            Text(value, style = Type.footnote.semibold, color = color, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
        if (extra != null) Box(Modifier.padding(start = 30.dp)) { extra() }
    }
}

package com.mosaiclens.apertureiq.ui

import androidx.activity.compose.BackHandler
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Balance
import androidx.compose.material.icons.filled.BarChart
import androidx.compose.material.icons.filled.CenterFocusWeak
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ChevronLeft
import androidx.compose.material.icons.filled.EmojiEvents
import androidx.compose.material.icons.filled.FlipToFront
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.ColorMatrix
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.zIndex
import com.mosaiclens.apertureiq.judge.QuickVerdict
import com.mosaiclens.apertureiq.judge.RGBImage
import com.mosaiclens.apertureiq.judge.Stage
import com.mosaiclens.apertureiq.platform.toBitmap
import kotlinx.coroutines.delay
import kotlin.math.sin

@Composable
fun rememberImage(img: RGBImage): ImageBitmap = remember(img) { img.toBitmap().asImageBitmap() }

val Stage.icon: ImageVector
    get() = when (this) {
        Stage.READING -> Icons.Filled.PhotoLibrary
        Stage.MEASURING -> Icons.Filled.BarChart
        Stage.SCORING -> Icons.Filled.Psychology
        Stage.AUTHENTICITY -> Icons.Filled.ImageSearch
        Stage.MATCHING -> Icons.Filled.FlipToFront
        Stage.OBJECTS -> Icons.Filled.CenterFocusWeak
        Stage.TESTING -> Icons.Filled.AutoFixHigh
        Stage.ATTENTION -> Icons.Filled.Visibility
        Stage.EDITING -> Icons.Filled.Tune
        Stage.COACHING -> Icons.AutoMirrored.Filled.Chat
    }

/** "< New comparison" capsule used at the top of both result screens. */
@Composable
fun BackPill(onClick: () -> Unit) {
    Row(
        Modifier.background(Color.White.copy(alpha = 0.12f), CircleShape).clip(CircleShape).clickable(onClick = onClick).padding(start = 8.dp, end = 14.dp, top = 9.dp, bottom = 9.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(Icons.Filled.ChevronLeft, null, Modifier.size(18.dp), tint = Color.White)
        Text("New comparison", style = Type.subheadline.semibold, color = Color.White)
    }
}

// ---- analysing

@Composable
fun AnalyzingView(model: AppModel) {
    BackHandler { model.cancel() }
    val current = model.stage?.ordinal ?: -1
    val progress by animateFloatAsState((current + 1f) / model.stages.size, tween(300), label = "progress")
    Box(Modifier.fillMaxSize()) {
        AmbientBackground()
        Column(Modifier.fillMaxSize().systemBarsPadding(), horizontalAlignment = Alignment.CenterHorizontally) {
            Spacer(Modifier.height(18.dp))
            Row(Modifier.padding(horizontal = 24.dp), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                ScanningPhoto(model.thumbs[0], 0, Modifier.weight(1f))
                ScanningPhoto(model.thumbs[1], 1, Modifier.weight(1f))
            }
            Spacer(Modifier.height(22.dp))
            Text("Analysing", style = Type.title, color = Color.White)
            Text(model.stage?.title ?: "Warming up the network…", style = Type.subheadline, color = Theme.dim, modifier = Modifier.padding(top = 6.dp))
            Spacer(Modifier.height(20.dp))
            Canvas(Modifier.fillMaxWidth().padding(horizontal = 40.dp).height(4.dp)) {
                val r = CornerRadius(size.height / 2)
                drawRoundRect(Color.White.copy(alpha = 0.15f), cornerRadius = r)
                drawRoundRect(Color.White, size = Size(size.width * progress, size.height), cornerRadius = r)
            }
            Spacer(Modifier.height(22.dp))
            Column(
                Modifier.weight(1f, fill = false).padding(horizontal = 20.dp).glassCard(18.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(11.dp),
            ) {
                for (s in model.stages) {
                    val done = s.ordinal < current; val active = s.ordinal == current
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Box(Modifier.width(22.dp), contentAlignment = Alignment.Center) {
                            when {
                                done -> Icon(Icons.Filled.CheckCircle, null, Modifier.size(20.dp), tint = Theme.good)
                                active -> CircularProgressIndicator(Modifier.size(16.dp), color = Color.White, strokeWidth = 2.dp)
                                else -> Icon(s.icon, null, Modifier.size(18.dp), tint = Color.White.copy(alpha = 0.3f))
                            }
                        }
                        Text(
                            s.title, style = if (active) Type.subheadline.semibold else Type.subheadline,
                            color = if (done || active) Color.White else Color.White.copy(alpha = 0.35f),
                        )
                    }
                }
            }
            Spacer(Modifier.weight(1f))
            Text("Cancel", style = Type.subheadline.semibold, color = Theme.dim, modifier = Modifier.clickable { model.cancel() }.padding(14.dp))
        }
    }
}

/** Photo with a sweeping scan line. */
@Composable
private fun ScanningPhoto(image: ImageBitmap?, index: Int, modifier: Modifier = Modifier) {
    var t by remember { mutableFloatStateOf(0f) }
    LaunchedEffect(Unit) {
        val start = withFrameNanos { it }
        while (true) withFrameNanos { t = (it - start) / 1e9f }
    }
    val tint = Theme.tint(index)
    val shape = RoundedCornerShape(20.dp)
    Box(modifier.aspectRatio(3f / 4f).clip(shape).background(Theme.card).border(1.5.dp, tint.copy(alpha = 0.8f), shape)) {
        if (image != null) Image(image, null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
        Canvas(Modifier.fillMaxSize()) {
            val phase = (sin(t * 1.6f + index * 1.3f) + 1) / 2
            val beam = 46.dp.toPx()
            val y = phase * (size.height - beam)
            // scan beam
            drawRect(
                Brush.verticalGradient(listOf(Color.Transparent, tint.copy(alpha = 0.55f), Color.White.copy(alpha = 0.9f), tint.copy(alpha = 0.55f), Color.Transparent), y, y + beam),
                Offset(0f, y), Size(size.width, beam), blendMode = BlendMode.Screen,
            )
            // grid
            val g = Color.White.copy(alpha = 0.25f)
            for (f in floatArrayOf(1f / 3, 2f / 3)) {
                drawLine(g, Offset(size.width * f, 0f), Offset(size.width * f, size.height), 0.7.dp.toPx())
                drawLine(g, Offset(0f, size.height * f), Offset(size.width, size.height * f), 0.7.dp.toPx())
            }
        }
        ShotTag(index, small = true, modifier = Modifier.padding(8.dp))
    }
}

// ---- the free tier's verdict

private val faded = ColorFilter.colorMatrix(ColorMatrix().apply { setToSaturation(0.35f) })

/**
 * The free tier's result: both shots, a crown on the winner, one line on how decisive it was.
 * No scores or metrics; the "why" is the Pro upsell underneath.
 */
@Composable
fun VerdictView(v: QuickVerdict, onNew: () -> Unit, onUpgrade: () -> Unit) {
    BackHandler(onBack = onNew)
    val w = v.winnerIndex
    var revealed by remember { mutableStateOf(false) }
    val view = LocalView.current
    LaunchedEffect(Unit) {
        delay(350)
        revealed = true
        Theme.success(view)
    }
    val reveal by animateFloatAsState(if (revealed) 1f else 0f, spring(dampingRatio = 0.62f, stiffness = 170f), label = "reveal")
    val fade by animateFloatAsState(if (revealed) 1f else 0f, tween(450), label = "fade")

    Box(Modifier.fillMaxSize()) {
        AmbientBackground(tintA = Theme.tint(w), tintB = Theme.violet)
        Column(Modifier.fillMaxSize().systemBarsPadding()) {
            Row(Modifier.padding(horizontal = 18.dp, vertical = 6.dp)) { BackPill(onNew) }
            Column(
                Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(bottom = 40.dp),
                horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(26.dp),
            ) {
                Text("THE VERDICT", style = Type.caption.heavy, letterSpacing = 2.sp, color = Theme.dim, modifier = Modifier.padding(top = 8.dp))
                // photos
                Row(Modifier.padding(top = 26.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    for (i in 0 until 2) {
                        val winner = i == w
                        val shape = RoundedCornerShape(22.dp)
                        val scale = 1f + reveal * (if (winner) 0.04f else -0.1f)
                        Box(Modifier.weight(1f).zIndex(if (winner) 1f else 0f).graphicsLayer { scaleX = scale; scaleY = scale }, contentAlignment = Alignment.TopCenter) {
                            Box(Modifier.graphicsLayer { alpha = if (winner) 1f else 1f - 0.4f * fade }) {
                                Image(
                                    rememberImage(v.images[i]), null, contentScale = ContentScale.Crop, colorFilter = if (revealed && !winner) faded else null,
                                    modifier = Modifier.fillMaxWidth().aspectRatio(3f / 4f).clip(shape)
                                        .border(if (winner && revealed) 3.dp else 1.5.dp, if (winner && revealed) Theme.gold else Theme.tint(i).copy(alpha = 0.6f), shape),
                                )
                                ShotTag(i, modifier = Modifier.padding(10.dp))
                            }
                            if (winner) {
                                Icon(
                                    Icons.Filled.EmojiEvents, "Winner", tint = Theme.gold,
                                    modifier = Modifier.size(40.dp).graphicsLayer { translationY = (-30 - 60 * (1 - reveal)).dp.toPx(); alpha = fade },
                                )
                            }
                            // the AI badge stays fully legible on the dimmed loser
                            v.authenticity.getOrNull(i)?.let { a ->
                                Box(Modifier.matchParentSize().padding(bottom = 10.dp).graphicsLayer { alpha = fade }, contentAlignment = Alignment.BottomCenter) { AuthBadge(a) }
                            }
                        }
                    }
                }
                // headline
                Column(
                    Modifier.graphicsLayer { alpha = fade; translationY = (1 - fade) * 14.dp.toPx() },
                    horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text("Shot ${if (w == 0) "A" else "B"} wins", style = Type.largeTitle, color = Color.White)
                    val close = v.confidence < 0.56
                    Pill(v.strength.first, if (close) Theme.gold else Theme.good, if (close) Icons.Filled.Balance else Icons.Filled.CheckCircle)
                    Text(v.strength.second, style = Type.subheadline, color = Theme.dim, textAlign = TextAlign.Center)
                    aiNote(v)?.let { note ->
                        Row(Modifier.padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            Icon(Icons.Filled.AutoAwesome, null, Modifier.size(15.dp), tint = Theme.bad)
                            Text(note, style = Type.footnote.semibold, color = Theme.bad, textAlign = TextAlign.Center)
                        }
                    }
                }
                // upsell
                Column(Modifier.graphicsLayer { alpha = fade }.glassCard(18.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Want to know why?", style = Type.headline, color = Color.White)
                        ProBadge()
                    }
                    Column(verticalArrangement = Arrangement.spacedBy(11.dp)) {
                        for ((icon, title) in ProFeatures.list.take(4)) {
                            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                Icon(icon, null, Modifier.width(24.dp).brushed(Theme.cta), tint = Color.White)
                                Text(title, style = Type.subheadline, color = Color.White, modifier = Modifier.weight(1f))
                                Icon(Icons.Filled.Lock, null, Modifier.size(13.dp), tint = Theme.dim)
                            }
                        }
                    }
                    Row(
                        Modifier.fillMaxWidth().background(Theme.cta, CircleShape).clip(CircleShape).clickable(onClick = onUpgrade).padding(vertical = 14.dp),
                        horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(Icons.Filled.AutoAwesome, null, Modifier.size(18.dp), tint = Color.White)
                        Spacer(Modifier.width(8.dp))
                        Text("See the full breakdown", style = Type.subheadline.bold, color = Color.White)
                    }
                }
            }
        }
    }
}

/** Calls out a shot that looks AI-made, since it changes what the verdict means. */
private fun aiNote(v: QuickVerdict): String? {
    val flagged = v.authenticity.indices.filter { v.authenticity[it].isAI }
    return when (flagged.size) {
        2 -> "Heads up: both shots look AI-generated."
        1 -> "Heads up: Shot ${if (flagged[0] == 0) "A" else "B"} looks AI-generated."
        else -> null
    }
}

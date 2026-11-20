package com.mosaiclens.apertureiq.ui

import android.view.HapticFeedbackConstants
import android.view.View
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.BlurOn
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Contrast
import androidx.compose.material.icons.filled.Crop
import androidx.compose.material.icons.filled.DarkMode
import androidx.compose.material.icons.filled.Exposure
import androidx.compose.material.icons.filled.FilterCenterFocus
import androidx.compose.material.icons.filled.Grain
import androidx.compose.material.icons.filled.Grid3x3
import androidx.compose.material.icons.filled.Lightbulb
import androidx.compose.material.icons.filled.Palette
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.Straighten
import androidx.compose.material.icons.filled.Thermostat
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material.icons.filled.WbTwilight
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sin

object Theme {
    val a = Color(0xFF4094FF)           // shot A: electric blue
    val b = Color(0xFFFF7A38)           // shot B: ember orange
    val gold = Color(0xFFFFCC40)
    val good = Color(0xFF40DB9E)
    val bad = Color(0xFFFF5C5C)
    val violet = Color(0xFF946BFF)
    val bg = Color(0xFF090A12)
    val card = Color.White.copy(alpha = 0.075f)
    val stroke = Color.White.copy(alpha = 0.09f)
    val dim = Color.White.copy(alpha = 0.55f)
    val pink = Color(0xFFF2549E)

    val ctaColors = listOf(Color(0xFF735CFF), pink, Color(0xFFFF8C40))
    val cta: Brush = Brush.horizontalGradient(ctaColors)

    fun tint(i: Int): Color = if (i == 0) a else b

    class Fix(val text: String, val color: Color, val icon: ImageVector)

    fun fixStyle(fix: String): Fix = when (fix) {
        "edit" -> Fix("Fix in the edit", good, Icons.Filled.Tune)
        "reshoot" -> Fix("Needs a reshoot", b, Icons.Filled.CameraAlt)
        else -> Fix("Shoot + edit", violet, Icons.Filled.AutoAwesome)
    }

    fun itemIcon(id: String): ImageVector = when (id) {
        "highlights" -> Icons.Filled.WbSunny
        "shadows" -> Icons.Filled.DarkMode
        "brightness" -> Icons.Filled.Exposure
        "contrast" -> Icons.Filled.Contrast
        "lighting" -> Icons.Filled.WbTwilight
        "muted", "garish", "palette" -> Icons.Filled.Palette
        "cast" -> Icons.Filled.Thermostat
        "sharpness" -> Icons.Filled.FilterCenterFocus
        "noise" -> Icons.Filled.Grain
        "dof" -> Icons.Filled.Camera
        "level" -> Icons.Filled.Straighten
        "distraction" -> Icons.Filled.WarningAmber
        "framing" -> Icons.Filled.Crop
        "composition" -> Icons.Filled.Grid3x3
        "content" -> Icons.Filled.Star
        else -> Icons.Filled.Lightbulb
    }

    fun tick(view: View) { view.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK) }
    fun success(view: View) { view.performHapticFeedback(if (android.os.Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.CONFIRM else HapticFeedbackConstants.LONG_PRESS) }
    fun error(view: View) { view.performHapticFeedback(if (android.os.Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.REJECT else HapticFeedbackConstants.LONG_PRESS) }
}

/** The iOS text styles the screens were designed with. */
object Type {
    val largeTitle = TextStyle(fontSize = 34.sp, fontWeight = FontWeight.ExtraBold, lineHeight = 40.sp)
    val title = TextStyle(fontSize = 28.sp, fontWeight = FontWeight.ExtraBold, lineHeight = 34.sp)
    val title2 = TextStyle(fontSize = 22.sp, fontWeight = FontWeight.Bold, lineHeight = 28.sp)
    val title3 = TextStyle(fontSize = 20.sp, fontWeight = FontWeight.Bold, lineHeight = 25.sp)
    val headline = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.SemiBold, lineHeight = 22.sp)
    val body = TextStyle(fontSize = 16.sp, lineHeight = 22.sp)
    val subheadline = TextStyle(fontSize = 15.sp, lineHeight = 20.sp)
    val footnote = TextStyle(fontSize = 13.sp, lineHeight = 18.sp)
    val caption = TextStyle(fontSize = 12.sp, lineHeight = 16.sp)
    val caption2 = TextStyle(fontSize = 11.sp, lineHeight = 14.sp)
}

fun TextStyle.w(weight: FontWeight) = copy(fontWeight = weight)
val TextStyle.semibold get() = copy(fontWeight = FontWeight.SemiBold)
val TextStyle.bold get() = copy(fontWeight = FontWeight.Bold)
val TextStyle.heavy get() = copy(fontWeight = FontWeight.ExtraBold)

/** Paint content (text, icons) with a gradient. */
fun Modifier.brushed(brush: Brush): Modifier = this
    .graphicsLayer { compositingStrategy = CompositingStrategy.Offscreen }
    .drawWithCache { onDrawWithContent { drawContent(); drawRect(brush, blendMode = BlendMode.SrcIn) } }

/** Slow-drifting coloured glows behind everything. */
@Composable
fun AmbientBackground(tintA: Color = Theme.a, tintB: Color = Theme.b) {
    var t by remember { mutableFloatStateOf(0f) }
    LaunchedEffect(Unit) {
        val start = withFrameNanos { it }
        while (true) withFrameNanos { t = (it - start) / 1e9f }
    }
    Canvas(Modifier.fillMaxSize().background(Theme.bg)) {
        val w = size.width; val h = size.height
        val d = density
        fun glow(c: Color, alpha: Float, cx: Float, cy: Float, r: Float) {
            drawCircle(Brush.radialGradient(listOf(c.copy(alpha = alpha), Color.Transparent), Offset(cx, cy), r), r, Offset(cx, cy))
        }
        glow(tintA, 0.36f, w * 0.15f + sin(t * 0.21f) * 40 * d, h * 0.2f + cos(t * 0.17f) * 50 * d, w * 0.95f)
        glow(tintB, 0.30f, w * 0.9f + cos(t * 0.19f) * 40 * d, h * 0.55f + sin(t * 0.23f) * 60 * d, w * 0.9f)
        glow(Theme.violet, 0.20f, w * 0.5f + sin(t * 0.13f) * 60 * d, h * 0.95f, w * 0.85f)
    }
}

fun Modifier.glassCard(padding: Dp = 16.dp, radius: Dp = 22.dp): Modifier = this
    .fillMaxWidth()
    .clip(RoundedCornerShape(radius))
    .background(Theme.card)
    .border(1.dp, Theme.stroke, RoundedCornerShape(radius))
    .padding(padding)

@Composable
fun SectionHeader(title: String, subtitle: String? = null, icon: ImageVector? = null) {
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (icon != null) Icon(icon, null, Modifier.size(22.dp).brushed(Theme.cta), tint = Color.White)
            Text(title, style = Type.title3, color = Color.White)
        }
        if (subtitle != null) Text(subtitle, style = Type.subheadline, color = Theme.dim)
    }
}

@Composable
fun ShotTag(index: Int, small: Boolean = false, modifier: Modifier = Modifier) {
    val s = if (small) 20.dp else 26.dp
    Box(modifier.size(s).background(Theme.tint(index), CircleShape), contentAlignment = Alignment.Center) {
        Text(if (index == 0) "A" else "B", fontSize = if (small) 11.sp else 13.sp, fontWeight = FontWeight.ExtraBold, color = Color.White, lineHeight = 13.sp)
    }
}

@Composable
fun Pill(text: String, color: Color = Color.White, icon: ImageVector? = null, modifier: Modifier = Modifier) {
    Row(
        modifier
            .background(color.copy(alpha = 0.14f), CircleShape)
            .border(1.dp, color.copy(alpha = 0.3f), CircleShape)
            .padding(horizontal = 10.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        if (icon != null) Icon(icon, null, Modifier.size(13.dp), tint = color)
        Text(text, style = Type.caption.semibold, color = color)
    }
}

/** Animated ring that fills to a 0...100 score, with the number counting up. */
@Composable
fun ScoreRing(value: Double, color: Color, size: Dp = 104.dp, line: Dp = 10.dp, label: String? = null) {
    val shown = remember { Animatable(0f) }
    LaunchedEffect(value) {
        kotlinx.coroutines.delay(150)
        shown.animateTo(value.toFloat(), spring(dampingRatio = 0.85f, stiffness = 30f))
    }
    Box(
        Modifier.size(size).clearAndSetSemantics { contentDescription = "${label ?: "Score"} ${value.roundToInt()} out of 100" },
        contentAlignment = Alignment.Center,
    ) {
        Canvas(Modifier.fillMaxSize()) {
            val sw = line.toPx()
            val inset = sw / 2
            val arc = Size(this.size.width - sw, this.size.height - sw)
            drawArc(Color.White.copy(alpha = 0.08f), 0f, 360f, false, Offset(inset, inset), arc, style = Stroke(sw))
            drawArc(
                Brush.sweepGradient(listOf(color.copy(alpha = 0.5f), color, color.copy(alpha = 0.5f)), center),
                -90f, 360f * shown.value.coerceIn(0f, 100f) / 100f, false, Offset(inset, inset), arc, style = Stroke(sw, cap = StrokeCap.Round),
            )
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("${shown.value.roundToInt()}", fontSize = (size.value * 0.32f).sp, fontWeight = FontWeight.ExtraBold, color = Color.White, lineHeight = (size.value * 0.34f).sp)
            if (label != null) Text(label, fontSize = (size.value * 0.1f).sp, fontWeight = FontWeight.SemiBold, color = Theme.dim, lineHeight = (size.value * 0.12f).sp)
        }
    }
}

/** Thin horizontal meter, 0...100. */
@Composable
fun Meter(value: Double, color: Color, height: Dp = 6.dp, modifier: Modifier = Modifier) {
    val shown = remember { Animatable(0f) }
    LaunchedEffect(value) {
        kotlinx.coroutines.delay(300)
        shown.animateTo(value.coerceIn(0.0, 100.0).toFloat(), tween(1000, easing = FastOutSlowInEasing))
    }
    Canvas(modifier.fillMaxWidth().height(height)) {
        val r = androidx.compose.ui.geometry.CornerRadius(size.height / 2)
        drawRoundRect(Color.White.copy(alpha = 0.08f), cornerRadius = r)
        drawRoundRect(color, size = Size(max(size.height, size.width * shown.value / 100f), size.height), cornerRadius = r)
    }
}

@Composable
fun Bullet(text: String, color: Color = Theme.dim) {
    Row(horizontalArrangement = Arrangement.spacedBy(9.dp)) {
        Box(Modifier.padding(top = 7.dp).size(5.dp).background(color, CircleShape))
        Text(text, style = Type.subheadline, color = Color.White)
    }
}

/** The app's wordmark: "Aperture" in white, "IQ" in the brand gradient. */
@Composable
fun Wordmark(size: Float = 38f, modifier: Modifier = Modifier) {
    Row(modifier.clearAndSetSemantics { contentDescription = "ApertureIQ" }) {
        val style = TextStyle(fontSize = size.sp, fontWeight = FontWeight.ExtraBold, lineHeight = (size * 1.15f).sp)
        Text("Aperture", style = style, color = Color.White, modifier = Modifier.brushed(Brush.verticalGradient(listOf(Color.White, Color.White.copy(alpha = 0.78f)))))
        Text("IQ", style = style, color = Color.White, modifier = Modifier.brushed(Theme.cta))
    }
}

@Composable
fun ProBadge(modifier: Modifier = Modifier) {
    Text(
        "PRO", fontSize = 11.sp, fontWeight = FontWeight.Black, letterSpacing = 1.sp, color = Color.White, lineHeight = 13.sp,
        modifier = modifier.background(Theme.cta, CircleShape).padding(horizontal = 7.dp, vertical = 3.dp),
    )
}

/** Six shutter blades around a lens opening; `open` 0 = closed iris, 1 = wide open. */
fun irisPath(size: Size, open: Float, blades: Int = 6): androidx.compose.ui.graphics.Path {
    val cx = size.width / 2; val cy = size.height / 2
    val r = minOf(size.width, size.height) / 2
    val hole = r * (0.08f + 0.5f * open)                       // radius of the opening
    val p = androidx.compose.ui.graphics.Path()
    val step = 2 * Math.PI / blades
    val twist = (1 - open) * 0.9 + 0.35                        // blades swing in as the iris closes
    for (i in 0 until blades) {
        val a0 = i * step; val a1 = a0 + step
        // each blade: from the opening's edge out to the rim, swept sideways by `twist`
        p.moveTo(cx + hole * cos(a0 + twist).toFloat(), cy + hole * sin(a0 + twist).toFloat())
        p.lineTo(cx + r * cos(a0).toFloat(), cy + r * sin(a0).toFloat())
        p.arcTo(androidx.compose.ui.geometry.Rect(cx - r, cy - r, cx + r, cy + r), Math.toDegrees(a0).toFloat(), Math.toDegrees(step).toFloat(), false)
        p.lineTo(cx + hole * cos(a1 + twist).toFloat(), cy + hole * sin(a1 + twist).toFloat())
        p.close()
    }
    return p
}

@Composable
fun Iris(open: Float, modifier: Modifier = Modifier, seams: Boolean = false) {
    Canvas(modifier) {
        val p = irisPath(size, open)
        drawPath(p, Brush.horizontalGradient(Theme.ctaColors))
        if (seams) drawPath(p, Color.Black.copy(alpha = 0.55f), style = Stroke(2.dp.toPx()))
    }
}

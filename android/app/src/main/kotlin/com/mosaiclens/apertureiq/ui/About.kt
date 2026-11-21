package com.mosaiclens.apertureiq.ui

import android.content.Context
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.ShoppingBag
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.mosaiclens.apertureiq.BuildConfig
import com.mosaiclens.apertureiq.judge.JudgeError
import com.mosaiclens.apertureiq.judge.usf
import com.mosaiclens.apertureiq.platform.Engine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import kotlin.math.abs
import kotlin.math.max

/** The app icon's artwork: the splash's iris, half open, on the app's dark background. */
@Composable
fun AppIconArt(modifier: Modifier = Modifier) {
    Canvas(modifier.background(Theme.bg)) {
        val s = size.width / 200f
        drawCircle(Brush.radialGradient(listOf(Theme.violet.copy(alpha = 0.55f), Color.Transparent), center, 120 * s), 120 * s)
        drawCircle(Brush.verticalGradient(listOf(Color.White.copy(alpha = 0.9f), Color.White.copy(alpha = 0.25f))), 75 * s, style = Stroke(6 * s))
        rotate(-18f) {
            val d = 134 * s
            val p = irisPath(androidx.compose.ui.geometry.Size(d, d), 0.42f)
            p.translate(androidx.compose.ui.geometry.Offset(center.x - d / 2, center.y - d / 2))
            drawPath(p, Brush.horizontalGradient(Theme.ctaColors, center.x - d / 2, center.x + d / 2))
            drawPath(p, Color.Black.copy(alpha = 0.55f), style = Stroke(2 * s))
        }
        drawCircle(Brush.radialGradient(listOf(Color.White.copy(alpha = 0.85f), Color.Transparent), center, 18 * s), 18 * s)
    }
}

private class Credit(val name: String, val detail: String, val license: String?)

private fun mit(holder: String) = """MIT License

Copyright (c) $holder

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""

private val credits = listOf(
    Credit(
        "Community Forensics",
        "The AI-image detector is the Community Forensics ViT-S/16-384 model by Jeongsoo Park and Andrew Owens (CVPR 2025), converted to ONNX.", mit("2025 Jeongsoo Park"),
    ),
    Credit(
        "DETR panoptic segmentation",
        "Object detection uses the DETR-ResNet50 panoptic segmentation model (Carion et al., Facebook AI Research), licensed under the Apache License 2.0.", null,
    ),
    Credit(
        "CLIP",
        "The twin network uses OpenAI's CLIP ViT-B/32 image encoder (Radford et al., 2021) to turn each photo into features, which ApertureIQ's own trained layers then score.", mit("2021 OpenAI"),
    ),
    Credit("ONNX Runtime", "The models run on this phone with Microsoft's ONNX Runtime, licensed under the MIT License.", null),
)

@Composable
private fun SheetBar(title: String, onDismiss: () -> Unit) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Spacer(Modifier.weight(1f))
        Text(title, style = Type.headline, color = Color.White)
        Box(Modifier.weight(1f), contentAlignment = Alignment.CenterEnd) {
            Text("Done", style = Type.headline, color = Theme.a, modifier = Modifier.clickable(onClick = onDismiss).padding(4.dp))
        }
    }
}

/** Version, privacy, purchase restore and third-party credits. */
@Composable
fun AboutSheet(store: Store, onDismiss: () -> Unit) {
    val scope = rememberCoroutineScope()
    LaunchedEffect(Unit) { store.message = null }
    Sheet(onDismiss) {
        Column(Modifier.fillMaxSize().systemBarsPadding()) {
            SheetBar("About", onDismiss)
            Column(
                Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(bottom = 30.dp),
                horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(22.dp),
            ) {
                Column(Modifier.padding(top = 8.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    AppIconArt(Modifier.size(96.dp).clip(RoundedCornerShape(22.dp)))
                    Wordmark(28f)
                    Text("Version ${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})", style = Type.caption, color = Theme.dim)
                }
                AboutSection("Privacy", Icons.Filled.Lock) {
                    Text(
                        "Every photo is analysed on this phone. Nothing is uploaded, there are no accounts, and the app collects no data. Your recent comparisons are kept as small copies in the app's own storage, without location, and are deleted with the app.",
                        style = Type.subheadline, color = Color.White,
                    )
                }
                AboutSection("Purchases", Icons.Filled.ShoppingBag) {
                    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text(
                            if (store.ownsPro) "Pro is unlocked on this Google account." else "Bought Pro on another device? Restore it here.",
                            style = Type.subheadline, color = Color.White,
                        )
                        Row(
                            Modifier.fillMaxWidth().background(Color.White.copy(alpha = 0.1f), CircleShape).clip(CircleShape)
                                .clickable(enabled = !store.busy) { scope.launch { store.restore() } }.padding(vertical = 11.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally), verticalAlignment = Alignment.CenterVertically,
                        ) {
                            if (store.busy) CircularProgressIndicator(Modifier.size(16.dp), color = Color.White, strokeWidth = 2.dp)
                            Text("Restore purchases", style = Type.subheadline.semibold, color = Color.White)
                        }
                        store.message?.let { Text(it, style = Type.footnote, color = Theme.dim) }
                    }
                }
                AboutSection("Acknowledgements", Icons.Filled.Favorite) {
                    Column(verticalArrangement = Arrangement.spacedBy(18.dp)) {
                        Text(
                            "ApertureIQ's photo judge is its own twin network: both photos go through the same model and are scored side by side. Its scoring layers were trained for ApertureIQ on five photo-rating datasets. It and the other features below build on these models.",
                            style = Type.footnote, color = Theme.dim,
                        )
                        for (c in credits) {
                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                Text(c.name, style = Type.subheadline.semibold, color = Color.White)
                                Text(c.detail, style = Type.footnote, color = Theme.dim)
                                c.license?.let { Text(it, fontSize = 10.sp, lineHeight = 13.sp, fontFamily = FontFamily.Monospace, color = Theme.dim) }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun AboutSection(title: String, icon: ImageVector, content: @Composable () -> Unit) {
    Column(Modifier.glassCard(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Icon(icon, null, Modifier.size(14.dp), tint = Theme.dim)
            Text(title.uppercase(), style = Type.caption.heavy, letterSpacing = 1.2.sp, color = Theme.dim)
        }
        content()
    }
}

// ---- self-test

/**
 * Compares this device's results for two bundled photos with the reference PyTorch outputs
 * (assets/parity.json): the check that the ONNX models and the Android decoding match training.
 */
object ParityCheck {
    @Serializable
    data class Expected(
        val file: String,
        val stdSize: List<Int>,
        val globalEmbedHead: List<Double>,
        val signals: Map<String, Double>,
        val scores: Map<String, Double>,
        val percentiles: Map<String, Double>,
    )

    /** Calls `log(text, ok)` for each line; ok == null is informational. */
    fun run(ctx: Context, log: (String, Boolean?) -> Unit) {
        try {
            val text = Engine.asset(ctx, "parity.json")?.decodeToString() ?: throw JudgeError.MissingResource("parity.json")
            val expected = Json { ignoreUnknownKeys = true }.decodeFromString(ListSerializer(Expected.serializer()), text)
            val judge = Engine.judge(ctx)
            val info = judge.info
            for (e in expected) {
                val bytes = Engine.asset(ctx, e.file)
                if (bytes == null) { log("${e.file}: not bundled in this build", null); continue }
                val t0 = System.nanoTime()
                val p = judge.score(bytes, e.file)
                log("${e.file}  (${usf("%.2f", (System.nanoTime() - t0) / 1e9)} s)", null)
                log("  standard size ${p.std.width} x ${p.std.height}", listOf(p.std.width, p.std.height) == e.stdSize)
                if (BuildConfig.DEBUG) runCatching {      // lets a desktop check compare the decoded pixels: adb run-as ... cat
                    java.io.File(ctx.filesDir, "selftest_${e.file}.rgb").writeBytes(p.std.data)
                }

                val embDiff = p.globalEmbedHead.zip(e.globalEmbedHead).maxOfOrNull { abs(it.first - it.second) } ?: 0.0
                val embScale = e.globalEmbedHead.maxOfOrNull { abs(it) } ?: 1.0
                log(usf("  encoder output: max diff %.3f (values up to %.2f)", embDiff, embScale), embDiff <= 0.15 * max(embScale, 1e-6))

                fun worst(keys: List<String>): Pair<String, Double> {
                    var w = Pair("", 0.0)
                    for (k in keys) {
                        val want = e.signals[k] ?: continue
                        val got = p.signals[k] ?: continue
                        val d = abs(got - want) / max(abs(want), 0.05)
                        if (d > w.second) w = Pair(k, d)
                    }
                    return w
                }
                val used = worst(info.signalsUsed)
                log(usf("  model-input signals: worst relative diff %.0f%% (%s)", used.second * 100, used.first), used.second <= 0.15)
                val extra = worst(e.signals.keys.filter { !info.signalsUsed.contains(it) }.sorted())
                log(usf("  coaching signals: worst relative diff %.0f%% (%s)", extra.second * 100, extra.first), extra.second <= 0.25)

                var w = Pair("", 0.0)
                for ((i, h) in info.heads.withIndex()) {
                    val want = e.percentiles[h.key] ?: continue
                    val d = abs(p.percentile[i] - want)
                    if (d > w.second) w = Pair(h.label, d)
                }
                // Android's JPEG decoder rounds chroma differently from the reference decoder (pixels differ by
                // a fraction of a level), which moves the less stable attributes by several points. Feeding the
                // tablet's own pixels to the PyTorch model reproduces the app's numbers exactly, so 8 is the
                // decoder allowance here; the two headline scores below keep the tight 3-point check.
                log(usf("  attribute percentiles: worst diff %.1f points (%s)", w.second, w.first), w.second <= 8)
                for (key in listOf("aesthetic", "technical")) {
                    val i = info.index(key) ?: continue
                    val want = e.percentiles[key] ?: continue
                    log(usf("  %s: %.1f here, %.1f reference", key, p.percentile[i], want), abs(p.percentile[i] - want) <= 3)
                }
            }
        } catch (t: Throwable) {
            log("Failed: ${t.localizedMessage ?: t.toString()}", false)
        }
    }
}

@Composable
fun SelfTestSheet(onDismiss: () -> Unit) {
    val ctx = LocalContext.current.applicationContext
    val lines = remember { mutableStateListOf<Pair<String, Boolean?>>() }
    var running by remember { mutableStateOf(true) }
    LaunchedEffect(Unit) {
        withContext(Dispatchers.Default) {
            ParityCheck.run(ctx) { text, ok ->
                android.util.Log.i("ApertureIQ", "SELFTEST ${if (ok == null) "    " else if (ok) "ok  " else "FAIL"} $text")
                launch(Dispatchers.Main) { lines.add(Pair(text, ok)) }
            }
        }
        running = false
    }
    Sheet(onDismiss) {
        Column(Modifier.fillMaxSize().systemBarsPadding()) {
            SheetBar("Self-test", onDismiss)
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(bottom = 30.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    "Compares this phone's results for two bundled photos with the reference PyTorch outputs. Overall scores within about 3 points mean the port is faithful; small differences come from image decoding and 16-bit model weights.",
                    style = Type.footnote, color = Theme.dim,
                )
                for ((text, ok) in lines) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        if (ok != null) Icon(if (ok) Icons.Filled.CheckCircle else Icons.Filled.WarningAmber, null, Modifier.size(16.dp), tint = if (ok) Theme.good else Theme.b)
                        Text(text, fontSize = 12.sp, lineHeight = 16.sp, fontFamily = FontFamily.Monospace, color = Color.White)
                    }
                }
                if (running) CircularProgressIndicator(Modifier.padding(top = 8.dp).size(22.dp), color = Color.White, strokeWidth = 2.dp)
            }
        }
    }
}

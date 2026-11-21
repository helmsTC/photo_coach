package com.mosaiclens.apertureiq.ui

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Balance
import androidx.compose.material.icons.filled.BarChart
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Compare
import androidx.compose.material.icons.filled.EmojiEvents
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.FlipToFront
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.NorthEast
import androidx.compose.material.icons.filled.School
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material.icons.filled.ThumbUp
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.ViewAgenda
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInParent
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.mosaiclens.apertureiq.judge.AttributeRow
import com.mosaiclens.apertureiq.judge.CoachItem
import com.mosaiclens.apertureiq.judge.Comparison
import com.mosaiclens.apertureiq.judge.Histogram
import com.mosaiclens.apertureiq.judge.Lens
import com.mosaiclens.apertureiq.judge.LookRecipe
import com.mosaiclens.apertureiq.judge.Measurement
import com.mosaiclens.apertureiq.judge.Preview
import com.mosaiclens.apertureiq.judge.Reason
import com.mosaiclens.apertureiq.judge.Authenticity
import com.mosaiclens.apertureiq.judge.Swatch
import com.mosaiclens.apertureiq.judge.Stage
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.runtime.DisposableEffect
import androidx.compose.material.icons.filled.CheckCircle
import com.mosaiclens.apertureiq.judge.usf
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** The Pro result, split into tabs so no single screen is a wall of text. */
enum class ResultTab(val title: String, val icon: ImageVector) {
    OVERVIEW("Overview", Icons.Filled.EmojiEvents),
    INSPECT("Inspect", Icons.Filled.Visibility),
    COACH("Coach", Icons.Filled.School),
    SCORES("Scores", Icons.Filled.BarChart),
    AI("AI check", Icons.Filled.ImageSearch);

    /** Needs the slow part of the analysis; shows a progress list until it arrives. */
    val late: Boolean get() = this == INSPECT || this == COACH
}

@Composable
fun ResultScreen(r: Comparison, debug: DebugOptions, working: Stage? = null, onNew: () -> Unit) {
    // `working` is the stage still running when the verdict is shown early; null once everything is in.
    val pending = working != null
    BackHandler(onBack = onNew)
    val view = LocalView.current
    DisposableEffect(view, pending) {
        view.keepScreenOn = pending
        onDispose { view.keepScreenOn = false }
    }
    val scope = rememberCoroutineScope()
    var tab by remember { mutableStateOf(ResultTab.entries.firstOrNull { it.name.equals(debug.tab, true) } ?: if (debug.lens != null) ResultTab.INSPECT else ResultTab.OVERVIEW) }
    var inspectPhoto by remember { mutableIntStateOf(debug.photo ?: r.loserIndex) }
    var lens by remember { mutableStateOf(Lens.entries.firstOrNull { it.name.equals(debug.lens, true) } ?: Lens.PHOTO) }
    var detailsPhoto by remember { mutableIntStateOf(debug.photo ?: r.loserIndex) }
    var expanded by remember(pending) { mutableStateOf(if (debug.expandAll) r.coaching.map { it.id }.toSet() else emptySet()) }
    var detailsY by remember { mutableFloatStateOf(0f) }
    var jumpToDetails by remember { mutableStateOf(false) }
    val go: (ResultTab) -> Unit = { t -> Theme.tick(view); tab = t }

    Box(Modifier.fillMaxSize()) {
        AmbientBackground(tintA = Theme.tint(r.winnerIndex), tintB = Theme.violet)
        Column(Modifier.fillMaxSize().statusBarsPadding()) {
            // top bar + tab bar
            Row(Modifier.padding(horizontal = 18.dp).padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                BackPill(onNew)
                Spacer(Modifier.weight(1f))
                Row(
                    Modifier.background(Color.White.copy(alpha = 0.1f), CircleShape).padding(horizontal = 10.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    if (pending) CircularProgressIndicator(Modifier.size(11.dp), color = Theme.dim, strokeWidth = 1.5.dp)
                    Text(if (pending) "Still analysing" else usf("%.1fs on device", r.seconds), style = Type.caption.w(FontWeight.Medium), color = Theme.dim)
                }
            }
            Row(
                Modifier.padding(horizontal = 18.dp, vertical = 8.dp).fillMaxWidth().background(Color.White.copy(alpha = 0.08f), CircleShape)
                    .border(1.dp, Theme.stroke, CircleShape).padding(4.dp),
                horizontalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                for (t in ResultTab.entries) {
                    val sel = tab == t
                    Box(Modifier.weight(1f)) {
                        Column(
                            Modifier.fillMaxWidth().clip(CircleShape).then(if (sel) Modifier.background(Theme.cta) else Modifier).clickable { go(t) }.padding(vertical = 7.dp),
                            horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(3.dp),
                        ) {
                            if (pending && t.late) CircularProgressIndicator(Modifier.size(16.dp).padding(1.dp), color = if (sel) Color.White else Theme.dim, strokeWidth = 2.dp)
                            else Icon(t.icon, null, Modifier.size(16.dp), tint = if (sel) Color.White else Theme.dim)
                            Text(t.title, fontSize = 10.sp, fontWeight = FontWeight.Bold, color = if (sel) Color.White else Theme.dim, maxLines = 1, lineHeight = 12.sp)
                        }
                        if (t == ResultTab.AI && r.extras.any { it.authenticity?.isAI == true }) {
                            Box(Modifier.align(Alignment.TopEnd).padding(top = 4.dp, end = 10.dp).size(7.dp).background(Theme.bad, CircleShape))
                        }
                    }
                }
            }

            Crossfade(tab, label = "tab", modifier = Modifier.weight(1f)) { t ->
                val scroll = rememberScrollState()
                if (t == ResultTab.SCORES) {
                    LaunchedEffect(jumpToDetails) {
                        if (jumpToDetails) { delay(350); scroll.animateScrollTo(detailsY.toInt(), tween(500)); jumpToDetails = false }
                    }
                }
                Column(
                    Modifier.fillMaxSize().verticalScroll(scroll).padding(horizontal = 18.dp).padding(top = 8.dp, bottom = 40.dp).navigationBarsPadding(),
                    verticalArrangement = Arrangement.spacedBy(if (t == ResultTab.OVERVIEW) 22.dp else 30.dp),
                ) {
                    if (working != null && t.late) StillWorking(t, working) else when (t) {
                        ResultTab.OVERVIEW -> OverviewTab(r, go, pending)
                        ResultTab.INSPECT -> Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                            SectionHeader("Inspect", "Switch lenses to see what the analysis sees.", Icons.Filled.Visibility)
                            Inspector(r, inspectPhoto, lens, { inspectPhoto = it }, { lens = it })
                        }
                        ResultTab.COACH -> {
                            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                                SectionHeader(
                                    "Your coaching plan", "How to bring ${r.loser.name} up to ${r.winner.name}, most important first.", Icons.Filled.School,
                                )
                                if (r.coaching.isEmpty()) {
                                    Text(
                                        "No single weakness stands out - the photos differ by small margins across the board. The Scores tab shows where.",
                                        style = Type.subheadline, color = Color.White, modifier = Modifier.glassCard(),
                                    )
                                }
                                for ((n, item) in r.coaching.withIndex()) {
                                    // "Show me" jumps to the tab and lens that illustrate the point.
                                    val l = Lens.fromOverlay(item.overlay)
                                    val showMe: (() -> Unit)? = when {
                                        l == Lens.OBJECTS && r.extras[r.loserIndex].objects.isEmpty() -> null
                                        l == Lens.FRAMING && r.extras[r.loserIndex].otherFrame == null -> null
                                        l != null -> ({ inspectPhoto = r.loserIndex; lens = l; go(ResultTab.INSPECT) })
                                        item.overlay == "hist" || item.overlay == "palette" -> ({ detailsPhoto = r.loserIndex; jumpToDetails = true; go(ResultTab.SCORES) })
                                        else -> null
                                    }
                                    CoachCard(n + 1, item, expanded.contains(item.id), { expanded = if (expanded.contains(item.id)) expanded - item.id else expanded + item.id }, showMe)
                                }
                            }
                            r.bestPreview?.let { EditTowardWinner(r, it) }
                        }
                        ResultTab.SCORES -> {
                            WhyItWon(r)
                            AttributeBreakdown(r)
                            Box(Modifier.onGloballyPositioned { detailsY = it.positionInParent().y }) { PhotoDetails(r, detailsPhoto) { detailsPhoto = it } }
                            Text(
                                "Scores are percentiles: 50 is a typical photo, 90 beats nine in ten human-rated reference photos. Gaps under about 10 points aren't meaningful.",
                                style = Type.caption, color = Theme.dim,
                            )
                        }
                        ResultTab.AI -> AuthenticityCard(r)
                    }
                }
            }
        }
    }
}

// ---- overview

/** The headline result at a glance, with tiles that lead into the other tabs. */
/** Shown in place of a tab whose content depends on the slow part of the analysis. */
@Composable
fun StillWorking(tab: ResultTab, stage: Stage) {
    val late = Stage.entries.filter { it.ordinal >= Stage.MATCHING.ordinal }
    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        SectionHeader(
            if (tab == ResultTab.COACH) "Your coaching plan" else "Inspect",
            "The verdict is in. This part takes longer: it tests every object in both photos.", if (tab == ResultTab.COACH) Icons.Filled.School else Icons.Filled.Visibility,
        )
        Column(Modifier.fillMaxWidth().glassCard(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            for (s in late) {
                val done = s.ordinal < stage.ordinal
                val now = s == stage || (s == Stage.OBJECTS && stage == Stage.TESTING)
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Box(Modifier.size(18.dp), contentAlignment = Alignment.Center) {
                        if (done && !now) Icon(Icons.Filled.CheckCircle, null, Modifier.size(18.dp), tint = Theme.good)
                        else if (now) CircularProgressIndicator(Modifier.size(16.dp), color = Color.White, strokeWidth = 2.dp)
                        else Box(Modifier.size(8.dp).background(Color.White.copy(alpha = 0.25f), CircleShape))
                    }
                    Text(s.title, style = Type.subheadline, color = if (done || now) Color.White else Theme.dim)
                }
            }
        }
    }
}

@Composable
fun OverviewTab(r: Comparison, go: (ResultTab) -> Unit, pending: Boolean = false) {
    val wi = r.winnerIndex
    // hero
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        val shape = RoundedCornerShape(26.dp)
        Box(Modifier.fillMaxWidth().aspectRatio(r.winner.aspect.toFloat()).clip(shape).border(2.dp, Theme.tint(wi).copy(alpha = 0.8f), shape)) {
            Image(rememberImage(r.winner.std), null, Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
            Box(Modifier.fillMaxSize().background(Brush.verticalGradient(0.5f to Color.Transparent, 1f to Color.Black.copy(alpha = 0.75f))))
            Row(
                Modifier.align(Alignment.BottomStart).padding(14.dp).background(Color(0xB3101018), CircleShape).padding(horizontal = 12.dp, vertical = 7.dp),
                verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Icon(Icons.Filled.EmojiEvents, null, Modifier.size(16.dp), tint = Theme.gold)
                Text("WINNER", style = Type.caption.heavy, letterSpacing = 1.5.sp, color = Color.White)
                ShotTag(wi, small = true)
            }
            r.extras[wi].authenticity?.let { a ->
                if (a.isAI || a.verdict == Authenticity.Verdict.aiEdited) AuthBadge(a, Modifier.align(Alignment.BottomEnd).padding(14.dp))
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(r.headline, style = Type.title, color = Color.White, modifier = Modifier.weight(1f))
            if (r.closeCall) Pill("Close call", Theme.gold, Icons.Filled.Balance)
        }
        Text(r.summary, style = Type.subheadline, color = Theme.dim, maxLines = 3, overflow = TextOverflow.Ellipsis)
    }

    Scoreboard(r)

    // The top reasons as compact chips: the full list lives in the Scores tab.
    if (r.reasons.isNotEmpty()) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Where ${r.winner.name} pulls ahead", style = Type.subheadline.bold, color = Color.White, modifier = Modifier.weight(1f))
                Row(Modifier.clickable { go(ResultTab.SCORES) }, verticalAlignment = Alignment.CenterVertically) {
                    Text("All scores", style = Type.caption.semibold, color = Theme.dim)
                    Icon(Icons.Filled.ChevronRight, null, Modifier.size(14.dp), tint = Theme.dim)
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (reason in r.reasons.take(3)) {
                    Column(
                        Modifier.weight(1f).heightIn(min = 74.dp).background(Theme.card, RoundedCornerShape(16.dp)).border(1.dp, Theme.stroke, RoundedCornerShape(16.dp)).padding(6.dp),
                        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(4.dp, Alignment.CenterVertically),
                    ) {
                        Text(usf("+%.0f", reason.gap), fontSize = 22.sp, fontWeight = FontWeight.ExtraBold, color = Theme.tint(wi), lineHeight = 24.sp)
                        Text(reason.label, style = Type.caption2.semibold, color = Theme.dim, maxLines = 2, textAlign = TextAlign.Center, overflow = TextOverflow.Ellipsis)
                    }
                }
            }
        }
    }

    // tiles
    val tiles = ArrayList<@Composable (Modifier) -> Unit>()
    if (!pending) r.coaching.firstOrNull()?.let { first ->
        val n = r.coaching.size
        tiles.add { m -> Tile(Theme.itemIcon(first.id), Theme.fixStyle(first.fix).color, "TOP FIX", first.title, "$n step${if (n == 1) "" else "s"} in your plan", m) { go(ResultTab.COACH) } }
    }
    r.bestPreview?.let { pv ->
        val d = (pv.overall - r.loser.overall).roundToInt()
        tiles.add { m ->
            Tile(
                Icons.Filled.AutoFixHigh, if (d > 0) Theme.good else if (d < 0) Theme.bad else Theme.dim, "ONE-TAP EDIT",
                if (d == 0) "No change" else "${if (d > 0) "+" else ""}$d point${if (abs(d) == 1) "" else "s"}", "${r.loser.name} with the winner's look", m,
            ) { go(ResultTab.COACH) }
        }
    }
    val au = r.extras.mapNotNull { it.authenticity }
    val flagged = au.indices.filter { au[it].isAI }
    val aiTitle = if (flagged.isEmpty()) (if (au.all { it.verdict == Authenticity.Verdict.noSigns }) "No AI signs" else "Mostly clear")
    else if (flagged.size == 2) "Both look AI" else "Shot ${if (flagged[0] == 0) "A" else "B"} looks AI"
    val aiFooter = if (au.map { it.title }.toSet().size <= 1) "Both shots checked" else au.indices.joinToString(" · ") { "${if (it == 0) "A" else "B"}: ${au[it].title}" }
    tiles.add { m -> Tile(Icons.Filled.ImageSearch, if (flagged.isEmpty()) Theme.good else Theme.bad, "AI CHECK", aiTitle, aiFooter, m) { go(ResultTab.AI) } }
    val relIcon = if (r.relation == "aligned") Icons.Filled.FlipToFront else if (r.relation == "similar") Icons.Filled.Compare else Icons.Filled.ViewAgenda
    val relTitle = if (r.relation == "aligned") "Same scene" else if (r.relation == "similar") "Similar subject" else "Different scenes"
    if (pending) tiles.add(0) { m ->
        Tile(Icons.Filled.School, Theme.violet, "COACHING", "Being written", "Objects, attention and edits are still running", m) { go(ResultTab.COACH) }
    }
    if (!pending) tiles.add { m ->
        Tile(relIcon, Theme.violet, "THE SCENES", relTitle, if (r.relation == "aligned") "Framing compared directly" else "Judged on their own merits", m) { go(ResultTab.INSPECT) }
    }
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        for (row in tiles.chunked(2)) {
            Row(Modifier.height(androidx.compose.foundation.layout.IntrinsicSize.Max), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                for (tile in row) tile(Modifier.weight(1f).fillMaxSize())
                if (row.size == 1) Spacer(Modifier.weight(1f))
            }
        }
    }

    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
        Icon(Icons.Filled.Lock, null, Modifier.size(12.dp), tint = Theme.dim)
        Spacer(Modifier.width(6.dp))
        Text("Analysed entirely on this phone.", style = Type.caption, color = Theme.dim)
    }
}

/** A tappable summary tile that leads into another tab. */
@Composable
fun Tile(icon: ImageVector, color: Color, kicker: String, title: String, footer: String, modifier: Modifier = Modifier, action: () -> Unit) {
    val shape = RoundedCornerShape(20.dp)
    Column(
        modifier.defaultMinSize(minHeight = 138.dp).clip(shape).background(Theme.card).border(1.dp, Theme.stroke, shape).clickable(onClick = action).padding(13.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(30.dp).background(color.copy(alpha = 0.16f), RoundedCornerShape(9.dp)), contentAlignment = Alignment.Center) {
                Icon(icon, null, Modifier.size(16.dp), tint = color)
            }
            Spacer(Modifier.weight(1f))
            Icon(Icons.Filled.NorthEast, null, Modifier.size(12.dp), tint = Theme.dim)
        }
        Text(kicker, fontSize = 10.sp, fontWeight = FontWeight.ExtraBold, letterSpacing = 0.8.sp, color = Theme.dim, lineHeight = 12.sp)
        Text(title, style = Type.subheadline.bold, color = Color.White, maxLines = 2, overflow = TextOverflow.Ellipsis)
        Spacer(Modifier.weight(1f))
        Text(footer, style = Type.caption2, color = Theme.dim, maxLines = 2, overflow = TextOverflow.Ellipsis)
    }
}

// ---- scoreboard

@Composable
fun Scoreboard(r: Comparison) {
    val bar = remember { Animatable(0.5f) }
    LaunchedEffect(r) { delay(400); bar.animateTo(r.pA.toFloat(), spring(dampingRatio = 0.8f, stiffness = 40f)) }
    val pa = (r.pA * 100).roundToInt()
    Column(Modifier.glassCard(18.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
        Row(verticalAlignment = Alignment.Top) {
            ScoreColumn(r, 0)
            Spacer(Modifier.weight(1f))
            Text("VS", fontSize = 15.sp, fontWeight = FontWeight.Black, color = Theme.dim, modifier = Modifier.padding(top = 44.dp))
            Spacer(Modifier.weight(1f))
            ScoreColumn(r, 1)
        }
        Column(Modifier.clearAndSetSemantics { contentDescription = "Viewers would prefer shot A $pa percent of the time" }, verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Canvas(Modifier.fillMaxWidth().height(10.dp)) {
                val rr = CornerRadius(size.height / 2)
                val wa = max(6.dp.toPx(), size.width * bar.value - 1.5.dp.toPx())
                drawRoundRect(Theme.a, size = Size(wa, size.height), cornerRadius = rr)
                drawRoundRect(Theme.b, Offset(wa + 3.dp.toPx(), 0f), Size(max(0f, size.width - wa - 3.dp.toPx()), size.height), rr)
            }
            Row {
                Text("$pa%", style = Type.caption.bold, color = Theme.a)
                Spacer(Modifier.weight(1f))
                Text("would pick each", style = Type.caption.bold, color = Theme.dim)
                Spacer(Modifier.weight(1f))
                Text("${100 - pa}%", style = Type.caption.bold, color = Theme.b)
            }
        }
    }
}

@Composable
private fun ScoreColumn(r: Comparison, i: Int) {
    val p = r.photo(i)
    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Box(contentAlignment = Alignment.TopEnd) {
            ScoreRing(p.overall, Theme.tint(i), label = "OVERALL")
            if (i == r.winnerIndex) Icon(Icons.Filled.EmojiEvents, "Winner", Modifier.offset(x = 4.dp, y = (-6).dp).size(22.dp), tint = Theme.gold)
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
            ShotTag(i, small = true)
            Text(p.name, style = Type.footnote.semibold, color = Color.White)
        }
        Column(Modifier.width(118.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
            for ((label, v) in listOf("Aesthetic" to p.percentile[0], "Technical" to p.percentile[1])) {
                Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    Row {
                        Text(label, style = Type.caption2, color = Theme.dim)
                        Spacer(Modifier.weight(1f))
                        Text("${v.roundToInt()}", style = Type.caption2.bold, color = Color.White)
                    }
                    Meter(v, Theme.tint(i), 4.dp)
                }
            }
        }
    }
}

// ---- why it won

@Composable
fun WhyItWon(r: Comparison) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        SectionHeader("Why ${r.winner.name} won", icon = Icons.Filled.EmojiEvents)
        Column(Modifier.glassCard(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            if (r.reasons.isEmpty()) Text("No single attribute separates them by a meaningful margin.", style = Type.subheadline, color = Theme.dim)
            for (reason in r.reasons) ReasonRow(reason, Theme.tint(r.winnerIndex), Theme.tint(r.loserIndex))
        }
        if (r.loserLeads.isNotEmpty()) {
            Column(Modifier.glassCard(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Icon(Icons.Filled.ThumbUp, null, Modifier.size(16.dp), tint = Theme.tint(r.loserIndex))
                    Text("Where ${r.loser.name} is ahead", style = Type.subheadline.bold, color = Color.White)
                }
                for (reason in r.loserLeads) ReasonRow(reason, Theme.tint(r.loserIndex), Theme.tint(r.winnerIndex), flipped = true)
            }
        }
    }
}

@Composable
fun ReasonRow(reason: Reason, lead: Color, trail: Color, flipped: Boolean = false) {
    val hi = if (flipped) reason.loserPct else reason.winnerPct
    val lo = if (flipped) reason.winnerPct else reason.loserPct
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(reason.label, style = Type.subheadline.semibold, color = Color.White, modifier = Modifier.weight(1f))
            Text(
                usf("+%.0f", reason.gap), style = Type.caption.heavy, color = lead,
                modifier = Modifier.background(lead.copy(alpha = 0.2f), CircleShape).padding(horizontal = 8.dp, vertical = 3.dp),
            )
        }
        Dumbbell(hi, lo, lead, trail)
    }
}

// ---- coaching card

@Composable
fun CoachCard(n: Int, item: CoachItem, expanded: Boolean, toggle: () -> Unit, showMe: (() -> Unit)?) {
    val fix = Theme.fixStyle(item.fix)
    val turn by animateFloatAsState(if (expanded) 180f else 0f, label = "chevron")
    Column(Modifier.glassCard(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(Modifier.clickable(onClickLabel = if (expanded) "Collapse" else "Expand", onClick = toggle), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Box(Modifier.size(42.dp).background(fix.color.copy(alpha = 0.18f), RoundedCornerShape(12.dp)), contentAlignment = Alignment.Center) {
                Icon(Theme.itemIcon(item.id), null, tint = fix.color)
            }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Text("$n. ${item.title}", style = Type.headline, color = Color.White)
                Pill(fix.text, fix.color, fix.icon)
            }
            Icon(Icons.Filled.ExpandMore, null, Modifier.padding(top = 4.dp).size(20.dp).rotate(turn), tint = Theme.dim)
        }
        AnimatedVisibility(expanded) {
            Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                @Composable
                fun block(title: String, icon: ImageVector, lines: List<String>, color: Color) {
                    if (lines.isEmpty()) return
                    Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            Icon(icon, null, Modifier.size(13.dp), tint = Theme.dim)
                            Text(title, style = Type.caption2.heavy, letterSpacing = 0.8.sp, color = Theme.dim)
                        }
                        for (l in lines) Bullet(l, color)
                    }
                }
                block("THE EVIDENCE", Icons.Filled.BarChart, item.evidence, Theme.dim)
                block("AT THE SHOOT", Icons.Filled.CameraAlt, item.shoot, Theme.b)
                block("IN THE EDIT", Icons.Filled.Tune, item.edit, Theme.good)
                if (showMe != null) {
                    Row(
                        Modifier.background(Color.White.copy(alpha = 0.1f), CircleShape).clip(CircleShape).clickable(onClick = showMe).padding(horizontal = 14.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Icon(Icons.Filled.Visibility, null, Modifier.size(16.dp), tint = Color.White)
                        Text("Show me", style = Type.subheadline.semibold, color = Color.White)
                    }
                }
            }
        }
    }
}

// ---- edit toward the winner

@Composable
fun EditTowardWinner(r: Comparison, initial: Preview) {
    var kind by remember(r) { mutableStateOf(initial.kind) }
    val pv = r.previews.firstOrNull { it.kind == kind } ?: initial
    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        SectionHeader(
            "Edit toward the winner",
            "${r.loser.name} with the winner's tone${if (r.recipe?.colorOK == true) " and colour" else ""} applied, re-scored by the same network.",
            Icons.Filled.AutoFixHigh,
        )
        if (r.previews.size > 1) Segmented(r.previews.map { it.title }, r.previews.indexOfFirst { it.kind == pv.kind }) { kind = r.previews[it].kind }
        BeforeAfter(rememberImage(r.loser.std), rememberImage(pv.image), pv.image.width.toFloat() / max(pv.image.height, 1), true)
        // score delta
        val before = r.loser.overall; val after = pv.overall
        val di = (after - before).roundToInt()
        Row(Modifier.glassCard(), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text("SCORE", style = Type.caption2.heavy, color = Theme.dim)
                Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("${before.roundToInt()}", style = Type.title2, color = Theme.dim)
                    Icon(Icons.AutoMirrored.Filled.ArrowForward, null, Modifier.padding(bottom = 6.dp).size(14.dp), tint = Theme.dim)
                    Text("${after.roundToInt()}", style = Type.title, color = Color.White)
                    Text(
                        if (di == 0) "±0" else usf("%+d", di), style = Type.subheadline.bold, modifier = Modifier.padding(bottom = 4.dp),
                        color = if (di > 0) Theme.good else if (di < 0) Theme.bad else Theme.dim,
                    )
                }
            }
            Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(if (pv.closure >= 0) "Closes ${(min(pv.closure, 1.0) * 100).roundToInt()}% of the gap" else "Doesn't help", style = Type.caption.semibold, color = Color.White)
                Meter(max(0.0, min(1.0, pv.closure)) * 100, Theme.good, 6.dp, Modifier.width(130.dp))
            }
        }
        val rec = r.recipe
        if (rec != null && rec.lines.isNotEmpty() && pv.kind == "edit") RecipeList(rec.lines)
    }
}

@Composable
fun Segmented(options: List<String>, selected: Int, onSelect: (Int) -> Unit) {
    Row(Modifier.fillMaxWidth().background(Color.White.copy(alpha = 0.08f), RoundedCornerShape(10.dp)).padding(3.dp)) {
        for ((i, o) in options.withIndex()) {
            val sel = i == selected
            Text(
                o, style = Type.footnote.semibold, color = if (sel) Color.Black else Color.White, textAlign = TextAlign.Center,
                modifier = Modifier.weight(1f).clip(RoundedCornerShape(8.dp)).background(if (sel) Color.White else Color.Transparent).clickable { onSelect(i) }.padding(vertical = 7.dp),
            )
        }
    }
}

@Composable
fun BeforeAfter(before: ImageBitmap, after: ImageBitmap, aspect: Float, comparable: Boolean) {
    val split = remember(after) { Animatable(if (comparable) 0.5f else 0f) }
    val scope = rememberCoroutineScope()
    LaunchedEffect(after) {
        if (!comparable) return@LaunchedEffect
        delay(400)
        split.animateTo(0.3f, tween(700))
        split.animateTo(0.5f, tween(800))
    }
    val shape = RoundedCornerShape(18.dp)
    BoxWithConstraints(
        Modifier.fillMaxWidth().aspectRatio(aspect).clip(shape).border(1.dp, Theme.stroke, shape)
            .then(
                if (comparable) Modifier
                    .pointerInput(after) { detectDragGestures { change, _ -> scope.launch { split.snapTo((change.position.x / size.width).coerceIn(0f, 1f)) } } }
                    .pointerInput(after) { detectTapGestures { p -> scope.launch { split.animateTo((p.x / size.width).coerceIn(0f, 1f), tween(200)) } } }
                else Modifier,
            ),
    ) {
        Image(after, null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
        if (comparable) {
            Image(before, null, Modifier.fillMaxSize().drawWithContent { clipRect(right = size.width * split.value) { this@drawWithContent.drawContent() } }, contentScale = ContentScale.Crop)
            Canvas(Modifier.fillMaxSize()) {
                val x = size.width * split.value
                drawLine(Color.Black.copy(alpha = 0.4f), Offset(x, 0f), Offset(x, size.height), 4.dp.toPx())
                drawLine(Color.White, Offset(x, 0f), Offset(x, size.height), 2.dp.toPx())
                drawCircle(Color.Black.copy(alpha = 0.3f), 19.dp.toPx(), Offset(x, size.height / 2))
                drawCircle(Color.White, 17.dp.toPx(), Offset(x, size.height / 2))
            }
            Icon(Icons.Filled.SwapHoriz, null, Modifier.offset(x = maxWidth * split.value - 10.dp, y = maxHeight / 2 - 10.dp).size(20.dp), tint = Color.Black)
        }
        Row(Modifier.fillMaxWidth().padding(10.dp)) {
            if (comparable) Tag("BEFORE")
            Spacer(Modifier.weight(1f))
            Tag("AFTER")
        }
    }
}

@Composable
private fun Tag(t: String) {
    Text(t, style = Type.caption2.heavy, letterSpacing = 1.sp, color = Color.White, modifier = Modifier.background(Color.Black.copy(alpha = 0.55f), CircleShape).padding(horizontal = 8.dp, vertical = 4.dp))
}

@Composable
fun RecipeList(lines: List<LookRecipe.Line>) {
    Column(Modifier.glassCard(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Icon(Icons.Filled.Tune, null, Modifier.size(13.dp), tint = Theme.dim)
            Text("THE RECIPE", style = Type.caption2.heavy, letterSpacing = 0.8.sp, color = Theme.dim)
        }
        for (l in lines) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(l.name, style = Type.subheadline, color = Color.White, modifier = Modifier.width(104.dp))
                Canvas(Modifier.weight(1f).height(14.dp)) {
                    val mid = size.width / 2; val cy = size.height / 2
                    val len = max(4.dp.toPx(), mid * min(1.0, abs(l.amount) / 100).toFloat())
                    val th = 4.dp.toPx(); val rr = CornerRadius(th / 2)
                    drawRoundRect(Color.White.copy(alpha = 0.08f), Offset(0f, cy - th / 2), Size(size.width, th), rr)
                    drawLine(Color.White.copy(alpha = 0.3f), Offset(mid, cy - 5.dp.toPx()), Offset(mid, cy + 5.dp.toPx()), 1.dp.toPx())
                    drawRoundRect(if (l.amount >= 0) Theme.good else Theme.b, Offset(if (l.amount >= 0) mid else mid - len, cy - th / 2), Size(len, th), rr)
                }
                Text(l.value, style = Type.caption.bold, color = Color.White, textAlign = TextAlign.End, modifier = Modifier.width(84.dp))
            }
        }
    }
}

// ---- attributes

@Composable
fun Dumbbell(a: Double, b: Double, colorA: Color, colorB: Color) {
    val t = remember { Animatable(0f) }
    LaunchedEffect(a, b) { delay(200); t.animateTo(1f, spring(dampingRatio = 0.8f, stiffness = 60f)) }
    Canvas(Modifier.fillMaxWidth().height(14.dp)) {
        val w = size.width; val cy = size.height / 2
        val xa = (w * (50 + (a - 50) * t.value) / 100).toFloat(); val xb = (w * (50 + (b - 50) * t.value) / 100).toFloat()
        val th = 3.dp.toPx()
        drawRoundRect(Color.White.copy(alpha = 0.07f), Offset(0f, cy - th / 2), Size(w, th), CornerRadius(th / 2))
        drawLine(Color.White.copy(alpha = 0.25f), Offset(w / 2, cy - 4.5.dp.toPx()), Offset(w / 2, cy + 4.5.dp.toPx()), 1.dp.toPx())
        if (abs(xa - xb) > 0.5f) {
            drawRect(Brush.horizontalGradient(listOf(colorB.copy(alpha = 0.6f), colorA.copy(alpha = 0.6f)), xb, xa), Offset(min(xa, xb), cy - th / 2), Size(abs(xa - xb), th))
        }
        drawCircle(colorB, 5.5.dp.toPx(), Offset(xb, cy))
        drawCircle(colorA, 6.5.dp.toPx(), Offset(xa, cy))
        drawCircle(Color.White.copy(alpha = 0.8f), 6.5.dp.toPx(), Offset(xa, cy), style = Stroke(1.5.dp.toPx()))
    }
}

@Composable
fun AttributeBreakdown(r: Comparison) {
    val groups = LinkedHashMap<String, ArrayList<AttributeRow>>()
    for (row in r.attributes) groups.getOrPut(row.group) { ArrayList() }.add(row)
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        SectionHeader("Attribute by attribute", "Percentile among human-rated photos; 50 is typical.", Icons.Filled.BarChart)
        Column(Modifier.glassCard(), verticalArrangement = Arrangement.spacedBy(18.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                for (i in 0 until 2) Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                    ShotTag(i, small = true)
                    Text(r.photo(i).name, style = Type.caption.semibold, color = Color.White)
                }
            }
            for ((g, rows) in groups) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(g.uppercase(), style = Type.caption2.heavy, letterSpacing = 0.8.sp, color = Theme.dim)
                    for (row in rows) {
                        Column(
                            Modifier.alpha(if (row.reliable) 1f else 0.55f)
                                .clearAndSetSemantics { contentDescription = "${row.label}: A ${row.a.roundToInt()}, B ${row.b.roundToInt()}" },
                            verticalArrangement = Arrangement.spacedBy(5.dp),
                        ) {
                            Row {
                                Text(row.label + (if (row.reliable) "" else " *"), style = Type.footnote, color = if (row.reliable) Color.White else Theme.dim, modifier = Modifier.weight(1f))
                                Text("${row.a.roundToInt()}", style = Type.footnote.bold, color = Theme.a, textAlign = TextAlign.End, modifier = Modifier.width(26.dp))
                                Text("${row.b.roundToInt()}", style = Type.footnote.bold, color = Theme.b, textAlign = TextAlign.End, modifier = Modifier.width(26.dp))
                            }
                            Dumbbell(row.a, row.b, Theme.a, Theme.b)
                        }
                    }
                }
            }
            if (r.attributes.any { !it.reliable }) {
                Text("* The network predicts this attribute only weakly on held-out photos; it is shown but never used for coaching.", style = Type.caption2, color = Theme.dim)
            }
        }
    }
}

// ---- details

@Composable
fun PhotoDetails(r: Comparison, photo: Int, setPhoto: (Int) -> Unit) {
    val p = r.photo(photo)
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        SectionHeader("Under the hood", "Measured straight off the pixels and the file.", Icons.Filled.Memory)
        Segmented(listOf(r.a.name, r.b.name), photo, setPhoto)
        Column(Modifier.glassCard(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            if (p.exif.chips.isNotEmpty() || p.exif.camera != null) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    p.exif.camera?.let { cam ->
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            Icon(Icons.Filled.CameraAlt, null, Modifier.size(14.dp), tint = Theme.dim)
                            Text(cam + (p.exif.lens?.let { " · $it" } ?: ""), style = Type.footnote, color = Theme.dim)
                        }
                    }
                    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        for (c in p.exif.chips) {
                            Text(c, style = Type.caption.semibold, color = Color.White, modifier = Modifier.background(Color.White.copy(alpha = 0.08f), CircleShape).padding(horizontal = 9.dp, vertical = 5.dp))
                        }
                    }
                }
            }
            Captioned("HISTOGRAM") { HistogramView(p.m.histogram) }
            Captioned("PALETTE") { PaletteStrip(p.m.palette) }
            Captioned("MEASUREMENTS") { MeasureGrid(p.m) }
        }
    }
}

@Composable
private fun Captioned(title: String, content: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(title, style = Type.caption2.heavy, letterSpacing = 0.8.sp, color = Theme.dim)
        content()
    }
}

@Composable
fun HistogramView(h: Histogram) {
    Canvas(Modifier.fillMaxWidth().height(90.dp).clip(RoundedCornerShape(10.dp)).background(Color.Black.copy(alpha = 0.3f))) {
        val peak = max(max(h.r.max(), h.g.max()), max(max(h.b.max(), h.l.max()), 1e-9))
        fun path(v: DoubleArray, closed: Boolean): Path {
            val p = Path()
            if (closed) p.moveTo(0f, size.height)
            for ((i, x) in v.withIndex()) {
                val px = size.width * i / (v.size - 1); val py = size.height * (1 - min(1.0, x / peak * 1.1)).toFloat()
                if (i == 0 && !closed) p.moveTo(px, py) else p.lineTo(px, py)
            }
            if (closed) { p.lineTo(size.width, size.height); p.close() }
            return p
        }
        drawPath(path(h.r, true), Color.Red.copy(alpha = 0.45f), blendMode = BlendMode.Screen)
        drawPath(path(h.g, true), Color.Green.copy(alpha = 0.45f), blendMode = BlendMode.Screen)
        drawPath(path(h.b, true), Color.Blue.copy(alpha = 0.5f), blendMode = BlendMode.Screen)
        drawPath(path(h.l, false), Color.White.copy(alpha = 0.85f), style = Stroke(1.2.dp.toPx()))
    }
}

@Composable
fun PaletteStrip(swatches: List<Swatch>) {
    Row(Modifier.fillMaxWidth().height(30.dp).clip(RoundedCornerShape(8.dp)), horizontalArrangement = Arrangement.spacedBy(2.dp)) {
        for (s in swatches) Box(Modifier.weight(max(0.01, s.share).toFloat()).fillMaxSize().background(Color(s.r, s.g, s.b)))
    }
}

@Composable
fun MeasureGrid(m: Measurement) {
    val cells = listOf(
        Triple("Brightness", usf("%.2f", m["luma_median"]), m["luma_median"] < 0.25 || m["luma_median"] > 0.7),
        Triple("Blown highlights", usf("%.1f%%", 100 * m["clip_hi"]), m["clip_hi"] >= 0.01),
        Triple("Crushed shadows", usf("%.1f%%", 100 * m["clip_lo"]), m["clip_lo"] >= 0.03),
        Triple("Contrast", usf("%.2f", m["contrast_rms"]), m["contrast_rms"] < 0.12),
        Triple("Colourfulness", usf("%.2f", m["colorfulness"]), false),
        Triple("Saturation", usf("%.0f%%", 100 * m["sat_mean"]), m["sat_mean"] > 0.62),
        Triple("Edge crispness", usf("%.2f", m["edge_crispness"]), false),
        Triple("Noise", usf("%.1f", 1000 * m["noise_luma"]), m["noise_luma"] > 0.006),
    )
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        for (row in cells.chunked(2)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for ((name, value, flag) in row) {
                    Column(Modifier.weight(1f).background(Color.White.copy(alpha = 0.04f), RoundedCornerShape(10.dp)).padding(10.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(name, style = Type.caption2, color = Theme.dim)
                        Text(value, style = Type.subheadline.bold, color = if (flag) Theme.b else Color.White)
                    }
                }
            }
        }
    }
}

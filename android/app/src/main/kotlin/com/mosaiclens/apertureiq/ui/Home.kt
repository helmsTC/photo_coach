package com.mosaiclens.apertureiq.ui

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.material.icons.automirrored.filled.CompareArrows
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.EmojiEvents
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.MoreHoriz
import androidx.compose.material.icons.filled.Verified
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.mosaiclens.apertureiq.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Shows the splash over the app until a tier is chosen, then opens onto the home screen. */
@Composable
fun RootView(model: AppModel) {
    var paywall by remember { mutableStateOf(false) }
    Box(Modifier.fillMaxSize().background(Theme.bg)) {
        AnimatedContent(
            model.phase, label = "phase",
            transitionSpec = { (fadeIn(tween(350)) + slideInVertically(tween(350)) { if (targetState > initialState) it / 8 else 0 }) togetherWith fadeOut(tween(250)) },
        ) { phase ->
            when (phase) {
                Phase.PICKING -> HomeView(model) { paywall = true }
                Phase.ANALYZING -> AnalyzingView(model)
                Phase.RESULT -> model.result?.let { ResultScreen(it, model.debug) { model.newComparison() } }
                Phase.VERDICT -> model.quick?.let { VerdictView(it, onNew = { model.newComparison() }, onUpgrade = { paywall = true }) }
            }
        }
        AnimatedVisibility(model.splash, enter = fadeIn(tween(0)), exit = fadeOut(tween(550)) + scaleOut(tween(550), targetScale = 1.25f)) {
            SplashView(model.store) { chosen ->
                model.tier = chosen
                model.splash = false
            }
        }
    }
    if (paywall) {
        PaywallSheet(model.store, onDismiss = { paywall = false }) {
            // upgraded mid-session: switch to Pro, and if a verdict is showing, rerun it as the full analysis
            model.tier = Tier.PRO
            if (model.phase == Phase.VERDICT) model.run()
        }
    }
}

// ---- splash

/** Startup animation: the iris spins shut, snaps open, the wordmark lands, then Free / Pro cards rise in. */
@Composable
fun SplashView(store: Store, onChoose: (Tier) -> Unit) {
    val open = remember { Animatable(1f) }
    val spin = remember { Animatable(-140f) }
    val mark = remember { Animatable(0f) }          // lens scale + opacity
    val flash = remember { Animatable(0f) }
    val ring = remember { Animatable(0f) }
    var titleShown by remember { mutableStateOf(false) }
    var taglineShown by remember { mutableStateOf(false) }
    var chooserShown by remember { mutableStateOf(false) }
    val view = LocalView.current

    LaunchedEffect(Unit) {
        // 1. lens arrives, iris spins closed
        launch { mark.animateTo(1f, spring(dampingRatio = 0.75f, stiffness = 120f)) }
        launch { open.animateTo(0f, tween(750, easing = FastOutSlowInEasing)) }
        launch { spin.animateTo(0f, tween(750, easing = FastOutSlowInEasing)) }
        delay(800)
        // 2. "click": iris snaps open with a flash and a shockwave
        Theme.tick(view)
        launch { open.animateTo(0.72f, spring(dampingRatio = 0.6f, stiffness = 320f)) }
        launch { spin.animateTo(60f, spring(dampingRatio = 0.6f, stiffness = 320f)) }
        launch { flash.animateTo(1f, tween(150)); flash.animateTo(0.35f, tween(600)) }
        launch { ring.animateTo(1f, tween(900, easing = LinearOutSlowInEasing)) }
        delay(150)
        // 3. wordmark and tagline
        titleShown = true
        delay(250)
        taglineShown = true
        delay(900)
        // 4. the logo lifts and the tier cards rise in
        chooserShown = true
    }

    val lift by animateFloatAsState(if (chooserShown) 0.62f else 1f, spring(dampingRatio = 0.85f, stiffness = 110f), label = "lift")
    val titleA by animateFloatAsState(if (titleShown) 1f else 0f, spring(dampingRatio = 0.8f, stiffness = 130f), label = "title")
    val tagA by animateFloatAsState(if (taglineShown) 1f else 0f, tween(500), label = "tagline")

    Box(Modifier.fillMaxSize().clickable(enabled = false) {}) {
        AmbientBackground(tintA = Theme.violet, tintB = Theme.b)
        Column(Modifier.fillMaxSize().systemBarsPadding(), horizontalAlignment = Alignment.CenterHorizontally) {
            Spacer(Modifier.weight(1f))
            Box(Modifier.height((200 * lift).dp), contentAlignment = Alignment.Center) {
                Canvas(Modifier.size(280.dp).graphicsLayer { val s = mark.value * lift; scaleX = s; scaleY = s; alpha = mark.value.coerceIn(0f, 1f) }) {
                    val c = center
                    // shockwave ring when the iris snaps open
                    if (ring.value > 0f && ring.value < 1f) {
                        drawCircle(Brush.horizontalGradient(Theme.ctaColors), 75.dp.toPx() * (1 + ring.value * 2.6f), c, alpha = (1 - ring.value) * 0.8f, style = Stroke(3.dp.toPx()))
                    }
                    drawCircle(Brush.radialGradient(listOf(Theme.violet.copy(alpha = 0.55f), Color.Transparent), c, 140.dp.toPx()), 140.dp.toPx(), c)
                    drawCircle(Brush.radialGradient(listOf(Theme.pink.copy(alpha = 0.25f), Color.Transparent), c, 120.dp.toPx()), 120.dp.toPx(), c)
                    // lens barrel
                    drawCircle(Brush.verticalGradient(listOf(Color.White.copy(alpha = 0.9f), Color.White.copy(alpha = 0.25f)), c.y - 84.dp.toPx(), c.y + 84.dp.toPx()), 84.dp.toPx(), c, style = Stroke(5.dp.toPx()))
                }
                // the blades: brand gradient with dark seams between them
                Iris(open.value, Modifier.size(152.dp).graphicsLayer { val s = mark.value * lift; scaleX = s; scaleY = s; alpha = mark.value.coerceIn(0f, 1f); rotationZ = spin.value }, seams = true)
                // glint of light through the opening
                Canvas(Modifier.size(60.dp).graphicsLayer { alpha = flash.value * mark.value.coerceIn(0f, 1f); scaleX = lift; scaleY = lift }) {
                    drawCircle(Brush.radialGradient(listOf(Color.White, Color.White.copy(alpha = 0.5f), Color.Transparent)))
                }
            }
            Spacer(Modifier.height(28.dp))
            Wordmark(44f, Modifier.graphicsLayer { alpha = titleA; translationY = (1 - titleA) * 18.dp.toPx(); val s = 0.92f + 0.08f * titleA; scaleX = s; scaleY = s })
            Spacer(Modifier.height(10.dp))
            Text(
                "Smarter shots, one comparison at a time", style = Type.subheadline.w(FontWeight.Medium), color = Theme.dim,
                modifier = Modifier.graphicsLayer { alpha = tagA; translationY = (1 - tagA) * 10.dp.toPx() },
            )
            Spacer(Modifier.weight(1f))
            AnimatedVisibility(chooserShown, enter = fadeIn(tween(400)) + slideInVertically(spring(dampingRatio = 0.85f, stiffness = 110f)) { it / 2 }) {
                TierChooser(store, onChoose, Modifier.padding(horizontal = 20.dp).padding(bottom = 12.dp))
            }
        }
    }
}

/** Free or Pro: the two ways into the app. */
@Composable
fun TierChooser(store: Store, onChoose: (Tier) -> Unit, modifier: Modifier = Modifier) {
    val view = LocalView.current
    Column(modifier, verticalArrangement = Arrangement.spacedBy(12.dp)) {
        // Free
        Row(
            Modifier.clip(RoundedCornerShape(22.dp)).clickable { Theme.tick(view); onChoose(Tier.FREE) }.glassCard(),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Box(Modifier.size(40.dp).background(Color.White.copy(alpha = 0.07f), CircleShape), contentAlignment = Alignment.Center) {
                Icon(Icons.Filled.EmojiEvents, null, tint = Theme.gold)
            }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                Text("Free", style = Type.headline, color = Color.White)
                Text("Pick two shots, see which one wins.", style = Type.footnote, color = Theme.dim)
            }
            Icon(Icons.Filled.ChevronRight, null, tint = Theme.dim)
        }
        // Pro
        Column(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(22.dp)).background(Theme.card).background(Theme.violet.copy(alpha = 0.12f))
                .border(1.5.dp, Theme.cta, RoundedCornerShape(22.dp)).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Pro", style = Type.headline, color = Color.White)
                ProBadge()
                Spacer(Modifier.weight(1f))
                if (store.ownsPro) {
                    Icon(Icons.Filled.Verified, null, Modifier.size(15.dp), tint = Theme.good)
                    Text("Unlocked", style = Type.caption.bold, color = Theme.good)
                }
            }
            Text(
                "The full coach: why it won, analysis lenses, object detection, a coaching plan and an edit toward the winner.",
                style = Type.footnote, color = Theme.dim,
            )
            PurchaseButton(store) { unlocked -> if (unlocked) onChoose(Tier.PRO) }
        }
    }
}

// ---- home

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun HomeView(model: AppModel, onUpgrade: () -> Unit) {
    var showSelfTest by remember { mutableStateOf(model.debug.selfTest) }
    var showAbout by remember { mutableStateOf(false) }
    var menu by remember { mutableStateOf(false) }
    val view = LocalView.current
    val tier = model.tier

    Box(Modifier.fillMaxSize()) {
        AmbientBackground()
        Column(
            Modifier.fillMaxSize().systemBarsPadding().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(top = 12.dp, bottom = 30.dp),
            verticalArrangement = Arrangement.spacedBy(26.dp),
        ) {
            // header
            Row(verticalAlignment = Alignment.CenterVertically) {
                Wordmark(34f)
                if (tier == Tier.PRO) ProBadge(Modifier.padding(start = 8.dp))
                Spacer(Modifier.weight(1f))
                if (tier == Tier.FREE) {
                    Row(
                        Modifier.padding(end = 8.dp).background(Theme.cta, CircleShape).clip(CircleShape).clickable(onClick = onUpgrade).padding(horizontal = 12.dp, vertical = 9.dp),
                        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp),
                    ) {
                        Icon(Icons.Filled.AutoAwesome, null, Modifier.size(14.dp), tint = Color.White)
                        Text("Go Pro", style = Type.caption.bold, color = Color.White)
                    }
                }
                Box {
                    CircleButton(Icons.Filled.MoreHoriz, "More") { menu = true }
                    DropdownMenu(menu, { menu = false }) {
                        DropdownMenuItem({ Text("About ApertureIQ") }, { menu = false; showAbout = true })
                        if (BuildConfig.DEBUG) DropdownMenuItem({ Text("Run self-test") }, { menu = false; showSelfTest = true })
                    }
                }
            }

            // the two slots
            Box(contentAlignment = Alignment.Center) {
                Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                    PhotoSlot(0, model.thumbs[0], Modifier.weight(1f).rotate(-2.5f)) { model.set(it, 0) }
                    PhotoSlot(1, model.thumbs[1], Modifier.weight(1f).rotate(2.5f)) { model.set(it, 1) }
                }
                val canSwap = model.data[0] != null || model.data[1] != null
                Box(
                    Modifier.size(56.dp).background(Theme.bg, CircleShape).border(2.5.dp, Theme.cta, CircleShape).clip(CircleShape)
                        .clickable(enabled = canSwap, onClickLabel = "Swap shots") { model.swap(); Theme.tick(view) },
                    contentAlignment = Alignment.Center,
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("VS", fontSize = 16.sp, fontWeight = FontWeight.Black, color = Color.White, lineHeight = 17.sp)
                        Icon(Icons.AutoMirrored.Filled.CompareArrows, null, Modifier.size(12.dp), tint = Theme.dim)
                    }
                }
            }

            // compare
            val glow by rememberInfiniteTransition(label = "glow").animateFloat(0.35f, 0.7f, infiniteRepeatable(tween(1600), RepeatMode.Reverse), label = "glow")
            Box(contentAlignment = Alignment.Center) {
                if (model.ready) {
                    Canvas(Modifier.fillMaxWidth().height(58.dp)) {
                        drawRoundRect(
                            Brush.radialGradient(listOf(Theme.pink.copy(alpha = glow * 0.6f), Color.Transparent), center, size.width * 0.6f),
                            Offset(-20.dp.toPx(), -14.dp.toPx()), androidx.compose.ui.geometry.Size(size.width + 40.dp.toPx(), size.height + 28.dp.toPx()), CornerRadius(40.dp.toPx()),
                        )
                    }
                }
                Row(
                    Modifier.fillMaxWidth().alpha(if (model.ready) 1f else 0.35f).background(Theme.cta, CircleShape).clip(CircleShape)
                        .clickable(enabled = model.ready) { model.run { ok -> if (ok) Theme.success(view) else Theme.error(view) } }.padding(vertical = 17.dp),
                    horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(Icons.Filled.AutoAwesome, null, Modifier.size(20.dp), tint = Color.White)
                    Spacer(Modifier.width(10.dp))
                    Text("Compare", style = Type.headline.bold, color = Color.White)
                }
            }

            model.error?.let { e ->
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Icon(Icons.Filled.WarningAmber, null, Modifier.size(16.dp), tint = Theme.bad)
                    Text(e, style = Type.footnote, color = Theme.bad)
                }
            }

            // recent
            val shown = model.history.visible(model.store.ownsPro)
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Recent", style = Type.subheadline.semibold, color = Theme.dim)
                    Spacer(Modifier.weight(1f))
                    if (model.store.ownsPro) Text("${shown.size} of 10", style = Type.caption, color = Theme.dim)
                    else Row(Modifier.clickable(onClick = onUpgrade)) {
                        Text("${shown.size} of 3 · ", style = Type.caption, color = Theme.dim)
                        Text("Pro keeps 10", style = Type.caption.bold, color = Theme.dim)
                    }
                }
                if (shown.isEmpty()) {
                    Row(Modifier.glassCard(14.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Icon(Icons.Filled.History, null, tint = Theme.dim)
                        Text(
                            "Your comparisons show up here so you can reopen them later. Pick two of your own photos to start (the sample pairs aren't saved).",
                            style = Type.footnote, color = Theme.dim,
                        )
                    }
                } else {
                    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        for (e in shown) {
                            var del by remember(e.id) { mutableStateOf(false) }
                            Box(Modifier.clip(RoundedCornerShape(18.dp)).combinedClickable(onLongClick = { del = true }, onLongClickLabel = "Delete") { Theme.tick(view); model.open(e) }) {
                                HistoryCard(e, model.history.thumbs(e))
                                DropdownMenu(del, { del = false }) {
                                    DropdownMenuItem({ Text("Delete") }, { del = false; model.history.delete(e) })
                                }
                            }
                        }
                    }
                }
            }

            // samples
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("No photos handy? Try a pair", style = Type.subheadline.semibold, color = Theme.dim)
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    for (pair in Samples.pairs) {
                        Box(Modifier.clip(RoundedCornerShape(18.dp)).clickable { model.load(pair); Theme.tick(view) }) { SampleCard(pair) }
                    }
                }
            }

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Lock, null, Modifier.size(12.dp), tint = Theme.dim)
                Spacer(Modifier.width(6.dp))
                Text("Everything runs on this phone. Your photos never leave it.", style = Type.caption, color = Theme.dim)
            }
        }
    }
    if (showSelfTest) SelfTestSheet { showSelfTest = false }
    if (showAbout) AboutSheet(model.store) { showAbout = false }
}

@Composable
fun PhotoSlot(index: Int, image: ImageBitmap?, modifier: Modifier = Modifier, onPick: (ByteArray?) -> Unit) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) scope.launch {
            val bytes = withContext(Dispatchers.IO) {
                try { ctx.contentResolver.openInputStream(uri)?.use { it.readBytes() } } catch (e: Exception) { null }
            }
            onPick(bytes)
        }
    }
    val tint = Theme.tint(index)
    val shape = RoundedCornerShape(24.dp)
    Box(
        modifier.aspectRatio(3f / 4f).clip(shape).background(Theme.card)
            .clickable(onClickLabel = if (index == 0) "Choose shot A" else "Choose shot B") {
                picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            },
    ) {
        if (image != null) {
            Image(image, null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
            Box(Modifier.fillMaxSize().border(2.dp, tint.copy(alpha = 0.9f), shape))
        } else {
            Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Icon(Icons.Filled.AddPhotoAlternate, null, Modifier.size(32.dp), tint = tint)
                Text(if (index == 0) "Your shot" else "Compare with", style = Type.footnote.semibold, color = tint)
                Text("Tap to choose", style = Type.caption2, color = Theme.dim)
            }
            Canvas(Modifier.fillMaxSize()) {
                val r = 24.dp.toPx()
                drawRoundRect(tint.copy(alpha = 0.5f), cornerRadius = CornerRadius(r), style = Stroke(2.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(7.dp.toPx(), 6.dp.toPx()))))
            }
        }
        ShotTag(index, modifier = Modifier.padding(10.dp))
    }
}

@Composable
fun SampleCard(pair: Samples.Pair) {
    val app = LocalContext.current.applicationContext as android.app.Application
    var images by remember(pair.title) { mutableStateOf<List<ImageBitmap>>(emptyList()) }
    LaunchedEffect(pair.title) {
        if (images.isEmpty()) images = withContext(Dispatchers.Default) {
            listOf(pair.a, pair.b).mapNotNull { n -> Samples.data(app, n)?.let { AppModel.thumbnail(it, 240) } }
        }
    }
    Column(
        Modifier.width(174.dp).background(Theme.card, RoundedCornerShape(18.dp)).border(1.dp, Theme.stroke, RoundedCornerShape(18.dp)).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(Modifier.size(150.dp, 84.dp), contentAlignment = Alignment.Center) {
            for ((i, img) in images.withIndex()) {
                Image(
                    img, null, contentScale = ContentScale.Crop,
                    modifier = Modifier.offset(x = if (i == 0) (-22).dp else 22.dp).rotate(if (i == 0) -6f else 6f).size(92.dp, 70.dp)
                        .clip(RoundedCornerShape(10.dp)).border(1.5.dp, Theme.tint(i), RoundedCornerShape(10.dp)),
                )
            }
        }
        Text(pair.title, style = Type.footnote.semibold, color = Color.White)
        Text(pair.subtitle, style = Type.caption2, color = Theme.dim)
    }
}

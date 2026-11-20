package com.mosaiclens.apertureiq.ui

import android.app.Activity
import android.content.Context
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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ManageSearch
import androidx.compose.material.icons.filled.BarChart
import androidx.compose.material.icons.filled.CenterFocusWeak
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.material.icons.filled.LockOpen
import androidx.compose.material.icons.filled.School
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.Verified
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.android.billingclient.api.AcknowledgePurchaseParams
import com.android.billingclient.api.BillingClient
import com.android.billingclient.api.BillingClientStateListener
import com.android.billingclient.api.BillingFlowParams
import com.android.billingclient.api.BillingResult
import com.android.billingclient.api.PendingPurchasesParams
import com.android.billingclient.api.ProductDetails
import com.android.billingclient.api.Purchase
import com.android.billingclient.api.PurchasesUpdatedListener
import com.android.billingclient.api.QueryProductDetailsParams
import com.android.billingclient.api.QueryPurchasesParams
import com.android.billingclient.api.acknowledgePurchase
import com.android.billingclient.api.queryProductDetails
import com.android.billingclient.api.queryPurchasesAsync
import com.mosaiclens.apertureiq.BuildConfig
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

/** Which experience the user is in this session. */
enum class Tier { FREE, PRO }

/**
 * ApertureIQ Pro: a one-time, non-consumable in-app product sold through Google Play Billing.
 * Ownership comes from Play's purchase records, so it survives reinstalls and follows the user's
 * Google account; "Restore purchases" just asks Play again.
 */
class Store(context: Context, private val scope: CoroutineScope) : PurchasesUpdatedListener {
    companion object {
        const val proID = "com.mosaiclens.apertureiq.pro"
        private const val prefs = "store"
        private const val debugKey = "debug_pro"
    }

    private val ctx = context.applicationContext
    var product by mutableStateOf<ProductDetails?>(null); private set
    var ownsPro by mutableStateOf(false); private set
    var busy by mutableStateOf(false); private set
    var loaded by mutableStateOf(false); private set
    var message by mutableStateOf<String?>(null)

    /** Debug builds only: Pro granted without Play (set by the debug unlock or the `pro` launch extra). */
    private var debugPro = BuildConfig.DEBUG && ctx.getSharedPreferences(prefs, Context.MODE_PRIVATE).getBoolean(debugKey, false)
    private var pending: CompletableDeferred<Boolean>? = null

    private val client: BillingClient = BillingClient.newBuilder(ctx)
        .setListener(this)
        .enablePendingPurchases(PendingPurchasesParams.newBuilder().enableOneTimeProducts().build())
        .build()

    init {
        ownsPro = debugPro
        scope.launch { load() }
    }

    /** True when this debug build can hand out Pro itself because Play has no product to sell. */
    val canDebugUnlock: Boolean get() = BuildConfig.DEBUG && loaded && product == null

    /** "Unlock Pro · $4.99", or just "Unlock Pro" until the price has loaded. */
    val unlockTitle: String
        get() = product?.oneTimePurchaseOfferDetails?.formattedPrice?.let { "Unlock Pro · $it" }
            ?: if (canDebugUnlock) "Unlock Pro (debug build)" else "Unlock Pro"

    fun forceDebugPro(on: Boolean) {
        if (!BuildConfig.DEBUG) return
        debugPro = on
        ctx.getSharedPreferences(prefs, Context.MODE_PRIVATE).edit().putBoolean(debugKey, on).apply()
        ownsPro = on || ownsPro
        if (!on) scope.launch { refresh() }
    }

    private suspend fun connect(): Boolean {
        if (client.isReady) return true
        return suspendCoroutine { cont ->
            var done = false
            try {
                client.startConnection(object : BillingClientStateListener {
                    override fun onBillingSetupFinished(result: BillingResult) {
                        if (!done) { done = true; cont.resume(result.responseCode == BillingClient.BillingResponseCode.OK) }
                    }

                    override fun onBillingServiceDisconnected() {}
                })
            } catch (e: Exception) {
                if (!done) { done = true; cont.resume(false) }
            }
        }
    }

    suspend fun load() {
        product = try {
            if (connect()) {
                val params = QueryProductDetailsParams.newBuilder().setProductList(
                    listOf(QueryProductDetailsParams.Product.newBuilder().setProductId(proID).setProductType(BillingClient.ProductType.INAPP).build()),
                ).build()
                client.queryProductDetails(params).productDetailsList?.firstOrNull()
            } else null
        } catch (e: Exception) {
            null
        }
        refresh()
        loaded = true
    }

    /** Re-reads what the user owns from Google Play. */
    suspend fun refresh() {
        var owns = false
        try {
            if (connect()) {
                val res = client.queryPurchasesAsync(QueryPurchasesParams.newBuilder().setProductType(BillingClient.ProductType.INAPP).build())
                for (p in res.purchasesList) if (settle(p)) owns = true
            }
        } catch (e: Exception) {
            // Play unreachable: keep whatever was known
            owns = ownsPro && !debugPro
        }
        ownsPro = owns || debugPro
    }

    /** True if this purchase grants Pro; acknowledges it if Play is still waiting for that. */
    private suspend fun settle(p: Purchase): Boolean {
        if (!p.products.contains(proID) || p.purchaseState != Purchase.PurchaseState.PURCHASED) return false
        if (!p.isAcknowledged) {
            try {
                client.acknowledgePurchase(AcknowledgePurchaseParams.newBuilder().setPurchaseToken(p.purchaseToken).build())
            } catch (e: Exception) {
                // retried on the next refresh; Play refunds purchases left unacknowledged for three days
            }
        }
        return true
    }

    override fun onPurchasesUpdated(result: BillingResult, purchases: MutableList<Purchase>?) {
        scope.launch {
            var owns = false
            when (result.responseCode) {
                BillingClient.BillingResponseCode.OK -> {
                    for (p in purchases.orEmpty()) {
                        if (settle(p)) owns = true
                        else if (p.products.contains(proID) && p.purchaseState == Purchase.PurchaseState.PENDING) {
                            message = "Purchase pending. Pro unlocks as soon as the payment clears."
                        }
                    }
                }
                BillingClient.BillingResponseCode.ITEM_ALREADY_OWNED -> { refresh(); owns = ownsPro }
                BillingClient.BillingResponseCode.USER_CANCELED -> {}
                else -> message = result.debugMessage.ifEmpty { "Google Play couldn't complete that purchase." }
            }
            if (owns) ownsPro = true
            pending?.complete(ownsPro)
            pending = null
        }
    }

    /** Returns true once Pro is owned. */
    suspend fun purchase(activity: Activity?): Boolean {
        val details = product
        if (details == null) {
            if (canDebugUnlock) { forceDebugPro(true); return true }
            message = "Google Play isn't reachable right now. Try again in a moment."
            return false
        }
        if (activity == null) return false
        busy = true
        try {
            val flow = BillingFlowParams.newBuilder().setProductDetailsParamsList(
                listOf(BillingFlowParams.ProductDetailsParams.newBuilder().setProductDetails(details).build()),
            ).build()
            val waiter = CompletableDeferred<Boolean>()
            pending = waiter
            val launched = client.launchBillingFlow(activity, flow)
            if (launched.responseCode != BillingClient.BillingResponseCode.OK) {
                pending = null
                if (launched.responseCode == BillingClient.BillingResponseCode.ITEM_ALREADY_OWNED) refresh()
                else message = launched.debugMessage.ifEmpty { "Google Play couldn't start the purchase." }
                return ownsPro
            }
            return waiter.await()
        } catch (e: Exception) {
            message = e.localizedMessage
            return ownsPro
        } finally {
            busy = false
        }
    }

    suspend fun restore() {
        busy = true
        try {
            if (!client.isReady) load() else refresh()
            message = if (ownsPro) null else "No Pro purchase found for this Google account."
        } finally {
            busy = false
        }
    }
}

/** What Pro adds, shown on the splash card and the paywall. */
object ProFeatures {
    val list: List<Pair<ImageVector, String>> = listOf(
        Icons.AutoMirrored.Filled.ManageSearch to "Why it won, in plain words",
        Icons.Filled.ImageSearch to "How the AI check decided, with the evidence",
        Icons.Filled.Visibility to "Lenses: clipping, focus, composition, AI attention",
        Icons.Filled.CenterFocusWeak to "Object detection: subject vs. distractions",
        Icons.Filled.School to "A step-by-step coaching plan",
        Icons.Filled.Tune to "A one-tap edit toward the winner",
        Icons.Filled.BarChart to "Full scores, histograms, palettes and EXIF",
    )
}

/** A full-screen sheet over the current screen. */
@Composable
fun Sheet(onDismiss: () -> Unit, content: @Composable () -> Unit) {
    Dialog(onDismiss, DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)) {
        Box(Modifier.fillMaxSize().background(Theme.bg)) { content() }
    }
}

@Composable
fun CircleButton(icon: ImageVector, description: String, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Box(
        modifier.size(38.dp).background(Color.White.copy(alpha = 0.12f), CircleShape).clickable(onClickLabel = description, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) { Icon(icon, description, Modifier.size(18.dp), tint = Color.White) }
}

/** Upgrade sheet, opened from inside the free experience. */
@Composable
fun PaywallSheet(store: Store, onDismiss: () -> Unit, onUnlocked: () -> Unit) {
    Sheet(onDismiss) {
        AmbientBackground(tintA = Theme.violet, tintB = Theme.b)
        Column(
            Modifier.fillMaxSize().systemBarsPadding().verticalScroll(rememberScrollState()).padding(horizontal = 22.dp).padding(bottom = 30.dp),
            horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(24.dp),
        ) {
            Box(Modifier.padding(top = 36.dp).size(84.dp), contentAlignment = Alignment.Center) {
                androidx.compose.foundation.Canvas(Modifier.fillMaxSize()) { drawCircle(Color.White.copy(alpha = 0.8f), style = Stroke(3.dp.toPx())) }
                Iris(0.55f, Modifier.size(76.dp))
            }
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Wordmark(32f)
                    ProBadge()
                }
                Text(
                    "Know exactly why one shot beats the other - and how to fix the other one.",
                    style = Type.subheadline, color = Theme.dim, textAlign = TextAlign.Center,
                )
            }
            Column(Modifier.glassCard(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                for ((icon, title) in ProFeatures.list) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Icon(icon, null, Modifier.width(26.dp).brushed(Theme.cta), tint = Color.White)
                        Text(title, style = Type.subheadline.w(FontWeight.Medium), color = Color.White)
                    }
                }
            }
            PurchaseButton(store) { unlocked -> if (unlocked) { onDismiss(); onUnlocked() } }
            Text(
                "One-time purchase. No subscription. Everything still runs on your phone.",
                style = Type.caption, color = Theme.dim, textAlign = TextAlign.Center,
            )
        }
        Box(Modifier.fillMaxSize().systemBarsPadding().padding(16.dp), contentAlignment = Alignment.TopEnd) {
            CircleButton(Icons.Filled.Close, "Close", onClick = onDismiss)
        }
    }
}

/** "Unlock Pro · $4.99" plus restore and any store message.  Calls back with whether Pro is now owned. */
@Composable
fun PurchaseButton(store: Store, done: (Boolean) -> Unit) {
    val scope = rememberCoroutineScope()
    val activity = LocalContext.current as? Activity
    val view = LocalView.current
    val available = store.ownsPro || store.product != null || store.canDebugUnlock
    val enabled = !store.busy && (available || !store.loaded)
    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(
            Modifier
                .fillMaxWidth()
                .alpha(if (available) 1f else 0.45f)
                .background(Theme.cta, CircleShape)
                .clickable(enabled = enabled) {
                    if (store.ownsPro) done(true)
                    else scope.launch {
                        val ok = store.purchase(activity)
                        if (ok) Theme.success(view)
                        done(ok)
                    }
                }
                .padding(vertical = 16.dp),
            horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically,
        ) {
            if (store.busy) CircularProgressIndicator(Modifier.size(18.dp), color = Color.White, strokeWidth = 2.dp)
            else Icon(if (store.ownsPro) Icons.Filled.Verified else Icons.Filled.LockOpen, null, Modifier.size(20.dp), tint = Color.White)
            Spacer(Modifier.width(10.dp))
            Text(if (store.ownsPro) "Continue with Pro" else store.unlockTitle, style = Type.headline.bold, color = Color.White)
        }
        if (!store.ownsPro) {
            Text(
                "Restore purchases", style = Type.footnote.semibold, color = Theme.dim,
                modifier = Modifier.clickable(enabled = !store.busy) { scope.launch { store.restore(); if (store.ownsPro) { Theme.success(view); done(true) } } }.padding(4.dp),
            )
            if (store.loaded && store.product == null && !store.canDebugUnlock) {
                Text("Google Play isn't available right now.", style = Type.caption, color = Theme.dim)
            }
        }
        store.message?.let { Text(it, style = Type.caption, color = Theme.gold, textAlign = TextAlign.Center) }
    }
}

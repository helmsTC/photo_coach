package com.mosaiclens.apertureiq.ui

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mosaiclens.apertureiq.BuildConfig
import com.mosaiclens.apertureiq.judge.Authenticity
import com.mosaiclens.apertureiq.judge.Comparison
import com.mosaiclens.apertureiq.judge.QuickVerdict
import com.mosaiclens.apertureiq.judge.Stage
import com.mosaiclens.apertureiq.platform.AndroidDecoder
import com.mosaiclens.apertureiq.platform.Engine
import com.mosaiclens.apertureiq.platform.toBitmap
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Built-in sample pairs (bundled demo photos). */
object Samples {
    class Pair(val title: String, val subtitle: String, val a: String, val b: String)

    val pairs = listOf(
        Pair("Lakeside château", "Same scene, two framings", "sample_louise_12", "sample_louise_13"),
        Pair("Boathouse", "Same subject, different light", "sample_louise_07", "sample_louise_09"),
        Pair("Moraine lookout", "Same view, two exposures", "sample_moraine_03", "sample_moraine_04"),
        Pair("Busy table", "Clutter vs a clean scene", "sample_zclutter_table", "parity_louise_11"),
    )

    fun data(app: Application, name: String): ByteArray? = Engine.asset(app, "$name.jpg")
}

enum class Phase { PICKING, ANALYZING, VERDICT, RESULT }

/** Launch options for screenshots and testing (debug builds only; see README). */
class DebugOptions(
    val noSplash: Boolean = false,
    val tier: Tier? = null,
    val autoSample: Int? = null,
    val tab: String? = null,
    val lens: String? = null,
    val photo: Int? = null,
    val selfTest: Boolean = false,
    val expandAll: Boolean = false,
)

class AppModel(private val app: Application) : AndroidViewModel(app) {
    val store = Store(app, viewModelScope)
    val history = History(app)

    var tier by mutableStateOf(Tier.FREE)
    var splash by mutableStateOf(true)
    var debug = DebugOptions(); private set

    val data = mutableStateListOf<ByteArray?>(null, null)
    val thumbs = mutableStateListOf<ImageBitmap?>(null, null)
    var phase by mutableStateOf(Phase.PICKING); private set
    var stage by mutableStateOf<Stage?>(null); private set
    var error by mutableStateOf<String?>(null)
    var result by mutableStateOf<Comparison?>(null); private set
    /** The verdict is on screen while the slower Pro sections (objects, attention, coaching) are still being worked out. */
    var pending by mutableStateOf(false); private set
    var quick by mutableStateOf<QuickVerdict?>(null); private set

    /** The saved entry being shown, so a re-run updates it instead of adding a duplicate. */
    private var entryID: String? = null
    /** AI checks saved with a reopened comparison, reused instead of checking the re-encoded copies. */
    private var known: List<Authenticity>? = null
    /** Samples aren't saved to history. */
    private var fromSample = false
    private var job: Job? = null
    private var debugApplied = false

    val ready: Boolean get() = data[0] != null && data[1] != null

    /** The free tier only runs the first four stages: read, measure, score, AI check. */
    val stages: List<Stage>
        get() = if (tier == Tier.FREE) listOf(Stage.READING, Stage.MEASURING, Stage.SCORING, Stage.AUTHENTICITY) else Stage.entries

    fun applyDebug(o: DebugOptions) {
        if (!BuildConfig.DEBUG || debugApplied) return
        debugApplied = true
        debug = o
        if (o.tier == Tier.PRO) store.forceDebugPro(true)
        if (o.noSplash || o.autoSample != null || o.selfTest) {
            splash = false
            tier = o.tier ?: Tier.PRO
        }
        o.autoSample?.let { n -> Samples.pairs.getOrNull(n)?.let { load(it); run() } }
    }

    fun set(bytes: ByteArray?, i: Int) {
        data[i] = bytes
        entryID = null
        known = null
        fromSample = false
        error = null
        thumbs[i] = null
        if (bytes != null) viewModelScope.launch {
            val t = withContext(Dispatchers.Default) { thumbnail(bytes, 900) }
            if (data[i] === bytes) {
                thumbs[i] = t
                if (t == null) { data[i] = null; error = "That photo could not be read." }
            }
        }
    }

    fun swap() {
        val d = data[0]; data[0] = data[1]; data[1] = d
        val t = thumbs[0]; thumbs[0] = thumbs[1]; thumbs[1] = t
        entryID = null
        known = known?.reversed()
    }

    fun load(pair: Samples.Pair) {
        set(Samples.data(app, pair.a), 0)
        set(Samples.data(app, pair.b), 1)
        fromSample = true
    }

    /** Reopens a saved comparison: same photos, analysed again in the current tier. */
    fun open(e: HistoryEntry) {
        val p = history.photos(e)
        if (p == null) { history.delete(e); return }
        set(p.first, 0)
        set(p.second, 1)
        entryID = e.id
        known = e.authenticity
        run()
    }

    private suspend fun remember(a: ByteArray, b: ByteArray, winner: Int, confidence: Double, authenticity: List<Authenticity>) {
        if (fromSample) return
        entryID = history.record(entryID, a, b, winner, confidence, tier == Tier.PRO, store.ownsPro, authenticity)
    }

    fun run(onDone: (Boolean) -> Unit = {}) {
        val a = data[0] ?: return
        val b = data[1] ?: return
        if (phase == Phase.ANALYZING) return
        error = null
        stage = null
        phase = Phase.ANALYZING
        val free = tier == Tier.FREE
        val knownNow = known
        job = viewModelScope.launch {
            try {
                val t0 = System.nanoTime()
                Engine.networks?.resetStats()
                val timing = com.mosaiclens.apertureiq.BuildConfig.DEBUG
                val progress: (Stage) -> Unit = { s ->
                    if (timing) android.util.Log.i("ApertureIQ", "STAGE %6.1f s  %s".format(java.util.Locale.US, (System.nanoTime() - t0) / 1e9, s.name))
                    viewModelScope.launch(Dispatchers.Main.immediate) { if (phase == Phase.ANALYZING || pending) stage = s }
                }
                if (free) {
                    val v = withContext(Dispatchers.Default) {
                        Engine.judge(app).quickVerdict(a, b, knownNow, progress) { ensureActive() }
                    }
                    quick = v
                    remember(a, b, v.winnerIndex, v.confidence, v.authenticity)
                    phase = Phase.VERDICT
                } else {
                    val early: (Comparison) -> Unit = { first ->
                        if (timing) android.util.Log.i("ApertureIQ", "STAGE %6.1f s  VERDICT SHOWN".format(java.util.Locale.US, (System.nanoTime() - t0) / 1e9))
                        viewModelScope.launch(Dispatchers.Main.immediate) {
                            if (phase == Phase.ANALYZING && job?.isActive == true) { result = first; pending = true; phase = Phase.RESULT }
                        }
                    }
                    val r = withContext(Dispatchers.Default) {
                        Engine.judge(app).compare(a, b, "Shot A", "Shot B", knownNow, progress, { ensureActive() }, early)
                    }
                    result = r
                    pending = false
                    remember(a, b, r.winnerIndex, r.confidence, r.extras.map { it.authenticity ?: Authenticity.decide(null, null, null) })
                    phase = Phase.RESULT
                }
                if (timing) android.util.Log.i("ApertureIQ", "STAGE %6.1f s  DONE".format(java.util.Locale.US, (System.nanoTime() - t0) / 1e9))
                if (timing) android.util.Log.i("ApertureIQ", "TIME in networks: ${Engine.networks?.stats()}")
                onDone(true)
            } catch (e: CancellationException) {
                pending = false
                phase = Phase.PICKING
                throw e
            } catch (e: Throwable) {
                pending = false
                result = null
                error = if (e is OutOfMemoryError) "Not enough memory to analyse these photos." else (e.localizedMessage ?: "Something went wrong.")
                phase = Phase.PICKING
                onDone(false)
            }
        }
    }

    fun cancel() {
        job?.cancel()
        pending = false
        phase = Phase.PICKING
    }

    fun newComparison() {
        if (pending) { job?.cancel(); pending = false }
        phase = Phase.PICKING
        result = null
        quick = null
    }

    companion object {
        fun thumbnail(bytes: ByteArray, maxSide: Int): ImageBitmap? = try {
            AndroidDecoder.pixels(bytes, maxSide)?.first?.toBitmap()?.asImageBitmap()
        } catch (e: Throwable) {
            null
        }
    }
}

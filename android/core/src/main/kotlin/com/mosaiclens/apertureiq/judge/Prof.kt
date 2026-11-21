package com.mosaiclens.apertureiq.judge

import java.util.concurrent.ConcurrentHashMap

/**
 * Wall-clock time per named step, summed across threads, for profiling on a device.
 * Costs two clock reads per step; the app prints [dump] in debug builds.
 */
object Prof {
    val nanos = ConcurrentHashMap<String, Long>()
    val calls = ConcurrentHashMap<String, Int>()

    inline fun <T> t(name: String, n: Int = 1, body: () -> T): T {
        val t0 = System.nanoTime()
        try { return body() } finally { nanos.merge(name, System.nanoTime() - t0, Long::plus); calls.merge(name, n, Int::plus) }
    }

    fun reset() { nanos.clear(); calls.clear() }

    fun dump(): String = nanos.entries.sortedByDescending { it.value }
        .joinToString(", ") { "${it.key} ${"%.1f".format(java.util.Locale.US, it.value / 1e9)}s/${calls[it.key]}" }
}

package com.github.nathanhowell.trailer

import org.openstreetmap.josm.spi.preferences.Config
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

/**
 * Finds the exported model on disk and keeps one session open.
 *
 * The weights are not in the jar. At ~99 MB they would double what a mapper
 * downloads to try the plugin, they change on a different schedule from the
 * code, and they are under a different licence — so they are a file the mapper
 * points at, and [ModelSpec] checks that file is one this build understands.
 *
 * The session is cached because building it means parsing 99 MB of graph, and
 * the action that uses it runs once per viewport. Cached on the *path*, so
 * pointing the preference at a different model swaps it without a restart.
 */
object ModelStore {

    /** Preference key holding the absolute path to an exported `.onnx`. */
    const val PREF_MODEL = "trailer.model.onnx"

    /** No model configured, or the configured one is not there. */
    class NotConfigured(message: String) : Exception(message)

    private var cached: Pair<Path, Inference>? = null

    /**
     * The sidecar `trailer export` wrote beside an `.onnx`.
     *
     * Mirrors Python's `Path.with_suffix(".json")` — replace the final
     * extension, do not append. `trailer.onnx` pairs with `trailer.json`, and
     * `dem1.v3.onnx` with `dem1.v3.json`, which is why this splits on the last
     * dot rather than stripping a known suffix.
     */
    fun sidecarFor(onnx: Path): Path {
        val name = onnx.fileName.toString()
        val dot = name.lastIndexOf('.')
        val stem = if (dot > 0) name.substring(0, dot) else name
        return onnx.resolveSibling("$stem.json")
    }

    /** Configured model path, or null if the mapper has not set one. */
    fun configuredPath(): Path? =
        Config.getPref().get(PREF_MODEL, "").trim()
            .takeIf { it.isNotEmpty() }
            ?.let { Paths.get(it) }

    /**
     * Open the configured model, reusing the session if it is already open.
     *
     * Every failure here is a message a mapper can act on. "Model failed to
     * load" would be true and useless; which file, and what is wrong with it,
     * is the difference between fixing a preference and filing a bug.
     */
    @Synchronized
    fun open(): Inference {
        val onnx = configuredPath() ?: throw NotConfigured(
            "No trail model is configured. Set the path to an exported .onnx " +
                "in Preferences, under $PREF_MODEL")
        cached?.let { (path, inference) -> if (path == onnx) return inference }

        if (!Files.isReadable(onnx)) throw NotConfigured(
            "Cannot read the configured model at $onnx")
        val sidecar = sidecarFor(onnx)
        if (!Files.isReadable(sidecar)) throw NotConfigured(
            "The model at $onnx has no sidecar beside it. `trailer export` " +
                "writes ${sidecar.fileName} next to the .onnx, and this plugin " +
                "needs it: it carries the window size, the tiling step and the " +
                "attribution, none of which are recoverable from the graph")

        close()
        val opened = Inference.open(onnx, sidecar)
        cached = onnx to opened
        return opened
    }

    /** Drop the cached session, e.g. when the plugin unloads. */
    @Synchronized
    fun close() {
        cached?.second?.close()
        cached = null
    }
}

package com.github.nathanhowell.trailer

import java.nio.file.Files
import java.nio.file.Path

/**
 * Decides whether a file a mapper picked is a model this build can use.
 *
 * Separate from the preferences dialog because it is the only part of choosing
 * a model with any judgement in it, and the dialog around it cannot be tested
 * without a running JOSM.
 *
 * The point is to fail at the moment of choosing. Deferring validation to the
 * first inference run means the mapper learns their file is wrong several
 * minutes and one 3DEP download later, with nothing on screen connecting the
 * failure to the choice that caused it.
 */
object ModelChoice {

    /** What a usable model turned out to be, in the terms a mapper cares about. */
    data class Summary(
        val variant: String,
        val resM: Double,
        val windowPx: Int,
        val tta: Boolean,
        val license: String,
        val attribution: String,
        val megabytes: Double,
    ) {
        /** One line for the dialog: what it is and what it will cost. */
        fun describe(): String = buildString {
            append(variant)
            append(" · ").append("%.2g".format(resM)).append(" m")
            append(" · ").append(windowPx).append(" px window")
            append(" · ").append("%.0f".format(megabytes)).append(" MB")
            if (tta) append(" · 8x test-time augmentation")
        }
    }

    sealed class Check {
        data class Ok(val summary: Summary) : Check()
        data class Bad(val reason: String) : Check()
    }

    /**
     * Inspect a candidate `.onnx` without building a session.
     *
     * The sidecar is read and validated in full; the graph is not loaded. That
     * is a deliberate line. Parsing ~99 MB of ONNX takes seconds, which is too
     * slow to sit behind a file chooser, and it is not where the interesting
     * failures are — a sidecar from an older `trailer export` is, and
     * [ModelSpec] catches every one of those. The graph's own consistency with
     * the sidecar is checked when [Inference] first opens it, which is the
     * first time it could possibly matter.
     */
    fun inspect(onnx: Path): Check {
        if (!Files.isRegularFile(onnx)) {
            return Check.Bad("There is no file at $onnx")
        }
        val sidecar = ModelStore.sidecarFor(onnx)
        if (!Files.isRegularFile(sidecar)) {
            return Check.Bad(
                "No sidecar beside this model. `trailer export` writes " +
                    "${sidecar.fileName} next to the .onnx, and it is needed: " +
                    "it carries the window size, the tiling step and the " +
                    "attribution, none of which can be recovered from the graph.")
        }
        val spec = try {
            ModelSpec(Files.readString(sidecar))
        } catch (ex: Exception) {
            // ModelSpec's messages are written for exactly this moment — they
            // say which field and why it matters — so they are shown as-is
            // rather than wrapped in something vaguer.
            return Check.Bad(ex.message ?: "Unreadable sidecar ${sidecar.fileName}")
        }
        return Check.Ok(Summary(
            variant = spec.variant,
            resM = spec.resM,
            windowPx = spec.inputPx,
            tta = spec.tta,
            license = spec.license,
            attribution = spec.attribution,
            megabytes = Files.size(onnx) / 1e6,
        ))
    }
}

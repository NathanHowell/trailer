package com.github.nathanhowell.trailer

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * The sidecar written beside an exported `.onnx`, read as a contract.
 *
 * Every number here is computed by the Python that trained the model and is used
 * rather than re-derived. That is not tidiness — it is the only structural
 * defence this side of the boundary has. The alternative was tried: Kotlin
 * computed the window step from its own copy of the formula, the copy was missing
 * the stride quantisation, and the two disagreed at overlap 0.7 on a stride-2
 * stem. Nothing caught it, because the test asserted Kotlin's own arithmetic and
 * the golden fixtures took `step` as an input rather than checking it.
 *
 * The rule that follows: if Python computes a number that affects how the raster
 * is cut up or reassembled, it belongs in the sidecar and is read from here. Only
 * quantities that depend on the *runtime* raster — how much padding this
 * particular viewport needs, where its windows fall — are computed in Kotlin, and
 * those are checked against generated Python values in `TilerTest`.
 */
class ModelSpec(json: String) {

    private val root: JsonNode = ObjectMapper().readTree(json)

    private fun int(name: String): Int = required(name).let {
        require(it.isNumber) { "$name is not a number in the model sidecar: $it" }
        it.asInt()
    }

    private fun required(name: String): JsonNode {
        val n = root.get(name)
        return n ?: throw IllegalArgumentException(
            "model sidecar has no '$name'; it was written by an older " +
                "`trailer export` than this plugin supports")
    }

    /** Which trained variant this is, e.g. `dem1`. */
    val variant: String = required("variant").asText()

    /** Ground sample distance the model expects on its input, in metres. */
    val resM: Double = required("res_m").asDouble()

    /** Ground sample distance of the output, in metres. */
    val outResM: Double = required("out_res_m").asDouble()

    /** Input window, in input pixels. Fixed: the U-Net needs H, W divisible by 32. */
    val inputPx: Int = int("input_px")

    /** Output window, in output pixels. */
    val outputPx: Int = int("output_px")

    /** Stem stride, i.e. how many input pixels make one output pixel. */
    val stride: Int = int("stride")

    /** Distance between window origins, in input pixels. Not recomputed here. */
    val stepPx: Int = int("step_px")

    /**
     * How the ragged tail of a raster is covered.
     *
     * Required rather than defaulted, and the reason is the whole point of this
     * class. The rule this replaced -- extend the raster, let the last window
     * hang off the end -- produced a false trail along the bottom and right of
     * every raster, because the model read the mirror seam as a tread. A
     * sidecar that does not say which rule its weights expect is a version
     * mismatch, and guessing the old one would silently reinstate the bug.
     */
    val edgeWindows: String = required("edge_windows").asText()

    /** How a viewport smaller than one window is extended to fill it. */
    val padMode: String = required("pad_mode").asText()

    /** Fraction of a window shared with its neighbour; informational. */
    val overlap: Double = required("overlap").asDouble()

    /**
     * Whether the 8-fold D4 average is baked into this graph.
     *
     * Baked, not switchable: ONNX cannot branch on it. Informational here — the
     * caller runs the graph the same way either way — but worth surfacing,
     * because it is the difference between one forward pass per window and eight.
     */
    val tta: Boolean = required("tta").asBoolean()

    /** Output tensor names, in order. The taper is the second one. */
    val outputs: List<String> = required("outputs").map { it.asText() }

    /** SPDX identifier for the weights, which are not under the plugin's licence. */
    val license: String = required("license").asText()

    /**
     * The notice the licence requires to be shown wherever the output is.
     *
     * Required, not optional, and deliberately not defaulted to a constant
     * compiled into the plugin. A model file that has lost its attribution is
     * one this plugin has no right to paint, and inventing the notice here
     * would let a stripped file paint anyway — which is the whole failure it
     * is meant to prevent. It is carried in the sidecar rather than read from
     * the repository because the weights are what gets downloaded.
     */
    val attribution: String = required("attribution").asText()

    init {
        require(license.isNotBlank()) { "model sidecar has a blank licence" }
        require(attribution.isNotBlank()) {
            "model sidecar has a blank attribution; the licence requires one"
        }
        require(inputPx == outputPx * stride) {
            "sidecar is inconsistent: input_px $inputPx is not output_px " +
                "$outputPx x stride $stride"
        }
        require(stepPx in 1..inputPx) {
            "step_px $stepPx is not a usable step for a $inputPx px window"
        }
        require(stepPx % stride == 0) {
            "step_px $stepPx is not a multiple of stride $stride, so windows " +
                "would land between output pixels"
        }
        require(edgeWindows == "flush") {
            "this plugin only implements flush edge windows, sidecar says " +
                "'$edgeWindows'"
        }
        require(padMode == "reflect") {
            "this plugin only implements reflect padding, sidecar says '$padMode'"
        }
        require(outputs.size == 2 && outputs[1] == "window_taper") {
            "expected the model to emit its own blending taper as a second " +
                "output; this one emits $outputs"
        }
    }

    /** Step in *output* pixels, which is what the blender accumulates in. */
    val bodyStepPx: Int get() = stepPx / stride
}

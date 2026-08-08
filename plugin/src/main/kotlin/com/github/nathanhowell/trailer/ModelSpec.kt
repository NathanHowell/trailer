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

    /** How the raster's far edges are extended so windows tile it exactly. */
    val padMode: String = required("pad_mode").asText()

    /** Fraction of a window shared with its neighbour; informational. */
    val overlap: Double = required("overlap").asDouble()

    init {
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
        require(padMode == "reflect") {
            "this plugin only implements reflect padding, sidecar says '$padMode'"
        }
    }

    /** Step in *output* pixels, which is what the blender accumulates in. */
    val bodyStepPx: Int get() = stepPx / stride
}

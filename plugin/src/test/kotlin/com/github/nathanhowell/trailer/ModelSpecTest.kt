package com.github.nathanhowell.trailer

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * `model-dem1.json` is a real sidecar, written by `trailer export` from a real
 * checkpoint. The point of using a captured one rather than a hand-written one is
 * that it is what the Python actually emits — if a field is renamed or dropped on
 * that side, this fails rather than the plugin silently falling back to a default.
 */
class ModelSpecTest {

    private fun fixture(name: String) =
        String(javaClass.getResourceAsStream("/$name")!!.readBytes(), Charsets.UTF_8)

    private val dem1 = ModelSpec(fixture("model-dem1.json"))

    /**
     * The licence fields every real sidecar carries.
     *
     * Spelled once so the hand-written cases below each vary exactly one thing.
     * They are required rather than defaulted, so omitting them here would make
     * every case fail for the wrong reason.
     */
    private val LICENCE = """"license":"CC-BY-SA-4.0","attribution":"(c) test""""

    @Test
    fun `reads a real exported sidecar`() {
        assertEquals("dem1", dem1.variant)
        assertEquals(1.0, dem1.resM, 1e-9)
        assertEquals(1.0, dem1.outResM, 1e-9)
        assertEquals(256, dem1.inputPx)
        assertEquals(256, dem1.outputPx)
        assertEquals(1, dem1.stride)
        assertEquals(128, dem1.stepPx)
        assertEquals("reflect", dem1.padMode)
        assertEquals(false, dem1.tta)
        assertEquals(listOf("trail_probability", "window_taper"), dem1.outputs)
    }

    @Test
    fun `reads the sidecar of a graph with the D4 average baked in`() {
        // Same export, --tta. Everything about tiling is identical; the only
        // difference is that each window costs eight forward passes instead of
        // one, which the caller cannot choose at runtime because ONNX cannot
        // branch on it.
        val tta = ModelSpec(fixture("model-dem1-tta.json"))
        assertEquals(true, tta.tta)
        assertEquals(dem1.stepPx, tta.stepPx)
        assertEquals(dem1.inputPx, tta.inputPx)
    }

    @Test
    fun `refuses a model that does not emit its own taper`() {
        // The taper is the model's second output precisely so that nothing here
        // recomputes it. A graph without one would mean someone had to.
        val e = assertFailsWith<IllegalArgumentException> {
            ModelSpec("""
                {"variant":"old","res_m":1.0,"out_res_m":1.0,
                 "input_px":256,"output_px":256,"stride":1,"overlap":0.5,
                 "step_px":128,"pad_mode":"reflect","tta":false,
                 "outputs":["trail_probability"],$LICENCE}
            """.trimIndent())
        }
        assertTrue(e.message!!.contains("taper"), e.message!!)
    }

    @Test
    fun `takes the step as given rather than deriving it`() {
        // The whole point of the sidecar. Kotlin's own formula was missing the
        // stride quantisation and disagreed with Python at overlap 0.7 on a
        // stride-2 stem: python 152, kotlin 153. There is now no Kotlin formula
        // to disagree with, and the value below is what infer.window_step
        // returns for (512, 0.7, 2).
        val strideTwo = ModelSpec("""
            {"variant":"future05","res_m":0.5,"out_res_m":1.0,
             "input_px":512,"output_px":256,"stride":2,"overlap":0.7,
             "step_px":152,"pad_mode":"reflect","tta":false,
             "outputs":["trail_probability","window_taper"],$LICENCE}
        """.trimIndent())
        assertEquals(152, strideTwo.stepPx, "the number Python computed, unmodified")
        assertEquals(76, strideTwo.bodyStepPx, "step in output pixels")
    }

    @Test
    fun `body step is the step in output pixels`() {
        assertEquals(128, dem1.bodyStepPx, "stride 1, so input and output agree")
    }

    @Test
    fun `rejects a sidecar whose window arithmetic does not close`() {
        // input_px must be output_px * stride. If it is not, either the export
        // is wrong or this plugin has misunderstood the model, and both are worse
        // than refusing to load.
        val e = assertFailsWith<IllegalArgumentException> {
            ModelSpec("""
                {"variant":"bad","res_m":0.5,"out_res_m":1.0,
                 "input_px":500,"output_px":256,"stride":2,"overlap":0.5,
                 "step_px":250,"pad_mode":"reflect","tta":false,
                 "outputs":["trail_probability","window_taper"],$LICENCE}
            """.trimIndent())
        }
        assertTrue(e.message!!.contains("input_px"), e.message!!)
    }

    @Test
    fun `rejects a step that would land windows between output pixels`() {
        // An origin that is not a multiple of the stride accumulates half an
        // output pixel away from where it belongs. It does not crash and it does
        // not look obviously wrong; it just softens everything.
        val e = assertFailsWith<IllegalArgumentException> {
            ModelSpec("""
                {"variant":"bad","res_m":0.5,"out_res_m":1.0,
                 "input_px":512,"output_px":256,"stride":2,"overlap":0.7,
                 "step_px":153,"pad_mode":"reflect","tta":false,
                 "outputs":["trail_probability","window_taper"],$LICENCE}
            """.trimIndent())
        }
        assertTrue(e.message!!.contains("multiple of stride"), e.message!!)
    }

    @Test
    fun `refuses a padding mode it does not implement`() {
        val e = assertFailsWith<IllegalArgumentException> {
            ModelSpec("""
                {"variant":"bad","res_m":1.0,"out_res_m":1.0,
                 "input_px":256,"output_px":256,"stride":1,"overlap":0.5,
                 "step_px":128,"pad_mode":"replicate","tta":false,
                 "outputs":["trail_probability","window_taper"],$LICENCE}
            """.trimIndent())
        }
        assertTrue(e.message!!.contains("replicate"), e.message!!)
    }

    @Test
    fun `says so when a field is missing rather than defaulting`() {
        // A sidecar from an older export is a version mismatch, not a reason to
        // guess. Guessing is how the plugin and Python drifted in the first place.
        val e = assertFailsWith<IllegalArgumentException> {
            ModelSpec("""{"variant":"old","res_m":1.0,"out_res_m":1.0,
                          "input_px":256,"output_px":256}""")
        }
        assertTrue(e.message!!.contains("stride"), e.message!!)
        assertTrue(e.message!!.contains("older"), e.message!!)
    }

    @Test
    fun `carries the weights' licence and attribution`() {
        // The weights are CC BY-SA and the code is MIT; they are different
        // artefacts with different terms, and the sidecar is what a downloaded
        // .onnx has instead of a repository.
        assertEquals("CC-BY-SA-4.0", dem1.license)
        assertTrue(dem1.attribution.contains("OpenStreetMap"), dem1.attribution)
        assertTrue(dem1.attribution.contains("3DEP"), dem1.attribution)
        assertTrue(dem1.attribution.contains("CC BY-SA"), dem1.attribution)
    }

    @Test
    fun `refuses a model that has lost its attribution`() {
        // Not defaulted to a constant compiled in here. Displaying the notice is
        // a condition of the licence, so a file that arrives without one is a
        // file this plugin has no right to paint — and a built-in fallback would
        // let exactly that file paint anyway.
        val e = assertFailsWith<IllegalArgumentException> {
            ModelSpec("""
                {"variant":"bad","res_m":1.0,"out_res_m":1.0,
                 "input_px":256,"output_px":256,"stride":1,"overlap":0.5,
                 "step_px":128,"pad_mode":"reflect","tta":false,
                 "outputs":["trail_probability","window_taper"],
                 "license":"CC-BY-SA-4.0","attribution":"  "}
            """.trimIndent())
        }
        assertTrue(e.message!!.contains("attribution"), e.message!!)
    }

    @Test
    fun `rejects a step of zero, which would loop forever`() {
        assertFailsWith<IllegalArgumentException> {
            ModelSpec("""
                {"variant":"bad","res_m":1.0,"out_res_m":1.0,
                 "input_px":256,"output_px":256,"stride":1,"overlap":1.0,
                 "step_px":0,"pad_mode":"reflect","tta":false,
                 "outputs":["trail_probability","window_taper"],$LICENCE}
            """.trimIndent())
        }
    }
}

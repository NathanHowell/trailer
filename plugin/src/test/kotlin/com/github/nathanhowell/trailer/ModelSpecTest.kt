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
             "step_px":152,"pad_mode":"reflect"}
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
                 "step_px":250,"pad_mode":"reflect"}
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
                 "step_px":153,"pad_mode":"reflect"}
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
                 "step_px":128,"pad_mode":"replicate"}
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
    fun `rejects a step of zero, which would loop forever`() {
        assertFailsWith<IllegalArgumentException> {
            ModelSpec("""
                {"variant":"bad","res_m":1.0,"out_res_m":1.0,
                 "input_px":256,"output_px":256,"stride":1,"overlap":1.0,
                 "step_px":0,"pad_mode":"reflect"}
            """.trimIndent())
        }
    }
}

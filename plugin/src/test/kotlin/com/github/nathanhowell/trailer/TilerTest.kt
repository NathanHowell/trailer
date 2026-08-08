package com.github.nathanhowell.trailer

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * Parity against `trailer.infer`, not against this file's own idea of correctness.
 *
 * `golden.json` is produced by `trailer golden` (src/trailer/golden.py), which
 * calls the actual Python functions the plugin is reimplementing, and the Maven
 * build regenerates it at generate-test-resources. A test that only checked
 * Kotlin against Kotlin would pass happily while the two drifted apart.
 */
class TilerTest {

    private val golden: JsonNode = ObjectMapper().readTree(
        javaClass.getResourceAsStream("/golden.json")
            ?: error("golden.json missing; run `uv run trailer golden`")
    )

    private fun JsonNode.floats(): FloatArray =
        FloatArray(size()) { get(it).floatValue() }

    private fun assertClose(expected: FloatArray, actual: FloatArray, tol: Float, what: String) {
        assertEquals(expected.size, actual.size, "$what: length")
        var worst = 0.0f
        var at = -1
        for (i in expected.indices) {
            val d = kotlin.math.abs(expected[i] - actual[i])
            if (d > worst) { worst = d; at = i }
        }
        assertTrue(worst <= tol) {
            "$what: max |diff| = $worst at index $at (tolerance $tol); " +
                "expected ${expected[at]}, got ${actual[at]}"
        }
    }

    @Test
    fun `hann taper matches torch`() {
        val size = golden["hann_size"].intValue()
        assertClose(golden["hann"].floats(), Tiler.hann2d(size), 1e-6f, "hann2d")
    }

    @Test
    fun `hann taper is strictly positive`() {
        // The +2-then-trim exists so the outer ring is not zero. If someone
        // "simplifies" it to a plain Hann window, the edges die and this catches it.
        val t = Tiler.hann2d(16)
        assertTrue(t.all { it > 0f }) { "taper has non-positive entries" }
        assertTrue(t.min() >= 1e-3f) { "taper floor not applied" }
    }

    @Test
    fun `padding and origins match python`() {
        for (c in golden["pad_cases"]) {
            val n = c["n"].intValue()
            val tile = c["tile"].intValue()
            val step = c["step"].intValue()
            val pad = Tiler.padAmount(n, tile, step)
            assertEquals(c["pad"].intValue(), pad, "padAmount(n=$n, tile=$tile, step=$step)")

            val expected = IntArray(c["origins"].size()) { c["origins"][it].intValue() }
            val actual = Tiler.origins(n + pad, tile, step)
            assertEquals(expected.toList(), actual.toList(),
                "origins(n=$n, tile=$tile, step=$step)")

            // The point of padding: the last window must land exactly on the end.
            if (actual.isNotEmpty()) {
                assertEquals(n + pad, actual.last() + tile,
                    "windows do not cover the padded axis for n=$n")
            }
        }
    }

    @Test
    fun `reflect padding matches torch`() {
        val r = golden["reflect"]
        val out = Tiler.reflectPad(
            r["src"].floats(), r["h"].intValue(), r["w"].intValue(),
            r["pad_h"].intValue(), r["pad_w"].intValue()
        )
        assertClose(r["out"].floats(), out, 1e-6f, "reflectPad")
    }

    @Test
    fun `blended output matches python`() {
        val b = golden["blend"]
        val tile = b["tile"].intValue()
        val blender = Tiler.Blender(b["h"].intValue(), b["w"].intValue(), tile)
        for (i in 0 until b["windows"].size()) {
            val w = b["windows"][i]
            blender.add(b["probs"][i].floats(), w["row"].intValue(), w["col"].intValue())
        }
        assertClose(b["out"].floats(), blender.result(), 1e-5f, "blend")
    }

    @Test
    fun `window layout matches python`() {
        val b = golden["blend"]
        val expected = (0 until b["windows"].size()).map {
            Tiler.Window(b["windows"][it]["row"].intValue(), b["windows"][it]["col"].intValue())
        }
        val actual = Tiler.windows(
            b["h"].intValue(), b["w"].intValue(), b["tile"].intValue(), b["step"].intValue()
        )
        assertEquals(expected, actual, "window layout")
    }

    @Test
    fun `blending a constant field reproduces the constant`() {
        // Partition-of-unity check, independent of the golden file: if the taper
        // normalisation is wrong, a uniform prediction comes back non-uniform and
        // the whole overlay picks up a grid of seams.
        val tile = 8
        val step = 4
        val h = 24
        val w = 24
        val blender = Tiler.Blender(h, w, tile)
        val flat = FloatArray(tile * tile) { 0.375f }
        for (win in Tiler.windows(h, w, tile, step)) blender.add(flat, win.row, win.col)
        val out = blender.result()
        assertClose(FloatArray(h * w) { 0.375f }, out, 1e-5f, "constant field")
    }

    // The step is no longer computed here, so there is nothing left to test: it
    // arrives as a number in the model sidecar. The test that used to live here
    // asserted Kotlin's own arithmetic against Kotlin's own expectations and
    // agreed with itself while disagreeing with Python. See ModelSpecTest.
}

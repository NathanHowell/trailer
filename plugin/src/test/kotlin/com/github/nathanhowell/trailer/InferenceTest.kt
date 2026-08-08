package com.github.nathanhowell.trailer

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * End-to-end parity: a whole raster through onnxruntime under the Kotlin tiler,
 * against what `trailer.infer.predict` produced from the same weights.
 *
 * This is the test the plugin actually needed. `TilerTest` checks each step
 * against Python separately, and every one of them can pass while the composed
 * result is wrong — windows blended in the wrong place, the crop taken from the
 * wrong corner, the taper read off the wrong output tensor. Those are not
 * hypothetical failure modes; a silent raster misregistration of exactly that
 * kind survived fourteen builds of this project.
 *
 * The weights are a fixed 3x3 convolution rather than the trained network, so
 * the fixture is 2.6 KB instead of 99 MB. Nothing is lost by that: what is under
 * test is the tiling and the session plumbing, and the trained weights' export
 * fidelity is a separate question already answered on the Python side. The stub
 * has a receptive field and is asymmetric in both axes precisely so a
 * misregistration shows up as a wrong *value* rather than only at the edges.
 */
class InferenceTest {

    private val golden: JsonNode = ObjectMapper().readTree(
        javaClass.getResourceAsStream("/golden.json")
            ?: error("golden.json missing; run `uv run trailer golden`")
    )

    private fun resourceBytes(name: String): ByteArray =
        javaClass.getResourceAsStream("/$name")?.readBytes()
            ?: error("$name missing; run `uv run trailer golden`")

    private fun JsonNode.floats(): FloatArray =
        FloatArray(size()) { get(it).floatValue() }

    private fun open(case: JsonNode): Inference {
        val spec = ModelSpec(String(resourceBytes(case["sidecar"].asText()),
                                    Charsets.UTF_8))
        val env = OrtEnvironment.getEnvironment()
        val session = env.createSession(resourceBytes(case["onnx"].asText()),
                                        OrtSession.SessionOptions())
        return Inference(session, spec, env)
    }

    @Test
    fun `whole-tile inference matches python`() {
        val cases = golden["tiles"]
        assertTrue(cases.size() >= 2, "expected the dem1 and stride-2 cases")

        for (case in cases) {
            val variant = case["variant"].asText()
            val h = case["h"].intValue()
            val w = case["w"].intValue()
            val expected = case["out"].floats()

            val actual = open(case).use { it.run(case["z"].floats(), h, w) }

            assertEquals(case["out_h"].intValue() * case["out_w"].intValue(),
                         actual.size,
                         "$variant: output should be the body grid, not the " +
                             "padded one")
            assertEquals(expected.size, actual.size, "$variant: length")

            var worst = 0.0f
            var at = -1
            for (i in expected.indices) {
                val d = kotlin.math.abs(expected[i] - actual[i])
                if (d > worst) { worst = d; at = i }
            }
            // Float32 throughout on both sides, and the accumulation order
            // differs, so exact equality is not the right ask. 1e-5 is far
            // tighter than any real disagreement would be: a window one pixel
            // out moves values by O(0.1).
            assertTrue(worst <= 1e-5f) {
                "$variant: max |diff| = $worst at index $at " +
                    "(row ${at / case["out_w"].intValue()}, " +
                    "col ${at % case["out_w"].intValue()}); " +
                    "expected ${expected[at]}, got ${actual[at]}"
            }
        }
    }

    @Test
    fun `the stride-2 path divides window origins down to the body grid`() {
        // No stride-2 variant is deployable, so this arithmetic would otherwise
        // ship having never run. It is also exactly where the step bug lived:
        // Kotlin's own formula lacked the stride quantisation and put windows
        // half an output pixel from where they belonged.
        val case = golden["tiles"].first { it["stride"].intValue() == 2 }
        val h = case["h"].intValue()
        val w = case["w"].intValue()

        val actual = open(case).use { it.run(case["z"].floats(), h, w) }

        assertEquals(h / 2, case["out_h"].intValue(), "python halved the height")
        assertEquals((h / 2) * (w / 2), actual.size,
                     "output should be half the input on each axis")
    }

    @Test
    fun `output is not accidentally uniform`() {
        // Cheap insurance against the whole comparison being two constant
        // rasters agreeing with each other, which would pass every assertion
        // above while testing nothing at all.
        val case = golden["tiles"].first()
        val out = open(case).use { it.run(case["z"].floats(),
                                          case["h"].intValue(),
                                          case["w"].intValue()) }
        val lo = out.min()
        val hi = out.max()
        assertTrue(hi - lo > 0.05f,
                   "probabilities span only $lo..$hi; the fixture is not " +
                       "exercising anything")
        assertTrue(lo >= 0f && hi <= 1f, "probabilities outside 0..1: $lo..$hi")
    }

    @Test
    fun `refuses a raster too small to reflect-pad`() {
        val case = golden["tiles"].first()
        val e = org.junit.jupiter.api.assertThrows<IllegalArgumentException> {
            open(case).use { it.run(FloatArray(4 * 4), 4, 4) }
        }
        assertTrue(e.message!!.contains("too small"), e.message!!)
    }
}

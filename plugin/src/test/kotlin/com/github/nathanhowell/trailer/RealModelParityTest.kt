package com.github.nathanhowell.trailer

import com.fasterxml.jackson.databind.ObjectMapper
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
// JUnit's rather than kotlin.test's: only these take a lazily-built message,
// and the diff message below is worth not formatting on every passing assert.
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue

/**
 * The trained model, on real elevation, through the Java runtime.
 *
 * `InferenceTest` runs a 2.6 KB stand-in, which is the right trade for a
 * fixture that lives in git: it exercises the tiling and the session plumbing,
 * and those are the parts the plugin reimplements. It cannot exercise the
 * trained graph — a ResNet-34 encoder, the im2col median filter that replaces
 * `scipy.ndimage.median_filter`, ~99 MB of initialisers, a 256 px window rather
 * than 32. Whether **onnxruntime's Java runtime** handles all of that is a
 * different question from whether Python's does, and no stub can answer it.
 *
 * Skipped unless pointed at a fixture, because the fixture cannot be committed:
 *
 * ```
 * uv run trailer parity --checkpoint runs/full/best.pt \
 *     --tile data/tiles/abandoned_south --out /tmp/parity
 * mvn -f plugin/pom.xml test -Dtest=RealModelParityTest -Dtrailer.parity=/tmp/parity
 * ```
 *
 * A skipped test is a weak thing, so this is deliberately not the only defence:
 * everything it covers that *can* be tested cheaply already is. What it adds is
 * the one claim nothing else can make.
 */
class RealModelParityTest {

    private fun fixture(): Path? =
        System.getProperty("trailer.parity")?.let { Paths.get(it) }
            ?.takeIf { Files.isDirectory(it) }

    private fun floats(p: Path, n: Int): FloatArray {
        val bytes = Files.readAllBytes(p)
        assertEquals(n * 4, bytes.size, "${p.fileName}: expected $n float32 values")
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(n).also { buf.asFloatBuffer().get(it) }
    }

    @Test
    fun `the trained model matches infer predict through the java runtime`() {
        val dir = fixture()
        assumeTrue(dir != null,
                   "set -Dtrailer.parity=<dir> from `trailer parity` to run this")
        dir!!

        val meta = ObjectMapper().readTree(Files.readString(dir.resolve("parity.json")))
        val h = meta["h"].intValue()
        val w = meta["w"].intValue()
        val outH = meta["out_h"].intValue()
        val outW = meta["out_w"].intValue()

        val z = floats(dir.resolve("z.f32"), h * w)
        val expected = floats(dir.resolve("expected.f32"), outH * outW)

        val actual = Inference.open(dir.resolve("model.onnx"),
                                    dir.resolve("model.json"))
            .use { it.run(z, h, w) }

        assertEquals(expected.size, actual.size, "output length")

        // NaN is the nodata contract, and it has to survive identically on both
        // sides: a NaN that became 0 would read as "confidently not a trail"
        // exactly where there is no ground truth at all.
        var worst = 0.0f
        var at = -1
        var nanMismatch = 0
        for (i in expected.indices) {
            val e = expected[i]
            val a = actual[i]
            if (e.isNaN() != a.isNaN()) { nanMismatch++; continue }
            if (e.isNaN()) continue
            val d = kotlin.math.abs(e - a)
            if (d > worst) { worst = d; at = i }
        }
        assertEquals(0, nanMismatch, "nodata disagreed on $nanMismatch cells")

        // Looser than the stub's 1e-5: this graph is thousands of ops deep and
        // the two runtimes reassociate float arithmetic differently. Python
        // measured onnx-vs-torch at 6.5e-5 on this same model, so anything at
        // that order is the runtimes, not the tiling. A misregistration would
        // land at O(0.1).
        assertTrue(worst <= 1e-3f) {
            "max |diff| = $worst at index $at (row ${at / outW}, col ${at % outW}); " +
                "expected ${expected[at]}, got ${actual[at]}"
        }

        // Guard against the whole comparison being two constant rasters: a model
        // that emitted the same number everywhere would pass everything above.
        val finite = actual.filter { !it.isNaN() }
        assertTrue(finite.isNotEmpty(), "everything was NaN")
        assertTrue(finite.max() - finite.min() > 0.05f,
                   "probabilities span only ${finite.min()}..${finite.max()}")
        println("real-model parity: max |diff| = $worst over ${expected.size} " +
                "cells, mean ${finite.average()}")
    }
}

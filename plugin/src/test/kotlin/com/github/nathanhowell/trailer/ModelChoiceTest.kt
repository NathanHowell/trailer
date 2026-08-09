package com.github.nathanhowell.trailer

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Uses the real stub export from `trailer golden` — an actual `.onnx` with an
 * actual sidecar beside it — so "a valid model" means what Python produces
 * rather than what this test imagines.
 */
class ModelChoiceTest {

    @TempDir
    lateinit var dir: Path

    private fun copyResource(name: String, to: Path) {
        Files.write(to, javaClass.getResourceAsStream("/$name")!!.readBytes())
    }

    /** A complete, valid model pair on disk. */
    private fun goodModel(): Path {
        val onnx = dir.resolve("trailer.onnx")
        copyResource("stub-dem1.onnx", onnx)
        copyResource("stub-dem1.json", dir.resolve("trailer.json"))
        return onnx
    }

    private fun bad(check: ModelChoice.Check): String {
        assertTrue(check is ModelChoice.Check.Bad, "expected a rejection, got $check")
        return (check as ModelChoice.Check.Bad).reason
    }

    @Test
    fun `accepts a real exported model`() {
        val ok = ModelChoice.inspect(goodModel())
        assertTrue(ok is ModelChoice.Check.Ok, "rejected a real export: $ok")
        val s = (ok as ModelChoice.Check.Ok).summary
        assertEquals("dem1", s.variant)
        assertEquals(1.0, s.resM, 1e-9)
        assertEquals(32, s.windowPx, "the stub export's window")
        assertEquals(false, s.tta)
        assertEquals("CC-BY-SA-4.0", s.license)
        assertTrue(s.attribution.contains("OpenStreetMap"), s.attribution)
        assertTrue(s.megabytes > 0.0)
    }

    @Test
    fun `describes what the mapper is about to run`() {
        val s = (ModelChoice.inspect(goodModel()) as ModelChoice.Check.Ok).summary
        val d = s.describe()
        assertTrue(d.contains("dem1"), d)
        assertTrue(d.contains("1.0 m"), d)
        assertTrue(d.contains("32 px"), d)
        assertTrue(!d.contains("augmentation"), "this export has no TTA: $d")
    }

    @Test
    fun `says so when the model has no sidecar`() {
        val onnx = dir.resolve("lonely.onnx")
        copyResource("stub-dem1.onnx", onnx)
        val reason = bad(ModelChoice.inspect(onnx))
        assertTrue(reason.contains("lonely.json"), reason)
        assertTrue(reason.contains("trailer export"), reason)
    }

    @Test
    fun `says so when there is no file at all`() {
        val reason = bad(ModelChoice.inspect(dir.resolve("absent.onnx")))
        assertTrue(reason.contains("absent.onnx"), reason)
    }

    @Test
    fun `rejects a directory picked by mistake`() {
        val reason = bad(ModelChoice.inspect(dir))
        assertTrue(reason.contains("no file"), reason)
    }

    @Test
    fun `passes ModelSpec's own words through for an outdated export`() {
        // The failure this whole check exists for. ModelSpec's messages name the
        // field and say why it matters, and they are written for exactly this
        // moment -- wrapping them in "invalid model file" would throw away the
        // only part that tells the mapper what to do.
        val onnx = dir.resolve("old.onnx")
        copyResource("stub-dem1.onnx", onnx)
        Files.writeString(dir.resolve("old.json"), """
            {"variant":"old","res_m":1.0,"out_res_m":1.0,
             "input_px":256,"output_px":256}
        """.trimIndent())
        val reason = bad(ModelChoice.inspect(onnx))
        assertTrue(reason.contains("stride"), reason)
        assertTrue(reason.contains("older"), reason)
    }

    @Test
    fun `rejects a model whose sidecar has lost its attribution`() {
        val onnx = dir.resolve("nolicence.onnx")
        copyResource("stub-dem1.onnx", onnx)
        Files.writeString(dir.resolve("nolicence.json"), """
            {"variant":"dem1","res_m":1.0,"out_res_m":1.0,
             "input_px":256,"output_px":256,"stride":1,"overlap":0.5,
             "step_px":128,"pad_mode":"reflect","tta":false,
             "outputs":["trail_probability","window_taper"],
             "license":"CC-BY-SA-4.0","attribution":""}
        """.trimIndent())
        assertTrue(bad(ModelChoice.inspect(onnx)).contains("attribution"))
    }

    @Test
    fun `rejects a sidecar that is not json at all`() {
        // Picking the .onnx of one export next to the .json of something else
        // entirely should not surface as a stack trace.
        val onnx = dir.resolve("mixed.onnx")
        copyResource("stub-dem1.onnx", onnx)
        Files.writeString(dir.resolve("mixed.json"), "not json {{{")
        assertTrue(bad(ModelChoice.inspect(onnx)).isNotEmpty())
    }
}

package com.github.nathanhowell.trailer

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FetchPlanTest {

    /** A real sidecar: dem1, 1 m, 256 px window. */
    private val spec = ModelSpec(
        String(javaClass.getResourceAsStream("/model-dem1.json")!!.readBytes(),
               Charsets.UTF_8))

    private fun view(w: Double, h: Double, cx: Double = 300_000.0,
                     cy: Double = 4_200_000.0) =
        Dem3dep.Bounds(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @Test
    fun `pixel size is exactly the model's resolution`() {
        // The whole point. 3DEP will resample anything to any grid asked for and
        // return something plausible, so a fetch at 1.03 m would look fine and be
        // feeding the model a scale it never trained on.
        for (span in listOf(300.0, 1000.0, 1517.0, 4096.0, 7999.5)) {
            val p = FetchPlan.forView(view(span, span * 0.7), spec)
            assertEquals(1.0, p.pixelM, 1e-12, "span $span")
            assertEquals(p.groundWidthM, p.bounds.width, 1e-9, "span $span width")
            assertEquals(p.groundHeightM, p.bounds.height, 1e-9, "span $span height")
        }
    }

    @Test
    fun `a small view is grown outward, not sampled finer`() {
        // A mapper zoomed in on one switchback still gets an answer, and gets it
        // by fetching more ground at 1 m rather than the same ground at 0.2 m --
        // which the model has never seen and 3DEP would have invented.
        val p = FetchPlan.forView(view(80.0, 60.0), spec)
        assertEquals(256, p.width)
        assertEquals(256, p.height)
        assertEquals(1.0, p.pixelM, 1e-12)
        assertTrue(p.bounds.width > 80.0, "should have grown past the viewport")
    }

    @Test
    fun `the grown window stays centred on the view`() {
        // Off-centre growth would put the mapper's actual area of interest near
        // an edge, which is where tiled inference is weakest.
        val v = view(80.0, 60.0, cx = 512_345.0, cy = 4_012_345.0)
        val p = FetchPlan.forView(v, spec)
        assertEquals((v.minX + v.maxX) / 2, (p.bounds.minX + p.bounds.maxX) / 2, 1e-9)
        assertEquals((v.minY + v.maxY) / 2, (p.bounds.minY + p.bounds.maxY) / 2, 1e-9)
    }

    @Test
    fun `covers the whole viewport, never less`() {
        // Rounding down would leave a strip of the view unpainted, which reads as
        // "the model found nothing there" rather than "nothing was asked".
        for (span in listOf(300.4, 1000.9, 2047.001)) {
            val v = view(span, span)
            val p = FetchPlan.forView(v, spec)
            assertTrue(p.bounds.minX <= v.minX + 1e-9, "span $span left")
            assertTrue(p.bounds.maxX >= v.maxX - 1e-9, "span $span right")
        }
    }

    @Test
    fun `refuses a view too wide to serve at model resolution`() {
        // 8000 px at 1 m is 8 km. Past that the only way to answer is to coarsen,
        // and coarsening is the failure Dem3dep's pixel-size guard exists to
        // catch -- doing it deliberately here would be worse, not better.
        val e = assertFailsWith<FetchPlan.TooLarge> {
            FetchPlan.forView(view(9_000.0, 1_000.0), spec)
        }
        assertTrue(e.message!!.contains("8.0 km"), e.message!!)
        assertTrue(e.message!!.contains("Zoom in"), e.message!!)
    }

    @Test
    fun `refuses a view too tall as well as too wide`() {
        assertFailsWith<FetchPlan.TooLarge> {
            FetchPlan.forView(view(1_000.0, 9_000.0), spec)
        }
    }

    @Test
    fun `accepts a projection whose metres are ground metres`() {
        // UTM at the centre of a zone: essentially 1:1, and the small scale
        // factor UTM does carry (0.9996 on the central meridian) is well inside
        // the tolerance.
        FetchPlan.checkTrueScale(1000.0, 1000.0, "EPSG:32611")
        FetchPlan.checkTrueScale(1000.0, 999.6, "EPSG:32611")
        FetchPlan.checkTrueScale(1000.0, 1004.0, "EPSG:32611")
    }

    @Test
    fun `refuses web mercator, whose metres are not ground metres`() {
        // At 38 degrees N the stretch is 1/cos(38) = 1.27. A fetch planned in
        // those units would ask 3DEP for a grid that is not 1 m on the ground,
        // and nothing downstream could tell: the raster is the right shape and
        // the elevations are real, the trails are just the wrong size.
        val e = assertFailsWith<FetchPlan.Distorted> {
            FetchPlan.checkTrueScale(1000.0, 788.0, "EPSG:3857")
        }
        assertTrue(e.message!!.contains("EPSG:3857"), e.message!!)
        assertTrue(e.message!!.contains("UTM"), e.message!!)
    }

    @Test
    fun `the scale check is symmetric about one`() {
        // Stretched and squashed are equally wrong; a check that only caught one
        // direction would pass on whichever projection happened to err the other
        // way.
        assertFailsWith<FetchPlan.Distorted> {
            FetchPlan.checkTrueScale(1000.0, 1020.0, "stretched")
        }
        assertFailsWith<FetchPlan.Distorted> {
            FetchPlan.checkTrueScale(1000.0, 980.0, "squashed")
        }
    }

    @Test
    fun `the limit is in ground metres, so it scales with model resolution`() {
        // A future 0.5 m model would hit the same 8000-pixel service limit at
        // half the ground distance. Nothing here hardcodes 8 km.
        val half = ModelSpec("""
            {"variant":"future05","res_m":0.5,"out_res_m":1.0,
             "input_px":512,"output_px":256,"stride":2,"overlap":0.5,
             "step_px":256,"edge_windows":"flush","pad_mode":"reflect","tta":false,
             "outputs":["trail_probability","window_taper"],
             "license":"CC-BY-SA-4.0","attribution":"(c) test"}
        """.trimIndent())
        val p = FetchPlan.forView(view(3_000.0, 3_000.0), half)
        assertEquals(0.5, p.pixelM, 1e-12)
        assertEquals(6000, p.width, "3 km at 0.5 m")

        val e = assertFailsWith<FetchPlan.TooLarge> {
            FetchPlan.forView(view(5_000.0, 5_000.0), half)
        }
        assertTrue(e.message!!.contains("4.0 km"), e.message!!)
    }
}

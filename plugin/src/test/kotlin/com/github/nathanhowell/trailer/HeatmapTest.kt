package com.github.nathanhowell.trailer

import org.junit.jupiter.api.Test
import java.awt.Color
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The overlay is the whole product — the plugin paints probability and a human
 * decides. So the ramp's legibility properties are asserted here rather than left
 * to whoever looks at a screenshot: a ramp that hides the weak end hides exactly
 * the faint trails the model exists to find.
 */
class HeatmapTest {

    private fun rgb(argb: Int) = Triple(
        argb ushr 16 and 0xFF, argb ushr 8 and 0xFF, argb and 0xFF)

    private fun alpha(argb: Int) = argb ushr 24 and 0xFF

    /** Rec. 709 relative luminance, the channel a colour-blind reader still has. */
    private fun luminance(argb: Int): Double {
        val (r, g, b) = rgb(argb)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    @Test
    fun `matches matplotlib plasma at its anchors`() {
        // Straight from matplotlib 'plasma' sampled at eighths. Pinned so a later
        // tweak to the ramp is a deliberate act rather than a drifting constant.
        val expected = listOf(
            Triple(13, 8, 135), Triple(76, 2, 161), Triple(126, 3, 168),
            Triple(170, 35, 149), Triple(204, 71, 120), Triple(230, 108, 92),
            Triple(248, 149, 64), Triple(253, 197, 39), Triple(240, 249, 33))
        for ((i, want) in expected.withIndex()) {
            assertEquals(want, rgb(Heatmap.ramp(i / 8f)), "anchor $i")
        }
    }

    @Test
    fun `the painted range never reaches down into shadow`() {
        // Plasma's own floor is luminance 18, indistinguishable from the dark side
        // of a hillshade once alpha is applied. What the layer actually paints
        // starts a quarter of the way up.
        assertEquals(rgb(Heatmap.ramp(Heatmap.RAMP_FLOOR)), rgb(Heatmap.colour(0f)))
        assertEquals(rgb(Heatmap.ramp(1f)), rgb(Heatmap.colour(1f)))
        assertTrue(luminance(Heatmap.colour(0f)) > 35,
                   "the weakest painted colour is too dark: ${luminance(Heatmap.colour(0f))}")
    }

    @Test
    fun `lightness increases monotonically across the whole ramp`() {
        // This is the accessibility property that matters. Hue is a redundant
        // second channel; confidence has to survive being read as brightness
        // alone, or the ramp is unreadable to a colour-blind mapper.
        var prev = -1.0
        for (i in 0 until Heatmap.LEVELS) {
            val l = luminance(Heatmap.colour(i / (Heatmap.LEVELS - 1f)))
            assertTrue(l >= prev - 0.5, "luminance dipped at $i: $prev -> $l")
            prev = l
        }
        assertTrue(luminance(Heatmap.colour(1f)) - luminance(Heatmap.colour(0f)) > 150,
                   "the ramp should span most of the available lightness range")
    }

    @Test
    fun `the ramp never lands in the green band`() {
        // The backdrop is aerial imagery of vegetated terrain or a grey hillshade.
        // A green overlay disappears into the first. Yellow (~60 degrees) is fine
        // and is where the ramp tops out; 75..160 is where it must not go.
        for (i in 0 until Heatmap.LEVELS) {
            val (r, g, b) = rgb(Heatmap.colour(i / (Heatmap.LEVELS - 1f)))
            val hue = Color.RGBtoHSB(r, g, b, null)[0] * 360f
            assertTrue(hue < 75f || hue > 160f,
                       "entry $i is green: hue $hue from ($r,$g,$b)")
        }
    }

    // ------------------------------------------------------------- palette

    @Test
    fun `nothing below the threshold is painted`() {
        val p = Heatmap.Palette(0.4f, 1f)
        for (i in 0 until Heatmap.index(0.4f)) {
            assertEquals(0, p.argb[i], "index $i should be fully transparent")
        }
        assertTrue(alpha(p.argb[Heatmap.index(0.4f)]) > 0, "the threshold itself paints")
    }

    @Test
    fun `anything shown is visible, and more confident is more opaque`() {
        val p = Heatmap.Palette(0.2f, 1f)
        val first = Heatmap.index(0.2f)
        // The floor is not zero: fading in from nothing would make the top half of
        // the slider's travel do its work invisibly.
        assertTrue(alpha(p.argb[first]) >= Math.round(Heatmap.ALPHA_MIN * 255f) - 1,
                   "just-visible pixels must actually be visible")
        var prev = -1
        for (i in first until Heatmap.LEVELS) {
            val a = alpha(p.argb[i])
            assertTrue(a >= prev, "alpha dipped at $i")
            prev = a
        }
        assertEquals(Math.round(Heatmap.ALPHA_MAX * 255f), alpha(p.argb[Heatmap.LEVELS - 1]))
    }

    @Test
    fun `a colour means the same probability whatever the threshold is`() {
        // The mapper is expected to sweep the threshold. If the ramp renormalised
        // into the visible range, every pixel would change colour as the slider
        // moved and "orange is about 0.75" would stop being true.
        val low = Heatmap.Palette(0.1f, 1f)
        val high = Heatmap.Palette(0.6f, 1f)
        for (i in Heatmap.index(0.6f) until Heatmap.LEVELS) {
            assertEquals(rgb(low.argb[i]), rgb(high.argb[i]), "colour moved at index $i")
            assertEquals(low.argb[i], high.argb[i], "alpha moved at index $i")
        }
    }

    @Test
    fun `opacity scales alpha and leaves colour alone`() {
        val full = Heatmap.Palette(0.0f, 1.0f)
        val half = Heatmap.Palette(0.0f, 0.5f)
        val i = Heatmap.index(0.8f)
        assertEquals(rgb(full.argb[i]), rgb(half.argb[i]))
        assertTrue(Math.abs(alpha(full.argb[i]) / 2 - alpha(half.argb[i])) <= 1,
                   "half opacity should halve alpha: " +
                       "${alpha(full.argb[i])} -> ${alpha(half.argb[i])}")
        assertEquals(0, alpha(Heatmap.Palette(0f, 0f).argb[i]), "opacity 0 paints nothing")
    }

    @Test
    fun `nodata is transparent even at threshold zero`() {
        val p = Heatmap.Palette(0.0f, 1.0f)
        assertTrue(alpha(p.argb[0]) > 0, "at threshold 0 even p=0 paints...")
        assertEquals(0, p.lookup(Float.NaN), "...but NaN is nodata, not p=0")
    }

    @Test
    fun `probabilities outside 0 to 1 clamp instead of throwing`() {
        val p = Heatmap.Palette(0.0f, 1.0f)
        assertEquals(p.argb[0], p.lookup(-0.5f))
        assertEquals(p.argb[Heatmap.LEVELS - 1], p.lookup(1.5f))
    }

    // -------------------------------------------------------------- render

    @Test
    fun `renders a raster through the table`() {
        val prob = floatArrayOf(0.0f, 0.5f, Float.NaN, 1.0f)
        val p = Heatmap.Palette(0.25f, 1f)
        val img = Heatmap.render(prob, 2, 2, p)
        assertEquals(2, img.width)
        assertEquals(2, img.height)
        assertEquals(0, img.getRGB(0, 0), "below threshold")
        assertEquals(p.argb[Heatmap.index(0.5f)], img.getRGB(1, 0))
        assertEquals(0, img.getRGB(0, 1), "nodata")
        assertEquals(p.argb[Heatmap.LEVELS - 1], img.getRGB(1, 1))
    }

    @Test
    fun `render refuses a raster that does not match its dimensions`() {
        assertFailsWith<IllegalArgumentException> {
            Heatmap.render(FloatArray(5), 2, 2, Heatmap.Palette(0.5f, 1f))
        }
    }
}

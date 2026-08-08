package com.github.nathanhowell.trailer

import java.awt.image.BufferedImage
import java.awt.image.DataBufferInt

/**
 * Turns a probability raster into a reviewable overlay.
 *
 * The product decision this file serves: the plugin never creates geometry. It
 * paints what the model believes and a human decides what is a trail. That makes
 * legibility a correctness property, not decoration — a ramp that hides the weak
 * end is a ramp that loses the faint trails this model exists to find.
 *
 * Everything is driven off a 256-entry ARGB lookup table. Colour and alpha are
 * a pure function of probability, so the per-pixel cost of a repaint is one
 * quantise, one array read and one array write; no colour arithmetic happens per
 * pixel. The table is rebuilt only when the mapper moves the threshold or opacity
 * slider — 256 entries, once per drag event, against a viewport that is millions
 * of pixels.
 */
object Heatmap {

    /** Entries in the lookup table; also the quantisation of probability. */
    const val LEVELS = 256

    /**
     * Alpha at the bottom and top of the visible range, before layer opacity.
     *
     * The floor is not zero, and it is high. Anything at or above the threshold is
     * something the mapper asked to see, so it has to survive the worst backdrop
     * it can land on. A low-confidence pixel at 0.35 alpha over the black band of
     * a hillshade is invisible — checked by rendering it, not by reasoning about
     * it. Alpha therefore carries less of the confidence signal than it could;
     * lightness and hue carry the rest.
     */
    const val ALPHA_MIN = 0.55f
    const val ALPHA_MAX = 0.92f

    /**
     * Where in [PLASMA] the visible range starts.
     *
     * Plasma's bottom is (13, 8, 135), luminance 18 — as dark as the shadow side
     * of a hillshade, so a weak detection painted with it reads as terrain rather
     * than as overlay. Starting a quarter of the way up trades a little of the
     * ramp's span for a floor that is unambiguously not shadow.
     */
    const val RAMP_FLOOR = 0.25f

    /**
     * Matplotlib's `plasma`, sampled at eighths and interpolated between.
     *
     * Chosen over the obvious heat ramp for three reasons that all matter here:
     *
     * - Its lightness increases monotonically from end to end, so confidence
     *   survives being read as brightness alone. That is what makes it legible to
     *   a colour-blind mapper: hue is a second, redundant channel rather than the
     *   only one. Pinned by a test.
     * - It contains no green. The backdrop is aerial imagery of vegetated terrain
     *   or a grey hillshade, and a green overlay disappears into the first.
     * - Its low end is a deep blue rather than black, so weak detections read as
     *   overlay rather than as terrain shadow on a hillshade.
     */
    private val PLASMA = intArrayOf(
        0x0D0887, 0x4C02A1, 0x7E03A8, 0xAA2395, 0xCC4778,
        0xE66C5C, 0xF89540, 0xFDC527, 0xF0F921,
    )

    /**
     * The overlay colour for a probability, `t` clamped to 0..1.
     *
     * Maps 0..1 onto [RAMP_FLOOR]..1 of the underlying ramp, so the darkest colour
     * ever painted is still clearly an overlay.
     */
    fun colour(t: Float): Int = ramp(RAMP_FLOOR + (1f - RAMP_FLOOR) * t.coerceIn(0f, 1f))

    /** Linear interpolation through [PLASMA], `t` clamped to 0..1. */
    fun ramp(t: Float): Int {
        val u = t.coerceIn(0f, 1f) * (PLASMA.size - 1)
        val i = u.toInt().coerceAtMost(PLASMA.size - 2)
        val f = u - i
        val a = PLASMA[i]
        val b = PLASMA[i + 1]
        val r = lerp(a ushr 16 and 0xFF, b ushr 16 and 0xFF, f)
        val g = lerp(a ushr 8 and 0xFF, b ushr 8 and 0xFF, f)
        val bl = lerp(a and 0xFF, b and 0xFF, f)
        return (r shl 16) or (g shl 8) or bl
    }

    private fun lerp(a: Int, b: Int, f: Float): Int =
        Math.round(a + (b - a) * f).coerceIn(0, 255)

    /**
     * The lookup table for one (threshold, opacity) setting.
     *
     * Colour is a function of the *raw* probability, not of the probability
     * rescaled into the visible range. That is deliberate: the mapper is expected
     * to sweep the threshold, and if the ramp were renormalised every pixel would
     * change colour as the slider moved. Here the threshold only decides what is
     * shown, never what a shown colour means, so "orange is about 0.75" stays true
     * across the whole sweep.
     */
    class Palette(threshold: Float, opacity: Float) {
        val threshold = threshold.coerceIn(0f, 1f)
        val opacity = opacity.coerceIn(0f, 1f)

        /** Packed ARGB, indexed by `probability * (LEVELS - 1)`. */
        val argb: IntArray = IntArray(LEVELS) { i ->
            val p = i.toFloat() / (LEVELS - 1)
            if (p < this.threshold) 0 else {
                val a = (ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * p) * this.opacity
                (Math.round(a * 255f).coerceIn(0, 255) shl 24) or colour(p)
            }
        }

        /** ARGB for one probability. NaN is nodata and never paints. */
        fun lookup(p: Float): Int =
            if (p.isNaN()) 0 else argb[index(p)]
    }

    /** Quantise a probability to a table index, clamping anything out of range. */
    fun index(p: Float): Int =
        Math.round(p.coerceIn(0f, 1f) * (LEVELS - 1))

    /**
     * Paint a probability raster into an image, one array lookup per pixel.
     *
     * Writes straight into the backing `int[]` rather than through `setRGB`, which
     * goes through a `ColorModel` conversion per call.
     */
    fun render(prob: FloatArray, width: Int, height: Int, palette: Palette): BufferedImage {
        require(prob.size == width * height) {
            "expected ${width * height} probabilities, got ${prob.size}"
        }
        val img = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
        val px = (img.raster.dataBuffer as DataBufferInt).data
        val lut = palette.argb
        for (i in prob.indices) {
            val p = prob[i]
            px[i] = if (p.isNaN()) 0 else lut[index(p)]
        }
        return img
    }
}

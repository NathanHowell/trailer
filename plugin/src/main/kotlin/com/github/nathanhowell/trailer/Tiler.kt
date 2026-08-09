package com.github.nathanhowell.trailer

import kotlin.math.max

/**
 * Hann-tapered overlapping-window tiling, mirroring `trailer.infer.predict`.
 *
 * The model runs on a fixed window — a ResNet-34 U-Net needs its input divisible
 * by 32, so the ONNX export pins H and W — which means whole-viewport inference
 * has to be stitched from tiles. Tiled inference on curvilinear features leaves
 * visible seams: a trail crossing a window boundary gets half its context on
 * each side and the model hedges. Overlapping windows under a 2-D Hann taper fix
 * that by making every pixel's final value dominated by the window that had it
 * nearest the centre.
 *
 * This file is the highest-risk part of the plugin, because it is a second
 * implementation of something Python already does. That is exactly the shape of
 * the bug that once misregistered two raster bands across fourteen builds
 * without anyone noticing. Everything here is therefore checked against
 * generated-from-Python golden values in `TilerTest`, not merely against its own
 * idea of what it should do.
 */
object Tiler {

    // There is deliberately no `hann2d` here either. The taper arrives as the
    // model's second output, `window_taper`, computed by the same `infer.hann2d`
    // that trained and validated the weights. Its definition has a detail that
    // exists precisely because it is not obvious — `hann_window(size + 2)` with
    // the ends trimmed, since a plain Hann is exactly zero at both ends and the
    // outermost row and column of every tile would otherwise contribute nothing.
    // A reimplementation that misses that is wrong only along tile edges, which
    // is exactly where seams live and where nobody looks.

    // There is deliberately no `step(tile, overlap)` here. The step depends on
    // the stem's stride as well as the overlap, and an origin that is not a
    // multiple of the stride lands between output pixels — a misregistration
    // that is worst at nothing in particular and so reads as general softening.
    // Kotlin computed it once, without the stride quantisation, and disagreed
    // with Python at overlap 0.7. It now comes from [ModelSpec.stepPx], which
    // carries the number `infer.window_step` produced at export time.

    /**
     * Padding needed on one axis so it can hold a single window.
     *
     * Mirrors `_pad_to`. A ragged tail is no longer padded: it is covered by a
     * flush final window instead, see [origins]. All that is left here is a
     * viewport smaller than one window, where there is no real ground to read.
     */
    fun padAmount(n: Int, tile: Int): Int = max(tile - n, 0)

    /**
     * Window origins along one axis of an `n` pixel raster, in input pixels.
     *
     * Stepped by `step` from zero, with the last window pulled back flush
     * against the far edge rather than hanging off a padded one.
     *
     * The previous rule extended the axis by reflection and let the final
     * window overhang. That put a mirror seam one window-edge away from real
     * output, and a mirrored hillslope is a symmetric V a few metres across --
     * the tread cross-section the model is trained to find. It fired on it
     * every time: on the trail-free control tile, *every* prediction above 0.5
     * sat in the outer four output pixels, against an interior rate of zero.
     * Cropping the padding off the output never helped, because the damage was
     * done to the input the window read.
     *
     * The flush origin is quantised down to `stride` for the same reason
     * [ModelSpec.stepPx] is a multiple of it: an origin between output pixels
     * misregisters the whole window by half an output pixel. Quantising down
     * strands nothing -- the leftover is by construction less than one output
     * pixel wide.
     *
     * Checked against `infer.window_origins` in `TilerTest`, not against this
     * file's own idea of the rule.
     */
    fun origins(n: Int, tile: Int, step: Int, stride: Int): IntArray {
        if (n < tile) return IntArray(0)
        val last = ((n - tile) / stride) * stride
        val stepped = last / step + 1
        // `last` coincides with the final stepped origin whenever the tail is
        // not ragged; emitting it twice would double-weight that window.
        val extra = if ((stepped - 1) * step == last) 0 else 1
        return IntArray(stepped + extra) { if (it < stepped) it * step else last }
    }

    /**
     * Reflect-pad the bottom and right edges, matching `F.pad(mode="reflect")`.
     *
     * Torch's reflect excludes the edge sample itself (`d c b | a b c d | c b a`),
     * so a run of length n reflects as index `2*(n-1) - i`. Getting this wrong is
     * invisible in the middle of a tile and wrong only along two edges, which is
     * precisely the kind of error that survives casual inspection.
     */
    fun reflectPad(src: FloatArray, h: Int, w: Int, padH: Int, padW: Int): FloatArray {
        require(src.size == h * w) { "expected ${h * w} samples, got ${src.size}" }
        require(padH < h && padW < w) {
            "reflect padding must be smaller than the source ($padH,$padW vs $h,$w)"
        }
        val H = h + padH
        val W = w + padW
        val out = FloatArray(H * W)
        for (r in 0 until H) {
            val sr = if (r < h) r else 2 * (h - 1) - r
            for (c in 0 until W) {
                val sc = if (c < w) c else 2 * (w - 1) - c
                out[r * W + c] = src[sr * w + sc]
            }
        }
        return out
    }

    /** A window's origin in the padded input grid. */
    data class Window(val row: Int, val col: Int)

    fun windows(paddedH: Int, paddedW: Int, tile: Int, step: Int,
                stride: Int): List<Window> {
        val rows = origins(paddedH, tile, step, stride)
        val cols = origins(paddedW, tile, step, stride)
        val out = ArrayList<Window>(rows.size * cols.size)
        for (r in rows) for (c in cols) out.add(Window(r, c))
        return out
    }

    /**
     * Accumulates tapered window predictions and normalises by the taper sum.
     *
     * Output lives on the *body* grid: the model always predicts at 1 m whatever
     * the input pixel size, so a window at input origin `row` lands at
     * `row / stride` in the output. For the deployable bare-earth variant stride
     * is 1, but the division is kept explicit rather than assumed away.
     */
    class Blender(private val height: Int, private val width: Int,
                  private val tile: Int, private val taper: FloatArray) {
        private val acc = FloatArray(height * width)
        private val den = FloatArray(height * width)

        init {
            require(taper.size == tile * tile) {
                "taper is ${taper.size} values for a $tile x $tile window; it " +
                    "should be the model's window_taper output"
            }
        }

        fun add(prob: FloatArray, row: Int, col: Int) {
            require(prob.size == tile * tile) {
                "expected ${tile * tile} predictions, got ${prob.size}"
            }
            for (r in 0 until tile) {
                val dst = row + r
                if (dst >= height) break
                for (c in 0 until tile) {
                    val dc = col + c
                    if (dc >= width) break
                    val t = taper[r * tile + c]
                    val i = dst * width + dc
                    acc[i] += prob[r * tile + c] * t
                    den[i] += t
                }
            }
        }

        fun result(): FloatArray = FloatArray(height * width) { i ->
            acc[i] / max(den[i], 1e-6f)
        }
    }
}

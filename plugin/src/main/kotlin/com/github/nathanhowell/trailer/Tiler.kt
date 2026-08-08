package com.github.nathanhowell.trailer

import kotlin.math.PI
import kotlin.math.cos
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

    /**
     * Separable 2-D Hann taper, flattened row-major.
     *
     * Matches `torch.hann_window(size + 2, periodic=False)[1:-1]` outer-multiplied
     * with itself and floored at 1e-3. The `size + 2` then trim is not incidental:
     * a plain Hann window is exactly zero at both ends, so the outermost row and
     * column of every tile would contribute nothing and the floor would be doing
     * all the work there. Trimming the zero endpoints keeps the taper strictly
     * positive across the whole window.
     */
    fun hann2d(size: Int): FloatArray {
        require(size > 0) { "window size must be positive, got $size" }
        val w = DoubleArray(size) { i ->
            0.5 - 0.5 * cos(2.0 * PI * (i + 1) / (size + 1))
        }
        val out = FloatArray(size * size)
        for (r in 0 until size) {
            for (c in 0 until size) {
                out[r * size + c] = max(w[r] * w[c], 1e-3).toFloat()
            }
        }
        return out
    }

    // There is deliberately no `step(tile, overlap)` here. The step depends on
    // the stem's stride as well as the overlap, and an origin that is not a
    // multiple of the stride lands between output pixels — a misregistration
    // that is worst at nothing in particular and so reads as general softening.
    // Kotlin computed it once, without the stride quantisation, and disagreed
    // with Python at overlap 0.7. It now comes from [ModelSpec.stepPx], which
    // carries the number `infer.window_step` produced at export time.

    /**
     * Padding needed on one axis so windows tile it exactly.
     *
     * Mirrors `_pad_to`. Note Python's `%` on a negative left operand returns a
     * non-negative result; Kotlin's `rem` does not, so this uses `mod`.
     */
    fun padAmount(n: Int, tile: Int, step: Int): Int {
        val short = max(tile - n, 0)
        val ragged = if (n > tile) (-(n - tile)).mod(step) else 0
        return short + ragged
    }

    /** Window origins along one axis, after padding. */
    fun origins(padded: Int, tile: Int, step: Int): IntArray {
        if (padded < tile) return IntArray(0)
        val n = (padded - tile) / step + 1
        return IntArray(n) { it * step }
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

    fun windows(paddedH: Int, paddedW: Int, tile: Int, step: Int): List<Window> {
        val rows = origins(paddedH, tile, step)
        val cols = origins(paddedW, tile, step)
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
    class Blender(private val height: Int, private val width: Int, private val tile: Int) {
        private val acc = FloatArray(height * width)
        private val den = FloatArray(height * width)
        private val taper = hann2d(tile)

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

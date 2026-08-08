package com.github.nathanhowell.trailer

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.file.Files
import java.nio.file.Path

/**
 * Whole-raster inference: reflect-pad, tile, run the graph, blend, crop.
 *
 * This is the composition step, and composition is where this project's worst
 * bug class lives. Every piece below is already checked against Python
 * individually — [Tiler.padAmount], [Tiler.origins], [Tiler.reflectPad],
 * [Tiler.Blender] — and two individually correct pieces wired together in the
 * wrong order still produce a plausible-looking raster. `InferenceTest` therefore
 * compares a finished raster against one `trailer.infer.predict` produced from
 * the same weights, rather than checking these steps again one at a time.
 *
 * Nothing here decides anything the model already decided. The window size,
 * step, padding mode and stride are read from [ModelSpec]; the blending taper is
 * the graph's own second output. What is left in Kotlin is only what depends on
 * the raster in front of it: how much padding it needs and where its windows
 * fall.
 */
class Inference(
    private val session: OrtSession,
    val spec: ModelSpec,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment(),
) : AutoCloseable {

    companion object {
        /** Load an exported `.onnx` and the sidecar written beside it. */
        fun open(onnx: Path, sidecar: Path): Inference {
            val spec = ModelSpec(Files.readString(sidecar))
            val env = OrtEnvironment.getEnvironment()
            val session = env.createSession(Files.readAllBytes(onnx),
                                            OrtSession.SessionOptions())
            return Inference(session, spec, env)
        }
    }

    private val inputName: String = session.inputNames.first()

    init {
        // The sidecar says what the graph emits and in what order; the graph
        // itself is the authority on the same question. If they disagree, one of
        // them was regenerated without the other, and every number downstream is
        // read off the wrong tensor.
        val actual = session.outputNames.toList()
        require(actual == spec.outputs) {
            "model emits $actual but its sidecar claims ${spec.outputs}; the " +
                "two were not written by the same export"
        }
        require(session.inputNames.size == 1) {
            "expected a single input, model takes ${session.inputNames}"
        }
    }

    /**
     * Probabilities for a `height x width` elevation raster, in metres.
     *
     * Returns the *body* grid: `height / stride` by `width / stride`, since the
     * model predicts at 1 m whatever it was fed. Padding added to make the
     * windows tile is cropped back off, so the result lines up with the input
     * cell for cell.
     */
    fun run(z: FloatArray, height: Int, width: Int): FloatArray {
        require(z.size == height * width) {
            "expected ${height * width} samples, got ${z.size}"
        }
        val tile = spec.inputPx
        val step = spec.stepPx
        val stride = spec.stride

        val padH = Tiler.padAmount(height, tile, step)
        val padW = Tiler.padAmount(width, tile, step)
        // Reflection cannot extend a run by more than its own length, in torch
        // or here. A raster this small is a caller error rather than something
        // to paper over, and saying so beats a confusing failure inside the pad.
        require(padH < height && padW < width) {
            "raster ${height}x$width is too small for a $tile px window: it " +
                "needs ${padH}x$padW of reflect padding, which is more than " +
                "there is to reflect"
        }

        val paddedH = height + padH
        val paddedW = width + padW
        val padded =
            if (padH == 0 && padW == 0) z
            else Tiler.reflectPad(z, height, width, padH, padW)

        val bodyH = paddedH / stride
        val bodyW = paddedW / stride
        val bt = spec.outputPx

        // Built on the first window, because the taper is the model's output
        // and there is nothing to build it from until the graph has run once.
        // Deliberately not precomputed here: recomputing it would be exactly
        // the duplication the second output exists to remove.
        var blender: Tiler.Blender? = null
        val crop = FloatArray(tile * tile)

        for (win in Tiler.windows(paddedH, paddedW, tile, step)) {
            for (r in 0 until tile) {
                System.arraycopy(padded, (win.row + r) * paddedW + win.col,
                                 crop, r * tile, tile)
            }
            OnnxTensor.createTensor(env, FloatBuffer.wrap(crop),
                                    longArrayOf(1, 1, tile.toLong(), tile.toLong()))
                .use { input ->
                    session.run(mapOf(inputName to input)).use { result ->
                        val prob = floats(result, 0, bt)
                        if (blender == null) {
                            blender = Tiler.Blender(bodyH, bodyW, bt,
                                                    floats(result, 1, bt))
                        }
                        // Window origins are in input pixels; the blender
                        // accumulates in body pixels. ModelSpec has already
                        // refused a step that is not a multiple of the stride,
                        // so this division is exact.
                        blender!!.add(prob, win.row / stride, win.col / stride)
                    }
                }
        }

        val full = (blender ?: error("no windows covered ${height}x$width"))
            .result()
        return cropTo(full, bodyH, bodyW, height / stride, width / stride)
    }

    /** One output tensor as a flat array, checked for the shape it should have. */
    private fun floats(result: OrtSession.Result, index: Int, bt: Int): FloatArray {
        val tensor = result.get(index) as OnnxTensor
        val buf = tensor.floatBuffer
        require(buf.remaining() == bt * bt) {
            "output ${spec.outputs[index]} has ${buf.remaining()} values, " +
                "expected ${bt * bt} for a $bt x $bt window"
        }
        val out = FloatArray(bt * bt)
        buf.get(out)
        return out
    }

    /** Drop the padded tail, so the result lines up with the input raster. */
    private fun cropTo(src: FloatArray, srcH: Int, srcW: Int,
                       h: Int, w: Int): FloatArray {
        require(h <= srcH && w <= srcW) { "cannot crop ${srcH}x$srcW to ${h}x$w" }
        if (h == srcH && w == srcW) return src
        val out = FloatArray(h * w)
        for (r in 0 until h) System.arraycopy(src, r * srcW, out, r * w, w)
        return out
    }

    override fun close() = session.close()
}

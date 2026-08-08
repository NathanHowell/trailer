package com.github.nathanhowell.trailer

import kotlin.math.ceil
import kotlin.math.max

/**
 * Turns the map viewport into a 3DEP request the model can actually eat.
 *
 * The one rule everything here serves: **the fetched pixel size must equal the
 * size the model was trained at**, exactly. It is tempting to ask the service
 * for whatever fits and let the model cope, and that is precisely the trap
 * [Dem3dep.MAX_SOURCE_PIXEL_M] exists to catch from the other direction — 3DEP
 * will happily resample 10 m data up to a 1 m grid and return something that
 * looks right. Asking for 2 m pixels because the viewport is wide is the same
 * mistake made deliberately, so it is refused rather than accommodated.
 *
 * That leaves two honest failure modes, handled differently on purpose. A view
 * too *large* for one request is refused, because the alternative is silently
 * coarsening. A view too *small* is expanded, because the model needs its full
 * window of context regardless and a mapper zoomed in on one switchback still
 * deserves an answer.
 */
object FetchPlan {

    /**
     * A request: where to fetch, at what size, and the pixel size that implies.
     *
     * [bounds] is not the viewport. It is the viewport grown to a whole number of
     * model-resolution pixels and re-centred, so that [pixelM] comes out exact
     * rather than nearly right.
     */
    data class Plan(val bounds: Dem3dep.Bounds, val width: Int, val height: Int,
                    val pixelM: Double) {
        val groundWidthM get() = width * pixelM
        val groundHeightM get() = height * pixelM
    }

    /**
     * How far the map projection may stretch a metre before we refuse to run.
     *
     * 1%, which at 1 m sampling is a centimetre — far below anything the model
     * can notice, and far above the rounding in a well-chosen projection.
     */
    const val MAX_SCALE_ERROR = 0.01

    /**
     * The projection's metres are not ground metres here.
     *
     * This is the trap worth being loud about. JOSM's default is Web Mercator,
     * whose east/north are nominally metres and are stretched by 1/cos(latitude)
     * — about 27% at 38°N, and more further north. A fetch planned in those
     * units asks 3DEP for a grid that is not 1 m on the ground, the service
     * cheerfully resamples to it, and the model reads terrain at a scale it
     * never trained on. Nothing downstream can detect this: the raster is the
     * right shape, the elevations are real, and the trails are simply the wrong
     * size.
     */
    class Distorted(val scale: Double, val projection: String) : Exception(
        "in $projection a projected metre is ${"%.3f".format(scale)} ground " +
            "metres here, so a fetch planned in these units would sample the " +
            "terrain at the wrong scale and the model would be reading a size " +
            "of trail it never trained on. Switch to the local UTM zone in " +
            "Preferences → Map Projection")

    /**
     * Refuse a projection that does not measure true metres at this location.
     *
     * Both arguments describe the same span: one measured in the projection's
     * east/north units, one measured on the ellipsoid. Conformal projections —
     * UTM, and Web Mercator — distort isotropically, so a single ratio captures
     * it and one span is enough to test.
     */
    fun checkTrueScale(projectedSpan: Double, groundSpan: Double,
                       projection: String) {
        require(projectedSpan > 0 && groundSpan > 0) {
            "degenerate span: $projectedSpan projected, $groundSpan ground"
        }
        val scale = groundSpan / projectedSpan
        if (kotlin.math.abs(scale - 1.0) > MAX_SCALE_ERROR) {
            throw Distorted(scale, projection)
        }
    }

    /** The view covers more ground than one request can serve at model resolution. */
    class TooLarge(val spanM: Double, val limitM: Double) : Exception(
        "this view spans ${"%.1f".format(spanM / 1000)} km, and 3DEP serves at " +
            "most ${Dem3dep.MAX_IMAGE_PX} pixels per request — " +
            "${"%.1f".format(limitM / 1000)} km at the resolution this model " +
            "needs. Zoom in, or the elevation would have to be coarsened and " +
            "the model would be reading interpolation")

    /**
     * Plan a fetch for [view], in projected metres.
     *
     * The returned bounds are centred on the view. Growing a small view outward
     * rather than upward — more ground at the same pixel size, never the same
     * ground at a finer one — is what keeps [Plan.pixelM] honest.
     */
    fun forView(view: Dem3dep.Bounds, spec: ModelSpec): Plan {
        val res = spec.resM
        require(res > 0) { "model declares a non-positive resolution: $res" }

        // Whole pixels, and never fewer than one model window: below that the
        // window could not be filled without reflecting more than there is to
        // reflect, and the context the model needs is not in the view anyway.
        val w = max(ceil(view.width / res).toInt(), spec.inputPx)
        val h = max(ceil(view.height / res).toInt(), spec.inputPx)

        val limitM = Dem3dep.MAX_IMAGE_PX * res
        if (w > Dem3dep.MAX_IMAGE_PX) throw TooLarge(view.width, limitM)
        if (h > Dem3dep.MAX_IMAGE_PX) throw TooLarge(view.height, limitM)

        val cx = (view.minX + view.maxX) / 2
        val cy = (view.minY + view.maxY) / 2
        val halfW = w * res / 2
        val halfH = h * res / 2
        return Plan(
            Dem3dep.Bounds(cx - halfW, cy - halfH, cx + halfW, cy + halfH),
            w, h, res)
    }
}

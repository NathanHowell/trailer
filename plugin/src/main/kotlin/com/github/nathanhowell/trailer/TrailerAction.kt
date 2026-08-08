package com.github.nathanhowell.trailer

import org.openstreetmap.josm.actions.JosmAction
import org.openstreetmap.josm.gui.MainApplication
import org.openstreetmap.josm.gui.PleaseWaitRunnable
import org.openstreetmap.josm.tools.Logging
import org.openstreetmap.josm.tools.Shortcut
import java.awt.event.ActionEvent
import javax.swing.JOptionPane

/**
 * "Trail probability for this view": fetch, infer, paint.
 *
 * The only place the tested pieces meet. [FetchPlan] decides what to ask 3DEP
 * for, [Dem3dep] fetches it and refuses anything that is not wholly 1 m LiDAR,
 * [Inference] runs the graph, and [TrailerLayer] paints the result. Each of
 * those is tested on its own; this class is deliberately thin, because it is the
 * part that cannot be tested without a running JOSM and therefore the part that
 * should contain as little judgement as possible.
 *
 * Every failure below is reported to the mapper as a sentence about what to do
 * next. The interesting failures here are not bugs — no 1 m LiDAR, view too
 * wide, wrong projection, no model configured — and each has a specific remedy.
 * A generic "inference failed" would turn all four into a shrug.
 */
class TrailerAction : JosmAction(
    "Trail probability for this view",
    null,
    "Fetch 1 m bare-earth elevation and paint where trails probably are",
    Shortcut.registerShortcut("trailer:run", "Trailer: trail probability",
                              java.awt.event.KeyEvent.VK_T, Shortcut.ALT_CTRL),
    true,
) {

    override fun actionPerformed(e: ActionEvent?) {
        val map = MainApplication.getMap()
        if (map == null || !MainApplication.isDisplayingMapView()) {
            report("Open a map view first.")
            return
        }
        MainApplication.worker.submit(Run(this))
    }

    private fun report(message: String) =
        JOptionPane.showMessageDialog(MainApplication.getMainFrame(), message,
                                      "Trail probability", JOptionPane.INFORMATION_MESSAGE)

    /**
     * The work, off the event thread.
     *
     * Fetching a few megabytes of GeoTIFF and running a U-Net over it takes
     * seconds, not milliseconds. On the EDT that would freeze JOSM, and a frozen
     * editor is indistinguishable from a crashed one.
     */
    private class Run(private val owner: TrailerAction) :
        PleaseWaitRunnable("Trail probability") {

        private var overlay: Overlay? = null
        private var failure: String? = null

        override fun cancel() {
            // Nothing to unwind: the fetch and the session are stateless from
            // here, and a half-built overlay is simply never installed.
        }

        override fun realRun() {
            try {
                overlay = build()
            } catch (ex: Dem3dep.UnsupportedCoverage) {
                failure = ex.message
            } catch (ex: FetchPlan.TooLarge) {
                failure = ex.message
            } catch (ex: FetchPlan.Distorted) {
                failure = ex.message
            } catch (ex: ModelStore.NotConfigured) {
                failure = ex.message
            } catch (ex: Exception) {
                // Genuinely unexpected, so it goes to the log as well as the
                // dialog. The four above are expected conditions with remedies.
                Logging.error(ex)
                failure = "Trail probability failed: ${ex.message ?: ex::class.java.simpleName}"
            }
        }

        private fun build(): Overlay {
            val inference = ModelStore.open()
            val spec = inference.spec

            val view = MainApplication.getMap().mapView
            val projection = view.projection
            val code = projection.toCode()
            val pb = view.projectionBounds

            // Ground truth before pixel arithmetic: if the projection's metres
            // are not metres here, every number after this is quietly wrong.
            val west = projection.eastNorth2latlon(
                org.openstreetmap.josm.data.coor.EastNorth(pb.minEast, pb.minNorth))
            val east = projection.eastNorth2latlon(
                org.openstreetmap.josm.data.coor.EastNorth(pb.maxEast, pb.minNorth))
            FetchPlan.checkTrueScale(pb.maxEast - pb.minEast,
                                     west.greatCircleDistance(east), code)

            val plan = FetchPlan.forView(
                Dem3dep.Bounds(pb.minEast, pb.minNorth, pb.maxEast, pb.maxNorth),
                spec)

            val epsg = code.substringAfter("EPSG:").toIntOrNull()
                ?: throw IllegalStateException(
                    "projection $code has no EPSG code, so 3DEP cannot be asked " +
                        "for data in it")

            progressMonitor.indeterminateSubTask(
                "Fetching ${"%.2g".format(plan.pixelM)} m elevation")
            // One catalog round trip, not two: the coverage that authorised the
            // fetch is the coverage the mapper is shown.
            val (grid, coverage) =
                Dem3dep.fetchDescribed(plan.bounds, epsg, plan.width, plan.height)

            progressMonitor.indeterminateSubTask(
                "Running the model over " +
                    "${"%.0f".format(plan.groundWidthM)} x " +
                    "${"%.0f".format(plan.groundHeightM)} m")
            val prob = inference.run(grid.values, grid.height, grid.width)

            val stride = spec.stride
            return Overlay(
                prob = prob,
                width = grid.width / stride,
                height = grid.height / stride,
                minEast = plan.bounds.minX, minNorth = plan.bounds.minY,
                maxEast = plan.bounds.maxX, maxNorth = plan.bounds.maxY,
                projection = code,
                source = coverage.describe(),
                model = "${spec.variant} ${spec.resM} m" + if (spec.tta) " +TTA" else "",
                attribution = spec.attribution,
            )
        }

        override fun finish() {
            failure?.let { owner.report(it); return }
            val o = overlay ?: return
            val existing = MainApplication.getLayerManager()
                .getLayersOfType(TrailerLayer::class.java).firstOrNull()
            if (existing != null) existing.setOverlay(o)
            else MainApplication.getLayerManager().addLayer(TrailerLayer(o))
        }
    }
}

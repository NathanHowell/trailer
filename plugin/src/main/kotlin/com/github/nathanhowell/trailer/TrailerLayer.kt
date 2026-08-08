package com.github.nathanhowell.trailer

import org.openstreetmap.josm.data.Bounds
import org.openstreetmap.josm.data.coor.EastNorth
import org.openstreetmap.josm.data.osm.visitor.BoundingXYVisitor
import org.openstreetmap.josm.data.projection.Projection
import org.openstreetmap.josm.gui.MapView
import org.openstreetmap.josm.gui.dialogs.LayerListDialog
import org.openstreetmap.josm.gui.dialogs.LayerListPopup
import org.openstreetmap.josm.gui.layer.Layer
import java.awt.Component
import java.awt.Graphics2D
import java.awt.Graphics
import java.awt.GridLayout
import java.awt.RenderingHints
import java.awt.event.ActionEvent
import java.awt.image.BufferedImage
import javax.swing.AbstractAction
import javax.swing.Action
import javax.swing.Icon
import javax.swing.JLabel
import javax.swing.JOptionPane
import javax.swing.JPanel
import javax.swing.JSlider

/**
 * The trail-probability overlay.
 *
 * This layer paints and nothing else. It never creates, edits or suggests OSM
 * geometry, and that is a product decision rather than an unfinished one: OSM has
 * strong and well-earned opinions about machine-generated ways, and a mapper
 * tracing what they judge to be real is the whole point. Anything here that
 * looked like a "convert to way" button would be a bug.
 *
 * A custom [Layer] rather than an imagery layer, for the reason recorded on the
 * bead: JOSM imagery layers carry 8-bit `BufferedImage` tiles, and what is being
 * shown is probability, whose weak end is exactly where the interesting faint
 * trails live. It also means the layer owns its own viewport events, which is
 * the natural trigger for fetching and inference.
 */
class TrailerLayer(private var overlay: Overlay) : Layer(NAME) {

    companion object {
        const val NAME = "Trail probability"

        /**
         * Default threshold.
         *
         * Low on purpose. The model is trained to buy recall at the cost of
         * precision, so a mapper who never touches the slider should see the
         * faint candidates and reject them, not miss them.
         */
        const val DEFAULT_THRESHOLD = 0.3f

        const val THRESHOLD_STEP = 0.05f
    }

    var threshold: Float = DEFAULT_THRESHOLD
        set(value) {
            val v = value.coerceIn(0f, 1f)
            if (v != field) {
                field = v
                repaintCache = null
                invalidate()
            }
        }

    /** Rebuilt only when the threshold or the layer's opacity actually changes. */
    private var repaintCache: BufferedImage? = null
    private var cachedOpacity: Double = -1.0
    private var cachedThreshold: Float = -1f

    private fun image(): BufferedImage {
        val cached = repaintCache
        if (cached != null && cachedOpacity == opacity && cachedThreshold == threshold) {
            return cached
        }
        // Opacity folds into the palette rather than into an AlphaComposite: the
        // table is 256 entries and the viewport is millions of pixels, so paying
        // for it once per settings change beats paying per pixel per repaint.
        val img = Heatmap.render(overlay.prob, overlay.width, overlay.height,
                                 Heatmap.Palette(threshold, opacity.toFloat()))
        repaintCache = img
        cachedOpacity = opacity
        cachedThreshold = threshold
        return img
    }

    fun setOverlay(o: Overlay) {
        overlay = o
        repaintCache = null
        invalidate()
    }

    override fun paint(g: Graphics2D, mv: MapView, bbox: Bounds) {
        if (overlay.projection != mv.projection.toCode()) return
        // Bilinear, because the overlay is 1 m and the mapper will be zoomed in
        // past that. Nearest-neighbour would show the model's grid as blocks and
        // invite tracing the quantisation rather than the terrain.
        val old = g.getRenderingHint(RenderingHints.KEY_INTERPOLATION)
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                           RenderingHints.VALUE_INTERPOLATION_BILINEAR)
        try {
            g.drawImage(image(), overlay.toScreen(mv.affineTransform), null)
        } finally {
            if (old != null) g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, old)
        }
    }

    override fun projectionChanged(oldValue: Projection?, newValue: Projection?) {
        // The overlay's extent is in the old projection's metres. Rather than
        // reproject a raster (and quietly resample probabilities), stop drawing;
        // the next fetch will produce one in the new projection.
        repaintCache = null
        invalidate()
    }

    // ------------------------------------------------------------------ chrome

    /** A swatch of the actual ramp, so the layer list shows what it means. */
    override fun getIcon(): Icon = object : Icon {
        override fun getIconWidth() = 16
        override fun getIconHeight() = 16
        override fun paintIcon(c: Component?, g: Graphics, x: Int, y: Int) {
            for (i in 0 until 16) {
                g.color = java.awt.Color(Heatmap.colour(i / 15f))
                g.fillRect(x, y + 15 - i, 16, 1)
            }
        }
    }

    override fun getToolTipText(): String =
        "$NAME at ${"%.2f".format(threshold)} — ${overlay.describe()}"

    override fun isMergable(other: Layer?): Boolean = false

    override fun mergeFrom(from: Layer?) =
        throw UnsupportedOperationException("probability layers do not merge")

    override fun visitBoundingBox(v: BoundingXYVisitor) {
        v.visit(EastNorth(overlay.minEast, overlay.minNorth))
        v.visit(EastNorth(overlay.maxEast, overlay.maxNorth))
    }

    /**
     * Provenance, which a mapper needs both to judge the overlay and to cite it.
     *
     * The survey date matters more than it looks: a trail cut after the LiDAR was
     * flown cannot appear here at all, and a mapper who does not know the date
     * will read that absence as the model saying no.
     */
    override fun getInfoComponent(): Any {
        val p = JPanel(GridLayout(0, 1))
        p.add(JLabel(NAME))
        p.add(JLabel("Elevation source: ${overlay.source}"))
        p.add(JLabel("Model: ${overlay.model}"))
        p.add(JLabel("Projection: ${overlay.projection}"))
        p.add(JLabel("Grid: ${overlay.width} x ${overlay.height} at " +
                     "${"%.2f".format(overlay.pixelWidthM)} m"))
        p.add(JLabel("Threshold: ${"%.2f".format(threshold)}"))
        p.add(JLabel("This layer never creates ways. Trace what you judge to be real."))
        return p
    }

    override fun getMenuEntries(): Array<Action> = arrayOf(
        LayerListDialog.getInstance().createShowHideLayerAction(),
        LayerListDialog.getInstance().createDeleteLayerAction(),
        SeparatorLayerAction.INSTANCE,
        ThresholdAction(this, +THRESHOLD_STEP),
        ThresholdAction(this, -THRESHOLD_STEP),
        SweepAction(this),
        SeparatorLayerAction.INSTANCE,
        LayerListPopup.InfoAction(this),
    )

    /** Nudge the threshold. Repeatable from the keyboard, which is how one sweeps. */
    private class ThresholdAction(val layer: TrailerLayer, val delta: Float) :
        AbstractAction(if (delta > 0) "Raise threshold" else "Lower threshold") {
        override fun actionPerformed(e: ActionEvent?) {
            layer.threshold += delta
        }
    }

    /**
     * A slider with live preview.
     *
     * Recall here is deliberately bought with precision, so no single threshold is
     * the right one and the mapper is expected to sweep. Live preview is the point
     * — an OK/Cancel dialog would make them guess.
     */
    private class SweepAction(val layer: TrailerLayer) : AbstractAction("Set threshold…") {
        override fun actionPerformed(e: ActionEvent?) {
            val start = layer.threshold
            val slider = JSlider(0, 100, Math.round(start * 100))
            slider.majorTickSpacing = 25
            slider.paintTicks = true
            slider.paintLabels = true
            slider.addChangeListener { layer.threshold = slider.value / 100f }
            val ok = JOptionPane.showConfirmDialog(
                null, slider, "Trail probability threshold",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE)
            if (ok != JOptionPane.OK_OPTION) layer.threshold = start
        }
    }
}

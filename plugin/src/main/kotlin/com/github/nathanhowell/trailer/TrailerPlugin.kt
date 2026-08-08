package com.github.nathanhowell.trailer

import org.openstreetmap.josm.gui.MainApplication
import org.openstreetmap.josm.gui.MainMenu
import org.openstreetmap.josm.gui.MapFrame
import org.openstreetmap.josm.plugins.Plugin
import org.openstreetmap.josm.plugins.PluginInformation
import org.openstreetmap.josm.tools.Logging

/**
 * Entry point. Deliberately does almost nothing yet.
 *
 * The plugin's job is to show a trail-probability heatmap over a mapper's
 * viewport, derived from USGS 3DEP bare-earth elevation, so a human can trace
 * what they judge to be real. It never creates geometry: OSM has strong and
 * well-earned opinions about machine-generated ways, and an import is
 * explicitly not the product.
 *
 * The path, end to end: [TrailerAction] plans a fetch with [FetchPlan],
 * [Dem3dep] fetches bare-earth elevation and refuses anything that is not wholly
 * 1 m LiDAR, [Inference] runs the graph [ModelStore] opened, and [TrailerLayer]
 * paints the result through [Heatmap].
 *
 * The weights are not in this jar. They are ~99 MB, change on a different
 * schedule from the code, and are under a different licence (CC BY-SA, against
 * the code's MIT), so a mapper points at a file rather than downloading one
 * bundled. Model distribution is tracked separately in beads.
 *
 * [Tiler] and [Inference] are parity-tested against generated Python values,
 * because they reimplement existing Python and are therefore the parts most
 * likely to drift.
 */
class TrailerPlugin(info: PluginInformation) : Plugin(info) {
    init {
        MainMenu.add(MainApplication.getMenu().imagerySubMenu, TrailerAction())
        Logging.info("trailer plugin loaded")
    }

    /**
     * Release the ONNX session when the last map view closes.
     *
     * The session holds a graph of the order of 100 MB off-heap, which the JVM's
     * heap pressure never sees and so never collects under.
     */
    override fun mapFrameInitialized(oldFrame: MapFrame?, newFrame: MapFrame?) {
        if (newFrame == null) ModelStore.close()
    }
}

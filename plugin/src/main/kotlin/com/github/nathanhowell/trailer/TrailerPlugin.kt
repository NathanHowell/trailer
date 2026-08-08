package com.github.nathanhowell.trailer

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
 * Still to come, tracked in beads under the JOSM plugin epic: the 3DEP
 * ImageServer client, the ONNX session wrapper, the heatmap layer, and model
 * distribution. [Tiler] is already here and parity-tested, because it is the
 * one part that reimplements existing Python and is therefore the one part most
 * likely to drift.
 */
class TrailerPlugin(info: PluginInformation) : Plugin(info) {
    init {
        Logging.info("trailer plugin loaded")
    }
}

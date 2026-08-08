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
 * Here already: [Dem3dep] fetches bare-earth elevation and refuses windows that
 * are not wholly 1 m LiDAR, [Tiler] reproduces the Python windowing, and
 * [TrailerLayer] paints an [Overlay] through [Heatmap]. Still to come, tracked in
 * beads under the JOSM plugin epic: the ONNX session wrapper that turns elevation
 * into probability and the model distribution around it. Until that lands nothing
 * constructs an [Overlay] at runtime — the rendering path is complete and tested,
 * but it has no source of data yet.
 *
 * [Tiler] is parity-tested against generated Python values because it is the one
 * part that reimplements existing Python and is therefore the one part most
 * likely to drift.
 */
class TrailerPlugin(info: PluginInformation) : Plugin(info) {
    init {
        Logging.info("trailer plugin loaded")
    }
}

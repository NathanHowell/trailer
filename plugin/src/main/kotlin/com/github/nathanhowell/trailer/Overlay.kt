package com.github.nathanhowell.trailer

import java.awt.geom.AffineTransform

/**
 * A probability raster pinned to the ground, with the provenance a mapper needs
 * to judge it.
 *
 * Held in projected coordinates (JOSM's "east/north" metres) rather than in
 * lat/lon, because that is the frame the model works in: windows are defined in
 * metres and the raster is square on the ground. [projection] records which
 * projection those numbers belong to, so a projection change invalidates the
 * overlay instead of silently smearing it across the map.
 */
class Overlay(
    val prob: FloatArray,
    val width: Int,
    val height: Int,
    val minEast: Double,
    val minNorth: Double,
    val maxEast: Double,
    val maxNorth: Double,
    val projection: String,
    /** From the 3DEP catalog: source, pixel size and survey date. */
    val source: String,
    /** Which exported model produced this, so a result can be reproduced. */
    val model: String,
    /**
     * The weights' attribution notice, verbatim from the model sidecar.
     *
     * A constructor argument rather than something the layer looks up, so a
     * painted overlay cannot exist without one. The weights are CC BY-SA and
     * are trained on ODbL geometry; showing this is a licence condition, and a
     * condition that is easy to satisfy only when you remember is one that will
     * eventually not be satisfied.
     */
    val attribution: String,
) {
    init {
        require(width > 0 && height > 0) { "empty overlay: $width x $height" }
        require(attribution.isNotBlank()) {
            "an overlay must carry the model's attribution notice"
        }
        require(prob.size == width * height) {
            "expected ${width * height} probabilities, got ${prob.size}"
        }
        require(maxEast > minEast && maxNorth > minNorth) {
            "empty ground extent: $minEast,$minNorth .. $maxEast,$maxNorth"
        }
    }

    /** Ground size of one raster pixel, in metres. */
    val pixelWidthM get() = (maxEast - minEast) / width
    val pixelHeightM get() = (maxNorth - minNorth) / height

    /**
     * Maps image pixels onto the ground.
     *
     * The Y scale is negative: raster row 0 is the *north* edge, while north
     * increases upward in projected coordinates. Getting this wrong flips the
     * overlay vertically, which on symmetrical terrain looks plausible — the same
     * class of error as the band misregistration that once survived fourteen
     * builds here.
     */
    fun toGround(): AffineTransform = AffineTransform(
        pixelWidthM, 0.0, 0.0, -pixelHeightM, minEast, maxNorth)

    /**
     * Image-to-screen transform, given the map view's own ground-to-screen one.
     *
     * Composing with the view's transform rather than mapping two corners by hand
     * means rotation and any future non-axis-aligned view come out right without
     * this code knowing about them.
     */
    fun toScreen(groundToScreen: AffineTransform): AffineTransform {
        val t = AffineTransform(groundToScreen)
        t.concatenate(toGround())
        return t
    }

    /** Probability at a raster cell, or NaN outside it. */
    fun at(row: Int, col: Int): Float =
        if (row < 0 || col < 0 || row >= height || col >= width) Float.NaN
        else prob[row * width + col]

    /** Probability at a ground position, nearest cell, NaN outside the overlay. */
    fun atGround(east: Double, north: Double): Float {
        val col = Math.floor((east - minEast) / pixelWidthM).toInt()
        val row = Math.floor((maxNorth - north) / pixelHeightM).toInt()
        return at(row, col)
    }

    /** One line of provenance, for the layer tooltip and the info dialog. */
    fun describe(): String = "$source · model $model · $projection"
}

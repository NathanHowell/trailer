package com.github.nathanhowell.trailer

import org.junit.jupiter.api.Test
import java.awt.geom.AffineTransform
import java.awt.geom.Point2D
import kotlin.math.PI
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class OverlayTest {

    /** 4x2 cells over 400 x 200 m, so pixel size is 100 m and asymmetry shows. */
    private fun overlay(prob: FloatArray = FloatArray(8)) = Overlay(
        prob, width = 4, height = 2,
        minEast = 1000.0, minNorth = 5000.0, maxEast = 1400.0, maxNorth = 5200.0,
        projection = "EPSG:32611", source = "USGS 1 Meter", model = "test")

    private fun assertPoint(x: Double, y: Double, p: Point2D, msg: String) {
        assertEquals(x, p.x, 1e-6, "$msg x")
        assertEquals(y, p.y, 1e-6, "$msg y")
    }

    private fun map(t: AffineTransform, x: Double, y: Double): Point2D =
        t.transform(Point2D.Double(x, y), null)

    @Test
    fun `raster row zero is the north edge`() {
        // The Y scale is negative because north increases upward while raster rows
        // run downward. Get it wrong and the overlay is mirrored, which over
        // symmetrical terrain looks entirely plausible.
        val t = overlay().toGround()
        assertPoint(1000.0, 5200.0, map(t, 0.0, 0.0), "top-left pixel -> NW corner")
        assertPoint(1400.0, 5000.0, map(t, 4.0, 2.0), "bottom-right pixel -> SE corner")
        assertPoint(1200.0, 5100.0, map(t, 2.0, 1.0), "centre")
    }

    @Test
    fun `pixel size is reported per axis`() {
        val o = overlay()
        assertEquals(100.0, o.pixelWidthM, 1e-9)
        assertEquals(100.0, o.pixelHeightM, 1e-9)
    }

    @Test
    fun `composes with the map view transform instead of mapping corners itself`() {
        // A plain two-corner mapping would silently drop rotation. JOSM can rotate
        // the map view, so the ground-to-screen transform is composed whole.
        val view = AffineTransform()
        view.translate(50.0, 20.0)
        view.rotate(PI / 6)
        view.scale(2.0, 2.0)

        val t = overlay().toScreen(view)
        // Every image corner must land where the view would put its ground point.
        for ((px, py) in listOf(0.0 to 0.0, 4.0 to 0.0, 0.0 to 2.0, 4.0 to 2.0)) {
            val ground = map(overlay().toGround(), px, py)
            assertPoint(map(view, ground.x, ground.y).x, map(view, ground.x, ground.y).y,
                        map(t, px, py), "corner $px,$py")
        }
    }

    @Test
    fun `reads probability back by ground position`() {
        val prob = FloatArray(8) { it.toFloat() }
        val o = overlay(prob)
        // Cell centres: east 1050,1150,1250,1350; north 5150 (row 0), 5050 (row 1).
        assertEquals(0f, o.atGround(1050.0, 5150.0))
        assertEquals(3f, o.atGround(1350.0, 5150.0))
        assertEquals(4f, o.atGround(1050.0, 5050.0), "row 1 is the southern row")
        assertEquals(7f, o.atGround(1350.0, 5050.0))
    }

    @Test
    fun `outside the overlay reads as nodata, not as zero probability`() {
        val o = overlay(FloatArray(8) { 0.9f })
        assertTrue(o.atGround(999.0, 5100.0).isNaN(), "west of the overlay")
        assertTrue(o.atGround(1401.0, 5100.0).isNaN(), "east")
        assertTrue(o.atGround(1200.0, 5201.0).isNaN(), "north")
        assertTrue(o.atGround(1200.0, 4999.0).isNaN(), "south")
        assertTrue(o.at(-1, 0).isNaN())
        assertTrue(o.at(0, 4).isNaN())
    }

    @Test
    fun `rejects a raster that does not match its declared size`() {
        assertFailsWith<IllegalArgumentException> {
            Overlay(FloatArray(7), 4, 2, 0.0, 0.0, 1.0, 1.0, "EPSG:4326", "s", "m")
        }
    }

    @Test
    fun `rejects an inverted or empty ground extent`() {
        assertFailsWith<IllegalArgumentException> {
            Overlay(FloatArray(8), 4, 2, 1400.0, 5000.0, 1000.0, 5200.0,
                    "EPSG:32611", "s", "m")
        }
    }

    @Test
    fun `describes its own provenance`() {
        val d = overlay().describe()
        assertTrue(d.contains("USGS 1 Meter"), d)
        assertTrue(d.contains("EPSG:32611"), d)
    }
}

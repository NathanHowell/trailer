package com.github.nathanhowell.trailer

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The decode path runs against a real captured service response
 * (`exportimage-64.tif`), not a synthesised TIFF. The point of the fixture is
 * that it is what ArcGIS actually emits — uncompressed, internally tiled, single
 * band float32 — which is what decides whether the JDK reader suffices. A TIFF
 * we wrote ourselves would test our writer.
 *
 * Expected values come from reading the same bytes with rasterio.
 */
class Dem3depTest {

    private fun fixture(name: String): ByteArray =
        javaClass.getResourceAsStream("/$name")?.readBytes()
            ?: error("missing test fixture $name")

    private val bounds = Dem3dep.Bounds(340_000.0, 4_070_000.0, 340_256.0, 4_070_256.0)

    @Test
    fun `decodes a real service response to metres`() {
        val g = Dem3dep.decodeTiff(fixture("exportimage-64.tif"))
        assertEquals(64, g.width)
        assertEquals(64, g.height)
        assertEquals(64 * 64, g.finiteCount(), "no nodata expected in this window")
        // rasterio on the same bytes: centre 3610.34326171875, corners as below.
        assertEquals(3610.3433f, g[32, 32], 1e-3f)
        assertEquals(3637.8367f, g[0, 0], 1e-3f)
        assertEquals(3624.2859f, g[0, 63], 1e-3f)
        assertEquals(3603.8933f, g[63, 0], 1e-3f)
        assertEquals(3589.9192f, g[63, 63], 1e-3f)
    }

    @Test
    fun `rejects a JSON error body served with HTTP 200`() {
        // The service reports failures this way, so status code is not a check.
        val body = """{"error":{"code":400,"message":"Unable to complete operation."}}"""
        val e = assertFailsWith<IllegalArgumentException> {
            Dem3dep.decodeTiff(body.toByteArray())
        }
        assertTrue(e.message!!.contains("Unable to complete"),
                   "the service's own message should survive: ${e.message}")
    }

    @Test
    fun `maps fill values to NaN rather than trusting the declared nodata`() {
        // nodata came back null on every tile tried, so the sentinel is unusable
        // and the guard is on magnitude instead.
        assertTrue(Dem3dep.sanitize(-3.4e38f).isNaN(), "float32 fill")
        assertTrue(Dem3dep.sanitize(-9999999f).isNaN(), "common DEM fill")
        assertTrue(Dem3dep.sanitize(Float.NaN).isNaN(), "NaN passes through")
        // Real elevation must survive, including below sea level: Badwater is
        // -86 m and Death Valley has 3DEP 1 m coverage.
        assertEquals(-86.0f, Dem3dep.sanitize(-86.0f))
        assertEquals(0.0f, Dem3dep.sanitize(0.0f))
        assertEquals(4421.0f, Dem3dep.sanitize(4421.0f), "Whitney")
    }

    // ------------------------------------------------------------- coverage

    private fun catalog(vararg rows: Pair<Double, String>) = """
        {"features":[${rows.joinToString(",") {
            """{"attributes":{"LowPS":${it.first},"title":"${it.second}"}}"""
        }}]}
    """.trimIndent()

    @Test
    fun `accepts a window with 1 m LiDAR coverage`() {
        // Shape of a real Sierra response.
        val c = Dem3dep.parseCoverage(catalog(
            10.3074 to "USGS 1/3 Arc Second n37w119 20260610",
            1.0 to "USGS 1 Meter 11 x37y408 CA_SierraNevada_B22",
            30.9221 to "USGS 1 Arc Second n37w119 20260610"))
        assertEquals(1.0, c.bestPixelSizeM, 1e-9)
        assertEquals(3, c.sourceCount)
        assertTrue(c.usable)
        assertTrue(c.title!!.contains("1 Meter"), "reports the source it would use")
    }

    @Test
    fun `refuses Alaska, where the best source is 5 m IFSAR`() {
        // Measured at Denali and the Brooks Range: best LowPS 5.0, no 1 m product.
        val c = Dem3dep.parseCoverage(catalog(
            5.0 to "USGS Alaska 5 Meter AK_IFSAR_2010 80",
            30.9221 to "USGS 1 Arc Second n63w151"))
        assertEquals(5.0, c.bestPixelSizeM, 1e-9)
        assertFalse(c.usable, "5 m must not pass; the model would read interpolation")
    }

    @Test
    fun `refuses a window served only from the 1_3 arc-second fallback`() {
        val c = Dem3dep.parseCoverage(catalog(10.3074 to "USGS 1/3 Arc Second"))
        assertFalse(c.usable)
    }

    @Test
    fun `treats an empty catalog as unusable rather than as 1 m`() {
        val c = Dem3dep.parseCoverage("""{"features":[]}""")
        assertEquals(0, c.sourceCount)
        assertFalse(c.usable, "no coverage must fail closed")
    }

    @Test
    fun `surfaces a catalog error instead of reporting no coverage`() {
        assertFailsWith<IllegalStateException> {
            Dem3dep.parseCoverage("""{"error":{"code":400,"message":"bad"}}""")
        }
    }

    // ------------------------------------------------------------- requests

    @Test
    fun `export request asks for float data in the target projection`() {
        val url = Dem3dep.exportImageUrl(bounds, 32611, 256, 256)
        // Each of these was verified against the live service; an 8-bit or
        // Web Mercator request would return something that looks fine and is not.
        for (expected in listOf("format=tiff", "pixelType=F32", "f=image",
                                "bboxSR=32611", "imageSR=32611", "size=256%2C256",
                                "noDataInterpretation=esriNoDataMatchAny")) {
            assertTrue(url.contains(expected), "missing $expected in $url")
        }
    }

    @Test
    fun `export request refuses to exceed the service limit`() {
        assertFailsWith<IllegalArgumentException> {
            Dem3dep.exportImageUrl(bounds, 32611, Dem3dep.MAX_IMAGE_PX + 1, 256)
        }
    }

    @Test
    fun `catalog request carries the geometry type`() {
        // Omitting it makes the service silently ignore the spatial filter and
        // return the entire national catalog, which parses fine and is nonsense.
        val url = Dem3dep.catalogUrl(bounds, 32611)
        assertTrue(url.contains("geometryType=esriGeometryEnvelope"), url)
        assertTrue(url.contains("where=Category+%3D+1"),
                   "must exclude overview pyramids: $url")
        assertTrue(url.contains("outFields=LowPS%2Ctitle"), url)
    }

    @Test
    fun `bounds reject an empty window`() {
        assertFailsWith<IllegalArgumentException> {
            Dem3dep.Bounds(1.0, 1.0, 1.0, 2.0)
        }
    }
}

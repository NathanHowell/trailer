package com.github.nathanhowell.trailer

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import org.openstreetmap.josm.tools.HttpClient
import java.io.ByteArrayInputStream
import java.net.URL
import java.net.URLEncoder
import javax.imageio.ImageIO

/**
 * Bare-earth elevation from the USGS 3DEP ImageServer.
 *
 * The model's input contract is elevation in metres, so this fetches *data*, not
 * a picture. That rules out the obvious shortcut of declaring 3DEP a JOSM
 * imagery layer and reusing tiles JOSM already has, for two independent reasons
 * that were both checked rather than assumed:
 *
 * 1. The service has no tiles at all — its metadata reports `cacheType = None`
 *    and `tileInfo = null`, so `exportImage` is the only way in.
 * 2. JOSM imagery layers deliver `BufferedImage`, i.e. 8 bits per channel. Over
 *    a tile with 200 m of relief that is ~0.78 m per level, against a tread
 *    signature of 15–100 mm. The signal is quantised away roughly 50-fold.
 *
 * So: `exportImage` with `format=tiff` and `pixelType=F32`, requested directly
 * in the target UTM zone. Never Web Mercator — at 37°N its scale factor is about
 * 1.25, so a "1 m" Mercator pixel is 0.8 m on the ground, and every window in
 * the model is defined in metres.
 */
object Dem3dep {

    const val BASE =
        "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"

    /** Service limit, confirmed from its metadata: one request covers 8000x8000. */
    const val MAX_IMAGE_PX = 8000

    /**
     * Coarsest source we will run the model on, in metres.
     *
     * 3DEP is 1 m only where LiDAR has been flown. Elsewhere it silently falls
     * back to 1/3 arc-second (10.3 m) or 1 arc-second (30.9 m) and resamples up
     * to whatever pixel size was asked for — the response looks like 1 m data and
     * is not. Alaska is served from 5 m IFSAR. A model fed any of those is
     * reading interpolation, so the fetch refuses instead.
     *
     * 1.5 rather than 1.0 because the catalog reports nominal product sizes and
     * we want the next real tier (5 m) to fail, not floating-point slop.
     */
    const val MAX_SOURCE_PIXEL_M = 1.5

    /**
     * Anything below this is nodata, however the service labelled it.
     *
     * It has returned `nodata = null` on every tile tried, so the declared
     * sentinel cannot be relied on.
     */
    const val ABSURD_M = -1.0e5f

    private val mapper = ObjectMapper()

    /**
     * The model's nodata contract is NaN, and the service's declared sentinel
     * cannot be trusted — it has reported `nodata = null` on every tile tried
     * while still returning large negative fill. So the guard is on magnitude.
     */
    fun sanitize(v: Float): Float = if (v.isNaN() || v < ABSURD_M) Float.NaN else v

    /** A projected bounding box, in the units of [epsg]. */
    data class Bounds(val minX: Double, val minY: Double,
                      val maxX: Double, val maxY: Double) {
        init {
            require(maxX > minX && maxY > minY) { "empty bounds: $this" }
        }
        val width get() = maxX - minX
        val height get() = maxY - minY
    }

    /** Elevation in metres, row-major, nodata as NaN. */
    class Grid(val values: FloatArray, val width: Int, val height: Int) {
        init {
            require(values.size == width * height) {
                "expected ${width * height} samples, got ${values.size}"
            }
        }
        operator fun get(row: Int, col: Int) = values[row * width + col]
        fun finiteCount() = values.count { it.isFinite() }
    }

    /**
     * What the service would actually serve here, from its own catalog.
     *
     * Deliberately metadata rather than a content test. A content-based detector
     * was calibrated first and abandoned: against 24 real 1 m tiles and copies
     * decimated to 3/8/10 m and resampled back, detail-energy ratios and
     * structure-function ratios all overlapped completely (real `hf/mf` 0.10–0.22
     * against 0.10–0.16 for 10 m-upsampled). The published 1 m product is itself
     * hydro-flattened and interpolated, so its 1 m detail band is already weak
     * enough to look like something upsampled. The catalog just says.
     */
    data class Coverage(val bestPixelSizeM: Double, val title: String?,
                        val sourceCount: Int) {
        val usable get() = bestPixelSizeM <= MAX_SOURCE_PIXEL_M
    }

    class UnsupportedCoverage(val coverage: Coverage) : Exception(
        "3DEP serves ${"%.1f".format(coverage.bestPixelSizeM)} m here " +
            "(${coverage.title ?: "unknown source"}); trail detection needs 1 m " +
            "LiDAR and will not run on resampled data")

    // ---------------------------------------------------------------- queries

    private fun query(params: List<Pair<String, String>>) = params.joinToString("&") {
        "${it.first}=${URLEncoder.encode(it.second, Charsets.UTF_8)}"
    }

    fun exportImageUrl(bounds: Bounds, epsg: Int, width: Int, height: Int): String {
        require(width in 1..MAX_IMAGE_PX && height in 1..MAX_IMAGE_PX) {
            "$width x $height exceeds the service limit of $MAX_IMAGE_PX"
        }
        return "$BASE/exportImage?" + query(listOf(
            "bbox" to "${bounds.minX},${bounds.minY},${bounds.maxX},${bounds.maxY}",
            "bboxSR" to "$epsg",
            "imageSR" to "$epsg",
            "size" to "$width,$height",
            "format" to "tiff",
            "pixelType" to "F32",
            "noDataInterpretation" to "esriNoDataMatchAny",
            "interpolation" to "RSP_BilinearInterpolation",
            "f" to "image",
        ))
    }

    fun catalogUrl(bounds: Bounds, epsg: Int): String {
        val envelope = """{"xmin":${bounds.minX},"ymin":${bounds.minY},""" +
            """"xmax":${bounds.maxX},"ymax":${bounds.maxY},""" +
            """"spatialReference":{"wkid":$epsg}}"""
        return "$BASE/query?" + query(listOf(
            "geometry" to envelope,
            // Omitting geometryType makes the service ignore the filter entirely
            // and return the whole national catalog, which looks like success.
            "geometryType" to "esriGeometryEnvelope",
            "inSR" to "$epsg",
            "spatialRel" to "esriSpatialRelIntersects",
            "where" to "Category = 1",   // primary rasters, not overview pyramids
            "outFields" to "LowPS,title",
            "returnGeometry" to "false",
            "f" to "json",
        ))
    }

    // ---------------------------------------------------------------- parsing

    /** Smallest `LowPS` among the primary rasters covering the window. */
    fun parseCoverage(json: String): Coverage {
        val root: JsonNode = mapper.readTree(json)
        root.get("error")?.let {
            throw IllegalStateException("3DEP catalog error: ${it.toString().take(300)}")
        }
        var best: JsonNode? = null
        var bestPs = Double.MAX_VALUE
        var n = 0
        for (f in root.path("features")) {
            val ps = f.path("attributes").path("LowPS")
            if (!ps.isNumber) continue
            n++
            if (ps.asDouble() < bestPs) {
                bestPs = ps.asDouble()
                best = f
            }
        }
        if (best == null) return Coverage(Double.MAX_VALUE, null, 0)
        return Coverage(bestPs, best.path("attributes").path("title").asText(null), n)
    }

    /**
     * Decode a float32 TIFF into elevation, mapping nodata to NaN.
     *
     * Uses the JDK's own TIFF reader rather than a bundled library. Note the
     * `read` rather than `readRaster` path: `TIFFImageReader.readRaster` throws
     * `UnsupportedOperationException`, while `read` returns a float raster. This
     * was verified byte-identical against rasterio on a real service response
     * (uncompressed, internally tiled, single band, little-endian).
     */
    fun decodeTiff(bytes: ByteArray): Grid {
        // The service reports failures as JSON with HTTP 200, so the status code
        // is not a success check. The magic number is.
        require(bytes.size >= 4) { "response too short to be a TIFF (${bytes.size} bytes)" }
        val magic = String(bytes, 0, 2, Charsets.US_ASCII)
        require(magic == "II" || magic == "MM") {
            "expected a TIFF, got: " + String(bytes, 0, minOf(bytes.size, 300), Charsets.UTF_8)
        }
        ImageIO.createImageInputStream(ByteArrayInputStream(bytes)).use { input ->
            val readers = ImageIO.getImageReaders(input)
            check(readers.hasNext()) { "no ImageIO reader for the response" }
            val reader = readers.next()
            try {
                reader.input = input
                val raster = reader.read(0).raster
                val w = raster.width
                val h = raster.height
                val out = FloatArray(w * h)
                var i = 0
                for (r in 0 until h) {
                    for (c in 0 until w) {
                        out[i++] = sanitize(raster.getSampleFloat(c, r, 0))
                    }
                }
                return Grid(out, w, h)
            } finally {
                reader.dispose()
            }
        }
    }

    // ---------------------------------------------------------------- network

    private fun get(url: String, timeoutMs: Int = 180_000): ByteArray =
        HttpClient.create(URL(url))
            .setConnectTimeout(30_000)
            .setReadTimeout(timeoutMs)
            .setHeader("Accept", "*/*")
            .connect()
            .also {
                if (it.responseCode !in 200..299) {
                    throw IllegalStateException(
                        "3DEP returned HTTP ${it.responseCode} ${it.responseMessage}")
                }
            }
            .content.use { it.readBytes() }

    fun coverage(bounds: Bounds, epsg: Int): Coverage =
        parseCoverage(String(get(catalogUrl(bounds, epsg), 90_000), Charsets.UTF_8))

    /**
     * Elevation for a window, at [width] x [height] samples.
     *
     * Checks coverage first. Fetching and then discovering the data was 10 m
     * wastes a multi-megabyte download, and — worse — nothing downstream could
     * tell afterwards.
     */
    fun fetch(bounds: Bounds, epsg: Int, width: Int, height: Int): Grid {
        val cov = coverage(bounds, epsg)
        if (!cov.usable) throw UnsupportedCoverage(cov)
        return decodeTiff(get(exportImageUrl(bounds, epsg, width, height)))
    }
}

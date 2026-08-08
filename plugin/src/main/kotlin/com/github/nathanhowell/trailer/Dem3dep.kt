package com.github.nathanhowell.trailer

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import org.openstreetmap.josm.tools.HttpClient
import java.io.ByteArrayInputStream
import java.net.URL
import java.net.URLEncoder
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneOffset
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
     * Sample spacing for the completeness test, in metres, and the grid bounds it
     * is clamped to. See [coveredFraction] for what this buys and what it misses.
     */
    const val COVERAGE_STEP_M = 50.0
    const val COVERAGE_MIN_SAMPLES = 16
    const val COVERAGE_MAX_SAMPLES = 128

    /** A window is complete enough to run on only if every sample is backed. */
    const val MIN_COVERED_FRACTION = 1.0

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
     *
     * [bestPixelSizeM] alone is not enough, which is why [coveredFraction] exists.
     * It is a minimum over rasters *intersecting* the window, so a view straddling
     * the edge of a LiDAR project reports 1.0 m on the strength of a raster that
     * clips one corner. Measured on a real 2 km window at the west edge of the
     * Oregon project near 421000E 4914000N: `LowPS` 1.0, and 76% of the window
     * served from the 10.3 m fallback. At 1 km the same window is 1.0 m by
     * `LowPS` and *nothing* by area.
     */
    data class Coverage(val bestPixelSizeM: Double, val title: String?,
                        val sourceCount: Int, val coveredFraction: Double,
                        val acquired: ClosedRange<LocalDate>? = null) {
        val fineEnough get() = bestPixelSizeM <= MAX_SOURCE_PIXEL_M
        val complete get() = coveredFraction >= MIN_COVERED_FRACTION
        val usable get() = fineEnough && complete

        /**
         * What a mapper needs to judge and cite this overlay.
         *
         * The acquisition date is a *range* because a window can legitimately span
         * two LiDAR projects flown years apart — the four-tile corner in the
         * Sierra fixture mixes 2020 and 2022 — and a trail cut after the older
         * survey will be missing from part of the view for reasons that have
         * nothing to do with the model.
         */
        fun describe(): String = buildString {
            append(title ?: "unknown source")
            append(" · ").append("%.2g".format(bestPixelSizeM)).append(" m")
            acquired?.let {
                append(" · flown ")
                append(if (it.start.isEqual(it.endInclusive)) "${it.start}"
                       else "${it.start} to ${it.endInclusive}")
            }
        }
    }

    class UnsupportedCoverage(val coverage: Coverage) : Exception(
        if (!coverage.fineEnough)
            "3DEP serves ${"%.1f".format(coverage.bestPixelSizeM)} m here " +
                "(${coverage.title ?: "unknown source"}); trail detection needs " +
                "1 m LiDAR and will not run on resampled data"
        else
            "only ${"%.0f".format(coverage.coveredFraction * 100)}% of this view " +
                "has 1 m LiDAR (${coverage.title ?: "unknown source"}); the rest " +
                "is resampled from a coarser source. Zoom in past the edge of the " +
                "LiDAR project and try again")

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
            // AcquisitionDate is for provenance, not for the coverage decision: a
            // mapper needs to know whether a trail could postdate the survey.
            "outFields" to "LowPS,title,AcquisitionDate",
            // Footprints, not just pixel sizes: intersecting is not covering, and
            // without the geometry there is no way to tell the difference.
            "returnGeometry" to "true",
            // In the same SR as the window, so the rings are comparable to it.
            "outSR" to "$epsg",
            "f" to "json",
        ))
    }

    // ---------------------------------------------------------------- parsing

    /**
     * A catalog footprint: rings of projected vertices, plus a bbox to reject on.
     *
     * ArcGIS densifies the edges — a 10 km raster comes back as an 85-vertex ring
     * — so these are not four-corner rectangles even when the footprint is one.
     * Requested in the window's own SR they are axis-aligned; reprojected across
     * a UTM zone they bow by ~600 m over 10 km, which is why the test below works
     * on the rings themselves and never on their bounding boxes.
     */
    private class Footprint(val rings: List<DoubleArray>) {
        var minX = Double.MAX_VALUE; var minY = Double.MAX_VALUE
        var maxX = -Double.MAX_VALUE; var maxY = -Double.MAX_VALUE

        init {
            for (r in rings) {
                var i = 0
                while (i < r.size) {
                    if (r[i] < minX) minX = r[i]
                    if (r[i] > maxX) maxX = r[i]
                    if (r[i + 1] < minY) minY = r[i + 1]
                    if (r[i + 1] > maxY) maxY = r[i + 1]
                    i += 2
                }
            }
        }

        /** Even-odd ray crossing, so an interior ring reads as a hole. */
        fun contains(x: Double, y: Double): Boolean {
            if (x < minX || x > maxX || y < minY || y > maxY) return false
            var odd = false
            for (r in rings) {
                val n = r.size / 2
                var j = n - 1
                for (i in 0 until n) {
                    val yi = r[2 * i + 1]
                    val yj = r[2 * j + 1]
                    if ((yi > y) != (yj > y)) {
                        val xi = r[2 * i]
                        val xj = r[2 * j]
                        if (x < xi + (y - yi) * (xj - xi) / (yj - yi)) odd = !odd
                    }
                    j = i
                }
            }
            return odd
        }
    }

    private fun footprint(f: JsonNode): Footprint? {
        val rings = f.path("geometry").path("rings")
        if (!rings.isArray || rings.isEmpty) return null
        val out = ArrayList<DoubleArray>(rings.size())
        for (ring in rings) {
            if (!ring.isArray || ring.size() < 3) continue
            val a = DoubleArray(ring.size() * 2)
            var i = 0
            for (p in ring) {
                if (p.size() < 2) return null
                a[i++] = p.get(0).asDouble()
                a[i++] = p.get(1).asDouble()
            }
            out.add(a)
        }
        return if (out.isEmpty()) null else Footprint(out)
    }

    /**
     * Fraction of [bounds] backed by a footprint at or below [MAX_SOURCE_PIXEL_M].
     *
     * Sampled on a grid whose endpoints are inclusive, so the window's corners and
     * edges are tested — that is where a project boundary actually clips a view.
     * Spacing is [COVERAGE_STEP_M] clamped to [COVERAGE_MIN_SAMPLES]..
     * [COVERAGE_MAX_SAMPLES] per axis, i.e. at worst 62 m across a full-size 8 km
     * request.
     *
     * What this misses, stated plainly: an uncovered strip narrower than the
     * spacing and falling between samples. That is not the failure this guards
     * against. Real gaps are LiDAR project boundaries, which are kilometres
     * across; the seams *inside* a project are not gaps at all, since adjacent
     * 10 km tiles are published with about 7 m of overlap.
     *
     * A feature with no usable ring contributes nothing, so a service that ignores
     * `returnGeometry` reads as no coverage rather than as full coverage.
     */
    private fun coveredFraction(features: List<JsonNode>, bounds: Bounds): Double {
        val prints = features.mapNotNull { footprint(it) }
        if (prints.isEmpty()) return 0.0
        val nx = samplesAcross(bounds.width)
        val ny = samplesAcross(bounds.height)
        var hit = 0
        for (i in 0 until nx) {
            val x = bounds.minX + bounds.width * i / (nx - 1.0)
            for (j in 0 until ny) {
                val y = bounds.minY + bounds.height * j / (ny - 1.0)
                if (prints.any { it.contains(x, y) }) hit++
            }
        }
        return hit.toDouble() / (nx * ny)
    }

    private fun samplesAcross(span: Double): Int =
        Math.ceil(span / COVERAGE_STEP_M).toInt().coerceIn(
            COVERAGE_MIN_SAMPLES, COVERAGE_MAX_SAMPLES)

    /**
     * Smallest `LowPS` among the primary rasters intersecting the window, and how
     * much of the window the fine ones actually cover.
     */
    fun parseCoverage(json: String, bounds: Bounds): Coverage {
        val root: JsonNode = mapper.readTree(json)
        root.get("error")?.let {
            throw IllegalStateException("3DEP catalog error: ${it.toString().take(300)}")
        }
        var best: JsonNode? = null
        var bestPs = Double.MAX_VALUE
        var n = 0
        val fine = ArrayList<JsonNode>()
        for (f in root.path("features")) {
            val ps = f.path("attributes").path("LowPS")
            if (!ps.isNumber) continue
            n++
            if (ps.asDouble() <= MAX_SOURCE_PIXEL_M) fine.add(f)
            if (ps.asDouble() < bestPs) {
                bestPs = ps.asDouble()
                best = f
            }
        }
        if (best == null) return Coverage(Double.MAX_VALUE, null, 0, 0.0)
        return Coverage(bestPs, best.path("attributes").path("title").asText(null),
                        n, coveredFraction(fine, bounds), acquired(fine))
    }

    /** Acquisition span of the rasters we would actually read, or null if unstated. */
    private fun acquired(features: List<JsonNode>): ClosedRange<LocalDate>? {
        val dates = features.mapNotNull {
            val ms = it.path("attributes").path("AcquisitionDate")
            // Epoch milliseconds, UTC. Absent on some older products.
            if (ms.isNumber && ms.asLong() > 0)
                Instant.ofEpochMilli(ms.asLong()).atZone(ZoneOffset.UTC).toLocalDate()
            else null
        }
        return if (dates.isEmpty()) null else dates.min()..dates.max()
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
        parseCoverage(String(get(catalogUrl(bounds, epsg), 90_000), Charsets.UTF_8),
                      bounds)

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

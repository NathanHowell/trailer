package com.github.nathanhowell.trailer

import org.junit.jupiter.api.Test
import java.nio.file.Paths
import kotlin.test.assertEquals

/**
 * Only the pure part. Opening a session needs a configured preference and a
 * JOSM `Config`, which is a running editor; the pairing rule below is the bit
 * with an actual decision in it, and it has to agree with Python.
 */
class ModelStoreTest {

    @Test
    fun `pairs an onnx with the sidecar python wrote beside it`() {
        // `cmd_export` uses `out.with_suffix(".json")`, which *replaces* the
        // final extension. Appending instead would look for trailer.onnx.json
        // and report a missing sidecar for a model that has one.
        assertEquals(Paths.get("/m/trailer.json"),
                     ModelStore.sidecarFor(Paths.get("/m/trailer.onnx")))
    }

    @Test
    fun `replaces only the last extension`() {
        // A versioned filename is the case that separates "replace the suffix"
        // from "strip everything after the first dot".
        assertEquals(Paths.get("/m/dem1.v3.json"),
                     ModelStore.sidecarFor(Paths.get("/m/dem1.v3.onnx")))
    }

    @Test
    fun `handles a name with no extension at all`() {
        assertEquals(Paths.get("/m/model.json"),
                     ModelStore.sidecarFor(Paths.get("/m/model")))
    }

    @Test
    fun `handles a dotfile, which has no stem to replace`() {
        // lastIndexOf('.') is 0 here; treating that as an extension would leave
        // an empty stem and produce ".json".
        assertEquals(Paths.get("/m/.onnx.json"),
                     ModelStore.sidecarFor(Paths.get("/m/.onnx")))
    }

    @Test
    fun `keeps the sidecar in the model's own directory`() {
        // resolveSibling, not resolve against the working directory: JOSM's cwd
        // is wherever it was launched from and has nothing to do with the model.
        val s = ModelStore.sidecarFor(Paths.get("/deep/nested/dir/m.onnx"))
        assertEquals(Paths.get("/deep/nested/dir"), s.parent)
    }
}

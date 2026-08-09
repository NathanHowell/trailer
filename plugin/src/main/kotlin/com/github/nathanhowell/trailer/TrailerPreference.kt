package com.github.nathanhowell.trailer

import org.openstreetmap.josm.gui.preferences.DefaultTabPreferenceSetting
import org.openstreetmap.josm.gui.preferences.PreferenceTabbedPane
import org.openstreetmap.josm.spi.preferences.Config
import java.awt.BorderLayout
import java.awt.Color
import java.awt.GridBagConstraints
import java.awt.GridBagLayout
import java.awt.Insets
import java.io.File
import java.nio.file.Path
import java.nio.file.Paths
import javax.swing.BorderFactory
import javax.swing.JButton
import javax.swing.JFileChooser
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.JTextArea
import javax.swing.JTextField
import javax.swing.filechooser.FileNameExtensionFilter

/**
 * Where a mapper points the plugin at a model.
 *
 * The weights are not in the jar — ~99 MB, a different release cadence, and a
 * different licence — so a path is the one thing this plugin genuinely has to
 * ask for. Until this existed the only way to set it was to hand-edit JOSM's
 * preferences file, which is not a thing to ask of anyone.
 *
 * The file is checked the moment it is chosen, by [ModelChoice], and what it
 * turned out to be is shown straight away. Validating on OK, or on first use,
 * would put the complaint minutes and one 3DEP download away from the choice
 * that caused it, with nothing on screen connecting the two.
 */
class TrailerPreference : DefaultTabPreferenceSetting(
    "preferences/imagery",
    "Trail probability",
    "Where to find the exported trail-detection model",
) {

    private val path = JTextField(40)
    private val status = JTextArea(4, 40).apply {
        isEditable = false
        isFocusable = false
        isOpaque = false
        lineWrap = true
        wrapStyleWord = true
        font = JLabel().font
    }

    override fun addGui(gui: PreferenceTabbedPane) {
        val panel = JPanel(GridBagLayout())
        panel.border = BorderFactory.createEmptyBorder(8, 8, 8, 8)
        val c = GridBagConstraints().apply {
            insets = Insets(4, 4, 4, 4)
            anchor = GridBagConstraints.WEST
            fill = GridBagConstraints.HORIZONTAL
        }

        c.gridx = 0; c.gridy = 0; c.weightx = 0.0
        panel.add(JLabel("Model (.onnx):"), c)

        path.text = Config.getPref().get(ModelStore.PREF_MODEL, "")
        c.gridx = 1; c.weightx = 1.0
        panel.add(path, c)

        val browse = JButton("Browse…")
        browse.addActionListener { choose() }
        c.gridx = 2; c.weightx = 0.0
        panel.add(browse, c)

        c.gridx = 1; c.gridy = 1; c.gridwidth = 2; c.weightx = 1.0
        panel.add(status, c)

        c.gridy = 2
        panel.add(JLabel("<html><body style='width:420px'>" +
            "Produced by <code>trailer export</code>, which writes the .onnx " +
            "and a .json sidecar beside it. Both are needed. The weights are " +
            "CC BY-SA 4.0 and are licensed separately from this plugin; the " +
            "attribution they carry is shown on the layer." +
            "</body></html>"), c)

        // Whatever is already configured gets the same scrutiny as a fresh
        // choice: a model that was valid when it was picked may have been moved
        // or replaced by an export from a newer trailer since.
        refresh()
        path.document.addDocumentListener(object : javax.swing.event.DocumentListener {
            override fun insertUpdate(e: javax.swing.event.DocumentEvent?) = refresh()
            override fun removeUpdate(e: javax.swing.event.DocumentEvent?) = refresh()
            override fun changedUpdate(e: javax.swing.event.DocumentEvent?) = refresh()
        })

        val wrapper = JPanel(BorderLayout())
        wrapper.add(panel, BorderLayout.NORTH)
        createPreferenceTabWithScrollPane(gui, wrapper)
    }

    private fun choose() {
        val fc = JFileChooser()
        fc.dialogTitle = "Select an exported trail model"
        fc.fileFilter = FileNameExtensionFilter("ONNX model (*.onnx)", "onnx")
        current()?.let { fc.currentDirectory = File(it.parent?.toString() ?: ".") }
        if (fc.showOpenDialog(path) == JFileChooser.APPROVE_OPTION) {
            path.text = fc.selectedFile.absolutePath
        }
    }

    private fun current(): Path? =
        path.text.trim().takeIf { it.isNotEmpty() }?.let { Paths.get(it) }

    private fun refresh() {
        val p = current()
        if (p == null) {
            say("No model selected. The overlay cannot run without one.",
                neutral = true)
            return
        }
        when (val check = ModelChoice.inspect(p)) {
            is ModelChoice.Check.Ok -> say("✓ " + check.summary.describe(),
                                           neutral = true)
            is ModelChoice.Check.Bad -> say(check.reason, neutral = false)
        }
    }

    private fun say(message: String, neutral: Boolean) {
        status.text = message
        // Not colour alone — the tick and the wording carry it too. Roughly a
        // fifth of men have some red-green deficiency, and a mapper who cannot
        // see the difference should still be able to read it.
        status.foreground = if (neutral) JLabel().foreground else Color(0xB0, 0x20, 0x00)
    }

    override fun ok(): Boolean {
        val text = path.text.trim()
        Config.getPref().put(ModelStore.PREF_MODEL, text.ifEmpty { null })
        // Drop any open session: the path may now point somewhere else, and a
        // cached model from the old one would keep painting as if nothing had
        // changed. ModelStore reopens lazily on the next run.
        ModelStore.close()
        return false   // no restart required
    }

    override fun isExpert(): Boolean = false
}

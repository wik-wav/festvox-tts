"""festvox_gui.py -- Windows-XP styled desktop GUI for the pure-Python
diphone renderer and real Festival synthesis through WSL.

PyQt5 + PyQtGraph. What it really does:
  * Languages = the engine's real front ends: Asaxi / English (CMU) / Japanese
  * Voicebanks = diphone DBs plus path-backed or installed Festival voices
  * Generate uses synth_diphone.render() or Festival/UniSyn TD-PSOLA in WSL
  * waveform view with draggable red phoneme boundaries (time-stretch DSP)
  * editable phoneme fields under the waveform; Re-render feeds your edited
    phone list straight back through the engine (e.g. override r -> rr)
  * waveform-aligned timing, pitch-curve and punctuation-intonation editors
  * visible automatic/manual recording takes and diagnostic output faults
  * sentence/phrase routing, cached projects, merged/separate WAV export

Run:  python festvox_gui.py
Deps: pip install -r requirements.txt   (optional: sounddevice, librosa,
      cmudict for English)
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import re
import sys
import uuid
import wave
from collections import OrderedDict
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt
    import pyqtgraph as pg
except Exception as e:  # pragma: no cover
    sys.stderr.write("This GUI needs PyQt5 and pyqtgraph:\n"
                     "    pip install -r requirements.txt\n\n%s\n" % e)
    raise

import festvox_core as fc

FESTVOX_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(FESTVOX_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(FESTVOX_TOOL_DIR))

import japanese_editing as je
import japanese_assembly as japanese_assembly
import japanese_frontend as japanese_frontend
import japanese_profiles as japanese_profiles
import japanese_devoicing as japanese_devoicing
import asaxi_editing as asaxi_editing
import asaxi_phonation as asaxi_phonation
import asaxi_prosody as asaxi_prosody
import diphone_loudness as diphone_loudness
import english_syllables as english_syllables
import join_spectrogram as join_spectrogram
import rendered_formant_diagnostic as rendered_formant_diagnostic
import vocal_tract as vocal_tract
import vocal_tract_validation as vocal_tract_validation

pg.setConfigOptions(antialias=True)

MIN_SEG = 0.010  # s, smallest phoneme duration a boundary can create
WAVEFORM_LEFT_LIMIT = -0.25  # s, shared pre-zero panning room
WAVEFORM_SUMMARY_BLOCK = 16
WAVEFORM_SUMMARY_GROWTH = 2
WAVEFORM_RAW_SAMPLES_PER_PIXEL = 2.0
WAVEFORM_CONNECTED_SAMPLES_PER_PIXEL = 16.0
BOUNDARY_DETAIL_MIN_PX = 18.0
BOUNDARY_OVERVIEW_BUCKET_PX = 6.0
PHONE_LABEL_MIN_PX = 24.0
PHONE_FIELD_MIN_PX = 24.0
PARAMETER_DETAIL_MIN_PX = 6.0
PARAMETER_OVERVIEW_BUCKET_PX = 6.0
PITCH_POINT_MIN_PX = 10.0
JOIN_DETAIL_MIN_PX = 10.0
JOIN_OVERVIEW_BUCKET_PX = 8.0
LINGUISTIC_DETAIL_MIN_PX = 10.0
LINGUISTIC_OVERVIEW_BUCKET_PX = 8.0
MIXED_SELECTION_DATA = "__festvox_mixed_selection__"
PHRASE_TEXT_SEPARATOR = " [pau] [pau] "
CONFIG_PATH = os.path.join(fc.GUI_DIR, "config.json")
SHORTCUT_SPECS = (
    ("undo", "Undo", "Ctrl+Z"),
    ("redo", "Redo", "Ctrl+Y"),
    ("new_sentence", "New sentence", "Ctrl+N"),
    ("open_project", "Open project", "Ctrl+O"),
    ("save_project", "Save project", "Ctrl+S"),
    ("export_audio", "Export audio", "Ctrl+E"),
    ("generate", "Generate audio", "Ctrl+R"),
    ("rerender", "Re-render", "R"),
    ("play", "Play", "Space"),
    ("stop", "Stop", "Escape"),
    ("select_all", "Select all", "Ctrl+A"),
    ("copy", "Copy selection", "Ctrl+C"),
    ("cut", "Cut selection", "Ctrl+X"),
    ("paste", "Paste", "Ctrl+V"),
    ("duplicate", "Duplicate selection", "Ctrl+D"),
    ("delete", "Delete selection", "Delete"),
)
DEFAULT_SHORTCUTS = {key: shortcut for key, _label, shortcut in SHORTCUT_SPECS}
SHORTCUT_LABELS = {key: label for key, label, _shortcut in SHORTCUT_SPECS}
# phones the English (kal) voice's "radio" phoneset knows -- inline [phones]
# outside this set (e.g. q, cl and other Lem-bank phones) are mapped to pau so
# they render as a brief silence instead of crashing Festival on English text
EN_PHONES = frozenset(
    "aa ae ah ao aw ax ay b ch d dh eh el em en er ey f g hh ih iy jh k l m n "
    "ng ow oy p r s sh t th uh uw v w y z zh pau".split())

# --------------------------------------------------------------- XP Luna theme
XP_QSS = """
* { font-size: 9pt; color: #000; }
QMainWindow, QWidget { background: #ECE9D8; }
QMenuBar { background: #ECE9D8; }
QMenuBar::item:selected { background: #316AC5; color: #fff; }
QMenu { background: #fff; border: 1px solid #808080; }
QMenu::item { min-height: 18px; padding: 3px 28px 3px 24px; }
QMenu::item:selected { background: #316AC5; color: #fff; }
QMenu::item:disabled {
    color: #65625C; background: #E2E0DA;
}
QMenu::item:disabled:selected { color: #65625C; background: #E2E0DA; }
QMenu::separator { height: 1px; background: #B8B5AC; margin: 3px 5px; }
QGroupBox { border: 1px solid #ACA899; margin-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 7px; padding: 0 2px; }
QLabel#hdr { font-weight: bold; }
QPushButton {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #FDFDFD, stop:1 #E3E0D2);
    border: 1px solid #707070; border-radius: 3px; padding: 4px 8px; min-height: 16px;
}
QPushButton:hover { border: 1px solid #E9A700; }
QPushButton:pressed { background: #DEDBCE; }
QPushButton:disabled { color: #A0A0A0; border: 1px solid #B4B0A4; }
QPushButton[renderPending="true"] {
    background: #FFD86A; border: 2px solid #A87500; font-weight: bold;
}
QPushButton[generatePending="true"] {
    background: #FFD86A; border: 2px solid #A87500; font-weight: bold;
}
QPushButton[sentenceGenerate="true"][generatePending="true"] {
    background: #FFF0B8; border: 1px solid #B58B2A; font-weight: bold;
}
QComboBox, QLineEdit, QListWidget, QSpinBox, QDoubleSpinBox {
    background: #fff; border: 1px solid #7F9DB9; padding: 2px; selection-background-color: #316AC5;
}
QComboBox::drop-down { border-left: 1px solid #7F9DB9; width: 16px; }
QListWidget::item:selected { background: #316AC5; color: #fff; }
QSlider::groove:horizontal { height: 4px; background: #B4B0A4; border: 1px solid #808080; }
QSlider::handle:horizontal { width: 11px; background: #ECE9D8; border: 1px solid #404040;
    margin: -6px 0; border-radius: 2px; }
QSlider::groove:vertical { width: 4px; background: #B4B0A4; border: 1px solid #808080; }
QSlider::handle:vertical { height: 11px; background: #ECE9D8; border: 1px solid #404040;
    margin: 0 -6px; border-radius: 2px; }
QStatusBar { background: #ECE9D8; border-top: 1px solid #ACA899; }
QTabWidget::pane { border: 1px solid #ACA899; background: #ECE9D8; }
QTabBar::tab { background: #D8D5C8; border: 1px solid #9C998F;
    padding: 5px 14px; margin-right: 1px; }
QTabBar::tab:selected { background: #FFFFFF; border-bottom-color: #FFFFFF; }
QFrame#sentenceRow { border-bottom: 1px solid #B7B3A7; background: #F4F2E9; }
QFrame#sentenceRow[pending="rerender"] { background: #E2E0DA; }
QFrame#sentenceRow[pending="generate"] { background: #E7E6E1; }
QFrame#sentenceRow[selected="true"] { background: #DCE9FF;
    border: 1px solid #316AC5; }
QFrame#sentenceRow[playing="true"] { background: #FFF7D6;
    border-left: 4px solid #D48A00; }
QFrame#phraseChip { background: #E8E5DA; border: 1px solid #8F8C82;
    border-radius: 3px; }
QFrame#phraseChip[pending="rerender"] { background: #DDDCD8; }
QFrame#phraseChip[pending="generate"] { background: #DEDDD8; }
QFrame#phraseChip[selected="true"] { background: #DCE9FF;
    border: 2px solid #316AC5; }
QFrame#phraseChip[playing="true"] { background: #FFF0B8;
    border: 2px solid #D48A00; }
QPlainTextEdit#sentenceText {
    background: #FFFFFF; border: 1px solid #A9B7C7; border-radius: 3px;
    padding: 6px; margin: 1px 0; font-weight: normal;
    selection-background-color: #316AC5;
}
QLabel#pendingBadge { border: 1px solid #8A6A18; padding: 3px 6px;
    border-radius: 2px; font-weight: bold; }
QLabel#pendingBadge[pending="rerender"] { background: #FFD86A; }
QWidget[gainPending="true"] QSlider::groove:horizontal {
    background: #E6BE55; border: 1px solid #9C761E;
}
QWidget[gainPending="true"] QDoubleSpinBox { background: #FFF3C2; }
QLineEdit#phon { background: #fff; border: 1px solid #7F9DB9; padding: 1px; }
QLineEdit#phon[dirty="true"] { background: #FFF3C2; }
QLineEdit#phon:read-only { background: #E4E2D6; color: #707070; }
QLineEdit#phon[selected="true"] { border: 2px solid #1E6FE0; background: #DCE9FF; }
"""


JAPANESE_FONT_SAMPLE = "\u65e5\u672c\u8a9e\u304b\u306a\u30ab\u30ca"


def configure_qt_high_dpi():
    """Enable Qt 5 high-DPI behavior before QApplication construction."""
    if QtWidgets.QApplication.instance() is not None:
        return False
    changed = False
    for name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        attribute = getattr(Qt, name, None)
        if attribute is not None:
            QtWidgets.QApplication.setAttribute(attribute, True)
            changed = True
    return changed


def font_has_japanese_glyphs(font):
    metrics = QtGui.QFontMetrics(font)
    return all(metrics.inFontUcs4(ord(character))
               for character in JAPANESE_FONT_SAMPLE)


def select_ui_font(point_size=9.0):
    """Choose an installed system UI font with verified Japanese glyphs."""
    candidates = (
        "Yu Gothic UI", "Meiryo UI", "Meiryo", "Noto Sans CJK JP",
        "Segoe UI",
    )
    families = [str(name) for name in QtGui.QFontDatabase().families()]
    by_folded = {name.casefold(): name for name in families}
    ordered = []
    for family in candidates:
        installed = by_folded.get(family.casefold())
        if installed and installed not in ordered:
            ordered.append(installed)
    ordered.extend(name for name in families if name not in ordered)
    for family in ordered:
        font = QtGui.QFont(family)
        font.setPointSizeF(float(point_size))
        if font_has_japanese_glyphs(font):
            return font
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)
    font.setPointSizeF(float(point_size))
    return font


class ArrowProxyStyle(QtWidgets.QProxyStyle):
    """Draw real painter arrows for combos and spin boxes.

    Qt style sheets do not support the CSS border-triangle trick reliably; on
    the Windows/Fusion combination it renders as a small rectangle instead.
    """
    def drawPrimitive(self, element, option, painter, widget=None):
        directions = {
            QtWidgets.QStyle.PE_IndicatorArrowDown: "down",
            QtWidgets.QStyle.PE_IndicatorArrowUp: "up",
            getattr(QtWidgets.QStyle, "PE_IndicatorSpinDown", -1): "down",
            getattr(QtWidgets.QStyle, "PE_IndicatorSpinUp", -2): "up",
        }
        direction = directions.get(element)
        if direction:
            rect = option.rect
            cx, cy = rect.center().x(), rect.center().y()
            if direction == "down":
                points = [QtCore.QPoint(cx - 4, cy - 2),
                          QtCore.QPoint(cx + 4, cy - 2),
                          QtCore.QPoint(cx, cy + 3)]
            else:
                points = [QtCore.QPoint(cx - 4, cy + 2),
                          QtCore.QPoint(cx + 4, cy + 2),
                          QtCore.QPoint(cx, cy - 3)]
            color = "#202020" if option.state & QtWidgets.QStyle.State_Enabled \
                else "#909090"
            painter.save()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QtGui.QColor(color))
            painter.drawPolygon(QtGui.QPolygon(points))
            painter.restore()
            return
        super().drawPrimitive(element, option, painter, widget)


# ------------------------------------------------------------------- widgets
class SpeedSlider(QtWidgets.QSlider):
    """Log-speed slider; double-click resets to x1.0 (value 0)."""
    def mouseDoubleClickEvent(self, ev):
        self.setValue(0)
        ev.accept()


class ResetSlider(QtWidgets.QSlider):
    """Slider whose double click restores a declared default value."""
    def __init__(self, orientation, default=0, parent=None):
        super().__init__(orientation, parent)
        self.default_value = int(default)

    def mouseDoubleClickEvent(self, ev):
        self.setValue(self.default_value)
        ev.accept()


def safe_gain_ceiling_db(peak, applied_gain_db=0.0):
    """Largest target gain which keeps a pre-gain waveform below 0 dBFS."""
    peak = max(0.0, float(peak or 0.0))
    if peak <= 1e-9:
        return 12.0
    # A cached waveform may already include gain. Recover its pre-gain peak
    # when only the applied value is known.
    pre_gain_peak = peak / (10.0 ** (float(applied_gain_db) / 20.0))
    return max(-60.0, min(12.0, -20.0 * np.log10(
        max(1e-9, pre_gain_peak))))


class GainControl(QtWidgets.QWidget):
    """Resettable gain slider/spin pair with shared clipping policy/state."""
    valueChanged = QtCore.pyqtSignal(float)
    clippingChanged = QtCore.pyqtSignal(bool)

    def __init__(self, value=0.0, parent=None):
        super().__init__(parent)
        self._syncing = False
        self._safe_max = 12.0
        self._available = False
        self.setProperty("gainPending", False)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.slider = ResetSlider(Qt.Horizontal, default=0)
        self.slider.setRange(-600, 120)
        self.slider.setSingleStep(5)
        self.slider.setPageStep(30)
        self.slider.setToolTip(
            "Output gain; double-click to reset to 0 dB. The upper limit "
            "uses the generated waveform's available headroom.")
        layout.addWidget(self.slider)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel("volume"))
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(-60.0, 12.0)
        self.spin.setDecimals(1)
        self.spin.setSingleStep(1.0)
        self.spin.setSuffix(" dB")
        row.addWidget(self.spin, 1)
        self.allow_clipping = QtWidgets.QCheckBox("Allow clipping")
        self.allow_clipping.setToolTip(
            "Allow gain above the waveform's measured headroom. Peaks may "
            "be flattened at 0 dBFS.")
        row.addWidget(self.allow_clipping)
        layout.addLayout(row)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.allow_clipping.toggled.connect(self._clipping_toggled)
        self.set_value(value, emit=False)
        self.set_audio_state(False)

    def value(self):
        return float(self.spin.value())

    def set_value(self, value, emit=False):
        value = float(max(-60.0, min(self.spin.maximum(), float(value))))
        self._syncing = True
        try:
            self.spin.setValue(value)
            self.slider.setValue(int(round(value * 10.0)))
        finally:
            self._syncing = False
        if emit:
            self.valueChanged.emit(self.value())

    def set_allow_clipping(self, enabled, emit=False):
        self._syncing = True
        try:
            self.allow_clipping.setChecked(bool(enabled))
        finally:
            self._syncing = False
        self._apply_ceiling(emit_clamp=emit)
        if emit:
            self.clippingChanged.emit(bool(enabled))

    def set_audio_state(self, available, peak=0.0, applied_gain_db=0.0):
        self._available = bool(available)
        self._safe_max = safe_gain_ceiling_db(peak, applied_gain_db)
        self.setEnabled(self._available)
        self._apply_ceiling(emit_clamp=False)

    def set_safe_ceiling(self, available, ceiling_db=12.0):
        self._available = bool(available)
        self._safe_max = max(-60.0, min(12.0, float(ceiling_db)))
        self.setEnabled(self._available)
        self._apply_ceiling(emit_clamp=False)

    def set_pending(self, pending):
        pending = bool(pending and self._available)
        if bool(self.property("gainPending")) == pending:
            return
        self.setProperty("gainPending", pending)
        self.style().unpolish(self)
        self.style().polish(self)

    def _apply_ceiling(self, emit_clamp=True):
        maximum = 12.0 if self.allow_clipping.isChecked() else self._safe_max
        maximum = max(-60.0, min(12.0, float(maximum)))
        old = self.value()
        self.spin.setMaximum(maximum)
        self.slider.setMaximum(int(round(maximum * 10.0)))
        if old > maximum + 1e-6:
            self.set_value(
                maximum, emit=bool(emit_clamp and not self._syncing))

    def _from_slider(self, value):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.spin.setValue(float(value) / 10.0)
        finally:
            self._syncing = False
        self.valueChanged.emit(self.value())

    def _from_spin(self, value):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.slider.setValue(int(round(float(value) * 10.0)))
        finally:
            self._syncing = False
        self.valueChanged.emit(float(value))

    def _clipping_toggled(self, enabled):
        self._apply_ceiling()
        if not self._syncing:
            self.clippingChanged.emit(bool(enabled))


class VocalTractControl(QtWidgets.QWidget):
    """Reference-bounded resonance control with an exact identity centre."""
    valueChanged = QtCore.pyqtSignal(float)
    chipmunkRangeChanged = QtCore.pyqtSignal(bool)

    def __init__(self, ratio=1.0, chipmunk_range=False, parent=None):
        super().__init__(parent)
        self.profile = vocal_tract.load_vocal_tract_range()
        self._syncing = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        slider_row = QtWidgets.QHBoxLayout()
        slider_row.setContentsMargins(0, 0, 0, 0)
        self.longer_label = QtWidgets.QLabel("longer")
        self.longer_label.setStyleSheet("color:#555")
        slider_row.addWidget(self.longer_label)
        self.slider = ResetSlider(Qt.Horizontal, default=500)
        self.slider.setRange(0, 1000)
        self.slider.setSingleStep(5)
        self.slider.setPageStep(40)
        self.slider.setAccessibleName("Vocal tract length")
        slider_row.addWidget(self.slider, 1)
        self.shorter_label = QtWidgets.QLabel("shorter")
        self.shorter_label.setStyleSheet("color:#555")
        slider_row.addWidget(self.shorter_label)
        layout.addLayout(slider_row)

        value_row = QtWidgets.QHBoxLayout()
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.addWidget(QtWidgets.QLabel("ratio"))
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setSingleStep(0.005)
        self.spin.setSuffix(" x")
        self.spin.setAccessibleName("Vocal tract length ratio")
        value_row.addWidget(self.spin, 1)
        self.reset = QtWidgets.QToolButton()
        self.reset.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        self.reset.setToolTip("Reset vocal tract length to the original voice")
        self.reset.setAccessibleName("Reset vocal tract length")
        value_row.addWidget(self.reset)
        layout.addLayout(value_row)

        self.chipmunk = QtWidgets.QCheckBox("Chipmunk range")
        self.chipmunk.setChecked(bool(chipmunk_range))
        self.chipmunk.setAccessibleName("Chipmunk range")
        self.chipmunk.setToolTip(
            "Unlocks the validated extended resonance range. It remains "
            "bounded for DSP safety.")
        layout.addWidget(self.chipmunk)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.reset.clicked.connect(lambda: self.set_ratio(1.0, emit=True))
        self.chipmunk.toggled.connect(self._range_toggled)
        self.set_state(ratio, chipmunk_range, emit=False)

    def ratio(self):
        return float(self.spin.value())

    def chipmunk_range(self):
        return bool(self.chipmunk.isChecked())

    def _bounds(self):
        return self.profile.bounds(self.chipmunk_range())

    def _update_tooltip(self):
        ratio = self.ratio()
        semitones = vocal_tract.ratio_to_formant_semitones(ratio)
        text = (
            "Changes apparent vocal size and gender presentation by shifting "
            "vocal-tract resonances independently of pitch.\n"
            "Effective target/source tract ratio: %.3f x; resonance shift: "
            "%+.2f semitones. Double-click the slider or use Reset for the "
            "original voice." % (ratio, semitones)
        )
        self.setToolTip(text)
        self.slider.setToolTip(text)
        self.spin.setToolTip(text)

    def set_state(self, ratio, chipmunk_range, emit=False):
        self._syncing = True
        try:
            self.chipmunk.setChecked(bool(chipmunk_range))
            lower, upper = self.profile.bounds(bool(chipmunk_range))
            self.spin.setRange(lower, upper)
            clamped = self.profile.clamp(ratio, bool(chipmunk_range))
            self.spin.setValue(clamped)
            position = vocal_tract.ratio_to_control_position(
                clamped, self.profile, bool(chipmunk_range))
            self.slider.setValue(int(round(position * 1000.0)))
        finally:
            self._syncing = False
        self._update_tooltip()
        if emit:
            self.chipmunkRangeChanged.emit(self.chipmunk_range())
            self.valueChanged.emit(self.ratio())

    def set_ratio(self, ratio, emit=False):
        clamped = self.profile.clamp(ratio, self.chipmunk_range())
        self._syncing = True
        try:
            self.spin.setValue(clamped)
            position = vocal_tract.ratio_to_control_position(
                clamped, self.profile, self.chipmunk_range())
            self.slider.setValue(int(round(position * 1000.0)))
        finally:
            self._syncing = False
        self._update_tooltip()
        if emit:
            self.valueChanged.emit(self.ratio())

    def _from_slider(self, value):
        if self._syncing:
            return
        ratio = vocal_tract.control_position_to_ratio(
            float(value) / 1000.0, self.profile, self.chipmunk_range())
        self.set_ratio(ratio, emit=True)

    def _from_spin(self, value):
        if self._syncing:
            return
        self.set_ratio(value, emit=True)

    def _range_toggled(self, enabled):
        if self._syncing:
            return
        before = self.ratio()
        self.set_state(before, bool(enabled), emit=False)
        self.chipmunkRangeChanged.emit(bool(enabled))
        if abs(self.ratio() - before) > 1e-9:
            self.valueChanged.emit(self.ratio())


class AvailabilityItemDelegate(QtWidgets.QStyledItemDelegate):
    """Make unavailable combo entries unambiguous without hiding them."""
    DISABLED_BACKGROUND = "#D9D6CE"
    DISABLED_TEXT = "#5F5C56"
    DISABLED_MARKER = "#8D887E"

    def paint(self, painter, option, index):
        if index.flags() & Qt.ItemIsEnabled:
            super().paint(painter, option, index)
            return
        styled = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        original_rect = QtCore.QRect(styled.rect)
        background = QtGui.QColor(self.DISABLED_BACKGROUND)
        text_color = QtGui.QColor(self.DISABLED_TEXT)
        styled.state &= ~(QtWidgets.QStyle.State_Selected |
                          QtWidgets.QStyle.State_MouseOver |
                          QtWidgets.QStyle.State_HasFocus |
                          QtWidgets.QStyle.State_Enabled)
        styled.backgroundBrush = QtGui.QBrush(background)
        for group in (QtGui.QPalette.Active, QtGui.QPalette.Inactive,
                      QtGui.QPalette.Disabled):
            styled.palette.setColor(group, QtGui.QPalette.Text, text_color)
            styled.palette.setColor(
                group, QtGui.QPalette.HighlightedText, text_color)
            styled.palette.setColor(group, QtGui.QPalette.Base, background)
        painter.save()
        painter.fillRect(original_rect, background)
        styled.rect = styled.rect.adjusted(6, 0, 0, 0)
        super().paint(painter, styled, index)
        painter.fillRect(
            QtCore.QRect(original_rect.left(), original_rect.top(), 3,
                         original_rect.height()),
            QtGui.QColor(self.DISABLED_MARKER))
        painter.restore()

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        return QtCore.QSize(hint.width(), max(22, hint.height()))


class ArrowComboBox(QtWidgets.QComboBox):
    """Combo box with an explicit painter arrow independent of QSS quirks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setItemDelegate(AvailabilityItemDelegate(self))

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        x = self.width() - 10
        y = self.height() // 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QtGui.QColor("#202020" if self.isEnabled()
                                     else "#909090"))
        painter.drawPolygon(QtGui.QPolygon([
            QtCore.QPoint(x - 4, y - 2), QtCore.QPoint(x + 4, y - 2),
            QtCore.QPoint(x, y + 3)]))


class AppliedUndoCommand(QtWidgets.QUndoCommand):
    """Undo command for an edit which was already applied by its widget."""
    def __init__(self, label, undo, redo):
        super().__init__(str(label))
        self._undo = undo
        self._redo = redo
        self._first_redo = True

    def undo(self):
        self._undo()

    def redo(self):
        if self._first_redo:
            self._first_redo = False
            return
        self._redo()


class ShortcutDialog(QtWidgets.QDialog):
    """Collision-checked persistent shortcut editor."""
    def __init__(self, shortcuts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(True)
        self.resize(430, 560)
        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.edits = {}
        for key, label, default in SHORTCUT_SPECS:
            edit = QtWidgets.QKeySequenceEdit(
                QtGui.QKeySequence(shortcuts.get(key, default)))
            if hasattr(edit, "setClearButtonEnabled"):
                edit.setClearButtonEnabled(True)
            form.addRow(label + ":", edit)
            self.edits[key] = edit
        scroll_host = QtWidgets.QWidget()
        scroll_host.setLayout(form)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(scroll_host)
        root.addWidget(scroll, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok |
            QtWidgets.QDialogButtonBox.Cancel |
            QtWidgets.QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self._accept_checked)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults)
        root.addWidget(buttons)

    @staticmethod
    def _portable(sequence):
        return sequence.toString(QtGui.QKeySequence.PortableText)

    def values(self):
        return {key: self._portable(edit.keySequence())
                for key, edit in self.edits.items()}

    def _restore_defaults(self):
        for key, edit in self.edits.items():
            edit.setKeySequence(QtGui.QKeySequence(DEFAULT_SHORTCUTS[key]))

    def _accept_checked(self):
        values = self.values()
        used = {}
        reserved = {"Tab", "Backtab", "Return", "Enter"}
        bare_allowed = {
            "rerender", "play", "stop", "delete",
        }
        for key, value in values.items():
            if not value:
                continue
            if value in reserved:
                QtWidgets.QMessageBox.warning(
                    self, "Keyboard Shortcuts",
                    "%s is reserved for normal control navigation." % value)
                return
            if ("+" not in value and len(value) == 1 and value.isalnum()
                    and key not in bare_allowed):
                QtWidgets.QMessageBox.warning(
                    self, "Keyboard Shortcuts",
                    "%s needs Ctrl, Alt, or Shift so typing remains safe." %
                    SHORTCUT_LABELS[key])
                return
            if value in used:
                QtWidgets.QMessageBox.warning(
                    self, "Keyboard Shortcuts",
                    "%s is assigned to both %s and %s." %
                    (value, SHORTCUT_LABELS[used[value]],
                     SHORTCUT_LABELS[key]))
                return
            used[value] = key
        self.accept()


# ------------------------------------------------------------------- audio play
class Player:
    """sounddevice if installed; else winsound (Windows stdlib -- always
    available on the target machine); else Qt Multimedia; else clear error."""
    def __init__(self):
        self.mode = None
        self._sd = None
        self._qmp = None
        self._tmp = None
        self._temp_paths = set()
        try:
            import sounddevice as sd
            self._sd = sd
            self.mode = "sd"
            return
        except Exception:
            pass
        if sys.platform == "win32":
            try:
                import winsound  # noqa: F401
                self.mode = "winsound"
                return
            except Exception:
                pass
        try:
            from PyQt5.QtMultimedia import QMediaPlayer
            self._qmp = QMediaPlayer()
            self.mode = "qt"
        except Exception:
            self.mode = None

    def play(self, samples, sr):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size <= 1:
            raise RuntimeError("Nothing to play -- generate audio first.")
        self.stop()
        if self.mode == "sd":
            self._sd.play(samples, int(sr))
        elif self.mode == "winsound":
            import tempfile, winsound
            self._tmp = os.path.join(tempfile.gettempdir(), "festvox_gui_play.wav")
            fc.write_wav(self._tmp, samples, sr)
            winsound.PlaySound(self._tmp,
                               winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif self.mode == "qt":
            from PyQt5.QtMultimedia import QMediaContent
            import tempfile
            handle = tempfile.NamedTemporaryFile(
                prefix="festvox_gui_play_", suffix=".wav", delete=False)
            self._tmp = handle.name
            self._temp_paths.add(self._tmp)
            handle.close()
            fc.write_wav(self._tmp, samples, sr)
            self._qmp.setMedia(QMediaContent(QtCore.QUrl.fromLocalFile(self._tmp)))
            self._qmp.play()
        else:
            raise RuntimeError("No audio backend. Install sounddevice:  "
                               "pip install sounddevice")

    def stop(self):
        if self.mode == "sd":
            self._sd.stop()
        elif self.mode == "winsound":
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        elif self.mode == "qt" and self._qmp is not None:
            self._qmp.stop()
            try:
                from PyQt5.QtMultimedia import QMediaContent
                self._qmp.setMedia(QMediaContent())
            except Exception:
                pass
        self._cleanup_temp()

    def _cleanup_temp(self):
        if self._tmp:
            self._temp_paths.add(self._tmp)
            self._tmp = None
        for path in list(self._temp_paths):
            try:
                os.remove(path)
                self._temp_paths.discard(path)
            except FileNotFoundError:
                self._temp_paths.discard(path)
            except OSError:
                # Qt Multimedia can briefly retain the file after setMedia().
                # Keep it queued and retry on the next stop or shutdown.
                pass

    def shutdown(self):
        self.stop()
        if self._qmp is not None:
            self._qmp.deleteLater()
            self._qmp = None
        self._cleanup_temp()


class _SynthesisTask(QtCore.QObject):
    """Run one blocking backend/DSP callable outside Qt's GUI thread."""
    succeeded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(object)

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.succeeded.emit(self.callback())
        except Exception as error:
            self.failed.emit(error)


# ---------------------------------------------------------- linked time tracks
class TimelinePlotWidget(pg.PlotWidget):
    """A parameter timeline which zooms the shared X view on mouse wheel."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_viewbox = None

    def _queue_visible_timer(self, timer):
        """Do not rebuild a hidden stacked parameter page on every zoom."""
        if self.isVisible() and not timer.isActive():
            timer.start(0)

    def _timeline_became_visible(self):
        """Subclasses schedule their current-view LOD from this hook."""

    def showEvent(self, event):
        super().showEvent(event)
        self._timeline_became_visible()

    def wheelEvent(self, ev):
        vb = self.zoom_viewbox or self.getViewBox()
        delta = ev.angleDelta().y()
        if not delta:
            return super().wheelEvent(ev)
        center = vb.mapSceneToView(self.mapToScene(ev.pos()))
        vb.scaleBy((0.82 if delta > 0 else 1.22, 1.0),
                   center=(center.x(), center.y()))
        ev.accept()


def _visible_lod_line(times, values, left, right, pixel_width,
                      points_per_pixel=1.0):
    """Clip a sorted line to the view and preserve min/max per pixel bucket."""
    x = np.asarray(times, np.float64)
    y = np.asarray(values, np.float64)
    if not len(x) or len(x) != len(y):
        return np.zeros(0, np.float64), np.zeros(0, np.float64)
    first = max(0, int(np.searchsorted(x, left, side='left')) - 1)
    last = min(len(x), int(np.searchsorted(x, right, side='right')) + 1)
    x, y = x[first:last], y[first:last]
    budget = max(16, int(max(1.0, pixel_width) * points_per_pixel))
    if len(x) <= budget:
        return x, y
    bucket_count = max(8, budget // 2)
    span = max(1e-12, float(right) - float(left))
    columns = np.floor((x - left) / span * bucket_count).astype(np.int64)
    columns = np.clip(columns, 0, bucket_count - 1)
    runs = np.r_[0, np.flatnonzero(np.diff(columns)) + 1]
    stops = np.r_[runs[1:], len(x)]
    selected = []
    for start, stop in zip(runs, stops):
        chunk = y[start:stop]
        if not len(chunk):
            continue
        low = start + int(np.argmin(chunk))
        high = start + int(np.argmax(chunk))
        selected.extend(sorted({low, high}))
    if not selected:
        return x[[0, -1]], y[[0, -1]]
    indices = np.unique(np.asarray([0] + selected + [len(x) - 1],
                                   np.int64))
    return x[indices], y[indices]


# ------------------------------------------------------- per-phone timing track
class TimingTrack(TimelinePlotWidget):
    """One vertical slider bar per phoneme, x-aligned with the waveform above
    (the plots are x-linked, so zoom/pan stay in sync). Drag a bar up to
    lengthen that phoneme, down to shorten (log2 scale, x0.25..x4) -- and
    keep dragging sideways to paint across several bars in one gesture. The
    stretch is applied on release. Right-click a bar to reset that phoneme's
    timing to exactly what the render produced. Re-render afterwards for
    optimal quality: the engine then re-synthesizes at the edited timings."""
    factorPreview = QtCore.pyqtSignal(int, float)    # live while dragging
    factorsCommitted = QtCore.pyqtSignal(object)     # {idx: factor} on release
    factorReset = QtCore.pyqtSignal(int)             # right click

    def __init__(self, parent=None):
        super().__init__(parent, background='#DCDCDC')
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideAxis('bottom')
        ax = self.getAxis('left')
        ax.setWidth(40)
        ax.setTicks([[(-2, "x0.25"), (-1, "x0.5"), (0, "x1"),
                      (1, "x2"), (2, "x4")]])
        self.setYRange(-2.1, 2.1, padding=0)
        self.setMinimumHeight(110)
        self.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.setToolTip("drag: adjust phoneme length (sweep across several "
                        "bars in one go)\nright-drag: reset crossed phonemes "
                        "to rendered timing\nafter adjusting, hit Re-render "
                        "Phonemes -- re-synthesis at the new timings sounds "
                        "better than the preview stretch")
        self._zero = pg.InfiniteLine(pos=0, angle=0,
                                     pen=pg.mkPen('#999999', width=1))
        self.addItem(self._zero)
        self._bars = None
        self._spans = []      # [(start, end)] per segment
        self._logf = []       # log2(factor) per segment
        self._phones = []
        self._timing_roles = []
        self._edit_vowels = True
        self._edit_consonants = True
        self._drag = False
        self._drag_mode = ""
        self._last_paint_x = None
        self._latched_value = None
        self._touched = {}    # idx -> factor changed during this gesture
        self._lod_detailed = True
        self._display_bar_count = 0
        self._lod_timer = QtCore.QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.timeout.connect(self._redraw_bars)
        self.getViewBox().sigXRangeChanged.connect(
            lambda *_args: self._schedule_lod_redraw())

    def _schedule_lod_redraw(self):
        self._queue_visible_timer(self._lod_timer)

    def _timeline_became_visible(self):
        self._schedule_lod_redraw()

    def _visible_span_range(self):
        if not self._spans:
            return range(0)
        left, right = self.getViewBox().viewRange()[0]
        lo, hi = 0, len(self._spans)
        while lo < hi:
            middle = (lo + hi) // 2
            if self._spans[middle][1] < left:
                lo = middle + 1
            else:
                hi = middle
        first = lo
        lo, hi = first, len(self._spans)
        while lo < hi:
            middle = (lo + hi) // 2
            if self._spans[middle][0] <= right:
                lo = middle + 1
            else:
                hi = middle
        return range(first, lo)

    def set_segments(self, spans, factors, phones=None, timing_roles=None):
        self._spans = [(float(a), float(b)) for a, b in spans]
        self._logf = [float(np.log2(max(0.125, min(8.0, f or 1.0))))
                      for f in factors]
        self._phones = [str(phone) for phone in (phones or [])]
        self._timing_roles = [str(role or "")
                              for role in (timing_roles or [])]
        self._redraw_bars()

    def set_filter(self, consonants=True, vowels=True):
        self._edit_consonants = bool(consonants)
        self._edit_vowels = bool(vowels)
        self._redraw_bars()

    def _editable(self, index):
        if not (0 <= index < len(self._spans)):
            return False
        phone = self._phones[index] if index < len(self._phones) else ""
        if phone == "pau":
            return False
        role = (self._timing_roles[index]
                if index < len(self._timing_roles) else "")
        return self._edit_vowels if fc.is_timing_nucleus(phone, role) \
            else self._edit_consonants

    def factors(self):
        return [float(2.0 ** v) for v in self._logf]

    def _redraw_bars(self):
        if self._bars is not None:
            self.removeItem(self._bars)
            self._bars = None
        if not self._spans:
            self._display_bar_count = 0
            return
        left, right = self.getViewBox().viewRange()[0]
        span = max(1e-9, right - left)
        width = max(32.0, self.getViewBox().sceneBoundingRect().width())
        indices = list(self._visible_span_range())
        detailed = len(indices) * PARAMETER_DETAIL_MIN_PX <= width
        self._lod_detailed = detailed
        groups = []
        if detailed:
            groups = [[index] for index in indices]
        else:
            buckets = {}
            for index in indices:
                a, b = self._spans[index]
                pixel = (((a + b) * 0.5) - left) / span * width
                bucket = int(np.floor(
                    pixel / PARAMETER_OVERVIEW_BUCKET_PX))
                buckets.setdefault(bucket, []).append(index)
            groups = [buckets[key] for key in sorted(buckets)]
        x0, x1, heights, brushes = [], [], [], []
        for group in groups:
            representative = max(group, key=lambda i: abs(self._logf[i]))
            value = self._logf[representative]
            x0.append(self._spans[group[0]][0])
            x1.append(self._spans[group[-1]][1])
            heights.append(value if abs(value) > 0.02 else
                           (0.02 if value >= 0 else -0.02))
            editable = any(self._editable(index) for index in group)
            edited = any(abs(self._logf[index]) > 0.02 for index in group)
            brushes.append(pg.mkBrush(
                '#B0B0B0' if not editable else
                '#FFCC33' if edited else '#C8C4B8'))
        self._display_bar_count = len(groups)
        if not groups:
            return
        self._bars = pg.BarGraphItem(
            x0=x0,
            x1=[max(b - 0.004, a + 0.004) if detailed else b
                for a, b in zip(x0, x1)],
            y0=0, y1=heights, brushes=brushes,
            pen=pg.mkPen('#7F3300', width=1))
        self.addItem(self._bars)

    def _seg_at(self, x):
        lo, hi = 0, len(self._spans)
        while lo < hi:
            middle = (lo + hi) // 2
            if self._spans[middle][1] < x:
                lo = middle + 1
            else:
                hi = middle
        if lo < len(self._spans):
            a, b = self._spans[lo]
            if a <= x <= b:
                return lo if self._editable(lo) else None
        return None

    def _view_pos(self, ev):
        return self.getPlotItem().getViewBox().mapSceneToView(
            self.mapToScene(ev.pos()))

    def mousePressEvent(self, ev):
        p = self._view_pos(ev)
        if ev.button() == Qt.RightButton:
            self._drag = True
            self._drag_mode = "reset"
            self._touched = {}
            self._last_paint_x = float(p.x())
            self._latched_value = None
            self._paint(
                p, previous_x=self._last_paint_x, reset=True
            )
        elif ev.button() == Qt.LeftButton:
            self._drag = True
            self._drag_mode = "paint"
            self._touched = {}
            self._last_paint_x = float(p.x())
            self._latched_value = (
                float(min(2.0, max(-2.0, p.y())))
                if ev.modifiers() & Qt.ShiftModifier else None)
            self._paint(p, previous_x=self._last_paint_x)
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag:
            self._paint(
                self._view_pos(ev), previous_x=self._last_paint_x,
                reset=self._drag_mode == "reset"
            )
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if self._drag:
            self._drag = False
            if self._touched:
                self.factorsCommitted.emit(dict(self._touched))
            self._touched = {}
            self._last_paint_x = None
            self._latched_value = None
            self._drag_mode = ""
        ev.accept()

    def _paint(self, p, previous_x=None, reset=False):
        """Set the bar under the cursor to the cursor height -- sweeping
        sideways paints every bar the pointer crosses."""
        current_x = float(p.x())
        start_x = current_x if previous_x is None else float(previous_x)
        lo, hi = sorted((start_x, current_x))
        touched = [index for index, (start, end) in enumerate(self._spans)
                   if end >= lo and start <= hi and self._editable(index)]
        idx = self._seg_at(current_x)
        if idx is not None:
            touched.append(idx)
        if not touched:
            self._last_paint_x = current_x
            return
        value = (
            0.0 if reset else
            self._latched_value if self._latched_value is not None else
            float(min(2.0, max(-2.0, p.y())))
        )
        for index in sorted(set(touched)):
            self._logf[index] = value
            self._touched[index] = float(2.0 ** value)
        self._redraw_bars()
        for index in sorted(set(touched)):
            self.factorPreview.emit(index, float(2.0 ** value))
        self._last_paint_x = current_x


class PitchTrack(TimelinePlotWidget):
    """Editable, waveform-aligned F0 curve with the generated contour below."""
    targetsCommitted = QtCore.pyqtSignal(object)
    overrideCleared = QtCore.pyqtSignal()
    viewChanged = QtCore.pyqtSignal(float, int)

    def __init__(self, parent=None):
        super().__init__(parent, background='#DCDCDC')
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideAxis('bottom')
        ax = self.getAxis('left')
        ax.setWidth(46)
        ax.setLabel("Hz")
        self.showGrid(x=False, y=True, alpha=0.28)
        self.setMinimumHeight(110)
        self.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.setToolTip(
            "drag/sweep: paint the pitch curve\n"
            "Shift-drag: hold the starting value while painting sideways\n"
            "right-drag: restore regions to generated F0\n"
            "Re-render applies the curve through Festival PSOLA")
        self._spans = []
        self._phones = []
        self._segment_ids = []
        self._ground = []
        self._ground_x = np.asarray([], dtype=np.float64)
        self._ground_y = np.asarray([], dtype=np.float64)
        self._pause_starts = np.asarray([], dtype=np.float64)
        self._pause_ends = np.asarray([], dtype=np.float64)
        self._pause_edges = np.asarray([], dtype=np.float64)
        self._times = []
        self._times_x = np.asarray([], dtype=np.float64)
        self._values = []
        self._active = False
        self._drag_mode = ""
        self._last_edit_point = None
        self._latched_value = None
        self._anchor_edges = True
        self._zoom = 2
        self._view_center = 160.0
        self._pan_start_pixel_y = None
        self._pan_start_center = None
        self._pan_start_span = None
        self._pan_height = None
        self._lod_detailed = True
        self._lod_symbols_visible = False
        self._lod_ground_points = 0
        self._lod_override_points = 0
        self._syllable_debug = {}
        self._syllable_rows = []
        self._syllable_starts = np.asarray([], dtype=np.float64)
        self._syllable_ends = np.asarray([], dtype=np.float64)
        self._syllable_labels = []
        self._syllable_lod_detailed = True
        self._syllable_display_count = 0
        self._lod_timer = QtCore.QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.timeout.connect(self._refresh_lod)
        self._ground_curve = self.plot(
            [], [], pen=pg.mkPen('#777777', width=2, style=Qt.DashLine))
        self._override_curve = self.plot(
            [], [], pen=pg.mkPen('#C05000', width=2), connect='finite')
        self._pause_curve = self.plot(
            [], [], pen=pg.mkPen('#7B7B7B', width=2), connect='finite')
        self._override_points = pg.ScatterPlotItem(
            [], [], size=7, brush=pg.mkBrush('#FFCC33'),
            pen=pg.mkPen('#7F3300'))
        self._pause_points = pg.ScatterPlotItem(
            [], [], size=7, brush=pg.mkBrush('#A8A8A8'),
            pen=pg.mkPen('#666666'))
        self.addItem(self._override_points)
        self.addItem(self._pause_points)
        self._ground_curve.setClipToView(True)
        self._override_curve.setClipToView(True)
        self._pause_curve.setClipToView(True)
        self._syllable_even_bands = pg.BarGraphItem(
            x0=[], x1=[], y0=[], y1=[],
            pen=None, brush=pg.mkBrush(84, 130, 142, 24))
        self._syllable_odd_bands = pg.BarGraphItem(
            x0=[], x1=[], y0=[], y1=[],
            pen=None, brush=pg.mkBrush(91, 141, 105, 20))
        self._syllable_boundaries = pg.PlotCurveItem(
            [], [], pen=pg.mkPen(
                QtGui.QColor(51, 92, 104, 155),
                width=1, style=Qt.DotLine),
            connect="finite")
        for item, z_value in (
                (self._syllable_even_bands, -100),
                (self._syllable_odd_bands, -100),
                (self._syllable_boundaries, -90)):
            item.setZValue(z_value)
            self.addItem(item)
        self._syllable_label_timer = QtCore.QTimer(self)
        self._syllable_label_timer.setSingleShot(True)
        self._syllable_label_timer.timeout.connect(
            self._refresh_syllable_labels)
        self._syllable_vertical_timer = QtCore.QTimer(self)
        self._syllable_vertical_timer.setSingleShot(True)
        self._syllable_vertical_timer.timeout.connect(
            self._refresh_syllable_vertical_geometry)
        self.getViewBox().sigXRangeChanged.connect(
            lambda *_args: self._schedule_lod_refresh())
        self.getViewBox().sigXRangeChanged.connect(
            lambda *_args: self._schedule_syllable_vertical_geometry())
        self.getViewBox().sigYRangeChanged.connect(
            lambda *_args: self._schedule_syllable_vertical_geometry())

    def _schedule_lod_refresh(self):
        self._queue_visible_timer(self._lod_timer)

    def _timeline_became_visible(self):
        self._schedule_lod_refresh()
        self._schedule_syllable_vertical_geometry()

    @staticmethod
    def _sample_many(points, times, default):
        xs = np.asarray(times, dtype=np.float64)
        if not len(xs):
            return []
        pts = sorted((float(t), float(f)) for t, f in (points or []))
        if not pts:
            return [float(default)] * len(xs)
        px = np.fromiter((point[0] for point in pts), dtype=np.float64,
                         count=len(pts))
        py = np.fromiter((point[1] for point in pts), dtype=np.float64,
                         count=len(pts))
        return np.interp(xs, px, py).astype(float).tolist()

    def _sample_ground(self, x, default):
        if not len(self._ground_x):
            return float(default)
        return float(np.interp(float(x), self._ground_x, self._ground_y))

    @staticmethod
    def _control_times(spans, phones, anchor_edges=True):
        if not anchor_edges:
            return [((float(a) + float(b)) / 2.0)
                    for (a, b), phone in zip(spans, phones)
                    if phone != "pau"]
        times, i = [], 0
        while i < len(spans):
            if phones[i] == "pau":
                a, b = spans[i]
                times.extend((float(a), (float(a) + float(b)) / 2.0,
                              float(b)))
                i += 1
                continue
            first = i
            while i < len(spans) and phones[i] != "pau":
                i += 1
            last = i - 1
            if anchor_edges:
                times.append(float(spans[first][0]))
            times.extend((float(a) + float(b)) / 2.0
                         for a, b in spans[first:i])
            if anchor_edges:
                times.append(float(spans[last][1]))
        out = []
        for value in times:
            if not out or abs(value - out[-1]) > 1e-7:
                out.append(value)
        return out

    def _prospective_control_times(
            self, spans, phones, generated, anchor_edges=True):
        _ = generated
        return self._control_times(spans, phones, anchor_edges)

    def set_data(self, spans, phones, generated, override=None, base=160.0,
                 anchor_edges=True, segment_ids=None):
        self._spans = [(float(a), float(b)) for a, b in spans]
        self._phones = [str(p) for p in phones]
        ids = [str(value or "") for value in (segment_ids or ())]
        self._segment_ids = (
            ids if len(ids) == len(self._spans)
            else [""] * len(self._spans)
        )
        pauses = sorted(
            (start, end) for (start, end), phone in
            zip(self._spans, self._phones) if phone == "pau"
        )
        self._pause_starts = np.asarray(
            [start for start, _end in pauses], dtype=np.float64)
        self._pause_ends = np.asarray(
            [end for _start, end in pauses], dtype=np.float64)
        self._pause_edges = np.asarray(sorted({
            value for span in pauses for value in span
        }), dtype=np.float64)
        self._ground = sorted((float(t), float(f)) for t, f in
                              (generated or []))
        self._ground_x = np.fromiter(
            (point[0] for point in self._ground), dtype=np.float64,
            count=len(self._ground))
        self._ground_y = np.fromiter(
            (point[1] for point in self._ground), dtype=np.float64,
            count=len(self._ground))
        self._anchor_edges = bool(anchor_edges)
        self._times = self._control_times(
            self._spans, self._phones, self._anchor_edges)
        self._times_x = np.asarray(self._times, dtype=np.float64)
        source = list(override or [])
        self._active = bool(source)
        self._values = self._sample_many(
            source if source else self._ground, self._times, base)
        visible = [f for _, f in self._ground] + list(self._values)
        self._view_center = float(np.median(visible)) if visible else float(base)
        self._redraw()
        self._refresh_syllable_geometry()

    @staticmethod
    def _relative_deviations(values, ground):
        safe_values = np.maximum(
            np.asarray(values, dtype=np.float64), 1.0e-9)
        safe_ground = np.maximum(
            np.asarray(ground, dtype=np.float64), 1.0e-9)
        return np.log2(safe_values / safe_ground)

    @staticmethod
    def _apply_relative_deviations(ground, deviations):
        return (
            np.asarray(ground, dtype=np.float64)
            * np.exp2(np.asarray(deviations, dtype=np.float64))
        )

    @staticmethod
    def _geometry_segments(spans, phones, segment_ids=None):
        ids = list(segment_ids or ())
        return [
            fc.Segment(
                str(phones[index]) if index < len(phones) else "",
                float(start), float(end),
                uid=(str(ids[index]) if index < len(ids) and ids[index]
                     else uuid.uuid4().hex),
            )
            for index, (start, end) in enumerate(spans)
        ]

    def update_geometry(self, spans, phones, generated, base=160.0,
                        anchor_edges=True, segment_ids=None):
        override = []
        if self._active and self._times and self._spans:
            old_ground = self._sample_many(
                self._ground, self._times, base)
            deviations = self._relative_deviations(
                self._values, old_ground)
            old_segments = [
                fc.Segment(
                    self._phones[index] if index < len(self._phones) else "",
                    float(start), float(end),
                    uid=(self._segment_ids[index]
                         if index < len(self._segment_ids) and
                         self._segment_ids[index] else uuid.uuid4().hex))
                for index, (start, end) in enumerate(self._spans)
            ]
            new_segments = self._geometry_segments(
                spans, phones, segment_ids)
            moved = fc.remap_targets_aligned(
                list(zip(self._times, deviations.astype(float))),
                old_segments, new_segments)
            if moved:
                new_times = self._prospective_control_times(
                    spans, phones, generated, anchor_edges)
                default_ground = (
                    float(np.median(
                        [value for _time, value in (generated or ())]))
                    if generated else float(base)
                )
                ground_values = np.asarray(
                    self._sample_many(
                        generated, new_times, default_ground),
                    dtype=np.float64)
                moved_x = np.asarray(
                    [time for time, _value in moved], dtype=np.float64)
                moved_y = np.asarray(
                    [value for _time, value in moved], dtype=np.float64)
                mapped_deviations = np.interp(
                    np.asarray(new_times, dtype=np.float64),
                    moved_x, moved_y, left=0.0, right=0.0)
                values = self._apply_relative_deviations(
                    ground_values, mapped_deviations)
                override = list(zip(
                    [float(time) for time in new_times],
                    values.astype(float).tolist(),
                ))
        self.set_data(
            spans, phones, generated, override, base,
            anchor_edges=anchor_edges, segment_ids=segment_ids)

    def set_linguistic_unit_debug(self, metadata=None):
        """Show diagnostic syllables or morae without changing the curve."""
        self._syllable_debug = dict(metadata or {})
        self._refresh_syllable_geometry()

    def set_syllable_debug(self, metadata=None):
        """Backward-compatible alias for the original English-only overlay."""
        self.set_linguistic_unit_debug(metadata)

    def _refresh_syllable_geometry(self):
        rows = []
        sources = (
            self._syllable_debug.get("units")
            or self._syllable_debug.get("syllables")
            or ()
        )
        for source in sources:
            if not isinstance(source, dict):
                continue
            try:
                first = int(source.get("phone_start"))
                last = int(source.get("phone_end"))
            except (TypeError, ValueError):
                continue
            if not (0 <= first < last <= len(self._spans)):
                continue
            start = float(self._spans[first][0])
            end = float(self._spans[last - 1][1])
            if end <= start:
                continue
            phones = [str(value) for value in source.get("phones") or ()]
            kind = str(source.get("kind") or "syllable")
            label = str(
                source.get("display_label")
                or " ".join(phones)
            )
            try:
                confidence = float(source.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            tooltip = str(source.get("tooltip") or "").strip()
            if not tooltip:
                if kind == "syllable":
                    tooltip = (
                        "English syllable %d\nonset: %s\nnucleus: %s\n"
                        "coda: %s\nconfidence: %.2f" % (
                            int(source.get("index", len(rows))) + 1,
                            " ".join(str(value) for value in
                                     source.get("onset") or ()) or "(none)",
                            " ".join(str(value) for value in
                                     source.get("nucleus") or ()) or "(none)",
                            " ".join(str(value) for value in
                                     source.get("coda") or ()) or "(none)",
                            confidence,
                        )
                    )
                else:
                    tooltip = "%s %d\nphones: %s" % (
                        kind.replace("_", " ").title(),
                        int(source.get("index", len(rows))) + 1,
                        " ".join(phones) or "(none)",
                    )
            rows.append({
                "index": int(source.get("index", len(rows))),
                "start": start,
                "end": end,
                "label": label,
                "kind": kind,
                "onset": " ".join(
                    str(value) for value in source.get("onset") or ()),
                "nucleus": " ".join(
                    str(value) for value in source.get("nucleus") or ()),
                "coda": " ".join(
                    str(value) for value in source.get("coda") or ()),
                "stress": source.get("stress"),
                "confidence": confidence,
                "tooltip": tooltip,
            })
        self._syllable_rows = rows
        self._syllable_starts = np.asarray(
            [row["start"] for row in rows], dtype=np.float64)
        self._syllable_ends = np.asarray(
            [row["end"] for row in rows], dtype=np.float64)
        self._refresh_syllable_vertical_geometry()

    def _schedule_syllable_vertical_geometry(self):
        self._queue_visible_timer(self._syllable_vertical_timer)

    def _refresh_syllable_vertical_geometry(self):
        try:
            view_box = self.getViewBox()
            view_range = view_box.viewRange()
        except (AttributeError, RuntimeError):
            # A queued zero-delay refresh can outlive pyqtgraph's PlotItem
            # during window teardown. There is no geometry left to update.
            return
        left, right = (float(view_range[0][0]), float(view_range[0][1]))
        width = max(32.0, float(view_box.sceneBoundingRect().width()))
        if len(self._syllable_rows):
            first = max(
                0, int(np.searchsorted(
                    self._syllable_ends, left, side="left")))
            last = min(
                len(self._syllable_rows),
                int(np.searchsorted(
                    self._syllable_starts, right, side="right")))
        else:
            first = last = 0
        visible = self._syllable_rows[first:last]
        detailed = len(visible) * LINGUISTIC_DETAIL_MIN_PX <= width
        self._syllable_lod_detailed = detailed
        if detailed:
            display_rows = [
                dict(row, parity=(first + offset) % 2)
                for offset, row in enumerate(visible)
            ]
        else:
            span = max(1.0e-9, right - left)
            buckets = {}
            for offset, row in enumerate(visible):
                middle = (row["start"] + row["end"]) * 0.5
                bucket = int(np.floor(
                    ((middle - left) / span * width) /
                    LINGUISTIC_OVERVIEW_BUCKET_PX))
                buckets.setdefault(bucket, []).append((first + offset, row))
            display_rows = []
            for bucket in sorted(buckets):
                group = buckets[bucket]
                display_rows.append({
                    "start": group[0][1]["start"],
                    "end": group[-1][1]["end"],
                    "parity": group[0][0] % 2,
                })
        self._syllable_display_count = len(display_rows)
        y_range = view_range[1]
        y0, y1 = float(y_range[0]), float(y_range[1])
        if not np.isfinite(y0 + y1) or y1 <= y0:
            y0, y1 = 0.0, 1.0
        for parity, item in (
                (0, self._syllable_even_bands),
                (1, self._syllable_odd_bands)):
            selected = [
                row for row in display_rows
                if row["parity"] == parity
            ]
            item.setOpts(
                x0=[row["start"] for row in selected],
                x1=[row["end"] for row in selected],
                y0=[y0] * len(selected),
                y1=[y1] * len(selected),
            )
        edges = sorted({
            float(value)
            for row in display_rows
            for value in (row["start"], row["end"])
        })
        x_values, y_values = [], []
        for value in edges:
            x_values.extend((value, value, np.nan))
            y_values.extend((y0, y1, np.nan))
        self._syllable_boundaries.setData(
            x_values, y_values, connect="finite")
        self._schedule_syllable_labels()

    def _schedule_syllable_labels(self):
        self._queue_visible_timer(self._syllable_label_timer)

    def _clear_syllable_labels(self):
        for item in self._syllable_labels:
            try:
                self.removeItem(item)
            except (AttributeError, RuntimeError):
                pass
        self._syllable_labels = []

    def _refresh_syllable_labels(self):
        self._clear_syllable_labels()
        if not self._syllable_rows:
            return
        try:
            view_box = self.getViewBox()
            view_range = view_box.viewRange()
        except (AttributeError, RuntimeError):
            return
        left, right = view_range[0]
        y0, y1 = view_range[1]
        label_y = float(y1) - max(
            1.0e-6, (float(y1) - float(y0)) * 0.02)
        width = max(1.0, float(view_box.width()))
        seconds_per_pixel = max(
            1.0e-9, (float(right) - float(left)) / width)
        first = max(
            0, int(np.searchsorted(
                self._syllable_ends, left, side="left")))
        last = min(
            len(self._syllable_rows),
            int(np.searchsorted(
                self._syllable_starts, right, side="right")))
        visible = [
            row for row in self._syllable_rows[first:last]
            if (row["end"] - row["start"]) /
            seconds_per_pixel >= 44.0
        ]
        if len(visible) > 24:
            step = int(math.ceil(len(visible) / 24.0))
            visible = visible[::step]
        for row in visible:
            stress = row["stress"]
            suffix = "" if stress is None else " [%s]" % stress
            item = pg.TextItem(
                "%d: %s%s" % (
                    row["index"] + 1, row["label"], suffix),
                color=QtGui.QColor("#315864"),
                fill=pg.mkBrush(236, 239, 236, 218),
                border=pg.mkPen(95, 119, 122, 125),
                anchor=(0.5, 0.0),
            )
            item.setZValue(20)
            item.setPos(
                (row["start"] + row["end"]) * 0.5, label_y)
            item.setToolTip(row["tooltip"])
            self.addItem(item)
            self._syllable_labels.append(item)

    def targets(self):
        return [(float(t), float(value))
                for t, value in zip(self._times, self._values)]

    def render_targets(self):
        """Overlay local control deviations without replacing generated F0.

        The visible controls are intentionally much sparser than Festival's
        generated Target relation. Re-rendering those controls directly would
        replace the detailed contour and alter untouched parts of a sentence.
        Preserve every generated target and interpolate only the logarithmic
        deviation introduced by the editable controls.
        """
        controls = self.targets()
        if not self._active or not controls or not self._ground:
            return controls

        control_times = np.asarray(
            [time for time, _value in controls], dtype=np.float64)
        control_values = np.asarray(
            [value for _time, value in controls], dtype=np.float64)
        base_default = float(np.median(self._ground_y))
        control_ground = np.asarray(
            self._sample_many(self._ground, control_times, base_default),
            dtype=np.float64)
        safe_ground = np.maximum(control_ground, 1.0e-9)
        deviations = np.log2(
            np.maximum(control_values, 1.0e-9) / safe_ground)
        deviations[np.abs(deviations) < 1.0e-10] = 0.0

        times = sorted(
            [float(time) for time, _value in self._ground] +
            control_times.astype(float).tolist())
        merged_times = []
        for value in times:
            if not merged_times or abs(value - merged_times[-1]) > 1.0e-7:
                merged_times.append(value)
        base_values = np.asarray(
            self._sample_many(self._ground, merged_times, base_default),
            dtype=np.float64)
        local_deviation = np.interp(
            np.asarray(merged_times, dtype=np.float64),
            control_times,
            deviations,
            left=0.0,
            right=0.0,
        )
        values = base_values * np.exp2(local_deviation)
        values = np.clip(values, fc.PITCH_MIN_HZ, fc.PITCH_MAX_HZ)
        return [
            (float(time), float(value))
            for time, value in zip(merged_times, values)
        ]

    def clear_override(self):
        base = float(np.median([f for _, f in self._ground])) \
            if self._ground else 160.0
        self._values = self._sample_many(self._ground, self._times, base)
        self._active = False
        self._redraw()
        self.overrideCleared.emit()

    def set_zoom(self, level):
        self._zoom = max(0, min(5, int(level)))
        self._apply_zoom()

    def zoom_level(self):
        return int(self._zoom)

    def view_center(self):
        return float(self._view_center)

    def recenter(self, value):
        self._view_center = max(
            fc.PITCH_MIN_HZ, min(fc.PITCH_MAX_HZ, float(value)))
        self._apply_zoom()

    def _apply_zoom(self):
        center = self._view_center
        span = (450.0, 320.0, 240.0, 160.0, 100.0, 60.0)[self._zoom]
        lo, hi = center - span / 2.0, center + span / 2.0
        if lo < fc.PITCH_MIN_HZ:
            hi += fc.PITCH_MIN_HZ - lo
            lo = fc.PITCH_MIN_HZ
        if hi > fc.PITCH_MAX_HZ:
            lo -= hi - fc.PITCH_MAX_HZ
            hi = fc.PITCH_MAX_HZ
        lo = max(fc.PITCH_MIN_HZ, lo)
        hi = min(fc.PITCH_MAX_HZ, hi)
        self._view_center = float((lo + hi) * .5)
        self.setYRange(lo, hi, padding=0)
        self.viewChanged.emit(self._view_center, self._zoom)

    def _redraw(self):
        self._refresh_lod()
        self._apply_zoom()

    def _is_pause_time(self, value):
        if not len(self._pause_starts):
            return False
        index = int(np.searchsorted(
            self._pause_starts, float(value), side="right")) - 1
        return bool(index >= 0 and
                    float(value) <= self._pause_ends[index] + 1e-9)

    def _split_pause_lines(self, x_values, y_values):
        """Split visible line segments at pause edges for exact coloring."""
        xs = np.asarray(x_values, dtype=np.float64)
        ys = np.asarray(y_values, dtype=np.float64)
        runs = {False: [], True: []}
        active_kind = None
        active_x, active_y = [], []

        def finish_run():
            nonlocal active_kind, active_x, active_y
            if active_kind is not None and len(active_x) >= 2:
                runs[active_kind].append((active_x, active_y))
            active_kind = None
            active_x, active_y = [], []

        for index in range(max(0, len(xs) - 1)):
            x0, x1 = float(xs[index]), float(xs[index + 1])
            y0, y1 = float(ys[index]), float(ys[index + 1])
            if not all(np.isfinite((x0, x1, y0, y1))) or x1 <= x0:
                finish_run()
                continue
            first = int(np.searchsorted(self._pause_edges, x0, side="right"))
            last = int(np.searchsorted(self._pause_edges, x1, side="left"))
            cuts = [x0]
            cuts.extend(float(value) for value in
                        self._pause_edges[first:last])
            cuts.append(x1)
            for start, end in zip(cuts, cuts[1:]):
                if end <= start:
                    continue
                start_ratio = (start - x0) / (x1 - x0)
                end_ratio = (end - x0) / (x1 - x0)
                start_y = y0 + (y1 - y0) * start_ratio
                end_y = y0 + (y1 - y0) * end_ratio
                kind = self._is_pause_time((start + end) * 0.5)
                contiguous = (
                    active_kind == kind and active_x and
                    abs(active_x[-1] - start) <= 1e-8 and
                    abs(active_y[-1] - start_y) <= 1e-5
                )
                if not contiguous:
                    finish_run()
                    active_kind = kind
                    active_x = [start]
                    active_y = [start_y]
                active_x.append(end)
                active_y.append(end_y)
        finish_run()

        def flatten(items):
            out_x, out_y = [], []
            for run_x, run_y in items:
                out_x.extend(run_x)
                out_y.extend(run_y)
                out_x.append(np.nan)
                out_y.append(np.nan)
            return (np.asarray(out_x, dtype=np.float64),
                    np.asarray(out_y, dtype=np.float64))

        speech_x, speech_y = flatten(runs[False])
        pause_x, pause_y = flatten(runs[True])
        return (
            speech_x, speech_y, pause_x, pause_y,
        )

    def _display_ground_arrays(self):
        return self._ground_x, self._ground_y

    def _show_control_symbols(self):
        return True

    def _refresh_lod(self):
        plot_item = getattr(self, "plotItem", None)
        if plot_item is None:
            return
        view_box = plot_item.getViewBox()
        left, right = view_box.viewRange()[0]
        width = max(32.0, view_box.sceneBoundingRect().width())
        ground_x, ground_y = self._display_ground_arrays()
        display_gx, display_gy = _visible_lod_line(
            ground_x, ground_y, left, right, width,
            points_per_pixel=1.5)
        self._ground_curve.setData(display_gx, display_gy)
        self._lod_ground_points = len(display_gx)

        tx = self._times_x
        ty = np.asarray(self._values, np.float64)
        if len(tx):
            first = max(0, int(np.searchsorted(tx, left, side='left')) - 1)
            last = min(len(tx),
                       int(np.searchsorted(tx, right, side='right')) + 1)
            visible_count = max(0, last - first)
        else:
            first = last = 0
            visible_count = 0
        detailed = visible_count * PITCH_POINT_MIN_PX <= width
        self._lod_detailed = detailed
        if detailed:
            display_tx, display_ty = tx[first:last], ty[first:last]
        else:
            display_tx, display_ty = _visible_lod_line(
                tx, ty, left, right, width, points_per_pixel=0.8)
        speech_x, speech_y, pause_x, pause_y = self._split_pause_lines(
            display_tx, display_ty)
        self._override_curve.setData(speech_x, speech_y, connect='finite')
        self._pause_curve.setData(pause_x, pause_y, connect='finite')
        self._lod_symbols_visible = bool(
            self._show_control_symbols() and detailed and len(display_tx)
        )
        if self._lod_symbols_visible:
            pause_mask = np.asarray(
                [self._is_pause_time(value) for value in display_tx],
                dtype=bool)
            self._override_points.setData(
                np.asarray(display_tx)[~pause_mask],
                np.asarray(display_ty)[~pause_mask])
            self._pause_points.setData(
                np.asarray(display_tx)[pause_mask],
                np.asarray(display_ty)[pause_mask])
        else:
            self._override_points.setData([], [])
            self._pause_points.setData([], [])
        self._lod_override_points = len(display_tx)

    def _view_pos(self, ev):
        return self.getPlotItem().getViewBox().mapSceneToView(
            self.mapToScene(ev.pos()))

    def _point_at(self, x):
        if not len(self._times_x) or not self._spans:
            return None
        lo, hi = 0, len(self._spans)
        while lo < hi:
            middle = (lo + hi) // 2
            if self._spans[middle][1] < x:
                lo = middle + 1
            else:
                hi = middle
        candidates = range(max(0, lo - 1), min(len(self._spans), lo + 2))
        valid = any(
            self._spans[index][0] <= x <= self._spans[index][1] and
            (self._anchor_edges or self._phones[index] != "pau")
            for index in candidates)
        if not valid:
            return None
        position = int(np.searchsorted(self._times_x, x, side='left'))
        if position <= 0:
            return 0
        if position >= len(self._times_x):
            return len(self._times_x) - 1
        before = position - 1
        return (before if abs(self._times_x[before] - x) <=
                abs(self._times_x[position] - x) else position)

    def _edit(self, p, reset=False, previous=None):
        current_x = float(p.x())
        current_idx = self._point_at(current_x)
        previous_idx = (self._point_at(float(previous.x()))
                        if previous is not None else current_idx)
        valid = [idx for idx in (current_idx, previous_idx)
                 if idx is not None]
        if not valid:
            self._last_edit_point = QtCore.QPointF(p)
            return
        first, last = min(valid), max(valid)
        indices = list(range(first, last + 1))
        if reset:
            for idx in indices:
                base = float(self._values[idx] if self._values else 160.0)
                self._values[idx] = self._sample_ground(
                    self._times[idx], base)
        else:
            current_y = (self._latched_value
                         if self._latched_value is not None else float(p.y()))
            previous_x = (float(previous.x()) if previous is not None
                          else current_x)
            previous_y = (self._latched_value
                          if self._latched_value is not None else
                          float(previous.y()) if previous is not None
                          else current_y)
            denominator = current_x - previous_x
            for idx in indices:
                if abs(denominator) <= 1e-9:
                    value = current_y
                else:
                    ratio = (self._times[idx] - previous_x) / denominator
                    ratio = max(0.0, min(1.0, ratio))
                    value = previous_y + (current_y - previous_y) * ratio
                self._values[idx] = max(
                    fc.PITCH_MIN_HZ, min(fc.PITCH_MAX_HZ, float(value)))
        self._active = True
        self._redraw()
        self._last_edit_point = QtCore.QPointF(p)

    def mousePressEvent(self, ev):
        p = self._view_pos(ev)
        if ev.button() == Qt.MiddleButton:
            self._drag_mode = "pan"
            self._pan_start_pixel_y = float(ev.globalPos().y())
            self._pan_start_center = float(self._view_center)
            y_range = self.getViewBox().viewRange()[1]
            self._pan_start_span = max(1.0, float(y_range[1] - y_range[0]))
            self._pan_height = max(
                1.0, float(self.getViewBox().sceneBoundingRect().height()))
        elif ev.button() == Qt.RightButton:
            self._drag_mode = "reset"
            self._last_edit_point = QtCore.QPointF(p)
            self._latched_value = None
            self._edit(p, reset=True, previous=self._last_edit_point)
        elif ev.button() == Qt.LeftButton:
            self._drag_mode = "paint"
            self._last_edit_point = QtCore.QPointF(p)
            self._latched_value = (
                max(fc.PITCH_MIN_HZ,
                    min(fc.PITCH_MAX_HZ, float(p.y())))
                if ev.modifiers() & Qt.ShiftModifier else None)
            self._edit(p, reset=False, previous=self._last_edit_point)
        else:
            self._drag_mode = ""
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag_mode == "pan":
            self._pan_to_pixel_y(float(ev.globalPos().y()))
        elif self._drag_mode:
            self._edit(self._view_pos(ev), self._drag_mode == "reset",
                       previous=self._last_edit_point)
        ev.accept()

    def _pan_to_pixel_y(self, pixel_y):
        if any(value is None for value in (
                self._pan_start_pixel_y, self._pan_start_center,
                self._pan_start_span, self._pan_height)):
            return
        pixel_delta = float(pixel_y) - self._pan_start_pixel_y
        delta_hz = pixel_delta * self._pan_start_span / self._pan_height
        self._view_center = max(
            fc.PITCH_MIN_HZ,
            min(fc.PITCH_MAX_HZ, self._pan_start_center + delta_hz))
        self._apply_zoom()

    def mouseReleaseEvent(self, ev):
        if self._drag_mode and self._drag_mode != "pan":
            self._drag_mode = ""
            self._last_edit_point = None
            self._latched_value = None
            self.targetsCommitted.emit(self.targets())
        else:
            self._drag_mode = ""
        self._pan_start_pixel_y = None
        self._pan_start_center = None
        self._pan_start_span = None
        self._pan_height = None
        ev.accept()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta:
            span = (450.0, 320.0, 240.0, 160.0, 100.0, 60.0)[self._zoom]
            self._view_center = max(
                fc.PITCH_MIN_HZ,
                min(fc.PITCH_MAX_HZ,
                    self._view_center + (-1 if delta > 0 else 1) *
                    span * 0.08))
            self._apply_zoom()
        ev.accept()


class VoicingTrack(PitchTrack):
    """Editable harmonic/aperiodic mixture on a stable zero-to-one scale."""

    def __init__(self, parent=None):
        self._visual_ground_x = np.asarray([], dtype=np.float64)
        self._visual_ground_y = np.asarray([], dtype=np.float64)
        super().__init__(parent)
        self.getAxis("left").setLabel("Voicing")
        self.getAxis("left").setWidth(54)
        self.setToolTip(
            "drag/sweep: paint harmonic-to-noise balance\n"
            "Shift-drag: hold the starting value while painting sideways\n"
            "right-drag: restore regions to generated voicing\n"
            "0 = measured aperiodic residual, 1 = measured harmonic residual"
        )
        self._ground_curve.setPen(
            pg.mkPen("#6F6F6F", width=2, style=Qt.DashLine)
        )
        self._override_curve.setPen(pg.mkPen("#23765B", width=2))
        self._override_points.setBrush(pg.mkBrush("#9AD8C2"))
        self._override_points.setPen(pg.mkPen("#174B3B"))
        # A pause label is not an edit mask: edge-labelled source units can
        # contain audible speech. Keep the manual curve visibly continuous so
        # users can paint those samples too.
        self._pause_curve.setPen(pg.mkPen("#23765B", width=2))
        self._view_center = 0.5
        self._zoom = 0
        self._apply_zoom()

    @staticmethod
    def _frame_control_times(spans, ground):
        times = [float(time) for time, _value in (ground or ())]
        # Exact phone edges keep the first and last partial analysis hops
        # editable. The measured 8 ms frame timestamps remain unchanged.
        times.extend(float(value) for span in spans for value in span)
        if not ground:
            for start, end in spans:
                start, end = float(start), float(end)
                count = max(1, int(math.ceil((end - start) / 0.008)))
                times.extend(np.linspace(
                    start, end, count + 1, endpoint=True
                ).astype(float).tolist())
        result = []
        for value in sorted(times):
            if not result or abs(value - result[-1]) > 1e-7:
                result.append(value)
        return result

    def _control_times(self, spans, phones, anchor_edges=True):
        """Expose every analysis frame to painting, without point handles."""
        _ = phones, anchor_edges
        return self._frame_control_times(spans, self._ground)

    def _prospective_control_times(
            self, spans, phones, generated, anchor_edges=True):
        _ = phones, anchor_edges
        return self._frame_control_times(spans, generated)

    @staticmethod
    def _relative_deviations(values, ground):
        return (
            np.asarray(values, dtype=np.float64)
            - np.asarray(ground, dtype=np.float64)
        )

    @staticmethod
    def _apply_relative_deviations(ground, deviations):
        return np.clip(
            np.asarray(ground, dtype=np.float64)
            + np.asarray(deviations, dtype=np.float64),
            0.0, 1.0,
        )

    @staticmethod
    def _coarse_ground(points, step_seconds=0.032):
        source = sorted((float(time), float(value))
                        for time, value in (points or ()))
        if len(source) <= 2:
            return source
        source_x = np.asarray([row[0] for row in source], np.float64)
        source_y = np.asarray([row[1] for row in source], np.float64)
        first, last = float(source_x[0]), float(source_x[-1])
        times = np.arange(first, last, float(step_seconds), dtype=np.float64)
        if not len(times) or abs(float(times[-1]) - last) > 1e-7:
            times = np.append(times, last)
        half = float(step_seconds) * .5
        values = []
        for time in times:
            mask = np.abs(source_x - time) <= half
            values.append(float(np.median(source_y[mask])) if np.any(mask)
                          else float(np.interp(time, source_x, source_y)))
        return list(zip(times.astype(float), values))

    def set_data(self, spans, phones, generated, override=None, base=1.0,
                 anchor_edges=True, segment_ids=None):
        visual = self._coarse_ground(generated)
        self._visual_ground_x = np.asarray(
            [row[0] for row in visual], np.float64
        )
        self._visual_ground_y = np.asarray(
            [row[1] for row in visual], np.float64
        )
        super().set_data(
            spans, phones, generated, override, base,
            anchor_edges=anchor_edges,
            segment_ids=segment_ids,
        )

    def _display_ground_arrays(self):
        return self._visual_ground_x, self._visual_ground_y

    def _show_control_symbols(self):
        # Every frame remains paintable through _point_at; hiding thousands of
        # handles keeps the detailed curve readable and prevents visual lag.
        return False

    def _apply_zoom(self):
        self._view_center = 0.5
        self.setYRange(-0.035, 1.035, padding=0)

    def set_zoom(self, _level):
        self._apply_zoom()

    def recenter(self, _value):
        self._apply_zoom()

    def clear_override(self):
        self._values = self._sample_many(
            self._ground, self._times, 1.0
        )
        self._active = False
        self._redraw()
        self.overrideCleared.emit()

    def _edit(self, p, reset=False, previous=None):
        current_x = float(p.x())
        current_idx = self._point_at(current_x)
        previous_idx = (self._point_at(float(previous.x()))
                        if previous is not None else current_idx)
        valid = [idx for idx in (current_idx, previous_idx)
                 if idx is not None]
        if not valid:
            self._last_edit_point = QtCore.QPointF(p)
            return
        indices = list(range(min(valid), max(valid) + 1))
        if reset:
            for idx in indices:
                self._values[idx] = self._sample_ground(
                    self._times[idx], self._values[idx]
                )
        else:
            current_y = (self._latched_value
                         if self._latched_value is not None
                         else max(0.0, min(1.0, float(p.y()))))
            previous_x = (float(previous.x()) if previous is not None
                          else current_x)
            previous_y = (self._latched_value
                          if self._latched_value is not None
                          else max(0.0, min(1.0, float(previous.y())))
                          if previous is not None else current_y)
            denominator = current_x - previous_x
            for idx in indices:
                if abs(denominator) <= 1e-9:
                    value = current_y
                else:
                    ratio = (self._times[idx] - previous_x) / denominator
                    ratio = max(0.0, min(1.0, ratio))
                    value = previous_y + (current_y - previous_y) * ratio
                self._values[idx] = max(0.0, min(1.0, float(value)))
        self._active = True
        self._redraw()
        self._last_edit_point = QtCore.QPointF(p)

    def mousePressEvent(self, ev):
        p = self._view_pos(ev)
        if ev.button() == Qt.RightButton:
            self._drag_mode = "reset"
            self._last_edit_point = QtCore.QPointF(p)
            self._latched_value = None
            self._edit(p, reset=True, previous=self._last_edit_point)
        elif ev.button() == Qt.LeftButton:
            self._drag_mode = "paint"
            self._last_edit_point = QtCore.QPointF(p)
            self._latched_value = (
                max(0.0, min(1.0, float(p.y())))
                if ev.modifiers() & Qt.ShiftModifier else None
            )
            self._edit(p, reset=False, previous=self._last_edit_point)
        else:
            self._drag_mode = ""
        ev.accept()

    def wheelEvent(self, ev):
        ev.accept()


class VocalTractTrack(PitchTrack):
    """Editable apparent-tract ratio, aligned to rendered waveform time."""

    def __init__(self, parent=None):
        self.profile = vocal_tract.load_vocal_tract_range()
        self._chipmunk_range = False
        super().__init__(parent)
        axis = self.getAxis("left")
        axis.setLabel("Tract ratio")
        axis.setWidth(62)
        self.setToolTip(
            "drag/sweep: paint apparent vocal-tract length\n"
            "Shift-drag: hold the starting ratio while painting sideways\n"
            "right-drag: restore the original 1.000 x voice\n"
            "Above 1 lowers resonances; below 1 raises resonances. Pitch and "
            "duration remain independent."
        )
        self._ground_curve.setPen(
            pg.mkPen("#6F6F6F", width=2, style=Qt.DashLine)
        )
        self._override_curve.setPen(pg.mkPen("#356B78", width=2))
        self._override_points.setBrush(pg.mkBrush("#A9D7DE"))
        self._override_points.setPen(pg.mkPen("#244A53"))
        self._pause_curve.setPen(pg.mkPen("#878787", width=2))
        self._view_center = 1.0
        self._zoom = 0
        self._apply_zoom()

    @staticmethod
    def _control_times(spans, phones, anchor_edges=True):
        _ = phones
        if not anchor_edges:
            return [(float(start) + float(end)) * .5
                    for start, end in spans]
        times = sorted(float(value) for span in spans for value in span)
        result = []
        for value in times:
            if not result or abs(value - result[-1]) > 1e-7:
                result.append(value)
        return result

    def chipmunk_range(self):
        return bool(self._chipmunk_range)

    def bounds(self):
        return self.profile.bounds(self._chipmunk_range)

    def _clamp(self, value):
        return self.profile.clamp(value, self._chipmunk_range)

    def _apply_zoom(self):
        lower, upper = self.bounds()
        padding = max(0.004, (upper - lower) * 0.025)
        self._view_center = 1.0
        self.setYRange(lower - padding, upper + padding, padding=0)

    def set_zoom(self, _level):
        self._apply_zoom()

    def recenter(self, _value=1.0):
        self._apply_zoom()

    def set_chipmunk_range(self, enabled):
        """Switch bounds and return whether existing audible values changed."""
        before = list(self._values)
        self._chipmunk_range = bool(enabled)
        self._values = [self._clamp(value) for value in self._values]
        changed = any(abs(a - b) > 1e-10
                      for a, b in zip(before, self._values))
        self._redraw()
        return changed

    def set_data(self, spans, phones, generated=None, override=None, base=1.0,
                 anchor_edges=True, segment_ids=None):
        duration = max((float(end) for _start, end in spans), default=0.0)
        ground = list(generated or ())
        if not ground:
            ground = [(0.0, 1.0), (duration, 1.0)]
        safe_override = [(float(time), self._clamp(value))
                         for time, value in (override or ())]
        super().set_data(
            spans, phones, ground, safe_override, 1.0,
            anchor_edges=anchor_edges,
            segment_ids=segment_ids,
        )
        self._values = [self._clamp(value) for value in self._values]
        self._redraw()

    def clear_override(self):
        self._values = self._sample_many(self._ground, self._times, 1.0)
        self._active = False
        self._redraw()
        self.overrideCleared.emit()

    def set_uniform_ratio(self, ratio, emit=True):
        value = self._clamp(ratio)
        if not self._times:
            return False
        if abs(value - 1.0) <= 1e-12:
            self._values = self._sample_many(self._ground, self._times, 1.0)
            self._active = False
        else:
            self._values = [value] * len(self._times)
            self._active = True
        self._redraw()
        if emit:
            self.targetsCommitted.emit(self.targets())
        return True

    def ratio_range(self):
        values = list(self._values) or [1.0]
        return min(values), max(values)

    def _edit(self, p, reset=False, previous=None):
        current_x = float(p.x())
        current_idx = self._point_at(current_x)
        previous_idx = (self._point_at(float(previous.x()))
                        if previous is not None else current_idx)
        valid = [idx for idx in (current_idx, previous_idx)
                 if idx is not None]
        if not valid:
            self._last_edit_point = QtCore.QPointF(p)
            return
        indices = list(range(min(valid), max(valid) + 1))
        if reset:
            for idx in indices:
                self._values[idx] = self._sample_ground(
                    self._times[idx], 1.0
                )
        else:
            current_y = self._clamp(
                self._latched_value
                if self._latched_value is not None else float(p.y())
            )
            previous_x = (float(previous.x()) if previous is not None
                          else current_x)
            previous_y = self._clamp(
                self._latched_value
                if self._latched_value is not None else
                float(previous.y()) if previous is not None else current_y
            )
            denominator = current_x - previous_x
            for idx in indices:
                if abs(denominator) <= 1e-9:
                    value = current_y
                else:
                    fraction = (self._times[idx] - previous_x) / denominator
                    fraction = max(0.0, min(1.0, fraction))
                    # Ratios interpolate in log space, matching the renderer.
                    value = math.exp(
                        math.log(previous_y) * (1.0 - fraction)
                        + math.log(current_y) * fraction
                    )
                self._values[idx] = self._clamp(value)
        self._active = any(abs(value - 1.0) > 1e-10
                           for value in self._values)
        self._redraw()
        self._last_edit_point = QtCore.QPointF(p)

    def mousePressEvent(self, ev):
        p = self._view_pos(ev)
        if ev.button() == Qt.RightButton:
            self._drag_mode = "reset"
            self._last_edit_point = QtCore.QPointF(p)
            self._latched_value = None
            self._edit(p, reset=True, previous=self._last_edit_point)
        elif ev.button() == Qt.LeftButton:
            self._drag_mode = "paint"
            self._last_edit_point = QtCore.QPointF(p)
            self._latched_value = (
                self._clamp(float(p.y()))
                if ev.modifiers() & Qt.ShiftModifier else None
            )
            self._edit(p, reset=False, previous=self._last_edit_point)
        else:
            self._drag_mode = ""
        ev.accept()

    def wheelEvent(self, ev):
        ev.accept()


class RecordingTrack(TimelinePlotWidget):
    """Visible, waveform-aligned automatic and manual OTO take choices."""
    overrideChanged = QtCore.pyqtSignal(int, object)
    detailsRequested = QtCore.pyqtSignal(object)
    pitchmarksRequested = QtCore.pyqtSignal(object)
    joinDiagnosticRequested = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent, background='#DCDCDC')
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.setYRange(0.0, 1.0, padding=0)
        self.setMinimumHeight(88)
        self._rows = []
        self._items = []
        self._labels = []
        self._lod_detailed = True
        self._display_row_count = 0
        self._lod_timer = QtCore.QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.timeout.connect(self._redraw)
        self.getViewBox().sigXRangeChanged.connect(
            lambda *_args: self._schedule_lod_redraw())

    def _schedule_lod_redraw(self):
        self._queue_visible_timer(self._lod_timer)

    def _timeline_became_visible(self):
        self._schedule_lod_redraw()

    def _visible_row_range(self):
        if not self._rows:
            return range(0)
        left, right = self.getViewBox().viewRange()[0]
        lo, hi = 0, len(self._rows)
        while lo < hi:
            middle = (lo + hi) // 2
            if self._rows[middle]["end"] < left:
                lo = middle + 1
            else:
                hi = middle
        first = lo
        lo, hi = first, len(self._rows)
        while lo < hi:
            middle = (lo + hi) // 2
            if self._rows[middle]["start"] <= right:
                lo = middle + 1
            else:
                hi = middle
        return range(first, lo)

    def set_data(self, segments, inventory, selected_units, overrides,
                 source_phones=None):
        self._rows = []
        selected_units = {int(k): str(v) for k, v in
                          dict(selected_units or {}).items()}
        overrides = {int(k): str(v) for k, v in dict(overrides or {}).items()}
        display_phones = [str(segment.phone) for segment in segments]
        source_phones = [str(phone) for phone in (source_phones or ())]
        if len(source_phones) != len(display_phones):
            source_phones = list(display_phones)
        for i in range(max(0, len(segments) - 1)):
            p1, p2 = segments[i].phone, segments[i + 1].phone
            source_p1, source_p2 = source_phones[i:i + 2]
            display_pair = "%s-%s" % (p1, p2)
            pair = "%s-%s" % (source_p1, source_p2)
            choices = list((inventory or {}).get(pair) or [])
            actual = selected_units.get(
                i, str(choices[0].get("left_name") or source_p1)
                if choices else source_p1)
            manual = overrides.get(i)
            valid_names = {
                str(c.get("left_name") or source_p1) for c in choices
            }
            visible_manual = manual if manual in valid_names else None
            key = visible_manual or actual
            choice = next((c for c in choices
                           if str(c.get("left_name") or source_p1) == key),
                          choices[0] if choices else {})
            take = str(choice.get("id") or "base")
            inspect_only = len(choices) < 2
            role_names = {
                "mora_cv": "CV", "phrase_start_cv": "Start CV",
                "vowel_blend": "Blend", "vcv_mora": "Unit",
                "vc_transition": "VC", "release": "Release",
                "special_mora": "Special", "generated_cv_bridge": "Bridge",
                "structural_consonant_hold": "Hold",
            }
            if take.startswith("jc_"):
                transition_kind = str(
                    choice.get("transition_kind") or "").lower()
                if not transition_kind:
                    vowels = {"a", "i", "u", "e", "o"}
                    transition_kind = (
                        "vv" if p1 in vowels and p2 in vowels else
                        "vc" if p1 in vowels else
                        "cv" if p2 in vowels else ""
                    )
                kind = {"vv": "VV", "vc": "VC", "cv": "CV"}.get(
                    transition_kind,
                    role_names.get(str(choice.get("role") or ""), "Unit"))
                alias = str(choice.get("alias") or "").strip()
                take_label = "%s: %s" % (kind, alias) if alias else kind
            else:
                take_label = take
            source_note = (
                " [%s source]" % pair
                if pair != display_pair else ""
            )
            self._rows.append({
                "index": i, "start": float(segments[i].start),
                "end": float(segments[i].end), "pair": pair,
                "display_pair": display_pair,
                "source_phones": (source_p1, source_p2),
                "choices": choices, "actual": actual,
                "manual": visible_manual, "pending_manual": manual,
                "choice": choice,
                "inspect_only": inspect_only,
                "label": (("Base" if take == "base" else "Base " + take_label)
                          if inspect_only else
                          ("Manual " if visible_manual else "Auto ") +
                          take_label) + source_note,
            })
        self._redraw()

    def _redraw(self):
        for item in self._items:
            self.removeItem(item)
        self._items = []
        self._labels = []
        if not self._rows:
            self._display_row_count = 0
            return
        left, right = self.getViewBox().viewRange()[0]
        span = max(1e-9, right - left)
        width = max(32.0, self.getViewBox().sceneBoundingRect().width())
        indices = list(self._visible_row_range())
        detailed = len(indices) * PARAMETER_DETAIL_MIN_PX <= width
        self._lod_detailed = detailed
        if detailed:
            groups = [[index] for index in indices]
        else:
            buckets = {}
            for index in indices:
                row = self._rows[index]
                pixel = (((row["start"] + row["end"]) * 0.5) - left) \
                    / span * width
                bucket = int(np.floor(
                    pixel / PARAMETER_OVERVIEW_BUCKET_PX))
                buckets.setdefault(bucket, []).append(index)
            groups = [buckets[key] for key in sorted(buckets)]
        self._display_row_count = len(groups)
        if not groups:
            return
        x0, x1, brushes = [], [], []
        representatives = []
        for group in groups:
            def priority(index):
                row = self._rows[index]
                return (2 if row["manual"] else
                        1 if not row["inspect_only"] else 0)
            representative = max(group, key=priority)
            row = self._rows[representative]
            representatives.append(representative)
            x0.append(self._rows[group[0]]["start"])
            x1.append(self._rows[group[-1]]["end"])
            color = ('#8A8A86' if row["inspect_only"] else
                     '#C98638' if row["manual"] else '#4F79B8')
            brushes.append(pg.mkBrush(color))
        bar = pg.BarGraphItem(
            x0=x0, x1=x1, y0=[0.08] * len(groups),
            y1=[0.92] * len(groups), brushes=brushes,
            pen=pg.mkPen('#505050'))
        self.addItem(bar)
        self._items.append(bar)
        if not detailed:
            return
        pixels_per_second = width / span
        for representative in representatives:
            row = self._rows[representative]
            pixels = (row["end"] - row["start"]) * pixels_per_second
            if pixels < 34:
                continue
            label = row["label"] if pixels >= 62 else \
                row["label"].split()[-1]
            metrics = QtGui.QFontMetrics(self.font())
            label = metrics.elidedText(
                label, Qt.ElideRight, max(8, int(pixels) - 8))
            text_item = pg.TextItem(label, color='#FFFFFF',
                                    anchor=(0.5, 0.5))
            text_item.setToolTip(row["label"])
            text_item.setPos((row["start"] + row["end"]) * 0.5, 0.5)
            self.addItem(text_item)
            self._items.append(text_item)
            self._labels.append((row, text_item))

    def _row_at(self, ev):
        pos = self.getPlotItem().getViewBox().mapSceneToView(
            self.mapToScene(ev.pos()))
        x = float(pos.x())
        lo, hi = 0, len(self._rows)
        while lo < hi:
            middle = (lo + hi) // 2
            if self._rows[middle]["end"] < x:
                lo = middle + 1
            else:
                hi = middle
        if lo < len(self._rows):
            row = self._rows[lo]
            if row["start"] <= x <= row["end"]:
                return row
        return None

    def _show_menu(self, row, global_pos, alternatives=True):
        menu = QtWidgets.QMenu()
        if alternatives and len(row["choices"]) >= 2:
            current = row["manual"]
            auto = menu.addAction("Auto (currently %s)" % row["actual"])
            auto.setCheckable(True)
            auto.setChecked(not current)
            auto.triggered.connect(
                lambda _on=False, i=row["index"]:
                self.overrideChanged.emit(i, None))
            menu.addSeparator()
            for number, choice in enumerate(row["choices"]):
                key = str(choice.get("left_name") or "")
                left = fc.choice_recorded_context(choice, "left")
                right = fc.choice_recorded_context(choice, "right")
                take = str(choice.get("id") or (number + 1))
                action = menu.addAction("%s: %s > %s > %s" %
                                        (take, left, row["pair"], right))
                action.setCheckable(True)
                action.setChecked(current == key)
                action.setToolTip(
                    "%s\nAlias: %s\nOTO line: %s" %
                    (str(choice.get("wav") or ""),
                     str(choice.get("alias") or ""),
                     str(choice.get("oto_line") or "unknown")))
                action.triggered.connect(
                    lambda _on=False, i=row["index"], k=key:
                    self.overrideChanged.emit(i, k))
            menu.addSeparator()
        elif row.get("manual"):
            reset = menu.addAction("Reset to automatic recording")
            reset.triggered.connect(
                lambda _on=False, i=row["index"]:
                self.overrideChanged.emit(i, None))
            menu.addSeparator()
        details = menu.addAction("Inspect recording source...")
        details.triggered.connect(
            lambda _on=False, data=dict(row):
            self.detailsRequested.emit(data))
        pitchmarks = menu.addAction("View PSOLA source pitchmarks...")
        pitchmarks.setToolTip(
            "Show the generated PM/F0 track UniSyn uses for this unit")
        pitchmarks.triggered.connect(
            lambda _on=False, data=dict(row):
            self.pitchmarksRequested.emit(data))
        joins = menu.addAction("Inspect join and UniSyn windows...")
        joins.setToolTip(
            "Show the rendered handoff, source-window geometry, pitch "
            "periods, and acoustic discontinuity evidence")
        joins.triggered.connect(
            lambda _on=False, data=dict(row):
            self.joinDiagnosticRequested.emit(data))
        menu.exec_(global_pos)

    def mousePressEvent(self, ev):
        row = self._row_at(ev)
        if row is None:
            ev.accept()
            return
        if ev.button() == Qt.LeftButton:
            self._show_menu(row, ev.globalPos(), alternatives=True)
        elif ev.button() == Qt.RightButton:
            self._show_menu(row, ev.globalPos(), alternatives=False)
        ev.accept()


class SourcePitchmarkDialog(QtWidgets.QDialog):
    """Read-only generated-unit waveform and UniSyn PM/F0 inspection."""

    def __init__(self, diagnostic, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.diagnostic = dict(diagnostic or {})
        pair = str(self.diagnostic.get("pair") or "unit")
        self.setWindowTitle("PSOLA source pitchmarks: " + pair)
        self.resize(920, 560)
        layout = QtWidgets.QVBoxLayout(self)
        marks = np.asarray(
            self.diagnostic.get("pitchmarks") or [], dtype=np.float64)
        track = np.asarray(
            self.diagnostic.get("f0_track") or [], dtype=np.float64)
        epoch_track = np.asarray(
            self.diagnostic.get("epoch_f0_track") or [], dtype=np.float64)
        track_kind = str(
            self.diagnostic.get("f0_track_kind") or "epoch-rate")
        f0_source = str(
            self.diagnostic.get("f0_source") or "pitchmark-intervals")
        faults = list(self.diagnostic.get("discontinuities") or [])
        positive_f0 = (
            track[:, 1][np.isfinite(track[:, 1]) & (track[:, 1] > 0.0)]
            if track.ndim == 2 and track.shape[1] >= 2 else np.asarray([])
        )
        median_f0 = float(np.median(positive_f0)) if positive_f0.size else 0.0
        summary = QtWidgets.QLabel(
            "%s  |  %s  |  %d pitchmarks  |  %s median %.1f Hz  |  %s%s" % (
                pair,
                str(self.diagnostic.get("wav_name") or "generated WAV"),
                len(marks),
                "analyzed F0" if track_kind == "analyzed" else "epoch rate",
                median_f0,
                f0_source,
                "  |  %d local period jump%s" %
                (len(faults), "" if len(faults) == 1 else "s")
                if faults else "  |  no large local period jumps",
            ))
        summary.setWordWrap(True)
        self.summary = summary
        layout.addWidget(summary)

        waveform = pg.PlotWidget(background="#DCDCDC")
        self.waveform_plot = waveform
        waveform.setLabel("left", "Amplitude")
        waveform.setLabel("bottom", "Source time", units="s")
        waveform.showGrid(x=True, y=True, alpha=.18)
        raw_samples = self.diagnostic.get("samples")
        samples = np.asarray(
            raw_samples if raw_samples is not None else [], dtype=np.float32)
        sr = max(1, int(self.diagnostic.get("sr") or 16000))
        times = np.arange(len(samples), dtype=np.float64) / float(sr)
        display_x, display_y = _visible_lod_line(
            times, samples, 0.0,
            len(samples) / float(sr) if len(samples) else 1.0,
            900.0, points_per_pixel=1.5)
        waveform.plot(display_x, display_y, pen=pg.mkPen("#2F65B0", width=1))
        if len(marks) and len(samples):
            positions = np.clip(
                np.rint(marks * sr).astype(np.int64), 0, len(samples) - 1)
            waveform.plot(
                marks, samples[positions], pen=None, symbol="o", symbolSize=4,
                symbolBrush=pg.mkBrush("#E0A000"),
                symbolPen=pg.mkPen("#805000"))
        source_slice = dict(self.diagnostic.get("source_slice") or {})
        for key, color in (("start", "#777777"),
                           ("phone_boundary", "#C05000"),
                           ("end", "#777777")):
            try:
                value = float(source_slice[key])
            except (KeyError, TypeError, ValueError):
                continue
            waveform.addItem(pg.InfiniteLine(
                value, angle=90, pen=pg.mkPen(color, width=1)))
        layout.addWidget(waveform, 1)

        f0_plot = pg.PlotWidget(background="#DCDCDC")
        self.f0_plot = f0_plot
        f0_plot.setXLink(waveform)
        f0_plot.setLabel("left", "Source F0", units="Hz")
        f0_plot.setLabel("bottom", "Source time", units="s")
        f0_plot.showGrid(x=True, y=True, alpha=.28)
        if track_kind == "analyzed" and epoch_track.size:
            f0_plot.plot(
                epoch_track[:, 0], epoch_track[:, 1],
                pen=pg.mkPen("#8A8A8A", width=1, style=Qt.DashLine),
                name="PSOLA epoch rate")
        if track.size and track.ndim == 2 and track.shape[1] >= 2:
            shown_f0 = np.asarray(track[:, 1], dtype=np.float64).copy()
            shown_f0[shown_f0 <= 0.0] = np.nan
            f0_plot.plot(
                track[:, 0], shown_f0,
                pen=pg.mkPen("#C05000", width=2),
                symbol="o", symbolSize=4,
                symbolBrush=pg.mkBrush("#FFCC33"),
                symbolPen=pg.mkPen("#7F3300"), connect="finite",
                name=("Analyzed voiced F0" if track_kind == "analyzed"
                      else "PSOLA epoch rate"))
        if track_kind == "analyzed":
            f0_plot.setToolTip(
                "Orange is analyzed voiced F0. The dashed line is the epoch "
                "rate PSOLA uses to traverse voiced and unvoiced regions.")
        if faults:
            f0_plot.plot(
                [float(row["time"]) for row in faults],
                [float(row["f0_hz"]) for row in faults],
                pen=None, symbol="x", symbolSize=12,
                symbolPen=pg.mkPen("#C00000", width=2))
        layout.addWidget(f0_plot, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class JoinDiagnosticTableModel(QtCore.QAbstractTableModel):
    """Virtual table for rendered unit handoffs.

    Render order is the editing-oriented default. Severity rank remains a
    measured column and can be selected explicitly in the inspector.
    """

    COLUMNS = (
        ("Rank", "severity_rank"), ("Time", "time"),
        ("Phone", "phone"), ("Voicing", "voicing"),
        ("Severity", "severity_score"),
        ("Dominant issue", "dominant_issue"),
        ("Level", "level_step_db"), ("F0", "f0_step_cents"),
        ("Phase", "phase_mismatch"), ("Spectrum", "spectral_step"),
        ("Position", "position_source"),
    )

    def __init__(self, joins, parent=None, order="rendered"):
        super().__init__(parent)
        self._source_rows = [dict(row) for row in (joins or [])]
        self.order = ""
        self.rows = []
        self.set_order(order)

    @staticmethod
    def _render_key(row):
        return (
            int(row.get("segment_index") or 0),
            float(row.get("time") or 0.0),
        )

    @staticmethod
    def _severity_key(row):
        return (
            int(row.get("severity_rank") or 10 ** 9),
            float(row.get("time") or 0.0),
        )

    def set_order(self, order):
        order = "severity" if str(order) == "severity" else "rendered"
        rows = sorted(
            self._source_rows,
            key=self._severity_key if order == "severity" else
            self._render_key,
        )
        if order == self.order and rows == self.rows:
            return
        self.beginResetModel()
        self.order = order
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and 0 <= section < len(self.COLUMNS):
            return self.COLUMNS[section][0]
        if orientation == Qt.Vertical:
            return str(section + 1)
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.rows)):
            return None
        row = self.rows[index.row()]
        key = self.COLUMNS[index.column()][1]
        if role == Qt.DisplayRole:
            if key == "severity_rank":
                return str(int(row.get(key) or index.row() + 1))
            if key == "time":
                return "%.3f s" % float(row.get("time") or 0.0)
            if key == "severity_score":
                return "%.2f" % float(row.get(key) or 0.0)
            if key == "dominant_issue":
                return str(row.get(key) or "OK").replace("_", " ").title()
            if key == "level_step_db":
                value = row.get(key)
                return "n/a" if value is None else "%+.2f dB" % float(value)
            if key == "f0_step_cents":
                value = row.get(key)
                return "n/a" if value is None else "%+.0f ct" % float(value)
            if key in ("phase_mismatch", "spectral_step"):
                value = row.get(key)
                return "n/a" if value is None else "%.3f" % float(value)
            if key == "position_source":
                return ("Estimated" if row.get("position_estimated") else
                        "Exact")
            if key == "voicing":
                return str(row.get(key) or "unknown").title()
            return str(row.get(key) or "")
        if role == Qt.TextAlignmentRole and key in {
                "severity_rank", "time", "severity_score", "level_step_db",
                "f0_step_cents", "phase_mismatch", "spectral_step"}:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.ForegroundRole and row.get("flagged"):
            return QtGui.QBrush(QtGui.QColor("#8D2C27"))
        if role == Qt.BackgroundRole and row.get("flagged"):
            return QtGui.QBrush(QtGui.QColor("#F2DEDA"))
        if role == Qt.FontRole and row.get("flagged"):
            font = QtGui.QFont()
            font.setBold(True)
            return font
        if role == Qt.ToolTipRole:
            def metric(name, pattern):
                value = row.get(name)
                return "n/a" if value is None else pattern % float(value)
            return (
                "%s\n%s -> %s\n%s -> %s\n"
                "Level %s; sample novelty %s; slope novelty %s\n"
                "F0 %s; zero/best period correlation %s / %s\n"
                "Spectral value/slope novelty %s / %s\n%s"
                % (str(row.get("classification_reason") or ""),
                   row.get("incoming_unit") or "?",
                   row.get("outgoing_unit") or "?",
                   row.get("incoming_wav") or "unknown source",
                   row.get("outgoing_wav") or "unknown source",
                   metric("level_step_db", "%+.2f dB"),
                   metric("sample_jump_novelty", "%.2f"),
                   metric("slope_jump_novelty", "%.2f"),
                   metric("f0_step_cents", "%+.0f cents"),
                   metric("zero_lag_period_correlation", "%.3f"),
                   metric("best_lag_period_correlation", "%.3f"),
                   metric("spectral_step_novelty", "%.2f"),
                   metric("spectral_slope_break_novelty", "%.2f"),
                   str(row.get("repair_recommendation") or "")))
        return None


class JoinDiscontinuityDialog(QtWidgets.QDialog):
    """Per-splice evidence plus honest, bounded UniSyn window controls."""

    def __init__(
            self, diagnostic, samples, *, focus_edge=None,
            requested_join_settings=None, effective_join_settings=None,
            editable=False, legacy_active=False, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.diagnostic = dict(diagnostic or {})
        self.samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.joins = [dict(row) for row in
                      self.diagnostic.get("joins") or []]
        self.frame_trajectory_records = [
            dict(row) for row in
            self.diagnostic.get("frame_trajectory_records") or []
        ]
        self._requested_join_settings = (
            fc.FestivalWSLBackend.normalize_join_settings(
                requested_join_settings))
        self._crossover_overrides = copy.deepcopy(
            self._requested_join_settings.get(
                "crossover_overrides") or {})
        self._effective_join_settings = dict(effective_join_settings or {})
        self._join_controls_editable = bool(editable)
        self._legacy_join_active = bool(legacy_active)
        self._window_control_guard = False
        self._window_preview_periods = (None, None)
        self._window_preview_time = 0.0
        self._window_preview_region = None
        self._window_left_handle = None
        self._window_right_handle = None
        self._crossover_control_guard = False
        self._crossover_preview_region = None
        self._rendered_crossover_region = None
        self._crossover_left_handle = None
        self._crossover_right_handle = None
        self._crossover_preview_time = 0.0
        self._crossover_preview_bounds = (0.0, 0.0)
        self._selected_crossover_unit = None
        self._selected_crossover_row = None
        self.setWindowTitle("Rendered join inspector")
        self.resize(1280, 880)
        layout = QtWidgets.QVBoxLayout(self)

        summary = dict(self.diagnostic.get("summary") or {})
        issue_counts = dict(summary.get("dominant_issue_counts") or {})
        issue_text = ", ".join(
            "%s %d" % (name.replace("_", " ").title(), int(count))
            for name, count in sorted(issue_counts.items())
            if name != "OK" and count)
        self.summary = QtWidgets.QLabel(
            "%d joins | %d flagged | %d exact, %d estimated | "
            "maximum severity %.2f | %d phase-stabilized epochs%s" % (
                int(summary.get("join_count") or 0),
                int(summary.get("flagged_join_count") or 0),
                int(summary.get("exact_splice_count") or 0),
                int(summary.get("estimated_splice_count") or 0),
                float(summary.get("maximum_severity") or 0.0),
                len(self.frame_trajectory_records),
                (" | " + issue_text) if issue_text else ""))
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        overview = QtWidgets.QWidget()
        overview_layout = QtWidgets.QVBoxLayout(overview)
        overview_layout.setContentsMargins(4, 4, 4, 4)
        self.tabs.addTab(overview, "Overview")

        duration = max(0.001, float(self.diagnostic.get("duration") or 0.0))
        self.waveform_plot = pg.PlotWidget(background="#DCDCDC")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.showGrid(x=True, y=True, alpha=.15)
        self.waveform_plot.setYRange(-1.05, 1.05, padding=0)
        self.waveform_plot.setLimits(xMin=0.0, xMax=duration)
        self._wave_cache = WaveformPeakCache(self.samples)
        self._wave_curve = self.waveform_plot.plot(
            [], [], pen=pg.mkPen("#2F65B0", width=1), connect="finite")
        self._wave_timer = QtCore.QTimer(self)
        self._wave_timer.setSingleShot(True)
        self._wave_timer.timeout.connect(self._redraw_waveform)
        self.waveform_plot.getViewBox().sigXRangeChanged.connect(
            lambda *_args: self._wave_timer.start(0))

        joins = [dict(row) for row in self.diagnostic.get("joins") or []]
        if joins:
            collars = pg.BarGraphItem(
                x0=[float(row["overlap_start"]) for row in joins],
                x1=[float(row["overlap_end"]) for row in joins],
                y0=[-1.05] * len(joins), y1=[1.05] * len(joins),
                brushes=[
                    pg.mkBrush(192, 80, 80, 65)
                    if row.get("flagged") else pg.mkBrush(214, 182, 76, 38)
                    for row in joins],
                pen=pg.mkPen(None))
            collars.setZValue(-4)
            self.waveform_plot.addItem(collars)
        segments = [dict(row) for row in
                    self.diagnostic.get("segments") or []]
        self._add_verticals(
            self.waveform_plot,
            [float(row["start"]) for row in segments] +
            ([float(segments[-1]["end"])] if segments else []),
            -1.05, 1.05, pg.mkPen("#C74B42", width=1,
                                  style=Qt.DashLine))
        self._add_verticals(
            self.waveform_plot,
            [float(row["time"]) for row in joins],
            -1.05, 1.05, pg.mkPen("#9A6500", width=1))
        if self.frame_trajectory_records:
            self.trajectory_markers = self.waveform_plot.plot(
                [float(row.get("time") or 0.0)
                 for row in self.frame_trajectory_records],
                [1.0] * len(self.frame_trajectory_records),
                pen=None, symbol="t1", symbolSize=9,
                symbolBrush=pg.mkBrush("#206C67"),
                symbolPen=pg.mkPen("#164B48"))
            self.trajectory_markers.setToolTip(
                "Measured same-recording pitch-period phase stabilization")
        else:
            self.trajectory_markers = None
        overview_layout.addWidget(self.waveform_plot, 3)

        self.span_plot = pg.PlotWidget(background="#E4E4E1")
        self.span_plot.setXLink(self.waveform_plot)
        self.span_plot.setMouseEnabled(x=True, y=False)
        self.span_plot.setYRange(0.0, 2.0, padding=0)
        self.span_plot.setFixedHeight(118)
        self.span_plot.getAxis("left").setTicks(
            [[(0.45, "Phones"), (1.45, "Units")]])
        self.span_plot.hideAxis("bottom")
        if segments:
            phone_bars = pg.BarGraphItem(
                x0=[float(row["start"]) for row in segments],
                x1=[float(row["end"]) for row in segments],
                y0=[0.06] * len(segments), y1=[0.86] * len(segments),
                brushes=[pg.mkBrush("#A9B8CA") for _row in segments],
                pen=pg.mkPen("#6F7D8D"))
            self.span_plot.addItem(phone_bars)
        self.units = [dict(row) for row in
                      self.diagnostic.get("units") or []]
        if self.units:
            unit_bars = pg.BarGraphItem(
                x0=[float(row["start"]) for row in self.units],
                x1=[float(row["end"]) for row in self.units],
                y0=[1.06] * len(self.units), y1=[1.86] * len(self.units),
                brushes=[pg.mkBrush("#4F79B8" if index % 2 == 0
                                    else "#6E8FC0")
                         for index, _row in enumerate(self.units)],
                pen=pg.mkPen("#435B78"))
            self.span_plot.addItem(unit_bars)
        self.segments = segments
        self._span_labels = []
        self._label_timer = QtCore.QTimer(self)
        self._label_timer.setSingleShot(True)
        self._label_timer.timeout.connect(self._redraw_span_labels)
        self.span_plot.getViewBox().sigXRangeChanged.connect(
            lambda *_args: self._label_timer.start(0))
        overview_layout.addWidget(self.span_plot)

        self.loudness_plot = pg.PlotWidget(background="#DCDCDC")
        self.loudness_plot.setXLink(self.waveform_plot)
        self.loudness_plot.setLabel("left", "K-weighted loudness", units="LKFS")
        self.loudness_plot.setLabel("bottom", "Rendered time", units="s")
        self.loudness_plot.showGrid(x=True, y=True, alpha=.25)
        curve = dict(self.diagnostic.get("join_curve") or {})
        curve_y = np.asarray(curve.get("levels_lkfs") or [], np.float64)
        curve_y[curve_y < -100.0] = np.nan
        self.loudness_plot.plot(
            np.asarray(curve.get("times") or [], np.float64), curve_y,
            pen=pg.mkPen("#7553A4", width=2), connect="finite")
        momentary = dict(self.diagnostic.get("momentary_curve") or {})
        momentary_y = np.asarray(
            momentary.get("levels_lkfs") or [], np.float64)
        momentary_y[momentary_y < -100.0] = np.nan
        self.loudness_plot.plot(
            np.asarray(momentary.get("times") or [], np.float64), momentary_y,
            pen=pg.mkPen("#777777", width=1, style=Qt.DashLine),
            connect="finite")
        flagged = [row for row in joins if row.get("flagged")]
        if flagged:
            self.loudness_plot.plot(
                [float(row["time"]) for row in flagged],
                [(float(row["before_lkfs"]) +
                  float(row["after_lkfs"])) * 0.5 for row in flagged],
                pen=None, symbol="d", symbolSize=9,
                symbolBrush=pg.mkBrush("#C05050"),
                symbolPen=pg.mkPen("#7F2020"))
        overview_layout.addWidget(self.loudness_plot, 2)

        order_row = QtWidgets.QHBoxLayout()
        order_row.addWidget(QtWidgets.QLabel("Join order:"))
        self.join_order = QtWidgets.QComboBox()
        self.join_order.addItem("Rendered phone order", "rendered")
        self.join_order.addItem("Worst first", "severity")
        self.join_order.setToolTip(
            "Rendered order follows the phones from left to right. "
            "Worst first uses the uncalibrated diagnostic severity rank.")
        self.join_order.currentIndexChanged.connect(
            self._join_order_changed)
        order_row.addWidget(self.join_order)
        order_row.addStretch(1)
        overview_layout.addLayout(order_row)

        self.table_model = JoinDiagnosticTableModel(
            joins, self, order="rendered")
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(130)
        self.table.selectionModel().currentRowChanged.connect(
            self._table_row_changed)
        self.table.doubleClicked.connect(lambda _index:
                                         self.tabs.setCurrentIndex(1))
        overview_layout.addWidget(self.table, 1)

        self._build_detail_tab()
        if self.frame_trajectory_records:
            self._build_source_trajectory_tab()

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.apply_window_button = buttons.addButton(
            "Apply join settings for re-render",
            QtWidgets.QDialogButtonBox.AcceptRole)
        self.apply_window_button.setEnabled(
            self._join_controls_editable and not self._legacy_join_active)
        self.apply_window_button.setToolTip(
            "Store the sentence crossover and UniSyn source-window settings. "
            "Re-render Phonemes applies them without changing timing, pitch, "
            "or recording choices.")
        self.apply_window_button.clicked.connect(self.accept)
        self.save_json_button = buttons.addButton(
            "Save JSON...", QtWidgets.QDialogButtonBox.ActionRole)
        self.save_json_button.clicked.connect(self._save_json)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.waveform_plot.setXRange(0.0, duration, padding=0)
        self._redraw_waveform()
        self._redraw_span_labels()
        selected_row = 0 if self.table_model.rowCount() else -1
        if focus_edge is not None:
            # Recording block i represents the i -> i+1 unit. Its join with
            # the preceding unit is rendered inside phone i, not phone i+1.
            target_segment = int(focus_edge)
            for row_index, row in enumerate(self.table_model.rows):
                if int(row.get("segment_index", -1)) == target_segment:
                    selected_row = row_index
                    break
        if selected_row >= 0:
            index = self.table_model.index(selected_row, 0)
            self.table.setCurrentIndex(index)
            self.table.scrollTo(index)

    def _build_source_trajectory_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        explanation = QtWidgets.QLabel(
            "These are measured discontinuities between adjacent source "
            "pitch periods inside one selected recording. The renderer moves "
            "only the source-frame center when non-wrapped correlation shows "
            "a strong, bounded phase improvement. Recording choices, phone "
            "timing, F0, and crossover lengths are unchanged. Legacy joins "
            "bypasses this step.")
        explanation.setWordWrap(True)
        explanation.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(explanation)
        columns = (
            "Time", "Phone", "Source frames", "Shift", "Correlation before",
            "Correlation after", "Improvement", "Reason")
        self.trajectory_table = QtWidgets.QTableWidget(
            len(self.frame_trajectory_records), len(columns))
        self.trajectory_table.setHorizontalHeaderLabels(columns)
        self.trajectory_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.trajectory_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.trajectory_table.setAlternatingRowColors(True)
        self.trajectory_table.verticalHeader().setVisible(False)
        for row_index, row in enumerate(self.frame_trajectory_records):
            values = (
                "%.4f s" % float(row.get("time") or 0.0),
                str(row.get("phone") or ""),
                "%s -> %s" % (
                    row.get("previous_source_frame", "?"),
                    row.get("source_frame", "?")),
                "%+d samples" % int(row.get(
                    "centre_offset_samples") or 0),
                "%.3f" % float(row.get("original_correlation") or 0.0),
                "%.3f" % float(row.get("corrected_correlation") or 0.0),
                "%+.3f" % float(row.get(
                    "correlation_improvement") or 0.0),
                str(row.get("reason") or ""),
            )
            for column_index, value in enumerate(values):
                self.trajectory_table.setItem(
                    row_index, column_index,
                    QtWidgets.QTableWidgetItem(value))
        self.trajectory_table.resizeColumnsToContents()
        self.trajectory_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.trajectory_table, 1)
        self.tabs.addTab(tab, "Source trajectory")

    def _join_order_changed(self, _index):
        if not hasattr(self, "table_model"):
            return
        current = self.table.currentIndex()
        selected_segment = None
        if current.isValid() and current.row() < len(self.table_model.rows):
            selected_segment = int(
                self.table_model.rows[current.row()].get(
                    "segment_index", -1))
        self.table_model.set_order(self.join_order.currentData())
        selected_row = 0 if self.table_model.rowCount() else -1
        if selected_segment is not None:
            for row_index, row in enumerate(self.table_model.rows):
                if int(row.get("segment_index", -1)) == selected_segment:
                    selected_row = row_index
                    break
        if selected_row >= 0:
            index = self.table_model.index(selected_row, 0)
            self.table.setCurrentIndex(index)
            self.table.scrollTo(index)

    def _build_detail_tab(self):
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(4, 4, 4, 4)
        self.detail_summary = QtWidgets.QLabel("Select a join in Overview.")
        self.detail_summary.setWordWrap(True)
        self.detail_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        tab_layout.addWidget(self.detail_summary)

        self.crossover_controls = QtWidgets.QGroupBox(
            "Rendered join crossover")
        crossover_layout = QtWidgets.QGridLayout(self.crossover_controls)
        crossover_layout.setColumnStretch(4, 1)
        crossover_layout.addWidget(
            QtWidgets.QLabel("Default duration:"), 0, 0)
        self.join_crossover_ms = QtWidgets.QDoubleSpinBox()
        self.join_crossover_ms.setRange(0.0, 100.0)
        self.join_crossover_ms.setSingleStep(2.0)
        self.join_crossover_ms.setDecimals(1)
        self.join_crossover_ms.setSuffix(" ms")
        self.join_crossover_ms.setValue(
            self._requested_join_settings["crossover_ms"])
        self.join_crossover_ms.setToolTip(
            "Requested total crossover duration. Milliseconds are the "
            "authoritative control; voiced edges snap inward to target "
            "pitchmarks, so the effective period count varies with F0.")
        crossover_layout.addWidget(self.join_crossover_ms, 0, 1)
        self.clear_selected_crossover = QtWidgets.QPushButton(
            "Use default for selected")
        self.clear_selected_crossover.setToolTip(
            "Remove the selected join's asymmetric left/right override.")
        crossover_layout.addWidget(
            self.clear_selected_crossover, 0, 2)
        self.reset_join_crossovers = QtWidgets.QPushButton(
            "Reset crossovers")
        crossover_layout.addWidget(
            self.reset_join_crossovers, 0, 3)
        self.crossover_explanation = QtWidgets.QLabel()
        self.crossover_explanation.setWordWrap(True)
        self.crossover_explanation.setTextInteractionFlags(
            Qt.TextSelectableByMouse)
        crossover_layout.addWidget(
            self.crossover_explanation, 1, 0, 1, 5)
        crossover_enabled = (
            self._join_controls_editable and
            not self._legacy_join_active)
        self.join_crossover_ms.setEnabled(crossover_enabled)
        self.clear_selected_crossover.setEnabled(False)
        self.reset_join_crossovers.setEnabled(crossover_enabled)
        self.join_crossover_ms.valueChanged.connect(
            self._crossover_control_changed)
        self.clear_selected_crossover.clicked.connect(
            self._clear_selected_crossover)
        self.reset_join_crossovers.clicked.connect(
            self._reset_join_crossovers)
        tab_layout.addWidget(self.crossover_controls)
        self._refresh_crossover_explanation()

        self.window_controls = QtWidgets.QGroupBox(
            "UniSyn source-window geometry")
        controls = QtWidgets.QGridLayout(self.window_controls)
        controls.setColumnStretch(4, 1)
        controls.addWidget(QtWidgets.QLabel("Method:"), 0, 0)
        self.join_window_mode = QtWidgets.QComboBox()
        self.join_window_mode.addItem("Voice policy", "voice")
        self.join_window_mode.addItem("Symmetric pitch periods", "symmetric")
        self.join_window_mode.addItem(
            "Asymmetric source periods (experimental)", "asymmetric")
        requested_mode = self._requested_join_settings["mode"]
        mode_index = self.join_window_mode.findData(requested_mode)
        self.join_window_mode.setCurrentIndex(max(0, mode_index))
        controls.addWidget(self.join_window_mode, 0, 1)
        controls.addWidget(QtWidgets.QLabel("Window radius:"), 0, 2)
        self.join_window_factor = QtWidgets.QDoubleSpinBox()
        self.join_window_factor.setRange(1.0, 1.25)
        self.join_window_factor.setSingleStep(0.01)
        self.join_window_factor.setDecimals(2)
        self.join_window_factor.setSuffix(" periods")
        self.join_window_factor.setValue(
            self._requested_join_settings["window_factor"])
        self.join_window_factor.setToolTip(
            "A bounded UniSyn analysis/synthesis-window multiplier. Wider "
            "windows can smear voiced audio, so the editor is limited to "
            "1.00-1.25 periods.")
        controls.addWidget(self.join_window_factor, 0, 3)
        self.reset_join_window = QtWidgets.QPushButton("Reset to voice policy")
        controls.addWidget(self.reset_join_window, 0, 4,
                           alignment=Qt.AlignLeft)
        self.join_window_explanation = QtWidgets.QLabel()
        self.join_window_explanation.setWordWrap(True)
        self.join_window_explanation.setTextInteractionFlags(
            Qt.TextSelectableByMouse)
        controls.addWidget(self.join_window_explanation, 1, 0, 1, 5)
        enabled = self._join_controls_editable and not self._legacy_join_active
        self.join_window_mode.setEnabled(enabled)
        self.join_window_factor.setEnabled(enabled)
        self.reset_join_window.setEnabled(enabled)
        self.join_window_mode.currentIndexChanged.connect(
            self._join_window_control_changed)
        self.join_window_factor.valueChanged.connect(
            self._join_window_control_changed)
        self.reset_join_window.clicked.connect(self._reset_join_window)
        tab_layout.addWidget(self.window_controls)
        self._refresh_join_window_explanation()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        plots = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(plots)
        grid.setContentsMargins(0, 0, 0, 0)
        self.local_waveform_plot = self._detail_plot(
            "Waveform and rendered handoff", "Amplitude", "Time", "s")
        self.difference_plot = self._detail_plot(
            "First difference", "Delta", "Time", "s")
        self.raw_period_plot = self._detail_plot(
            "Final left and first right periods", "Amplitude", "Phase")
        self.normalised_period_plot = self._detail_plot(
            "DC-removed, RMS-normalised periods", "Normalised", "Phase")
        self.aligned_period_plot = self._detail_plot(
            "Best phase alignment", "Normalised", "Phase")
        self.rms_plot = self._detail_plot(
            "Local per-period or per-frame RMS", "RMS", "Time", "s")
        self.f0_plot = self._detail_plot(
            "Local F0", "F0", "Time", "s")
        self.spectral_plot = self._detail_plot(
            "Amplitude-independent spectral trajectories",
            "Feature projection", "Time", "s")
        self.formant_plot = self._detail_plot(
            "F1-F4 trajectories and splice extrapolation",
            "Frequency", "Time", "s")
        self.formant_plot.setLabel("left", "Frequency", units="Hz")
        self.spectral_envelope_plot = self._detail_plot(
            "Amplitude-independent spectral envelopes",
            "Relative level", "Frequency")
        self.spectral_envelope_plot.setLabel(
            "bottom", "Frequency", units="Hz")
        self.spectral_envelope_plot.setLabel(
            "left", "Relative level", units="dB")
        self.formant_balance_plot = self._detail_plot(
            "Formant energy-balance trajectories",
            "Log relative energy", "Time", "s")
        detail_plots = (
            self.local_waveform_plot, self.difference_plot,
            self.raw_period_plot, self.normalised_period_plot,
            self.aligned_period_plot, self.rms_plot, self.f0_plot,
            self.spectral_plot, self.formant_plot,
            self.spectral_envelope_plot, self.formant_balance_plot)
        for plot in detail_plots:
            plot.setMinimumHeight(215)
        for index, plot in enumerate(detail_plots):
            grid.addWidget(plot, index // 2, index % 2)
        scroll.setWidget(plots)
        tab_layout.addWidget(scroll, 1)
        self.tabs.addTab(tab, "Selected Join")

    def requested_join_settings(self):
        settings = fc.FestivalWSLBackend.normalize_join_settings({
            "mode": self.join_window_mode.currentData() or "voice",
            "window_factor": self.join_window_factor.value(),
            "crossover_ms": self.join_crossover_ms.value(),
            "crossover_overrides": copy.deepcopy(
                self._crossover_overrides),
        })
        return ({} if settings ==
                fc.FestivalWSLBackend.normalize_join_settings(None)
                else settings)

    def _selected_requested_crossover(self):
        unit_index = self._selected_crossover_unit
        override = (
            self._crossover_overrides.get(str(unit_index))
            if unit_index is not None else None)
        if override:
            return (
                float(override["left_ms"]),
                float(override["right_ms"]),
                True,
            )
        half = float(self.join_crossover_ms.value()) * 0.5
        return half, half, False

    def _crossover_control_changed(self, *_args):
        if self._crossover_control_guard:
            return
        self._refresh_crossover_explanation()
        self._update_crossover_preview()

    def _clear_selected_crossover(self):
        if self._selected_crossover_unit is None:
            return
        self._crossover_overrides.pop(
            str(self._selected_crossover_unit), None)
        self._refresh_crossover_explanation()
        self._update_crossover_preview()

    def _reset_join_crossovers(self):
        self._crossover_control_guard = True
        try:
            self.join_crossover_ms.setValue(40.0)
            self._crossover_overrides.clear()
        finally:
            self._crossover_control_guard = False
        self._refresh_crossover_explanation()
        self._update_crossover_preview()

    def _refresh_crossover_explanation(self):
        row = self._selected_crossover_row or {}
        left_ms, right_ms, overridden = (
            self._selected_requested_crossover())
        self.clear_selected_crossover.setEnabled(
            bool(overridden) and self._join_controls_editable and
            not self._legacy_join_active)
        if row:
            if row.get("crossover_active"):
                rendered = (
                    "Rendered: %.1f ms across %d target-epoch intervals "
                    "(%s). " % (
                        float(row.get("crossover_effective_ms") or 0.0),
                        int(row.get(
                            "crossover_epoch_intervals") or 0),
                        str(row.get("crossover_reason") or "applied"),
                    )
                )
            else:
                rendered = "Rendered: bypassed (%s). " % str(
                    row.get("crossover_reason") or
                    "no crossover evidence")
            selected = (
                "Selected request: %.1f ms left + %.1f ms right%s. " % (
                    left_ms, right_ms,
                    " (override)" if overridden else " (default)",
                )
            )
        else:
            selected = ""
            rendered = (
                "Select a rendered join to inspect its effective span. ")
        if self._legacy_join_active:
            prefix = (
                "Fault Mode > Legacy joins is active. Stock Festival is "
                "used exactly and crossover editing is disabled. ")
        elif not self._join_controls_editable:
            prefix = (
                "This renderer does not expose editable crossovers. ")
        else:
            prefix = ""
        self.crossover_explanation.setText(
            prefix + selected + rendered +
            "Green handles/shading are the requested left/right crossover. "
            "Dark green is the rendered pitchmark-snapped span. "
            "Milliseconds stay authoritative; period count is diagnostic. "
            "Phone-class caps protect stops, fricatives, pauses, and short "
            "contexts. Dragging never changes phone timing, F0 targets, or "
            "contextual/manual recording choices.")

    def _reset_join_window(self):
        self._window_control_guard = True
        try:
            self.join_window_mode.setCurrentIndex(
                self.join_window_mode.findData("voice"))
            self.join_window_factor.setValue(1.0)
        finally:
            self._window_control_guard = False
        self._join_window_control_changed()

    def _join_window_control_changed(self, *_args):
        if self._window_control_guard:
            return
        self._refresh_join_window_explanation()
        self._update_window_preview()

    def _refresh_join_window_explanation(self):
        effective = self._effective_join_settings
        if effective:
            effective_method = (
                "symmetric" if effective.get("window_symmetric", True) else
                "asymmetric")
            effective_factor = float(effective.get(
                "window_factor",
                self._requested_join_settings["window_factor"]))
            rendered = "Rendered setting: %s, %.2f periods (%s). " % (
                effective_method, effective_factor,
                str(effective.get("source") or "not recorded"))
        else:
            rendered = (
                "The older render did not record its effective UniSyn "
                "geometry. ")
        if self._legacy_join_active:
            prefix = (
                "Fault Mode > Legacy joins is active. It owns this render "
                "and fixes the historical symmetric 1.00-period geometry; "
                "manual controls are disabled. ")
        elif not self._join_controls_editable:
            prefix = (
                "This renderer does not expose editable UniSyn windows. ")
        else:
            prefix = ""
        self.join_window_explanation.setText(
            prefix +
            rendered +
            "The blue handles in "
            "the waveform preview show the requested source-window extent "
            "around this join. Drag either handle to adjust the bounded "
            "factor. Festival applies this setting to every pitchmark in "
            "the sentence; it does not move the red splice, change phone "
            "timing/F0, or replace contextual/manual recordings. Legend: "
            "blue handles/shading = requested source window; red = exact "
            "splice; gold = rendered Unit-map handoff; dotted grey = "
            "rendered pitchmarks.")

    @staticmethod
    def _valid_period_seconds(value, fallback):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return fallback
        return value if np.isfinite(value) and value > 1e-5 else fallback

    def _add_window_preview(self, row, when, start, end):
        fallback = 1.0 / 120.0
        left_period = self._valid_period_seconds(
            row.get("left_period_seconds"), fallback)
        right_period = self._valid_period_seconds(
            row.get("right_period_seconds"), left_period)
        self._window_preview_periods = (left_period, right_period)
        self._window_preview_time = float(when)
        factor = float(self.join_window_factor.value())
        left = when - left_period * factor
        right = when + right_period * factor
        self._window_preview_region = pg.LinearRegionItem(
            values=(left, right), movable=False,
            brush=pg.mkBrush(52, 126, 170, 34),
            pen=pg.mkPen(None))
        self._window_preview_region.setZValue(-1)
        self.local_waveform_plot.addItem(self._window_preview_region)
        handle_pen = pg.mkPen("#287EA8", width=2)
        hover_pen = pg.mkPen("#42A9D4", width=3)
        self._window_left_handle = pg.InfiniteLine(
            pos=left, angle=90, movable=(
                self._join_controls_editable and
                not self._legacy_join_active),
            pen=handle_pen, hoverPen=hover_pen)
        self._window_right_handle = pg.InfiniteLine(
            pos=right, angle=90, movable=(
                self._join_controls_editable and
                not self._legacy_join_active),
            pen=handle_pen, hoverPen=hover_pen)
        self._window_left_handle._join_window_side = -1
        self._window_right_handle._join_window_side = 1
        self._window_left_handle.setBounds((
            max(start, when - left_period * 1.25),
            min(end, when - left_period)))
        self._window_right_handle.setBounds((
            max(start, when + right_period),
            min(end, when + right_period * 1.25)))
        tooltip = (
            "Drag to change the sentence-wide UniSyn window radius. "
            "This preview uses the selected join's measured source period.")
        self._window_left_handle.setToolTip(tooltip)
        self._window_right_handle.setToolTip(tooltip)
        self._window_left_handle.sigPositionChanged.connect(
            self._window_handle_moved)
        self._window_right_handle.sigPositionChanged.connect(
            self._window_handle_moved)
        self._window_left_handle.setZValue(18)
        self._window_right_handle.setZValue(18)
        self.local_waveform_plot.addItem(self._window_left_handle)
        self.local_waveform_plot.addItem(self._window_right_handle)

    def _window_handle_moved(self, line):
        if self._window_control_guard:
            return
        left_period, right_period = self._window_preview_periods
        period = left_period if line._join_window_side < 0 else right_period
        if not period:
            return
        factor = abs(float(line.value()) -
                     self._window_preview_time) / period
        factor = max(1.0, min(1.25, factor))
        self._window_control_guard = True
        try:
            self.join_window_factor.setValue(factor)
        finally:
            self._window_control_guard = False
        self._refresh_join_window_explanation()
        self._update_window_preview()

    def _update_window_preview(self):
        if (self._window_preview_region is None or
                self._window_left_handle is None or
                self._window_right_handle is None):
            return
        left_period, right_period = self._window_preview_periods
        if not left_period or not right_period:
            return
        factor = float(self.join_window_factor.value())
        when = self._window_preview_time
        left = when - left_period * factor
        right = when + right_period * factor
        self._window_control_guard = True
        try:
            self._window_left_handle.setValue(left)
            self._window_right_handle.setValue(right)
            self._window_preview_region.setRegion((left, right))
        finally:
            self._window_control_guard = False

    def _add_crossover_preview(self, row, when, start, end):
        self._selected_crossover_row = row
        try:
            self._selected_crossover_unit = int(
                row.get("unit_index",
                        int(row.get("segment_index") or 1) - 1))
        except (TypeError, ValueError):
            self._selected_crossover_unit = None
        left_ms, right_ms, _overridden = (
            self._selected_requested_crossover())
        self._crossover_preview_time = float(when)
        self._crossover_preview_bounds = (float(start), float(end))
        left = max(float(start), when - left_ms / 1000.0)
        right = min(float(end), when + right_ms / 1000.0)

        rendered_start = row.get("crossover_start")
        rendered_end = row.get("crossover_end")
        if (row.get("crossover_active") and
                rendered_start is not None and
                rendered_end is not None and
                float(rendered_end) > float(rendered_start)):
            self._rendered_crossover_region = pg.LinearRegionItem(
                values=(float(rendered_start), float(rendered_end)),
                movable=False,
                brush=pg.mkBrush(24, 111, 82, 70),
                pen=pg.mkPen("#176F52", width=1))
            self._rendered_crossover_region.setZValue(-1)
            self.local_waveform_plot.addItem(
                self._rendered_crossover_region)

        self._crossover_preview_region = pg.LinearRegionItem(
            values=(left, right), movable=False,
            brush=pg.mkBrush(54, 156, 116, 34),
            pen=pg.mkPen(None))
        self._crossover_preview_region.setZValue(-2)
        self.local_waveform_plot.addItem(
            self._crossover_preview_region)
        movable = (
            self._join_controls_editable and
            not self._legacy_join_active and
            self._selected_crossover_unit is not None)
        pen = pg.mkPen("#28956D", width=2)
        hover = pg.mkPen("#45C994", width=3)
        self._crossover_left_handle = pg.InfiniteLine(
            pos=left, angle=90, movable=movable,
            pen=pen, hoverPen=hover)
        self._crossover_right_handle = pg.InfiniteLine(
            pos=right, angle=90, movable=movable,
            pen=pen, hoverPen=hover)
        self._crossover_left_handle._join_crossover_side = -1
        self._crossover_right_handle._join_crossover_side = 1
        self._crossover_left_handle.setBounds((
            max(float(start), when - 0.1), when))
        self._crossover_right_handle.setBounds((
            when, min(float(end), when + 0.1)))
        tooltip = (
            "Drag a requested crossover edge. Duration is stored in "
            "milliseconds and snapped inward to rendered pitchmarks.")
        self._crossover_left_handle.setToolTip(tooltip)
        self._crossover_right_handle.setToolTip(tooltip)
        self._crossover_left_handle.sigPositionChanged.connect(
            self._crossover_handle_moved)
        self._crossover_right_handle.sigPositionChanged.connect(
            self._crossover_handle_moved)
        for handle in (
                self._crossover_left_handle,
                self._crossover_right_handle):
            handle.setZValue(19)
            self.local_waveform_plot.addItem(handle)
        self._refresh_crossover_explanation()

    def _crossover_handle_moved(self, line):
        if (self._crossover_control_guard or
                self._selected_crossover_unit is None or
                self._crossover_left_handle is None or
                self._crossover_right_handle is None):
            return
        when = self._crossover_preview_time
        left_ms = max(
            0.0, (when - float(
                self._crossover_left_handle.value())) * 1000.0)
        right_ms = max(
            0.0, (float(
                self._crossover_right_handle.value()) - when) * 1000.0)
        if left_ms + right_ms > 100.0:
            if getattr(line, "_join_crossover_side", 0) < 0:
                left_ms = max(0.0, 100.0 - right_ms)
            else:
                right_ms = max(0.0, 100.0 - left_ms)
        self._crossover_overrides[str(
            self._selected_crossover_unit)] = {
                "left_ms": round(left_ms, 3),
                "right_ms": round(right_ms, 3),
            }
        self._refresh_crossover_explanation()
        self._update_crossover_preview()

    def _update_crossover_preview(self):
        if (self._crossover_preview_region is None or
                self._crossover_left_handle is None or
                self._crossover_right_handle is None):
            return
        left_ms, right_ms, _overridden = (
            self._selected_requested_crossover())
        when = self._crossover_preview_time
        left = when - left_ms / 1000.0
        right = when + right_ms / 1000.0
        start, end = self._crossover_preview_bounds
        left = max(max(start, when - 0.1), min(when, left))
        right = max(when, min(min(end, when + 0.1), right))
        self._crossover_control_guard = True
        try:
            self._crossover_left_handle.setValue(left)
            self._crossover_right_handle.setValue(right)
            self._crossover_preview_region.setRegion((left, right))
        finally:
            self._crossover_control_guard = False

    @staticmethod
    def _detail_plot(title, left_label, bottom_label, units=None):
        plot = pg.PlotWidget(background="#E2E2E0")
        plot.setTitle(title)
        plot.setLabel("left", left_label)
        plot.setLabel("bottom", bottom_label, units=units)
        plot.showGrid(x=True, y=True, alpha=.22)
        return plot

    @staticmethod
    def _metric(row, key, pattern):
        value = row.get(key)
        return "n/a" if value is None else pattern % float(value)

    def _save_json(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save join diagnostics", "join_discontinuities.json",
            "JSON files (*.json);;All files (*)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".json"
        try:
            Path(path).write_text(
                json.dumps(self.diagnostic, indent=2, ensure_ascii=False)
                + "\n", encoding="utf-8")
        except OSError as error:
            QtWidgets.QMessageBox.warning(
                self, "Save join diagnostics", str(error))

    @staticmethod
    def _plot_message(plot, message):
        plot.clear()
        plot.setXRange(0.0, 1.0, padding=0)
        plot.setYRange(0.0, 1.0, padding=0)
        label = pg.TextItem(message, color="#555555", anchor=(0.5, 0.5))
        label.setPos(0.5, 0.5)
        plot.addItem(label)

    @staticmethod
    def _splice_line(plot, when):
        line = pg.InfiniteLine(
            pos=float(when), angle=90,
            pen=pg.mkPen("#B12E2A", width=2))
        line.setZValue(20)
        plot.addItem(line)

    def _plot_period_pair(self, plot, period_data, right_key):
        plot.clear()
        phase = np.asarray(period_data.get("phase") or [], np.float64)
        left_key = ("left_raw" if right_key == "right_raw" else
                    "left_normalised")
        left = np.asarray(period_data.get(left_key) or [], np.float64)
        right = np.asarray(period_data.get(right_key) or [], np.float64)
        count = min(len(phase), len(left), len(right))
        if count < 2:
            self._plot_message(plot, "Unavailable for this join")
            return
        plot.plot(phase[:count], left[:count],
                  pen=pg.mkPen("#2F65B0", width=2), name="Left")
        plot.plot(phase[:count], right[:count],
                  pen=pg.mkPen("#B05B32", width=2), name="Right")
        plot.setXRange(0.0, 1.0, padding=0)
        plot.enableAutoRange(axis="y", enable=True)

    def _populate_join_detail(self, row):
        when = float(row.get("time") or 0.0)
        issue = str(row.get("dominant_issue") or "OK").replace("_", " ")
        exact = "estimated phone centre" if row.get(
            "position_estimated") else str(row.get("position_source") or
                                             "exact handoff")
        context = str(row.get("phone_context_string") or
                      row.get("phone") or "?")
        expected_burst = bool(row.get("broadband_context_may_be_expected"))
        burst_note = (
            " Current phone is a stop/affricate: a brief broadband release "
            "can be legitimate and is marked for review, not hidden."
            if expected_burst else "")
        if row.get("crossover_active"):
            crossover_note = (
                "Crossover: %.1f ms, %d epoch intervals, %s (%s)\n" % (
                    float(row.get("crossover_effective_ms") or 0.0),
                    int(row.get("crossover_epoch_intervals") or 0),
                    str(row.get("crossover_reason") or "applied"),
                    str(row.get("crossover_context") or "unknown"),
                )
            )
        else:
            crossover_note = "Crossover: bypassed (%s)\n" % str(
                row.get("crossover_reason") or "not recorded")
        self.detail_summary.setText(
            "Rank %(rank)d | %(time).6f s | phones %(context)s | "
            "%(voicing)s | "
            "%(position)s\n%(crossover)s"
            "Dominant: %(issue)s | severity %(severity).2f | "
            "level %(level)s | sample novelty %(sample)s | slope novelty "
            "%(slope)s | F0 %(f0)s | zero/best correlation %(zero)s / "
            "%(best)s | spectral value/slope novelty %(spec)s / %(traj)s\n"
            "Formants: %(formants)s | frequency/balance novelty "
            "%(formant_frequency)s / %(formant_balance)s\n"
            "%(reason)s%(burst_note)s Recommendation: %(repair)s" % {
                "rank": int(row.get("severity_rank") or 0),
                "time": when, "context": context,
                "voicing": str(row.get("voicing") or "unknown"),
                "position": exact, "issue": issue,
                "crossover": crossover_note,
                "severity": float(row.get("severity_score") or 0.0),
                "level": self._metric(row, "level_step_db", "%+.2f dB"),
                "sample": self._metric(row, "sample_jump_novelty", "%.2f"),
                "slope": self._metric(row, "slope_jump_novelty", "%.2f"),
                "f0": self._metric(row, "f0_step_cents", "%+.0f cents"),
                "zero": self._metric(
                    row, "zero_lag_period_correlation", "%.3f"),
                "best": self._metric(
                    row, "best_lag_period_correlation", "%.3f"),
                "spec": self._metric(
                    row, "spectral_step_novelty", "%.2f"),
                "traj": self._metric(
                    row, "spectral_slope_break_novelty", "%.2f"),
                "formants": (
                    "%d measured, %d classification-grade (confidence %.2f)" % (
                        int(row.get("formant_measured_track_count") or 0),
                        int(row.get("formant_classification_track_count") or 0),
                        float(row.get("formant_tracking_confidence") or 0.0))
                    if row.get("formants_available") else
                    "unavailable (%s)" % str(
                        row.get("formants_unavailable_reason") or "unknown")),
                "formant_frequency": self._metric(
                    row, "formant_frequency_jump_novelty", "%.2f"),
                "formant_balance": self._metric(
                    row, "formant_balance_novelty", "%.2f"),
                "reason": str(row.get("classification_reason") or ""),
                "burst_note": burst_note,
                "repair": str(row.get("repair_recommendation") or ""),
            })
        plot_data = dict(row.get("plot_data") or {})
        start = max(0.0, float(plot_data.get("context_start") or when - .04))
        total_duration = len(self.samples) / max(
            1, int(self.diagnostic.get("sample_rate") or 1))
        end = min(
            total_duration,
            float(plot_data.get("context_end") or when + .04))
        requested_span_ms = max(
            float(self.join_crossover_ms.value()),
            float(row.get("crossover_requested_left_ms") or 0.0) +
            float(row.get("crossover_requested_right_ms") or 0.0))
        preview_radius = max(0.08, requested_span_ms / 1000.0 + 0.025)
        start = max(0.0, min(start, when - preview_radius))
        end = min(total_duration, max(end, when + preview_radius))
        if row.get("crossover_start") is not None:
            start = max(0.0, min(
                start, float(row["crossover_start"]) - 0.01))
        if row.get("crossover_end") is not None:
            end = min(total_duration, max(
                end, float(row["crossover_end"]) + 0.01))
        sr = max(1, int(self.diagnostic.get("sample_rate") or 1))
        first = max(0, int(np.floor(start * sr)))
        last = min(len(self.samples), int(np.ceil(end * sr)) + 1)
        times = np.arange(first, last, dtype=np.float64) / sr
        local = self.samples[first:last]

        self.local_waveform_plot.clear()
        self._window_preview_region = None
        self._window_left_handle = None
        self._window_right_handle = None
        self._crossover_preview_region = None
        self._rendered_crossover_region = None
        self._crossover_left_handle = None
        self._crossover_right_handle = None
        if local.size:
            self.local_waveform_plot.plot(
                times, local, pen=pg.mkPen("#2F65B0", width=1))
            handoff_start = float(row.get("handoff_start") or when)
            handoff_end = float(row.get("handoff_end") or when)
            if handoff_end > handoff_start:
                region = pg.LinearRegionItem(
                    values=(handoff_start, handoff_end), movable=False,
                    brush=pg.mkBrush(211, 163, 58, 42),
                    pen=pg.mkPen(None))
                region.setZValue(-2)
                self.local_waveform_plot.addItem(region)
            self._add_window_preview(row, when, start, end)
            self._add_crossover_preview(row, when, start, end)
            self._splice_line(self.local_waveform_plot, when)
            for mark in plot_data.get("pitchmarks") or []:
                self.local_waveform_plot.addItem(pg.InfiniteLine(
                    pos=float(mark), angle=90,
                    pen=pg.mkPen("#7A7A7A", width=1,
                                 style=Qt.DotLine)))
            self.local_waveform_plot.setXRange(start, end, padding=.01)
            self.local_waveform_plot.enableAutoRange(axis="y", enable=True)
        else:
            self._plot_message(self.local_waveform_plot,
                               "Insufficient waveform context")

        self.difference_plot.clear()
        if local.size >= 2:
            self.difference_plot.plot(
                times[1:], np.diff(local.astype(np.float64)),
                pen=pg.mkPen("#5A4A91", width=1))
            self._splice_line(self.difference_plot, when)
            self.difference_plot.setXRange(start, end, padding=.01)
            self.difference_plot.enableAutoRange(axis="y", enable=True)
        else:
            self._plot_message(self.difference_plot,
                               "Insufficient waveform context")

        periods = dict(plot_data.get("periods") or {})
        self._plot_period_pair(self.raw_period_plot, periods, "right_raw")
        self._plot_period_pair(
            self.normalised_period_plot, periods, "right_normalised")
        self._plot_period_pair(
            self.aligned_period_plot, periods, "right_best_aligned")

        self.rms_plot.clear()
        rms_rows = list(plot_data.get("period_rms") or [])
        if rms_rows:
            for side, color in (("left", "#2F65B0"),
                                ("right", "#B05B32")):
                rows = [item for item in rms_rows if item.get("side") == side]
                self.rms_plot.plot(
                    [float(item["time"]) for item in rows],
                    [float(item["rms"]) for item in rows],
                    pen=pg.mkPen(color, width=2), symbol="o", symbolSize=5,
                    symbolBrush=pg.mkBrush(color))
            self._splice_line(self.rms_plot, when)
            self.rms_plot.enableAutoRange()
        else:
            self._plot_message(self.rms_plot, "Insufficient RMS context")

        self.f0_plot.clear()
        f0_rows = list(plot_data.get("local_f0") or [])
        if f0_rows:
            for side, color in (("left", "#2F65B0"),
                                ("right", "#B05B32")):
                rows = [item for item in f0_rows if item.get("side") == side]
                self.f0_plot.plot(
                    [float(item["time"]) for item in rows],
                    [float(item["f0_hz"]) for item in rows],
                    pen=pg.mkPen(color, width=2), symbol="o", symbolSize=5,
                    symbolBrush=pg.mkBrush(color))
            self._splice_line(self.f0_plot, when)
            self.f0_plot.enableAutoRange()
        else:
            self._plot_message(
                self.f0_plot, "Unavailable for unvoiced or silent join")

        self.spectral_plot.clear()
        spectral = dict(plot_data.get("spectral_trajectory") or {})
        left_times = np.asarray(spectral.get("left_times") or [], np.float64)
        right_times = np.asarray(spectral.get("right_times") or [], np.float64)
        left_values = np.asarray(
            spectral.get("left_projection") or [], np.float64)
        right_values = np.asarray(
            spectral.get("right_projection") or [], np.float64)
        if left_times.size and right_times.size:
            self.spectral_plot.plot(
                left_times, left_values, pen=None, symbol="o", symbolSize=6,
                symbolBrush=pg.mkBrush("#2F65B0"))
            self.spectral_plot.plot(
                right_times, right_values, pen=None, symbol="o", symbolSize=6,
                symbolBrush=pg.mkBrush("#B05B32"))
            for side_times, intercept_key, slope_key, color in (
                    (left_times, "left_intercept", "left_slope", "#2F65B0"),
                    (right_times, "right_intercept", "right_slope", "#B05B32")):
                intercept = float(spectral.get(intercept_key) or 0.0)
                slope = float(spectral.get(slope_key) or 0.0)
                line_times = np.asarray(
                    ([float(np.min(side_times)), when]
                     if side_times is left_times else
                     [when, float(np.max(side_times))]), np.float64)
                self.spectral_plot.plot(
                    line_times, intercept + slope * (line_times - when),
                    pen=pg.mkPen(color, width=2))
            self._splice_line(self.spectral_plot, when)
            self.spectral_plot.plot(
                [when, when],
                [float(spectral.get("left_intercept") or 0.0),
                 float(spectral.get("right_intercept") or 0.0)],
                pen=pg.mkPen("#B12E2A", width=3))
            self.spectral_plot.enableAutoRange()
        else:
            self._plot_message(self.spectral_plot,
                               "Insufficient spectral context")

        formants = dict(plot_data.get("formants") or {})
        tracks = list(formants.get("tracks") or [])
        self.formant_plot.clear()
        formant_colors = ("#2F65B0", "#2B8A6E", "#A06D22", "#8D4E8E")
        if tracks:
            for track_index, track in enumerate(tracks):
                color = formant_colors[track_index % len(formant_colors)]
                for side, symbol, times_key, values_key in (
                        ("left", "o", "left_times", "left_values"),
                        ("right", "t", "right_times", "right_values")):
                    track_times = np.asarray(
                        track.get(times_key) or [], np.float64)
                    track_values = np.asarray(
                        track.get(values_key) or [], np.float64)
                    if not track_times.size or not track_values.size:
                        continue
                    self.formant_plot.plot(
                        track_times, track_values, pen=None, symbol=symbol,
                        symbolSize=6, symbolBrush=pg.mkBrush(color),
                        symbolPen=pg.mkPen(color))
                    intercept = float(track.get(
                        "%s_intercept" % side) or 0.0)
                    slope = float(track.get("%s_slope" % side) or 0.0)
                    line_times = np.asarray(
                        ([float(np.min(track_times)), when]
                         if side == "left" else
                         [when, float(np.max(track_times))]), np.float64)
                    self.formant_plot.plot(
                        line_times,
                        intercept + slope * (line_times - when),
                        pen=pg.mkPen(color, width=2,
                                     style=(Qt.SolidLine if side == "left"
                                            else Qt.DashLine)))
                label = pg.TextItem(
                    str(track.get("name") or "F?"), color=color,
                    anchor=(0.0, 0.5))
                label.setPos(when, float(track.get("right_intercept") or
                                        track.get("left_intercept") or 0.0))
                self.formant_plot.addItem(label)
            self._splice_line(self.formant_plot, when)
            rejected_count = (len(formants.get("left_rejected") or []) +
                              len(formants.get("right_rejected") or []))
            if rejected_count:
                note = pg.TextItem(
                    "%d rejected/ambiguous candidates retained in JSON" %
                    rejected_count, color="#555555", anchor=(0.0, 1.0))
                note.setPos(when, 0.0)
                self.formant_plot.addItem(note)
            self.formant_plot.enableAutoRange()
        else:
            self._plot_message(
                self.formant_plot,
                "Unavailable for unvoiced, silent, or ambiguous context")

        self.spectral_envelope_plot.clear()
        envelopes = dict(formants.get("spectral_envelopes") or {})
        frequencies = np.asarray(
            envelopes.get("frequencies_hz") or [], np.float64)
        left_envelope = np.asarray(envelopes.get("left_db") or [], np.float64)
        right_envelope = np.asarray(
            envelopes.get("right_db") or [], np.float64)
        envelope_count = min(
            len(frequencies), len(left_envelope), len(right_envelope))
        if envelope_count:
            self.spectral_envelope_plot.plot(
                frequencies[:envelope_count], left_envelope[:envelope_count],
                pen=pg.mkPen("#2F65B0", width=2))
            self.spectral_envelope_plot.plot(
                frequencies[:envelope_count], right_envelope[:envelope_count],
                pen=pg.mkPen("#B05B32", width=2, style=Qt.DashLine))
            self.spectral_envelope_plot.enableAutoRange()
        else:
            self._plot_message(
                self.spectral_envelope_plot,
                "No stable spectral envelope available")

        self.formant_balance_plot.clear()
        balance = dict(formants.get("balance") or {})
        balance_plotted = False
        for side, style in (("left", Qt.SolidLine),
                            ("right", Qt.DashLine)):
            balance_times = np.asarray(
                balance.get("%s_times" % side) or [], np.float64)
            balance_values = np.asarray(
                balance.get("%s_values" % side) or [], np.float64)
            if balance_values.ndim != 2 or not balance_times.size:
                continue
            count = min(len(balance_times), len(balance_values))
            for formant_index in range(balance_values.shape[1]):
                color = formant_colors[
                    formant_index % len(formant_colors)]
                self.formant_balance_plot.plot(
                    balance_times[:count],
                    balance_values[:count, formant_index],
                    pen=pg.mkPen(color, width=2, style=style),
                    symbol="o" if side == "left" else "t",
                    symbolSize=5, symbolBrush=pg.mkBrush(color))
                balance_plotted = True
        if balance_plotted:
            self._splice_line(self.formant_balance_plot, when)
            self.formant_balance_plot.enableAutoRange()
        else:
            self._plot_message(
                self.formant_balance_plot,
                "Insufficient formant-energy context")

    @staticmethod
    def _add_verticals(plot, values, low, high, pen):
        if not values:
            return
        x = np.repeat(np.asarray(values, np.float64), 3)
        y = np.tile(np.asarray((low, high, np.nan), np.float64), len(values))
        plot.plot(x, y, pen=pen, connect="finite")

    def _redraw_waveform(self):
        left, right = self.waveform_plot.getViewBox().viewRange()[0]
        width = max(64, int(
            self.waveform_plot.getViewBox().sceneBoundingRect().width()))
        x, y = self._wave_cache.display(
            left, right, int(self.diagnostic.get("sample_rate") or 1), width)
        self._wave_curve.setData(x, y, connect="finite")

    def _redraw_span_labels(self):
        for item in self._span_labels:
            self.span_plot.removeItem(item)
        self._span_labels = []
        left, right = self.span_plot.getViewBox().viewRange()[0]
        span = max(1e-9, right - left)
        width = max(64.0,
                    self.span_plot.getViewBox().sceneBoundingRect().width())
        for rows, y, key in (
                (self.segments, 0.46, "phone"),
                (self.units, 1.46, "pair")):
            visible = [row for row in rows
                       if float(row["end"]) >= left and
                       float(row["start"]) <= right]
            for row in visible[:120]:
                pixels = ((float(row["end"]) - float(row["start"]))
                          / span * width)
                if pixels < 28.0:
                    continue
                label = str(row.get(key) or "")
                metrics = QtGui.QFontMetrics(self.font())
                label = metrics.elidedText(
                    label, Qt.ElideRight, max(8, int(pixels) - 6))
                item = pg.TextItem(
                    label, color="#24313F" if y < 1.0 else "#FFFFFF",
                    anchor=(0.5, 0.5))
                item.setToolTip(str(row.get(key) or ""))
                item.setPos(
                    (float(row["start"]) + float(row["end"])) * 0.5, y)
                self.span_plot.addItem(item)
                self._span_labels.append(item)

    def _table_row_changed(self, current, _previous):
        if not current.isValid():
            return
        row = self.table_model.rows[current.row()]
        when = float(row.get("time") or 0.0)
        collar = max(
            float(row.get("incoming_collar_ms") or 0.0),
            float(row.get("outgoing_collar_ms") or 0.0)) / 1000.0
        radius = max(0.08, collar + 0.05)
        duration = float(self.diagnostic.get("duration") or 0.0)
        self.waveform_plot.setXRange(
            max(0.0, when - radius), min(duration, when + radius),
            padding=0)
        self._populate_join_detail(row)


# Compatibility for callers and third-party scripts using the original name.
JoinLoudnessDialog = JoinDiscontinuityDialog


class RenderedFormantDialog(QtWidgets.QDialog):
    """Whole-render formant view with inspectable potential jump markers."""

    FORMANT_COLORS = ("#FFD04A", "#42D6A4", "#DF80FF", "#62C8FF")

    def __init__(self, diagnostic, samples, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.diagnostic = dict(diagnostic or {})
        self.samples = np.asarray(samples, np.float32).reshape(-1)
        self.phones = [dict(row) for row in
                       self.diagnostic.get("phones") or ()]
        self.jumps = [dict(row) for row in
                      self.diagnostic.get("jumps") or ()]
        self.setWindowTitle("Rendered formants and potential jumps")
        self.resize(1280, 820)
        layout = QtWidgets.QVBoxLayout(self)

        self.summary = QtWidgets.QLabel(
            "%d accepted frames, %d rejected | %d phones analyzed | "
            "%d potential jumps" % (
                int(self.diagnostic.get("accepted_frame_count") or 0),
                int(self.diagnostic.get("rejected_frame_count") or 0),
                int(self.diagnostic.get("analyzed_phone_count") or 0),
                int(self.diagnostic.get("potential_jump_count") or 0),
            )
        )
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        note = QtWidgets.QLabel(
            "Tracks are measured from the final synthesized waveform. The "
            "spectrogram is frequency-smoothed only for readability. Markers "
            "identify potential abrupt formant changes; legitimate consonant "
            "and vowel transitions can also be abrupt, so inspect the phone "
            "context and exact join evidence before changing unit selection."
            " Track colors: F1 yellow, F2 green, F3 purple, F4 cyan."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#4D5156;")
        layout.addWidget(note)

        self.formant_plot = pg.PlotWidget(background="#1E2430")
        self.formant_plot.setLabel("left", "Frequency", units="Hz")
        self.formant_plot.setLabel("bottom", "Rendered time", units="s")
        self.formant_plot.showGrid(x=True, y=True, alpha=.12)
        duration = max(
            0.001, float(self.diagnostic.get("duration_seconds") or 0.0)
        )
        sample_rate = max(1, int(self.diagnostic.get("sample_rate") or 1))
        maximum_hz = min(6000.0, sample_rate * 0.5)
        self.formant_plot.setLimits(
            xMin=0.0, xMax=duration, yMin=0.0, yMax=maximum_hz
        )
        self.formant_plot.setYRange(0.0, maximum_hz, padding=0)
        self._add_spectrogram(sample_rate, duration, maximum_hz)
        self._add_tracks()
        self._add_phone_boundaries(maximum_hz)
        self._add_jump_markers(maximum_hz)
        layout.addWidget(self.formant_plot, 5)

        self.phone_plot = pg.PlotWidget(background="#E6E7E5")
        self.phone_plot.setXLink(self.formant_plot)
        self.phone_plot.setMouseEnabled(x=True, y=False)
        self.phone_plot.setYRange(0.0, 1.0, padding=0)
        self.phone_plot.setFixedHeight(74)
        self.phone_plot.hideAxis("left")
        self.phone_plot.hideAxis("bottom")
        self._phone_labels = []
        self._label_timer = QtCore.QTimer(self)
        self._label_timer.setSingleShot(True)
        self._label_timer.timeout.connect(self._redraw_phone_labels)
        self.phone_plot.getViewBox().sigXRangeChanged.connect(
            lambda *_args: self._label_timer.start(0)
        )
        self._add_phone_strip()
        layout.addWidget(self.phone_plot)

        self.table = QtWidgets.QTableWidget(len(self.jumps), 8)
        self.table.setHorizontalHeaderLabels((
            "Rank", "Time", "Context", "Type", "Severity",
            "Largest jump", "Novelty", "Evidence",
        ))
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self._populate_jump_table()
        self.table.currentCellChanged.connect(self._jump_row_changed)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(140)
        layout.addWidget(self.table, 2)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.save_json_button = buttons.addButton(
            "Save JSON...", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.save_json_button.clicked.connect(self._save_json)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.formant_plot.setXRange(0.0, duration, padding=0)
        self._redraw_phone_labels()

    def _add_spectrogram(self, sample_rate, duration, maximum_hz):
        if not self.samples.size or sample_rate < 8000:
            return
        hop = min(1024, max(128, int(math.ceil(
            self.samples.size / 4000.0
        ))))
        _times, frequencies, decibels = join_spectrogram.spectrogram_db(
            self.samples, sample_rate, fft_size=1024, hop_size=hop
        )
        keep = frequencies <= maximum_hz
        visible_frequencies = frequencies[keep]
        display = vocal_tract_validation.formant_view_spectrogram_db(
            decibels[keep], visible_frequencies
        )
        rgb = join_spectrogram._spectrogram_rgb(display)
        item = pg.ImageItem(
            np.transpose(rgb, (1, 0, 2)), axisOrder="col-major"
        )
        item.setRect(QtCore.QRectF(0.0, 0.0, duration, maximum_hz))
        item.setZValue(-20)
        self.formant_plot.addItem(item)
        self.spectrogram_item = item

    def _add_tracks(self):
        for phone in self.phones:
            frames = [dict(row) for row in phone.get("frames") or ()]
            if not frames:
                continue
            times = np.asarray([float(row.get("time") or 0.0)
                                for row in frames], np.float64)
            for index, color in enumerate(self.FORMANT_COLORS):
                values = np.asarray([
                    (row.get("formants_hz") or [None] * 4)[index]
                    for row in frames
                ], dtype=object)
                values = np.asarray([
                    float(value) if value is not None else np.nan
                    for value in values
                ], np.float64)
                if np.count_nonzero(np.isfinite(values)) < 2:
                    continue
                self.formant_plot.plot(
                    times, values, pen=pg.mkPen("#151A22", width=5),
                    connect="finite"
                )
                self.formant_plot.plot(
                    times, values, pen=pg.mkPen(color, width=2.3),
                    connect="finite"
                )

    def _add_phone_boundaries(self, maximum_hz):
        boundaries = [float(row.get("start") or 0.0)
                      for row in self.phones]
        if self.phones:
            boundaries.append(float(self.phones[-1].get("end") or 0.0))
        JoinDiscontinuityDialog._add_verticals(
            self.formant_plot, boundaries, 0.0, maximum_hz,
            pg.mkPen(230, 110, 100, 125, width=1, style=Qt.DashLine),
        )

    def _add_jump_markers(self, maximum_hz):
        if not self.jumps:
            return
        frame_step = max(
            .004, float(self.diagnostic.get("analysis_frame_step_seconds")
                        or .010)
        )
        for row in self.jumps:
            when = float(row.get("time") or 0.0)
            exact = bool(row.get("exact_splice_evidence"))
            color = (210, 62, 54, 48) if exact else (230, 150, 45, 42)
            region = pg.LinearRegionItem(
                values=(when - frame_step, when + frame_step),
                movable=False, brush=pg.mkBrush(*color),
                pen=pg.mkPen(None),
            )
            region.setZValue(-5)
            self.formant_plot.addItem(region)
        self.formant_plot.plot(
            [float(row.get("time") or 0.0) for row in self.jumps],
            [maximum_hz * .975] * len(self.jumps),
            pen=None, symbol="t", symbolSize=12,
            symbolBrush=[
                pg.mkBrush("#D23E36") if row.get("exact_splice_evidence")
                else pg.mkBrush("#E6962D") for row in self.jumps
            ],
            symbolPen=pg.mkPen("#FFFFFF", width=1),
        )

    def _add_phone_strip(self):
        if not self.phones:
            return
        bars = pg.BarGraphItem(
            x0=[float(row.get("start") or 0.0) for row in self.phones],
            x1=[float(row.get("end") or 0.0) for row in self.phones],
            y0=[.08] * len(self.phones), y1=[.92] * len(self.phones),
            brushes=[pg.mkBrush("#AAB9CB" if index % 2 == 0 else "#BCC7D2")
                     for index, _row in enumerate(self.phones)],
            pen=pg.mkPen("#6F7D8D"),
        )
        self.phone_plot.addItem(bars)

    def _redraw_phone_labels(self):
        for item in self._phone_labels:
            self.phone_plot.removeItem(item)
        self._phone_labels = []
        left, right = self.phone_plot.getViewBox().viewRange()[0]
        visible_span = max(1.0e-9, right - left)
        width = max(64.0,
                    self.phone_plot.getViewBox().sceneBoundingRect().width())
        for row in self.phones:
            start, end = float(row["start"]), float(row["end"])
            if end < left or start > right:
                continue
            pixels = (end - start) / visible_span * width
            if pixels < 26.0:
                continue
            text = str(row.get("phone") or "")
            text = QtGui.QFontMetrics(self.font()).elidedText(
                text, Qt.ElideRight, max(8, int(pixels) - 6)
            )
            item = pg.TextItem(text, color="#24313F", anchor=(.5, .5))
            item.setPos((start + end) * .5, .5)
            item.setToolTip(str(row.get("phone") or ""))
            self.phone_plot.addItem(item)
            self._phone_labels.append(item)

    def _populate_jump_table(self):
        for row_index, row in enumerate(self.jumps):
            maximum = row.get("max_delta_cents")
            novelty = row.get("novelty")
            values = (
                str(int(row.get("rank") or row_index + 1)),
                "%.3f s" % float(row.get("time") or 0.0),
                "%s -> %s" % (row.get("left_phone") or "?",
                               row.get("right_phone") or "?"),
                str(row.get("kind") or "").replace("_", " ").title(),
                "%.2f" % float(row.get("severity") or 0.0),
                "n/a" if maximum is None else "%+.0f cents" % float(maximum),
                "n/a" if novelty is None else "%.2f" % float(novelty),
                "Exact splice" if row.get("exact_splice_evidence")
                else "Estimated trajectory",
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(str(row.get("interpretation") or ""))
                if column in {0, 1, 4, 5, 6}:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row_index, column, item)
        self.table.resizeColumnsToContents()

    def _jump_row_changed(self, row, _column, _old_row, _old_column):
        if not 0 <= row < len(self.jumps):
            return
        when = float(self.jumps[row].get("time") or 0.0)
        duration = float(self.diagnostic.get("duration_seconds") or 0.0)
        radius = max(.10, duration * .035)
        self.formant_plot.setXRange(
            max(0.0, when - radius), min(duration, when + radius), padding=0
        )

    def _save_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save rendered formant diagnostic",
            "rendered_formants.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self.diagnostic, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            QtWidgets.QMessageBox.warning(
                self, "Save rendered formant diagnostic", str(error)
            )


class IntonationTrack(TimelinePlotWidget):
    """Clickable phrase blocks for punctuation-shaped intonation."""
    blocksCommitted = QtCore.pyqtSignal(object)

    COLORS = {
        ".": "#6E8B74", "?": "#4A78C2", "!": "#C05A48",
        ",": "#A58A3A", ":": "#7665A8", ";": "#4E8B8B",
    }

    def __init__(self, parent=None):
        super().__init__(parent, background='#DCDCDC')
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.setYRange(0.0, 1.0, padding=0)
        self.setMinimumHeight(110)
        self.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.setToolTip("click a phrase block to choose its intonation")
        self._blocks = []
        self._bars = []
        self._labels = []
        self._lod_detailed = True
        self._display_block_count = 0
        self._lod_timer = QtCore.QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.timeout.connect(self._redraw)
        self.getViewBox().sigXRangeChanged.connect(
            lambda *_args: self._schedule_lod_redraw())

    def _schedule_lod_redraw(self):
        self._queue_visible_timer(self._lod_timer)

    def _timeline_became_visible(self):
        self._schedule_lod_redraw()

    def set_blocks(self, blocks):
        self._blocks = [dict(b) for b in (blocks or [])]
        self._redraw()

    def blocks(self):
        return [dict(b) for b in self._blocks]

    def _redraw(self):
        for item in self._bars + self._labels:
            self.removeItem(item)
        self._bars, self._labels = [], []
        if not self._blocks:
            self._display_block_count = 0
            return
        left, right = self.getViewBox().viewRange()[0]
        span = max(1e-9, right - left)
        width = max(32.0, self.getViewBox().sceneBoundingRect().width())
        indices = [index for index, block in enumerate(self._blocks)
                   if float(block["end"]) >= left and
                   float(block["start"]) <= right]
        detailed = len(indices) * PARAMETER_DETAIL_MIN_PX <= width
        self._lod_detailed = detailed
        if detailed:
            groups = [[index] for index in indices]
        else:
            buckets = {}
            for index in indices:
                block = self._blocks[index]
                middle = (float(block["start"]) +
                          float(block["end"])) * 0.5
                bucket = int(np.floor(
                    ((middle - left) / span * width) /
                    PARAMETER_OVERVIEW_BUCKET_PX))
                buckets.setdefault(bucket, []).append(index)
            groups = [buckets[key] for key in sorted(buckets)]
        self._display_block_count = len(groups)
        if not groups:
            return
        x0, x1, brushes, representatives = [], [], [], []
        punctuation_priority = {"!": 5, "?": 4, ";": 3,
                                ":": 2, ",": 1, ".": 0}
        for group in groups:
            representative = max(
                group, key=lambda index: punctuation_priority.get(
                    str(self._blocks[index].get("kind") or "."), 0))
            representatives.append(representative)
            x0.append(float(self._blocks[group[0]]["start"]))
            x1.append(float(self._blocks[group[-1]]["end"]))
            kind = str(self._blocks[representative].get("kind") or ".")
            brushes.append(pg.mkBrush(self.COLORS.get(kind, "#777777")))
        bar = pg.BarGraphItem(
            x0=x0, x1=x1, y0=[0.08] * len(groups),
            y1=[0.92] * len(groups), brushes=brushes,
            pen=pg.mkPen('#505050', width=1))
        self.addItem(bar)
        self._bars.append(bar)
        if not detailed:
            return
        pixels_per_second = width / span
        for representative in representatives:
            block = self._blocks[representative]
            a, b = float(block["start"]), float(block["end"])
            if (b - a) * pixels_per_second < 18:
                continue
            kind = str(block.get("kind") or ".")
            label = pg.TextItem(kind, color='#FFFFFF', anchor=(0.5, 0.5))
            label.setPos((a + b) / 2.0, 0.5)
            self.addItem(label)
            self._labels.append(label)

    def _view_pos(self, ev):
        return self.getPlotItem().getViewBox().mapSceneToView(
            self.mapToScene(ev.pos()))

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.accept()
            return
        x = self._view_pos(ev).x()
        idx = next((i for i, b in enumerate(self._blocks)
                    if float(b["start"]) <= x <= float(b["end"])), None)
        if idx is None:
            ev.accept()
            return
        menu = QtWidgets.QMenu()
        choices = (("Statement", "."), ("Question", "?"),
                   ("Exclamation", "!"), ("Continuation", ","),
                   ("Colon", ":"), ("Semicolon", ";"))
        for label, kind in choices:
            action = menu.addAction("%s  %s" % (kind, label))
            action.setData(kind)
        chosen = menu.exec_(ev.globalPos())
        if chosen is not None:
            self._blocks[idx]["kind"] = str(chosen.data())
            self._redraw()
            self.blocksCommitted.emit(self.blocks())
        ev.accept()


class StableWaveformViewBox(pg.ViewBox):
    """Keep waveform auto-fit horizontal without moving its vertical center."""
    def enableAutoRange(self, axis=None, enable=True, x=None, y=None):
        if x is not None or y is not None:
            if x is not None:
                super().enableAutoRange(axis='x', enable=x)
            super().enableAutoRange(axis='y', enable=False)
            return
        if axis in (None, getattr(pg.ViewBox, "XYAxes", 2)):
            # PlotItem's Auto button calls this with no axis.  Persistent X
            # auto-range feeds every LOD redraw back into the range solver and
            # can make the waveform appear stuck.  Treat the button as a
            # one-shot horizontal fit instead.
            if enable:
                provider = getattr(self, "fitRangeProvider", None)
                fitted = provider() if callable(provider) else None
                if fitted and float(fitted[1]) > float(fitted[0]):
                    self.setXRange(
                        float(fitted[0]), float(fitted[1]), padding=0.01)
                else:
                    self.autoRange()
            super().enableAutoRange(x=False, y=False)
            return
        if axis in ('y', getattr(pg.ViewBox, "YAxis", 1)):
            super().enableAutoRange(axis='y', enable=False)
            return
        super().enableAutoRange(axis=axis, enable=enable)

    def autoRange(self, *args, **kwargs):
        super().autoRange(*args, **kwargs)
        super().enableAutoRange(x=False, y=False)
        self.setYRange(-1.05, 1.05, padding=0)


class SelectableWaveformViewBox(StableWaveformViewBox):
    """Left drag selects time; middle drag retains optional timeline panning."""
    selectionDragged = QtCore.pyqtSignal(float, float, bool, bool)
    selectionMoveDragged = QtCore.pyqtSignal(float, float, bool)
    contextRequested = QtCore.pyqtSignal(float, object)

    def _scroll_x(self, delta):
        left, right = self.viewRange()[0]
        shift = (right - left) * (-0.12 if delta > 0 else 0.12)
        self.setXRange(left + shift, right + shift, padding=0)

    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & Qt.ShiftModifier:
            delta = float(ev.delta())
            if delta:
                self._scroll_x(delta)
            ev.accept()
            return
        super().wheelEvent(ev, axis=axis)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.LeftButton:
            start = self.mapToView(ev.buttonDownPos()).x()
            end = self.mapToView(ev.pos()).x()
            if ev.modifiers() & Qt.ShiftModifier:
                self.selectionMoveDragged.emit(
                    float(start), float(end), bool(ev.isFinish()))
            else:
                self.selectionDragged.emit(
                    float(start), float(end), bool(ev.isFinish()), False)
            ev.accept()
            return
        if ev.button() == Qt.RightButton:
            if ev.isFinish():
                when = float(self.mapToView(ev.pos()).x())
                self.contextRequested.emit(when, ev.screenPos())
            ev.accept()
            return
        super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            when = float(self.mapToView(ev.pos()).x())
            self.selectionDragged.emit(when, when, True,
                                       bool(ev.modifiers() & Qt.ShiftModifier))
            ev.accept()
            return
        if ev.button() == Qt.RightButton:
            when = float(self.mapToView(ev.pos()).x())
            self.contextRequested.emit(when, ev.screenPos())
            ev.accept()
            return
        super().mouseClickEvent(ev)


class WaveformPeakCache:
    """Audacity-style multiresolution peaks for viewport-sized waveform draws.

    Summary levels grow 16, 32, 64... samples per entry. This avoids the
    expensive middle-zoom cliff caused by choosing between only 256-sample
    and 65,536-sample summaries. At useful editing scales, extrema remain in
    sample order so the trace is continuous rather than a row of peak sticks.
    """
    def __init__(self, samples=None):
        self.samples = np.zeros(0, np.float32)
        self._summaries = {}
        self.last_block_size = 1
        self.last_source_count = 0
        self.last_output_points = 0
        self.last_mode = "line"
        self.set_samples(samples if samples is not None else [])

    def set_samples(self, samples):
        self.samples = np.asarray(samples, np.float32).reshape(-1)
        self._summaries = {}

    def clear_summaries(self):
        """Drop only reproducible LOD arrays, retaining the source waveform."""
        self._summaries.clear()

    def cache_info(self):
        return {
            "entries": len(self._summaries),
            "bytes": sum(
                int(minimum.nbytes) + int(maximum.nbytes)
                for minimum, maximum in self._summaries.values()
            ),
        }

    def _summary(self, block_size):
        block_size = int(block_size)
        cached = self._summaries.get(block_size)
        if cached is not None:
            return cached
        if (block_size < WAVEFORM_SUMMARY_BLOCK or
                block_size % WAVEFORM_SUMMARY_BLOCK):
            raise ValueError("unsupported waveform summary block")
        ratio = block_size // WAVEFORM_SUMMARY_BLOCK
        if ratio & (ratio - 1):
            raise ValueError("waveform summary block must be a power-of-two "
                             "multiple of the base block")
        if not len(self.samples):
            result = (np.zeros(0, np.float32), np.zeros(0, np.float32))
        elif block_size == WAVEFORM_SUMMARY_BLOCK:
            starts = np.arange(0, len(self.samples), block_size,
                               dtype=np.int64)
            result = (
                np.minimum.reduceat(self.samples, starts),
                np.maximum.reduceat(self.samples, starts))
        else:
            lower_min, lower_max = self._summary(
                block_size // WAVEFORM_SUMMARY_GROWTH)
            starts = np.arange(
                0, len(lower_min), WAVEFORM_SUMMARY_GROWTH, dtype=np.int64)
            result = (np.minimum.reduceat(lower_min, starts),
                      np.maximum.reduceat(lower_max, starts))
        self._summaries[block_size] = result
        return result

    def _connected_display(self, data_start, data_end, view_start,
                           view_samples, sr, pixels):
        """Preserve the chronological extrema in each screen column."""
        indices = np.arange(data_start, data_end, dtype=np.int64)
        values = self.samples[data_start:data_end]
        columns = np.floor(
            (indices - view_start) / view_samples * pixels).astype(np.int64)
        valid = (columns >= 0) & (columns < pixels)
        indices = indices[valid]
        values = values[valid]
        columns = columns[valid]
        if not len(indices):
            return np.zeros(0, np.float64), np.zeros(0, np.float32)
        runs = np.r_[0, np.flatnonzero(np.diff(columns)) + 1]
        stops = np.r_[runs[1:], len(indices)]
        selected = [0]
        for start, stop in zip(runs, stops):
            chunk = values[start:stop]
            if not len(chunk):
                continue
            low = start + int(np.argmin(chunk))
            high = start + int(np.argmax(chunk))
            selected.extend((low, high) if low <= high else (high, low))
        selected.append(len(indices) - 1)
        positions = np.unique(np.asarray(selected, dtype=np.int64))
        return (indices[positions].astype(np.float64) / float(sr),
                values[positions])

    @staticmethod
    def _block_size(samples_per_pixel):
        block_size = WAVEFORM_SUMMARY_BLOCK
        while (block_size * WAVEFORM_SUMMARY_GROWTH <=
               samples_per_pixel):
            block_size *= WAVEFORM_SUMMARY_GROWTH
        return block_size

    def display(self, start_s, end_s, sr, pixel_width):
        """Return raw samples up close, otherwise one min/max line per pixel."""
        sr = max(1, int(sr))
        pixels = max(1, int(pixel_width))
        view_start = float(min(start_s, end_s)) * sr
        view_end = float(max(start_s, end_s)) * sr
        view_samples = max(1.0, view_end - view_start)
        data_start = max(0, int(np.floor(view_start)))
        data_end = min(len(self.samples), int(np.ceil(view_end)))
        if data_end <= data_start:
            self.last_source_count = self.last_output_points = 0
            return np.zeros(0, np.float64), np.zeros(0, np.float32)

        samples_per_pixel = view_samples / float(pixels)
        if samples_per_pixel <= WAVEFORM_RAW_SAMPLES_PER_PIXEL:
            first = max(0, data_start - 1)
            last = min(len(self.samples), data_end + 1)
            x = np.arange(first, last, dtype=np.float64) / float(sr)
            y = self.samples[first:last]
            self.last_block_size = 1
            self.last_source_count = len(y)
            self.last_output_points = len(y)
            self.last_mode = "line"
            return x, y

        if samples_per_pixel <= WAVEFORM_CONNECTED_SAMPLES_PER_PIXEL:
            x, y = self._connected_display(
                data_start, data_end, view_start, view_samples, sr, pixels)
            self.last_block_size = 1
            self.last_source_count = data_end - data_start
            self.last_output_points = len(y)
            self.last_mode = "line"
            return x, y

        block_size = self._block_size(samples_per_pixel)
        first_block = data_start // block_size
        last_block = int(np.ceil(data_end / float(block_size)))
        all_minimum, all_maximum = self._summary(block_size)
        last_block = min(last_block, len(all_minimum))
        minimum = all_minimum[first_block:last_block].copy()
        maximum = all_maximum[first_block:last_block].copy()
        if not len(minimum):
            self.last_source_count = self.last_output_points = 0
            return np.zeros(0, np.float64), np.zeros(0, np.float32)

        # Cached edge blocks can extend outside the viewport. Recalculate
        # those two blocks from the exact visible samples so offscreen peaks
        # never leak into the display.
        edge_blocks = {first_block, last_block - 1}
        for absolute_block in edge_blocks:
            position = absolute_block - first_block
            start = max(data_start, absolute_block * block_size)
            stop = min(data_end, (absolute_block + 1) * block_size)
            if 0 <= position < len(minimum) and stop > start:
                edge = self.samples[start:stop]
                minimum[position] = np.min(edge)
                maximum[position] = np.max(edge)

        block_starts = np.arange(first_block, last_block,
                                 dtype=np.float64) * block_size
        block_ends = np.minimum(len(self.samples), block_starts + block_size)
        centers = ((np.maximum(block_starts, data_start) +
                    np.minimum(block_ends, data_end)) * 0.5)
        columns = np.floor(
            (centers - view_start) / view_samples * pixels).astype(np.int64)
        valid = (columns >= 0) & (columns < pixels)
        columns = columns[valid]
        minimum = minimum[valid]
        maximum = maximum[valid]
        if not len(columns):
            self.last_source_count = self.last_output_points = 0
            return np.zeros(0, np.float64), np.zeros(0, np.float32)

        runs = np.r_[0, np.flatnonzero(np.diff(columns)) + 1]
        unique_columns = columns[runs]
        peak_min = np.minimum.reduceat(minimum, runs)
        peak_max = np.maximum.reduceat(maximum, runs)
        times = (view_start + (unique_columns + 0.5) *
                 view_samples / pixels) / float(sr)
        # Two continuous envelope outlines are considerably cheaper for Qt
        # to paint than thousands of independent vertical peak sticks, and
        # remain readable at the intermediate zoom levels used for editing.
        x = np.concatenate((times, np.asarray([np.nan]), times))
        y = np.concatenate((peak_min, np.asarray([np.nan], np.float32),
                            peak_max))
        self.last_block_size = block_size
        self.last_source_count = len(minimum)
        self.last_output_points = len(y)
        self.last_mode = "envelope"
        return x, y


class TimelineRuler(TimelinePlotWidget):
    """Shared seconds ruler and the only draggable playhead surface."""
    timeChanged = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent, background="#D5D5D1")
        self._duration = 1.0
        self._setting = False
        self._dragging = False
        self.setFixedHeight(34)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideAxis("left")
        self.setYRange(0.0, 1.0, padding=0)
        self.getPlotItem().setContentsMargins(0, 0, 0, 0)
        axis = self.getAxis("bottom")
        axis.setHeight(22)
        axis.setStyle(tickLength=-6, autoExpandTextSpace=False,
                      tickTextHeight=12)
        self.cursor = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen("#276A50", width=2),
            hoverPen=pg.mkPen("#3E9A73", width=3))
        try:
            self.cursor.addMarker("v", position=0.08, size=10)
        except AttributeError:
            pass
        self.cursor.setZValue(10)
        self.cursor.setToolTip("Drag playback and phrase-split position")
        self.cursor.sigPositionChanged.connect(self._cursor_changed)
        self.addItem(self.cursor)
        self.set_duration(1.0)

    def set_duration(self, duration):
        self._duration = max(0.0, float(duration))
        limit = max(0.01, self._duration)
        self.cursor.setBounds((0.0, limit))
        self.setLimits(xMin=WAVEFORM_LEFT_LIMIT,
                       xMax=max(limit * 1.6, limit + 1.5))
        self.set_time(min(float(self.cursor.value()), self._duration))

    def set_time(self, when):
        value = max(0.0, min(self._duration, float(when)))
        self._setting = True
        try:
            self.cursor.setValue(value)
        finally:
            self._setting = False

    def time(self):
        return max(0.0, min(self._duration, float(self.cursor.value())))

    def _cursor_changed(self):
        if not self._setting:
            self.timeChanged.emit(self.time())

    def _set_from_event(self, event):
        point = self.getViewBox().mapSceneToView(self.mapToScene(event.pos()))
        self.set_time(float(point.x()))
        self.timeChanged.emit(self.time())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._set_from_event(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._set_from_event(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self._set_from_event(event)
            event.accept()
            return
        super().mouseReleaseEvent(event)


# --------------------------------------------------------------- waveform editor
class WaveformEditor(QtWidgets.QWidget):
    """Waveform + draggable boundaries + duration-aligned editable phoneme
    fields. Boundary drags time-stretch that segment (post-DSP); phoneme text
    edits are collected by the main window's Re-render action."""
    audioChanged = QtCore.pyqtSignal()
    phonesEdited = QtCore.pyqtSignal()
    rerenderRequested = QtCore.pyqtSignal()
    regionCutRequested = QtCore.pyqtSignal(int, int)
    faultTargetRequested = QtCore.pyqtSignal(object)
    playheadChanged = QtCore.pyqtSignal(float)
    timingEditCommitted = QtCore.pyqtSignal(object, object)
    structureEditCommitted = QtCore.pyqtSignal(object, object, str)
    selectionChanged = QtCore.pyqtSignal(object)
    joinCrossoverCommitted = QtCore.pyqtSignal(int, float, float)

    def __init__(self, stretch_hook=None, parent=None):
        super().__init__(parent)
        self.sr = 16000
        self.base_audio = []     # per-segment ORIGINAL audio (stretch source)
        self.base_durs = []      # per-segment ORIGINAL durations
        self.segments = []       # current fc.Segment list
        self.audio = np.zeros(1, np.float32)
        # Long utterances keep only visible controls alive. The lists retain
        # stable segment-index slots so the existing editing API is unchanged.
        self.boundaries = []     # optional pg.InfiniteLine per boundary
        self.fields = []         # optional QLineEdit per segment
        self._dirty_phone_indices = set()
        self.focused_idx = None  # phoneme box the user touched last
        self.stretch_hook = stretch_hook
        self.sustain_hook = None
        self.use_sustain = True
        self.variant_menu_hook = None
        self.selected_range = None
        self._last_selection_signal = None
        self._selection_anchor = None
        self._boundary_drag_snapshot = None
        self._selection_move_state = None
        self._paste_index = None
        self._drag_factors_before = None
        self._scroll_sync = False
        self._fault_regions = []
        self._fault_mode_active = False
        self._pending_action = ""
        self._pending_reason = ""
        self._pending_visual_key = None
        self._setting_pending_visual = False
        self._workspace_duration = 0.0
        self._waveform_cache = WaveformPeakCache(self.audio)
        self._waveform_cache_key = (id(self.audio), len(self.audio), self.sr)
        self._visible_field_indices = set()
        self._visible_boundary_indices = set()
        self._boundary_overview_signature = None
        self._boundary_lod_detailed = True
        self._phone_label_signature = None
        self._styled_focus_idx = None
        self._join_overlay_visible = False
        self._join_overlay_editable = False
        self._join_overlay_records = []
        self._join_overlay_base_segments = []
        self._requested_join_settings = (
            fc.FestivalWSLBackend.normalize_join_settings(None))
        self._join_overlay_spans = None
        self._join_overlay_signature = None
        self._join_overlay_lod_detailed = True
        self._join_overlay_display_count = 0
        self._redrawing_join_overlays = False
        self._selected_join_record = None
        self._join_editor_unit_index = None
        self._join_editor_center = 0.0
        self._join_editor_bounds = (0.0, 0.0)
        self._join_editor_max_seconds = 0.100
        self._join_editor_rendered_ms = 0.0
        self._join_handle_guard = False
        self._visible_refresh_timer = QtCore.QTimer(self)
        self._visible_refresh_timer.setSingleShot(True)
        self._visible_refresh_timer.timeout.connect(self._refresh_visible_view)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        viewbox = SelectableWaveformViewBox()
        viewbox.fitRangeProvider = lambda: (0.0, self.duration() or 1.0)
        self.plot = pg.PlotWidget(background='#B8B8B8', viewBox=viewbox)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.setMenuEnabled(False)
        # zoomable/pannable working space: wheel = x-zoom, drag = pan;
        # extra room to the right for phoneme lengthenings
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.plot.enableAutoRange(axis='y', enable=False)
        self.plot.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.plot.setMinimumHeight(170)
        self.curve = self.plot.plot([], [], pen=pg.mkPen('#1010C0', width=1))
        self.join_overlay_curve = self.plot.plot(
            [], [], pen=pg.mkPen('#A46B13', width=1), connect='finite')
        self.join_overlay_curve.setZValue(9)
        self.join_overlay_curve.setToolTip(
            "Rendered unit joins; shaded spans are effective crossovers")
        self.join_overlay_curve.hide()
        self.join_marker_scatter = pg.ScatterPlotItem(
            size=9, symbol='d', pen=pg.mkPen('#754A05', width=1),
            brush=pg.mkBrush('#E3A43A'), hoverable=True,
            hoverPen=pg.mkPen('#3E2A08', width=2),
            hoverBrush=pg.mkBrush('#FFD06A'))
        self.join_marker_scatter.setZValue(13)
        self.join_marker_scatter.setToolTip(
            "Click a join to edit its crossover edges")
        self.join_marker_scatter.sigClicked.connect(
            self._join_marker_clicked)
        self.plot.addItem(self.join_marker_scatter)
        self.join_marker_scatter.hide()
        self._join_overlay_spans = pg.BarGraphItem(
            x0=[], x1=[], y0=[], y1=[],
            pen=pg.mkPen(None), brush=pg.mkBrush(48, 132, 107, 34))
        self._join_overlay_spans.setZValue(-5)
        self._join_overlay_spans.setToolTip(
            "Effective rendered crossover span")
        self.plot.addItem(self._join_overlay_spans)
        self._join_overlay_spans.hide()
        self.join_edit_region = pg.LinearRegionItem(
            movable=False, brush=pg.mkBrush(42, 145, 112, 45),
            pen=pg.mkPen(None))
        self.join_edit_region.setZValue(-3)
        self.plot.addItem(self.join_edit_region)
        self.join_edit_region.hide()
        self.join_edit_label = pg.TextItem(
            "", color='#174F40', anchor=(0.5, 1.0),
            fill=pg.mkBrush(236, 241, 235, 220),
            border=pg.mkPen(64, 112, 96, 150))
        self.join_edit_label.setZValue(15)
        self.plot.addItem(self.join_edit_label)
        self.join_edit_label.hide()
        self.join_left_handle = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen('#167A5D', width=2),
            hoverPen=pg.mkPen('#22A47C', width=3))
        self.join_right_handle = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen('#167A5D', width=2),
            hoverPen=pg.mkPen('#22A47C', width=3))
        for handle, side in (
                (self.join_left_handle, "left"),
                (self.join_right_handle, "right")):
            handle.setZValue(14)
            handle.setToolTip(
                "Drag the %s crossover edge; re-render applies it" % side)
            handle.sigPositionChanged.connect(
                lambda _line, value=side:
                self._join_handle_changed(value))
            handle.sigPositionChangeFinished.connect(
                self._join_handle_finished)
            self.plot.addItem(handle)
            handle.hide()
        self.boundary_overview = self.plot.plot(
            [], [], pen=pg.mkPen('#D00000', width=1),
            connect='finite')
        self.boundary_overview.setZValue(8)
        self.waveform_zero = pg.InfiniteLine(
            pos=0.0, angle=0, movable=False,
            pen=pg.mkPen('#777777', width=1))
        self.waveform_zero.setZValue(-20)
        self.plot.addItem(self.waveform_zero)
        self.playhead = pg.InfiniteLine(
            pos=0.0, angle=90, movable=False,
            pen=pg.mkPen('#276A50', width=2),
            hoverPen=pg.mkPen('#3E9A73', width=3))
        self.playhead.setZValue(20)
        self.playhead.setToolTip("Playback position; drag it on the ruler above")
        self.plot.addItem(self.playhead)
        self.phone_labels = []   # in-plot phone name tags (zoom-aware)
        self.timeline = TimelineRuler()
        self.timeline.zoom_viewbox = self.plot.getViewBox()
        self.timeline.timeChanged.connect(self.set_playhead)
        lay.addWidget(self.timeline)
        lay.addWidget(self.plot, 1)

        # phoneme boxes are positioned absolutely and follow the waveform's
        # zoom/pan (sigXRangeChanged) so they stay aligned with the audio
        self.fields_host = QtWidgets.QWidget()
        self.fields_host.setFixedHeight(30)
        lay.addWidget(self.fields_host)
        self.plot.getViewBox().sigXRangeChanged.connect(
            lambda *_: self._schedule_visible_refresh())
        self.plot.getViewBox().sigXRangeChanged.connect(
            self._sync_timeline_range)
        self.plot.getViewBox().sigXRangeChanged.connect(
            lambda *_: self._sync_scrollbar_from_view())
        viewbox.selectionDragged.connect(self._on_selection_drag)
        viewbox.selectionMoveDragged.connect(self._on_selection_move_drag)
        viewbox.contextRequested.connect(self._selection_menu)
        self.hscroll = QtWidgets.QScrollBar(Qt.Horizontal)
        self.hscroll.setEnabled(False)
        self.hscroll.setRange(0, 0)
        self.hscroll.valueChanged.connect(self._scroll_to)
        lay.addWidget(self.hscroll)

    def _schedule_visible_refresh(self):
        if not self._visible_refresh_timer.isActive():
            self._visible_refresh_timer.start(0)

    def _ensure_waveform_cache(self):
        key = (id(self.audio), len(self.audio), self.sr)
        if key != self._waveform_cache_key:
            self._waveform_cache.set_samples(self.audio)
            self._waveform_cache_key = key

    def clear_display_cache(self):
        self._waveform_cache.clear_summaries()

    def _render_visible_waveform(self):
        self._ensure_waveform_cache()
        view = self.plot.getViewBox().viewRange()[0]
        width = max(32, int(round(
            self.plot.getViewBox().sceneBoundingRect().width())))
        x, y = self._waveform_cache.display(
            view[0], view[1], self.sr, width)
        self.curve.setData(x, y, connect='finite')

    def _refresh_visible_view(self):
        self._render_visible_waveform()
        self._update_boundary_lod()
        self._layout_fields()
        self._update_phone_labels()
        self._redraw_join_overlays()

    def _sync_timeline_range(self, _viewbox=None, ranges=None):
        """Mirror numeric seconds, avoiding pyqtgraph's geometry X-link offset."""
        if ranges is None:
            ranges = self.plot.getViewBox().viewRange()[0]
        elif len(ranges) == 2 and isinstance(ranges[0], (list, tuple)):
            ranges = ranges[0]
        self.timeline.setXRange(float(ranges[0]), float(ranges[1]), padding=0)

    # -- data in -------------------------------------------------------------
    def set_synthesis(self, syn: fc.Synthesis):
        self.sr = syn.sr
        self.segments = copy.deepcopy(syn.segments)
        self._join_overlay_base_segments = copy.deepcopy(syn.segments)
        self.audio = np.asarray(syn.samples, np.float32)
        self.base_audio, self.base_durs = [], []
        self._dirty_phone_indices.clear()
        self._chunks = []        # current per-segment audio (stretch cache)
        for s in self.segments:
            a = int(round(s.start * self.sr)); b = int(round(s.end * self.sr))
            a = max(0, min(a, len(self.audio))); b = max(a, min(b, len(self.audio)))
            # Editing replaces these immutable source views; it never writes
            # into the rendered sentence buffer in place.
            self.base_audio.append(self.audio[a:b])
            self.base_durs.append(max(s.dur, 1e-4))
            self._chunks.append(self.audio[a:b])
        self._rebuild_boundaries()
        self._rebuild_fields()
        self._redraw()
        self.focused_idx = None
        self.clear_selection()
        self.set_fault_events(syn.fault_events)
        self.set_join_overlays(getattr(syn, "splice_records", ()))
        if getattr(self, "sel_region", None) is not None:
            self.sel_region.hide()
        dur = self.duration() or 1.0
        self._workspace_duration = dur
        self.playhead.setBounds((0.0, dur))
        self.timeline.set_duration(dur)
        self.set_playhead(0.0)
        self.plot.setXRange(0, dur, padding=0.02)   # reset view on new render

    @staticmethod
    def _join_record_time(record, sample_rate):
        for key in ("time", "splice_time_seconds"):
            try:
                value = float(record.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        try:
            sample = float(record.get("splice_sample"))
        except (TypeError, ValueError):
            return None
        value = sample / max(1.0, float(sample_rate))
        return value if math.isfinite(value) else None

    def set_join_overlays(self, records):
        self.clear_join_selection(redraw=False)
        self._join_overlay_records = [
            dict(record) for record in (records or ())
            if isinstance(record, dict)
        ]
        self._join_overlay_signature = None
        self._redraw_join_overlays()

    def set_join_overlay_visible(self, visible):
        self._join_overlay_visible = bool(visible)
        if not self._join_overlay_visible:
            self.clear_join_selection(redraw=False)
        self._join_overlay_signature = None
        self._redraw_join_overlays()

    def set_join_overlay_editable(self, editable):
        self._join_overlay_editable = bool(editable)
        if not self._join_overlay_editable:
            self.clear_join_selection(redraw=False)
        self._join_overlay_signature = None
        self._redraw_join_overlays()

    def set_requested_join_settings(self, settings):
        self._requested_join_settings = (
            fc.FestivalWSLBackend.normalize_join_settings(settings))
        self._join_overlay_signature = None
        self._redraw_join_overlays()

    def _map_rendered_time(self, when, preferred_segment=None):
        if not self._join_overlay_base_segments or not self.segments:
            return float(when)
        index = None
        try:
            candidate = int(preferred_segment)
        except (TypeError, ValueError):
            candidate = -1
        if (0 <= candidate < len(self._join_overlay_base_segments) and
                candidate < len(self.segments)):
            index = candidate
        else:
            for position, segment in enumerate(
                    self._join_overlay_base_segments):
                if segment.start <= float(when) <= segment.end:
                    index = position
                    break
        if index is None or index >= len(self.segments):
            return float(when)
        before = self._join_overlay_base_segments[index]
        after = self.segments[index]
        if before.dur <= 1e-9:
            return float(after.start)
        fraction = (float(when) - before.start) / before.dur
        fraction = max(0.0, min(1.0, fraction))
        return after.start + fraction * after.dur

    def _join_overlay_geometry(self, record):
        segment_index = record.get("segment_index")
        original_time = self._join_record_time(record, self.sr)
        if original_time is None:
            return None
        when = self._map_rendered_time(original_time, segment_index)
        start = end = when
        try:
            original_start = float(record.get("crossover_start"))
            original_end = float(record.get("crossover_end"))
        except (TypeError, ValueError):
            original_start = original_end = original_time
        if math.isfinite(original_start) and math.isfinite(original_end):
            start = self._map_rendered_time(
                original_start, segment_index)
            end = self._map_rendered_time(original_end, segment_index)
        return when, min(start, end), max(start, end)

    def _hide_join_editor(self):
        self.join_edit_region.hide()
        self.join_edit_label.hide()
        self.join_left_handle.hide()
        self.join_right_handle.hide()
        self._join_editor_unit_index = None

    def clear_join_selection(self, redraw=True):
        """Dismiss the join editor without changing any requested crossover."""
        changed = (
            self._selected_join_record is not None or
            self._join_editor_unit_index is not None
        )
        self._selected_join_record = None
        self._hide_join_editor()
        QtWidgets.QToolTip.hideText()
        if redraw and changed:
            self._redraw_join_overlays()

    def _join_requested_edges(self, record, when):
        try:
            unit_index = int(record.get("unit_index"))
        except (TypeError, ValueError):
            return None
        overrides = dict(
            self._requested_join_settings.get("crossover_overrides") or {})
        override = dict(overrides.get(str(unit_index)) or {})
        total_ms = float(self._requested_join_settings.get(
            "crossover_ms") or 0.0)
        left_ms = float(override.get("left_ms", total_ms * 0.5))
        right_ms = float(override.get("right_ms", total_ms * 0.5))
        try:
            segment_index = int(record.get("segment_index"))
        except (TypeError, ValueError):
            segment_index = -1
        if 0 <= segment_index < len(self.segments):
            lower = float(self.segments[segment_index].start)
            upper = float(self.segments[segment_index].end)
        else:
            lower, upper = 0.0, self.duration()
        try:
            context_cap_ms = float(
                record.get("crossover_context_cap_ms"))
        except (TypeError, ValueError):
            context_cap_ms = 100.0
        context_cap_ms = max(0.0, min(100.0, context_cap_ms))
        requested_total = left_ms + right_ms
        if requested_total > context_cap_ms and requested_total > 1.0e-9:
            scale = context_cap_ms / requested_total
            left_ms *= scale
            right_ms *= scale
        return (
            unit_index,
            max(lower, when - left_ms / 1000.0),
            min(upper, when + right_ms / 1000.0),
            lower,
            upper,
            context_cap_ms / 1000.0,
        )

    def _position_join_editor(self):
        if (not self._join_overlay_visible or
                not self._join_overlay_editable or
                self._selected_join_record is None):
            self._hide_join_editor()
            return
        if not (0 <= self._selected_join_record <
                len(self._join_overlay_records)):
            self._hide_join_editor()
            return
        record = self._join_overlay_records[self._selected_join_record]
        geometry = self._join_overlay_geometry(record)
        if geometry is None:
            self._hide_join_editor()
            return
        when, _rendered_start, _rendered_end = geometry
        requested = self._join_requested_edges(record, when)
        if requested is None:
            self._hide_join_editor()
            return
        unit_index, left, right, lower, upper, maximum_seconds = requested
        self._join_handle_guard = True
        try:
            self._join_editor_unit_index = unit_index
            self._join_editor_center = when
            self._join_editor_bounds = (lower, upper)
            self._join_editor_max_seconds = maximum_seconds
            try:
                self._join_editor_rendered_ms = float(
                    record.get("crossover_effective_ms") or 0.0)
            except (TypeError, ValueError):
                self._join_editor_rendered_ms = 0.0
            self.join_left_handle.setBounds((lower, when))
            self.join_right_handle.setBounds((when, upper))
            self.join_left_handle.setValue(left)
            self.join_right_handle.setValue(right)
            self.join_edit_region.setRegion((left, right))
            self.join_edit_region.show()
            self._update_join_edit_label()
            cap_text = "%.1f ms" % (maximum_seconds * 1000.0)
            for handle in (
                    self.join_left_handle, self.join_right_handle):
                handle.setToolTip(
                    "Drag this requested crossover edge; the selected "
                    "phone's acoustic cap is " + cap_text)
            self.join_left_handle.show()
            self.join_right_handle.show()
        finally:
            self._join_handle_guard = False

    def _join_marker_clicked(self, _item, points, _event):
        if not points:
            return
        try:
            selected = int(points[0].data())
        except (TypeError, ValueError):
            return
        if selected == self._selected_join_record:
            self.clear_join_selection()
            return
        self._selected_join_record = selected
        self._redraw_join_overlays()

    def _update_join_edit_label(self):
        center = self._join_editor_center
        requested_ms = max(
            0.0,
            (float(self.join_right_handle.value()) -
             float(self.join_left_handle.value())) * 1000.0)
        cap_ms = self._join_editor_max_seconds * 1000.0
        self.join_edit_label.setText(
            "requested %.1f ms | rendered %.1f ms | cap %.1f ms" %
            (requested_ms, self._join_editor_rendered_ms, cap_ms))
        self.join_edit_label.setPos(center, -0.82)
        self.join_edit_label.show()

    def _join_handle_changed(self, side):
        if self._join_handle_guard:
            return
        center = self._join_editor_center
        lower, upper = self._join_editor_bounds
        left = max(lower, min(center, float(
            self.join_left_handle.value())))
        right = max(center, min(upper, float(
            self.join_right_handle.value())))
        maximum_seconds = self._join_editor_max_seconds
        if right - left > maximum_seconds:
            if side == "left":
                left = max(lower, right - maximum_seconds)
            else:
                right = min(upper, left + maximum_seconds)
        self._join_handle_guard = True
        try:
            self.join_left_handle.setValue(left)
            self.join_right_handle.setValue(right)
            self.join_edit_region.setRegion((left, right))
            self._update_join_edit_label()
        finally:
            self._join_handle_guard = False

    def _join_handle_finished(self, _line):
        if self._join_handle_guard or self._join_editor_unit_index is None:
            return
        center = self._join_editor_center
        left_ms = max(
            0.0, (center - float(self.join_left_handle.value())) * 1000.0)
        right_ms = max(
            0.0, (float(self.join_right_handle.value()) - center) * 1000.0)
        self.joinCrossoverCommitted.emit(
            int(self._join_editor_unit_index),
            round(left_ms, 3),
            round(right_ms, 3),
        )

    def _redraw_join_overlays(self):
        if self._redrawing_join_overlays:
            return
        self._redrawing_join_overlays = True
        try:
            if (not self._join_overlay_visible or
                    not self._join_overlay_records):
                signature = ("hidden",)
                if signature != self._join_overlay_signature:
                    self.join_overlay_curve.setData([], [])
                    self.join_overlay_curve.hide()
                    self.join_marker_scatter.setData([])
                    self.join_marker_scatter.hide()
                    self._join_overlay_spans.setOpts(
                        x0=[], x1=[], y0=[], y1=[])
                    self._join_overlay_spans.hide()
                    self._join_overlay_signature = signature
                    self._join_overlay_display_count = 0
                self._hide_join_editor()
                return

            duration = self.duration()
            left, right = self.plot.getViewBox().viewRange()[0]
            span = max(1.0e-9, float(right) - float(left))
            width = max(
                32.0,
                float(self.plot.getViewBox().sceneBoundingRect().width()))
            visible = []
            for record_index, record in enumerate(
                    self._join_overlay_records):
                geometry = self._join_overlay_geometry(record)
                if geometry is None:
                    continue
                when, start, end = geometry
                if when < 0.0 or when > duration:
                    continue
                if end < left or start > right:
                    continue
                visible.append((record_index, when, start, end))

            detailed = len(visible) * JOIN_DETAIL_MIN_PX <= width
            self._join_overlay_lod_detailed = detailed
            if detailed:
                display = visible
            else:
                buckets = {}
                for row in visible:
                    pixel = (row[1] - left) / span * width
                    bucket = int(np.floor(
                        pixel / JOIN_OVERVIEW_BUCKET_PX))
                    buckets.setdefault(bucket, []).append(row)
                display = []
                for bucket in sorted(buckets):
                    group = buckets[bucket]
                    center = (
                        bucket + 0.5) * JOIN_OVERVIEW_BUCKET_PX
                    representative = next(
                        (row for row in group
                         if row[0] == self._selected_join_record),
                        min(
                            group,
                            key=lambda row: abs(
                                ((row[1] - left) / span * width) -
                                center),
                        ),
                    )
                    display.append((
                        representative[0],
                        representative[1],
                        min(row[2] for row in group),
                        max(row[3] for row in group),
                    ))

            pending = self._pending_action == "rerender"
            signature = (
                pending, detailed, self._selected_join_record,
                tuple(
                    (index, round(when, 7), round(start, 7), round(end, 7))
                    for index, when, start, end in display
                ),
            )
            if signature == self._join_overlay_signature:
                self._position_join_editor()
                return
            self._join_overlay_signature = signature
            self._join_overlay_display_count = len(display)

            times = [row[1] for row in display]
            spots = [{
                "pos": (when, 0.96),
                "data": record_index,
                "brush": pg.mkBrush(
                    '#2E8B6C' if record_index ==
                    self._selected_join_record else '#E3A43A'),
            } for record_index, when, _start, _end in display]
            spans = [
                (max(0.0, start), min(duration, end))
                for _index, _when, start, end in display
                if start < end and end >= 0.0 and start <= duration
            ]
            if times:
                x = np.repeat(np.asarray(times, np.float64), 3)
                y = np.tile(
                    np.asarray((-1.0, 1.0, np.nan), np.float64),
                    len(times))
                marker_color = '#7F7B73' if pending else '#A46B13'
                self.join_overlay_curve.setPen(
                    pg.mkPen(marker_color, width=1))
                self.join_overlay_curve.setData(
                    x, y, connect='finite')
                self.join_overlay_curve.show()
            else:
                self.join_overlay_curve.setData([], [])
                self.join_overlay_curve.hide()
            self.join_marker_scatter.setData(spots)
            self.join_marker_scatter.setVisible(bool(spots))
            if spans:
                self._join_overlay_spans.setOpts(
                    x0=[start for start, _end in spans],
                    x1=[end for _start, end in spans],
                    y0=[-1.0] * len(spans),
                    y1=[1.0] * len(spans),
                    brushes=[
                        pg.mkBrush(112, 112, 108, 30) if pending else
                        pg.mkBrush(48, 132, 107, 34)
                        for _span in spans
                    ],
                    pen=pg.mkPen(None),
                )
                self._join_overlay_spans.show()
            else:
                self._join_overlay_spans.setOpts(
                    x0=[], x1=[], y0=[], y1=[])
                self._join_overlay_spans.hide()
            self._position_join_editor()
        finally:
            self._redrawing_join_overlays = False

    def factors(self):
        """Current per-segment duration factor vs the rendered original."""
        return [max(1e-3, s.dur) / max(1e-3, b)
                for s, b in zip(self.segments, self.base_durs)]

    def set_factor(self, idx, factor):
        """Stretch segment idx to factor x its ORIGINAL duration (velocity
        bar / project restore). Total length changes; only this segment's
        audio is recomputed."""
        if not (0 <= idx < len(self.segments)):
            return
        factor = float(max(0.125, factor))
        delta = (self.base_durs[idx] * factor) - self.segments[idx].dur
        self.segments[idx].end += delta
        for j in range(idx + 1, len(self.segments)):
            self.segments[j].start += delta
            self.segments[j].end += delta
        self._rebuild_audio(only=idx)
        self._redraw()
        self.audioChanged.emit()

    def set_factors(self, changes: dict):
        """Apply several {idx: factor} at once (one audio rebuild)."""
        touched = []
        for idx, factor in sorted(changes.items()):
            if not (0 <= idx < len(self.segments)):
                continue
            factor = float(max(0.125, factor))
            self.segments[idx].end = (self.segments[idx].start
                                      + self.base_durs[idx] * factor)
            touched.append(idx)
        if not touched:
            return
        for idx in touched:
            self._chunks[idx] = self._stretch_one(idx)
        t = 0.0
        for seg, ch in zip(self.segments, self._chunks):
            seg.start = t
            seg.end = t + len(ch) / self.sr
            t = seg.end
        self.audio = fc.blend_junctions(self._chunks, self.sr)
        self._redraw()
        self.audioChanged.emit()

    def duration(self):
        return len(self.audio) / float(self.sr) if self.sr else 0.0

    def playhead_time(self):
        limit = max(self.duration(), self._workspace_duration)
        return max(0.0, min(limit, float(self.playhead.value())))

    def set_playhead(self, when):
        limit = max(self.duration(), self._workspace_duration)
        value = max(0.0, min(limit, float(when)))
        self.playhead.setValue(value)
        self.timeline.set_time(value)
        self.playheadChanged.emit(value)

    def set_workspace_duration(self, duration):
        self._workspace_duration = max(self.duration(), float(duration or 0.0))
        self.playhead.setBounds((0.0, max(.01, self._workspace_duration)))
        self.timeline.set_duration(self._workspace_duration)
        self.plot.setLimits(
            xMin=WAVEFORM_LEFT_LIMIT,
            xMax=max(self._workspace_duration * 1.3,
                     self._workspace_duration + 1.5))

    def set_pending_action(self, action="", reason=""):
        """Visually distinguish current audio from pending parameters."""
        action = str(action or "")
        if action not in ("rerender", "generate"):
            action = ""
        reason = str(reason or "") if action else ""
        timing_pending = (
            action == "rerender" and
            reason == "Phoneme timing changed")
        visual_key = (action, timing_pending)
        self._pending_action = action
        self._pending_reason = reason
        if (visual_key == self._pending_visual_key or
                self._setting_pending_visual):
            return
        curve_color = ("#667FAF" if timing_pending else
                       {"": "#1010C0", "rerender": "#777777",
                        "generate": "#865757"}[action])
        boundary_color = ("#D00000" if timing_pending else
                          {"": "#D00000", "rerender": "#8A8A8A",
                           "generate": "#9A5A52"}[action])
        hover_color = ("#FF3030" if timing_pending else
                       {"": "#FF3030", "rerender": "#AAAAAA",
                        "generate": "#C87868"}[action])
        self._setting_pending_visual = True
        try:
            self.curve.setPen(pg.mkPen(curve_color, width=1))
            self.boundary_overview.setPen(pg.mkPen(boundary_color, width=1))
            for line in self.boundaries:
                if line is None:
                    continue
                line.setPen(pg.mkPen(
                    boundary_color, width=1, style=Qt.DashLine))
                line.setHoverPen(pg.mkPen(
                    hover_color, width=2, style=Qt.DashLine))
            self._render_visible_waveform()
            self._redraw_join_overlays()
            self._pending_visual_key = visual_key
        finally:
            self._setting_pending_visual = False

    def phone_list(self):
        """Current phoneme labels, in order, as typed (fields may hold several
        space-separated phones, or be emptied to delete a phone)."""
        out = []
        for seg in self.segments:
            out.extend(p for p in seg.phone.replace(",", " ").split() if p)
        return out

    def selected_indices(self):
        if self.selected_range:
            first, last = self.selected_range
            if 0 <= first <= last < len(self.segments):
                return first, last
        if (self.focused_idx is not None and
                0 <= self.focused_idx < len(self.segments)):
            return self.focused_idx, self.focused_idx
        return None

    def structure_snapshot(self):
        return {
            "segments": copy.deepcopy(self.segments),
            # Structural operations replace arrays instead of mutating their
            # samples. Share those immutable buffers between undo snapshots.
            "chunks": [np.asarray(item, np.float32)
                       for item in self._chunks],
            "base_audio": [np.asarray(item, np.float32)
                           for item in self.base_audio],
            "base_durs": list(self.base_durs),
            "selected": self.selected_indices(),
            "fault_events": copy.deepcopy(self._fault_events),
        }

    def restore_structure(self, snapshot):
        snapshot = dict(snapshot or {})
        self.segments = copy.deepcopy(snapshot.get("segments") or [])
        self._chunks = [np.asarray(item, np.float32).copy()
                        for item in snapshot.get("chunks") or []]
        self.base_audio = [np.asarray(item, np.float32).copy()
                           for item in snapshot.get("base_audio") or []]
        self.base_durs = [float(value) for value in
                          snapshot.get("base_durs") or []]
        if not (len(self.segments) == len(self._chunks) ==
                len(self.base_audio) == len(self.base_durs)):
            return False
        self._reanchor_blend()
        self._rebuild_boundaries()
        self._rebuild_fields()
        self.set_fault_events(snapshot.get("fault_events") or [])
        selected = snapshot.get("selected")
        if selected:
            self._set_selected_range(*selected)
        else:
            self.focused_idx = None
            self.clear_selection()
            self._highlight_selection()
        self._redraw()
        self.audioChanged.emit()
        self.phonesEdited.emit()
        return True

    def copy_selection_payload(self):
        selected = self.selected_indices()
        if not selected:
            return None
        first, last = selected
        return {
            "kind": "festvox-phone-region",
            "sr": int(self.sr),
            "segments": copy.deepcopy(self.segments[first:last + 1]),
            "chunks": [np.asarray(item, np.float32).copy()
                       for item in self._chunks[first:last + 1]],
            "base_audio": [np.asarray(item, np.float32).copy()
                           for item in self.base_audio[first:last + 1]],
            "base_durs": list(self.base_durs[first:last + 1]),
        }

    @staticmethod
    def _resample_clip(samples, source_sr, target_sr):
        samples = np.asarray(samples, np.float32)
        if not samples.size or int(source_sr) == int(target_sr):
            return samples.copy()
        count = max(1, int(round(
            len(samples) * float(target_sr) / max(1, int(source_sr)))))
        return np.interp(
            np.linspace(0.0, 1.0, count, endpoint=False),
            np.linspace(0.0, 1.0, len(samples), endpoint=False),
            samples).astype(np.float32)

    def paste_region_payload(self, payload, at=None):
        payload = dict(payload or {})
        if payload.get("kind") != "festvox-phone-region":
            return False
        source_segments = [
            fc.Segment(
                segment.phone, segment.start, segment.end,
                timing_role=getattr(segment, "timing_role", ""))
            for segment in copy.deepcopy(payload.get("segments") or [])]
        source_chunks = list(payload.get("chunks") or [])
        source_base = list(payload.get("base_audio") or [])
        source_durs = list(payload.get("base_durs") or [])
        count = len(source_segments)
        if not count or not (count == len(source_chunks) ==
                             len(source_base) == len(source_durs)):
            return False
        source_sr = int(payload.get("sr") or self.sr)
        source_chunks = [self._resample_clip(
            item, source_sr, self.sr) for item in source_chunks]
        source_base = [self._resample_clip(
            item, source_sr, self.sr) for item in source_base]
        placeholder = (len(self.segments) == 1 and
                       not self.segments[0].phone.strip() and
                       not np.any(self._chunks[0]))
        if placeholder:
            for rows in (self.segments, self._chunks,
                         self.base_audio, self.base_durs):
                rows.clear()
            at = 0
        if at is None:
            selected = self.selected_indices()
            at = (self._paste_index if self._paste_index is not None else
                  selected[1] + 1 if selected else len(self.segments))
        at = max(0, min(len(self.segments), int(at)))
        self._paste_index = None
        self.segments[at:at] = source_segments
        self._chunks[at:at] = source_chunks
        self.base_audio[at:at] = source_base
        self.base_durs[at:at] = [max(1e-4, float(value))
                                  for value in source_durs]
        self._finish_structure_change(at, at + count - 1)
        return True

    def delete_selection(self):
        selected = self.selected_indices()
        if not selected:
            return False
        first, last = selected
        for rows in (self.segments, self._chunks,
                     self.base_audio, self.base_durs):
            del rows[first:last + 1]
        if not self.segments:
            duration = 0.06
            silence = np.zeros(max(1, int(round(duration * self.sr))),
                               np.float32)
            self.segments = [fc.Segment("", 0.0, duration)]
            self._chunks = [silence.copy()]
            self.base_audio = [silence.copy()]
            self.base_durs = [duration]
        focus = min(first, len(self.segments) - 1)
        self._finish_structure_change(focus, focus)
        self._paste_index = min(first, len(self.segments))
        return True

    def duplicate_selection(self):
        payload = self.copy_selection_payload()
        return bool(payload and self.paste_region_payload(payload))

    def move_selection(self, target):
        selected = self.selected_indices()
        if not selected:
            return False
        first, last = selected
        target = max(0, min(len(self.segments), int(target)))
        if first <= target <= last + 1:
            return False
        moving = list(range(first, last + 1))
        count = len(moving)
        insert_at = target - sum(index < target for index in moving)
        lists = (self.segments, self._chunks,
                 self.base_audio, self.base_durs)
        blocks = [rows[first:last + 1] for rows in lists]
        for rows in lists:
            del rows[first:last + 1]
        for rows, block in zip(lists, blocks):
            rows[insert_at:insert_at] = block
        self._finish_structure_change(insert_at, insert_at + count - 1)
        return True

    def _finish_structure_change(self, first, last):
        self._reanchor_blend()
        self._rebuild_boundaries()
        self._rebuild_fields()
        self.set_fault_events([])
        self._set_selected_range(first, last)
        self._redraw()
        self.audioChanged.emit()
        self.phonesEdited.emit()

    # -- boundaries ----------------------------------------------------------
    def _boundary_pens(self):
        timing_pending = (
            self._pending_action == "rerender" and
            getattr(self, "_pending_reason", "") ==
            "Phoneme timing changed")
        boundary_color = ("#D00000" if timing_pending else
                          {"": "#D00000", "rerender": "#8A8A8A",
                           "generate": "#9A5A52"}.get(
                               self._pending_action, "#D00000"))
        hover_color = ("#FF3030" if timing_pending else
                       {"": "#FF3030", "rerender": "#AAAAAA",
                        "generate": "#C87868"}.get(
                            self._pending_action, "#FF3030"))
        return (
            pg.mkPen(boundary_color, width=1, style=Qt.DashLine),
            pg.mkPen(hover_color, width=2, style=Qt.DashLine),
        )

    def _ensure_boundary(self, index):
        if not (0 <= index < len(self.boundaries)):
            return None
        line = self.boundaries[index]
        if line is not None:
            return line
        pen, hoverpen = self._boundary_pens()
        line = pg.InfiniteLine(
            pos=self.segments[index].end, angle=90, movable=True,
            pen=pen, hoverPen=hoverpen)
        line.seg_index = index
        line.setToolTip(
            "Drag to resize this phone and move everything after it.\n"
            "Shift-drag to move only this boundary, keeping the next "
            "phone's far edge fixed.")
        line.sigDragged.connect(self._on_drag)
        line.sigPositionChangeFinished.connect(self._on_drag_finish)
        self.plot.addItem(line)
        self.boundaries[index] = line
        return line

    def _release_boundary(self, index):
        if not (0 <= index < len(self.boundaries)):
            return
        line = self.boundaries[index]
        if line is None:
            return
        self.plot.removeItem(line)
        self.boundaries[index] = None
        try:
            line.deleteLater()
        except AttributeError:
            pass

    def _rebuild_boundaries(self):
        for ln in self.boundaries:
            if ln is not None:
                self.plot.removeItem(ln)
                try:
                    ln.deleteLater()
                except AttributeError:
                    pass
        self.boundaries = [None] * max(0, len(self.segments) - 1)
        self._visible_boundary_indices = set()
        self._boundary_overview_signature = None
        self.boundary_overview.setData([], [])

    def _selection_spans_multiple_phones(self):
        """Only group selections alter boundary-drag timing behavior."""
        return bool(self.selected_range and
                    self.selected_range[0] < self.selected_range[1])

    def _on_drag(self, line):
        """RIPPLE drag: the boundary changes THIS phoneme's length; every
        later phoneme keeps its duration and slides along (no right-hand
        clamp -- phonemes can be lengthened arbitrarily)."""
        i = line.seg_index
        if self._drag_factors_before is None:
            self._drag_factors_before = self.factors()
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if self._selection_spans_multiple_phones():
            first, last = self.selected_range
            if i == last:
                line.drag_mode = "selection-right"
                self._stretch_selection(float(line.value()), from_left=False)
                return
            if (modifiers & Qt.ShiftModifier) and i == first - 1:
                line.drag_mode = "selection-left"
                self._stretch_selection(float(line.value()), from_left=True)
                return
        if modifiers & Qt.ShiftModifier:
            line.drag_mode = "fixed"
            low = self.segments[i].start + MIN_SEG
            high = (self.segments[i + 1].end - MIN_SEG
                    if i + 1 < len(self.segments) else low)
            x = max(low, min(high, float(line.value())))
            line.setValue(x)
            self.segments[i].end = x
            if i + 1 < len(self.segments):
                self.segments[i + 1].start = x
            return
        line.drag_mode = "ripple"
        left = self.segments[i].start
        x = float(max(left + MIN_SEG, line.value()))
        if x != line.value():
            line.setValue(x)
        delta = x - self.segments[i].end
        if abs(delta) < 1e-9:
            return
        self.segments[i].end = x
        for j in range(i + 1, len(self.segments)):
            self.segments[j].start += delta
            self.segments[j].end += delta
        for j in self._visible_boundary_indices:
            if j > i:
                boundary = self.boundaries[j]
                if boundary is not None:
                    boundary.setValue(self.segments[j].end)

    def _on_drag_finish(self, line):
        self._on_drag(line)
        i = line.seg_index
        mode = getattr(line, "drag_mode", "ripple")
        if (self._selection_spans_multiple_phones() and
                mode.startswith("selection")):
            first, last = self.selected_range
            affected = list(range(first, last + 1))
            if mode == "selection-left" and first > 0:
                affected.insert(0, first - 1)
            for index in affected:
                self._chunks[index] = self._stretch_one(index)
            self._reanchor_from_chunks()
        elif mode == "fixed":
            for index in (i, i + 1):
                if 0 <= index < len(self.segments):
                    self._chunks[index] = self._stretch_one(index)
            self._reanchor_from_chunks()
        else:
            self._rebuild_audio(only=i)   # later chunks just shift
        self._boundary_drag_snapshot = None
        line.drag_mode = ""
        self._redraw()
        self.audioChanged.emit()
        before = self._drag_factors_before
        self._drag_factors_before = None
        if before is not None and before != self.factors():
            self.timingEditCommitted.emit(before, self.factors())

    # -- audio rebuild via time-stretch (WSOLA) -------------------------------
    def _stretch_one(self, i):
        seg = self.segments[i]
        factor = max(seg.dur, 1e-4) / max(self.base_durs[i], 1e-4)
        sustain = None
        timing_nucleus = fc.is_timing_nucleus(
            seg.phone, getattr(seg, "timing_role", "")
        )
        if (self.use_sustain and fc.is_vowel_phone(seg.phone) and
                callable(self.sustain_hook) and factor > 3.0):
            try:
                loaded = self.sustain_hook(seg.phone)
                if loaded:
                    sustain, sustain_sr = loaded
                    if int(sustain_sr) != int(self.sr):
                        count = max(1, int(round(
                            len(sustain) * self.sr / float(sustain_sr))))
                        sustain = np.interp(
                            np.linspace(0, 1, count, endpoint=False),
                            np.linspace(0, 1, len(sustain), endpoint=False),
                            sustain).astype(np.float32)
            except Exception:
                sustain = None
        if self.stretch_hook is not None:
            y = fc.time_stretch(self.base_audio[i], self.sr, factor,
                                hook=self.stretch_hook)
        elif timing_nucleus:
            y = fc.stretch_segment(
                self.base_audio[i], self.sr, factor, sustain=sustain,
                use_sustain=self.use_sustain)
        else:
            y = fc.time_stretch(self.base_audio[i], self.sr, factor)
        n = max(1, int(round(seg.dur * self.sr)))
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        else:
            y = y[:n]
        return y.astype(np.float32)

    def _reanchor_from_chunks(self):
        t = 0.0
        for seg, chunk in zip(self.segments, self._chunks):
            seg.start = t
            seg.end = t + len(chunk) / float(self.sr)
            t = seg.end
        self.audio = fc.blend_junctions(self._chunks, self.sr)

    def _stretch_selection(self, boundary, from_left=False):
        if not self._selection_spans_multiple_phones():
            return
        first, last = self.selected_range
        if self._boundary_drag_snapshot is None:
            self._boundary_drag_snapshot = {
                "durs": [segment.dur for segment in self.segments],
                "starts": [segment.start for segment in self.segments],
                "ends": [segment.end for segment in self.segments],
                "left": self.segments[first].start,
                "right": self.segments[last].end,
            }
        snap = self._boundary_drag_snapshot
        anchor = snap["right"] if from_left else snap["left"]
        indices = list(range(first, last + 1))
        vowels = [index for index in indices
                  if fc.is_timing_nucleus(
                      self.segments[index].phone,
                      getattr(self.segments[index], "timing_role", ""))]
        adjustable = vowels
        if not adjustable and all(
                self.segments[index].phone == "pau" for index in indices):
            adjustable = indices
        if not adjustable:
            return
        fixed = sum(snap["durs"][index] for index in indices
                    if index not in adjustable)
        minimum_selection = fixed + MIN_SEG * len(adjustable)
        if from_left:
            low = (self.segments[first - 1].start + MIN_SEG
                   if first > 0 else 0.0)
            high = snap["right"] - minimum_selection
            boundary = max(low, min(high, float(boundary)))
        else:
            boundary = max(snap["left"] + minimum_selection,
                           float(boundary))
        total = max(MIN_SEG, (anchor - boundary if from_left
                              else boundary - anchor))
        adjustable_old = sum(snap["durs"][index] for index in adjustable)
        adjustable_total = max(MIN_SEG * len(adjustable), total - fixed)
        scale = adjustable_total / max(MIN_SEG, adjustable_old)
        durs = list(snap["durs"])
        for index in adjustable:
            durs[index] = max(MIN_SEG, snap["durs"][index] * scale)
        if from_left:
            cursor = snap["right"]
            for index in range(last, first - 1, -1):
                self.segments[index].end = cursor
                self.segments[index].start = cursor - durs[index]
                cursor = self.segments[index].start
            if first > 0:
                self.segments[first - 1].end = cursor
        else:
            cursor = snap["left"]
            for index in indices:
                self.segments[index].start = cursor
                self.segments[index].end = cursor + durs[index]
                cursor = self.segments[index].end
            shift = cursor - snap["right"]
            for index in range(last + 1, len(self.segments)):
                self.segments[index].start = snap["starts"][index] + shift
                self.segments[index].end = snap["ends"][index] + shift
        for index in self._visible_boundary_indices:
            if 0 <= index < len(self.boundaries):
                boundary = self.boundaries[index]
                if boundary is not None:
                    boundary.setValue(self.segments[index].end)

    def _rebuild_audio(self, only=None):
        """Recompute stretched audio. `only=idx` re-stretches just that
        segment (velocity drag); None re-stretches everything (boundary
        drags change two neighbours, so both get fresh chunks)."""
        if not self.segments:
            self.audio = np.zeros(1, np.float32)
            return
        if len(getattr(self, "_chunks", [])) != len(self.segments):
            self._chunks = [None] * len(self.segments)
            only = None
        if only is None:
            for i in range(len(self.segments)):
                self._chunks[i] = self._stretch_one(i)
        else:
            self._chunks[only] = self._stretch_one(only)
        # re-anchor the timeline and join with de-click smoothing
        t = 0.0
        for seg, ch in zip(self.segments, self._chunks):
            seg.start = t
            seg.end = t + len(ch) / self.sr
            t = seg.end
        self.audio = fc.blend_junctions(self._chunks, self.sr)

    # -- phoneme fields (zoom-following, aligned with the waveform) -----------
    def _create_field(self, index):
        if not (0 <= index < len(self.fields)):
            return None
        field = self.fields[index]
        if field is not None:
            return field
        seg = self.segments[index]
        field = QtWidgets.QLineEdit(seg.phone, self.fields_host)
        field.setObjectName("phon")
        field.setAlignment(Qt.AlignCenter)
        field.seg_index = index
        field.setProperty("selected", index == self.focused_idx)
        field.setProperty("dirty", index in self._dirty_phone_indices)
        if seg.phone == "pau":
            field.setReadOnly(True)
            field.setToolTip(
                "pause / silence -- click to select, then "
                "- Phone (or right-click) removes it")
        else:
            field.setToolTip(
                "type a phone from the bank (e.g. r -> rr), "
                "space-separate to insert, clear to delete;\n"
                "right-click to insert/delete phones;\n"
                "Enter or the Re-render button applies it")
            field.textEdited.connect(
                lambda txt, idx=index: self._on_phone_edit(idx, txt))
            field.returnPressed.connect(self.rerenderRequested.emit)
        field.setContextMenuPolicy(Qt.CustomContextMenu)
        field.customContextMenuRequested.connect(
            lambda _pos, idx=index: self._field_menu(idx))
        field.installEventFilter(self)
        field.hide()
        self.fields[index] = field
        return field

    def _release_field(self, index):
        if not (0 <= index < len(self.fields)):
            return
        field = self.fields[index]
        if field is None:
            return
        field.hide()
        field.deleteLater()
        self.fields[index] = None

    def _rebuild_fields(self):
        for f in self.fields:
            if f is not None:
                f.deleteLater()
        self.fields = [None] * len(self.segments)
        self._visible_field_indices = set()
        self._styled_focus_idx = None
        self._layout_fields()

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.FocusIn:
            idx = getattr(obj, "seg_index", None)
            if idx is not None:
                self.set_selected(int(idx))
        return super().eventFilter(obj, ev)

    def _visible_segment_range(self):
        if not self.segments:
            return range(0)
        left, right = self.plot.getViewBox().viewRange()[0]
        lo, hi = 0, len(self.segments)
        while lo < hi:
            middle = (lo + hi) // 2
            if self.segments[middle].end < left:
                lo = middle + 1
            else:
                hi = middle
        first = lo
        lo, hi = first, len(self.segments)
        while lo < hi:
            middle = (lo + hi) // 2
            if self.segments[middle].start <= right:
                lo = middle + 1
            else:
                hi = middle
        return range(first, lo)

    def _layout_fields(self):
        """Place each phoneme box exactly under its segment, in the current
        view (pan/zoom aware). Boxes squeezed below ~14 px are hidden."""
        if not self.fields:
            return
        vb = self.plot.getViewBox()
        view_left, view_right = vb.viewRange()[0]
        view_span = max(1e-9, view_right - view_left)
        w_widget = self.fields_host.width() or self.plot.width()
        try:
            left_px = self.plot.mapFromScene(vb.mapViewToScene(
                QtCore.QPointF(view_left, 0.0))).x()
            right_px = self.plot.mapFromScene(vb.mapViewToScene(
                QtCore.QPointF(view_right, 0.0))).x()
        except Exception:
            left_px, right_px = 0.0, float(w_widget)
        pixels_per_second = (right_px - left_px) / view_span
        shown = set()
        geometry = {}
        for index in self._visible_segment_range():
            if not (0 <= index < len(self.fields)):
                continue
            seg = self.segments[index]
            sx = left_px + (seg.start - view_left) * pixels_per_second
            ex = left_px + (seg.end - view_left) * pixels_per_second
            sx, ex = min(sx, ex), max(sx, ex)
            sx = max(-2.0, sx)
            ex = min(float(w_widget) + 2.0, ex)
            if ex - sx >= PHONE_FIELD_MIN_PX:
                shown.add(index)
                geometry[index] = (sx, ex)
        for index in self._visible_field_indices - shown:
            self._release_field(index)
        for index in shown:
            f = self._create_field(index)
            if f is None:
                continue
            sx, ex = geometry[index]
            f.setGeometry(int(sx) + 1, 2, int(ex - sx) - 2, 25)
            f.show()
        self._visible_field_indices = shown

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._schedule_visible_refresh()

    # -- add / delete phones ---------------------------------------------------
    def _field_menu(self, idx):
        if not (0 <= idx < len(self.segments)):
            return
        self.set_selected(idx)
        m = QtWidgets.QMenu()
        m.addAction("Insert blank phone before",
                    lambda: self._insert_phone(idx, True))
        m.addAction("Insert blank phone after",
                    lambda: self._insert_phone(idx, False))
        if len(self.segments) > 1:
            m.addAction("Delete this phone",
                        lambda: self._delete_phone(idx))
        if callable(self.variant_menu_hook):
            self.variant_menu_hook(m, idx)
        m.exec_(QtGui.QCursor.pos())

    def _insert_phone(self, idx, before=False):
        """Insert a BLANK phone field in a nominal 60 ms slot next to segment
        `idx` (silent placeholder until Re-render). The new field is selected
        and focused so a phone can be typed straight in."""
        sr = self.sr or 22050
        d = 0.06
        sil = np.zeros(max(1, int(d * sr)), np.float32)
        if not self.segments:
            at = 0
        else:
            idx = max(0, min(idx, len(self.segments) - 1))
            at = idx if before else idx + 1
        self.segments.insert(at, fc.Segment("", 0.0, d))
        self._chunks.insert(at, sil.copy())
        self.base_audio.insert(at, sil.copy())
        self.base_durs.insert(at, d)
        self._reanchor_blend()
        self._rebuild_boundaries()
        self._rebuild_fields()
        self._redraw()
        self.set_selected(at)
        if at < len(self.fields):
            field = self._create_field(at)
            if field is not None:
                field.setFocus()
        self.audioChanged.emit()
        self.phonesEdited.emit()

    def _delete_phone(self, idx):
        """Remove the phone field at `idx` entirely -- pau included."""
        if not (0 <= idx < len(self.segments)) or len(self.segments) <= 1:
            return
        for lst in (self.segments, self._chunks, self.base_audio,
                    self.base_durs):
            if idx < len(lst):
                del lst[idx]
        self._reanchor_blend()
        self._rebuild_boundaries()
        self._rebuild_fields()
        self._redraw()
        self.set_selected(min(idx, len(self.segments) - 1))
        self.audioChanged.emit()
        self.phonesEdited.emit()

    def _reanchor_blend(self):
        """Re-lay the timeline from the current chunk lengths and re-blend the
        preview audio (no re-stretch), after an insert/delete."""
        t = 0.0
        for seg, ch in zip(self.segments, self._chunks):
            seg.start = t
            seg.end = t + len(ch) / self.sr
            t = seg.end
        self.audio = fc.blend_junctions(self._chunks, self.sr)

    def set_selected(self, idx):
        """Select a phone / timeline position (click or the +/- buttons)."""
        self.clear_join_selection()
        self._paste_index = None
        self.clear_selection(notify=False)
        self.focused_idx = (idx if (idx is not None and self.segments
                            and 0 <= idx < len(self.segments)) else None)
        self._highlight_selection()
        self._notify_selection()

    def _set_selected_range(self, first, last):
        self.clear_join_selection()
        if not self.segments:
            self.clear_selection()
            return
        first = max(0, min(len(self.segments) - 1, int(first)))
        last = max(first, min(len(self.segments) - 1, int(last)))
        self._paste_index = None
        self.selected_range = (first, last)
        self.focused_idx = None
        self._highlight_selection()
        if getattr(self, "multi_region", None) is None:
            self.multi_region = pg.LinearRegionItem(
                movable=False, brush=pg.mkBrush(64, 132, 210, 58),
                pen=pg.mkPen('#275E9E', width=2))
            self.multi_region.setZValue(-4)
            self.plot.addItem(self.multi_region)
        self.multi_region.setRegion((self.segments[first].start,
                                     self.segments[last].end))
        self.multi_region.show()
        self._notify_selection()

    def _on_selection_drag(self, start, end, finished, shift=False):
        self.clear_join_selection()
        if not self.segments:
            return
        a, b = sorted((float(start), float(end)))
        if finished and abs(b - a) < 0.004:
            clicked = next((index for index, segment in enumerate(self.segments)
                            if segment.start <= a < segment.end), None)
            if clicked is None and a == self.segments[-1].end:
                clicked = len(self.segments) - 1
            touched = [] if clicked is None else [clicked]
        else:
            touched = [index for index, segment in enumerate(self.segments)
                       if segment.end >= a and segment.start <= b]
        if not touched:
            self.clear_selection()
            return
        if shift and self.selected_range:
            touched.extend(self.selected_range)
        self._set_selected_range(min(touched), max(touched))

    def _notify_selection(self):
        selected = self.selected_indices()
        value = tuple(selected) if selected is not None else None
        if value != self._last_selection_signal:
            self._last_selection_signal = value
            self.selectionChanged.emit(value)

    def clear_selection(self, notify=True):
        self.selected_range = None
        self._boundary_drag_snapshot = None
        if getattr(self, "multi_region", None) is not None:
            self.multi_region.hide()
        if notify:
            self._notify_selection()

    def _on_selection_move_drag(self, start, end, finished):
        selected = self.selected_indices()
        if self._selection_move_state is None:
            if not selected:
                self._on_selection_drag(start, end, finished, True)
                return
            first, last = selected
            block_start = self.segments[first].start
            block_end = self.segments[last].end
            if not block_start <= float(start) <= block_end:
                self._on_selection_drag(start, end, finished, True)
                return
            self._set_selected_range(first, last)
            self._selection_move_state = {
                "first": first, "last": last,
                "grab": float(start) - block_start,
                "duration": block_end - block_start,
                "before": self.structure_snapshot(),
                "target": first,
            }
            if getattr(self, "move_preview_region", None) is None:
                self.move_preview_region = pg.LinearRegionItem(
                    movable=False, brush=pg.mkBrush(34, 104, 214, 92),
                    pen=pg.mkPen('#1E6FE0', width=2))
                self.move_preview_region.setZValue(-2)
                self.plot.addItem(self.move_preview_region)
            if getattr(self, "move_drop_line", None) is None:
                self.move_drop_line = pg.InfiniteLine(
                    pos=0.0, angle=90, movable=False,
                    pen=pg.mkPen('#1E6FE0', width=3))
                self.move_drop_line.setZValue(24)
                self.plot.addItem(self.move_drop_line)
        state = self._selection_move_state
        pointer = float(end)
        target = next((index for index, segment in enumerate(self.segments)
                       if pointer < (segment.start + segment.end) * 0.5),
                      len(self.segments))
        state["target"] = target
        drop_time = (self.segments[target].start
                     if target < len(self.segments) else
                     self.segments[-1].end)
        preview_start = max(0.0, pointer - state["grab"])
        self.move_preview_region.setRegion(
            (preview_start, preview_start + state["duration"]))
        self.move_preview_region.show()
        self.move_drop_line.setValue(drop_time)
        self.move_drop_line.show()
        if not finished:
            return
        before = state["before"]
        target = state["target"]
        self._selection_move_state = None
        self.move_preview_region.hide()
        self.move_drop_line.hide()
        if self.move_selection(target):
            self.structureEditCommitted.emit(
                before, self.structure_snapshot(), "move phoneme region")

    def _selection_menu(self, when, screen_pos):
        self.clear_join_selection()
        if not self.selected_range:
            self._on_selection_drag(when, when, True)
        if not self.selected_range:
            return
        first, last = self.selected_range
        if not (self.segments[first].start <= when <=
                self.segments[last].end):
            self.clear_selection()
            return
        menu = QtWidgets.QMenu()
        menu.addAction(
            "Cut selection to new sentence",
            lambda: self.regionCutRequested.emit(first, last))
        if self._fault_mode_active:
            menu.addSeparator()
            heard = [dict(event) for event in self._fault_events
                     if event.get("kind") == "pitch_glitch" and
                     first <= int(event.get("segment", -1)) <= last and
                     event.get("broken_hz") is not None]
            label = ("Pin selected broken pitch fault"
                     if len(heard) == 1 else
                     "Pin %d selected broken pitch faults" % len(heard))
            pin_action = menu.addAction(
                label if heard else "Select a highlighted pitch fault to pin",
                lambda rows=heard:
                self.faultTargetRequested.emit(rows))
            pin_action.setEnabled(bool(heard))
            menu.addAction(
                "Use random broken pitch locations",
                lambda: self.faultTargetRequested.emit(None))
        menu.addSeparator()
        menu.addAction("Clear selection", self.clear_selection)
        point = screen_pos.toPoint() if hasattr(screen_pos, "toPoint") \
            else QtGui.QCursor.pos()
        menu.exec_(point)

    def set_fault_events(self, events):
        self._fault_events = [dict(event) for event in (events or [])]
        self._update_fault_regions()

    def set_fault_mode_active(self, active):
        self._fault_mode_active = bool(active)

    def _update_fault_regions(self):
        for item in self._fault_regions:
            self.plot.removeItem(item)
        self._fault_regions = []
        for event in getattr(self, "_fault_events", []):
            if event.get("kind") != "pitch_glitch":
                continue
            try:
                segment = self.segments[int(event.get("segment"))]
            except (IndexError, TypeError, ValueError):
                continue
            color = ((190, 62, 155, 72) if event.get("pinned")
                     else (230, 142, 38, 72))
            region = pg.LinearRegionItem(
                values=(segment.start, segment.end), movable=False,
                brush=pg.mkBrush(*color), pen=pg.mkPen(color[:3], width=2))
            region.setZValue(-3)
            self.plot.addItem(region)
            self._fault_regions.append(region)

    def _sync_scrollbar_from_view(self):
        if self._scroll_sync or not hasattr(self, "hscroll"):
            return
        view = self.plot.getViewBox().viewRange()[0]
        width = max(0.001, view[1] - view[0])
        duration = self.duration() or 1.0
        workspace = max(duration * 1.6, duration + 1.5)
        maximum = max(0.0, workspace - width)
        self._scroll_sync = True
        try:
            self.hscroll.setRange(0, int(round(maximum * 1000)))
            self.hscroll.setPageStep(max(1, int(round(width * 1000))))
            self.hscroll.setValue(int(round(max(0.0, view[0]) * 1000)))
            self.hscroll.setEnabled(maximum > 0.001)
        finally:
            self._scroll_sync = False

    def _scroll_to(self, value):
        if self._scroll_sync:
            return
        view = self.plot.getViewBox().viewRange()[0]
        width = max(0.001, view[1] - view[0])
        start = float(value) / 1000.0
        self.plot.setXRange(start, start + width, padding=0)

    def _highlight_selection(self):
        changed = {self._styled_focus_idx, self.focused_idx}
        for i in changed:
            if i is None or not (0 <= i < len(self.fields)):
                continue
            f = self.fields[i]
            if f is None:
                continue
            f.setProperty("selected", i == self.focused_idx)
            f.style().unpolish(f)
            f.style().polish(f)
        self._styled_focus_idx = self.focused_idx
        if getattr(self, "sel_region", None) is None:
            self.sel_region = pg.LinearRegionItem(
                movable=False, brush=pg.mkBrush(60, 130, 230, 45))
            self.sel_region.setZValue(-5)
            self.plot.addItem(self.sel_region)
        if self.focused_idx is not None and self.focused_idx < len(self.segments):
            s = self.segments[self.focused_idx]
            self.sel_region.setRegion((s.start, s.end))
            self.sel_region.show()
        else:
            self.sel_region.hide()

    def _mark_dirty(self, idx):
        self._dirty_phone_indices.add(int(idx))
        if idx < len(self.fields):
            w = self.fields[idx]
            if w is not None:
                w.setProperty("dirty", True)
                w.style().unpolish(w); w.style().polish(w)
        self.phonesEdited.emit()

    def _on_phone_edit(self, idx, txt):
        if 0 <= idx < len(self.segments):
            self.segments[idx].phone = txt
            self._mark_dirty(idx)

    # -- draw -----------------------------------------------------------------
    def _redraw(self):
        self._ensure_waveform_cache()
        dur = self.duration() or 1.0
        # keep the user's zoom/pan; just extend the workspace limits so
        # there is room to lengthen phonemes and pan past the end
        self.plot.setLimits(xMin=WAVEFORM_LEFT_LIMIT,
                            xMax=max(dur * 1.6, dur + 1.5),
                            yMin=-1.3, yMax=1.3)
        self.playhead.setBounds((0.0, dur))
        self.timeline.set_duration(max(dur, self._workspace_duration))
        self.plot.enableAutoRange(axis='y', enable=False)
        self.plot.setYRange(-1.05, 1.05, padding=0)
        self._update_boundary_lod(force=True)
        # Zoom-aware phone tags are virtualized to the visible range.
        self._update_phone_labels()
        self._layout_fields()
        self._render_visible_waveform()
        if self.selected_range and getattr(self, "multi_region", None):
            first, last = self.selected_range
            if last < len(self.segments):
                self.multi_region.setRegion((self.segments[first].start,
                                             self.segments[last].end))
        self._update_fault_regions()
        self._redraw_join_overlays()
        self._sync_scrollbar_from_view()

    def _update_phone_labels(self):
        left, right = self.plot.getViewBox().viewRange()[0]
        span = max(1e-9, right - left)
        width = max(32.0, self.plot.getViewBox().sceneBoundingRect().width())
        pixels_per_second = width / span
        wanted = []
        for index in self._visible_segment_range():
            seg = self.segments[index]
            if (seg.phone != "pau" and
                    seg.dur * pixels_per_second >= PHONE_LABEL_MIN_PX):
                wanted.append((index, seg.phone,
                               round((seg.start + seg.end) * 0.5, 7)))
        signature = tuple(wanted)
        if signature == self._phone_label_signature:
            return
        self._phone_label_signature = signature
        for it in self.phone_labels:
            self.plot.removeItem(it)
        self.phone_labels = []
        for index, _phone, _middle in wanted:
            seg = self.segments[index]
            ti = pg.TextItem(seg.phone, color='#404040', anchor=(0.5, 0.0))
            ti.setPos((seg.start + seg.end) / 2.0, 1.02)
            self.plot.addItem(ti)
            self.phone_labels.append(ti)

    def _update_boundary_lod(self, force=False):
        if not self.boundaries or not self.segments:
            self.boundary_overview.setData([], [])
            self._boundary_overview_signature = ()
            return
        left, right = self.plot.getViewBox().viewRange()[0]
        span = max(1e-9, right - left)
        width = max(32.0, self.plot.getViewBox().sceneBoundingRect().width())
        visible_segments = self._visible_segment_range()
        first = max(0, visible_segments.start - 1)
        stop = min(len(self.boundaries), visible_segments.stop)
        indices = list(range(first, stop))
        detailed = (len(indices) * BOUNDARY_DETAIL_MIN_PX <= width)
        self._boundary_lod_detailed = detailed
        if detailed:
            visible = set(indices)
            for index in self._visible_boundary_indices - visible:
                self._release_boundary(index)
            for index in visible:
                boundary = self._ensure_boundary(index)
                if boundary is None:
                    continue
                if force or index not in self._visible_boundary_indices:
                    boundary.setValue(self.segments[index].end)
                boundary.show()
            self._visible_boundary_indices = visible
            if self._boundary_overview_signature != ("detail",):
                self.boundary_overview.setData([], [])
                self._boundary_overview_signature = ("detail",)
            return

        for index in self._visible_boundary_indices:
            self._release_boundary(index)
        self._visible_boundary_indices = set()
        buckets = {}
        for index in indices:
            when = self.segments[index].end
            pixel = (when - left) / span * width
            bucket = int(np.floor(pixel / BOUNDARY_OVERVIEW_BUCKET_PX))
            left_pause = self.segments[index].phone == "pau"
            right_pause = (index + 1 < len(self.segments) and
                           self.segments[index + 1].phone == "pau")
            priority = 2 if left_pause and right_pause else \
                1 if left_pause or right_pause else 0
            center = (bucket + 0.5) * BOUNDARY_OVERVIEW_BUCKET_PX
            candidate = (priority, -abs(pixel - center), index)
            if bucket not in buckets or candidate > buckets[bucket]:
                buckets[bucket] = candidate
        chosen = sorted(candidate[2] for candidate in buckets.values())
        signature = tuple((index, round(self.segments[index].end, 7))
                          for index in chosen)
        if not force and signature == self._boundary_overview_signature:
            return
        self._boundary_overview_signature = signature
        times = np.asarray([self.segments[index].end for index in chosen],
                           np.float64)
        x = np.repeat(times, 3)
        y = np.empty(len(x), np.float32)
        for position, index in enumerate(chosen):
            left_pause = self.segments[index].phone == "pau"
            right_pause = (index + 1 < len(self.segments) and
                           self.segments[index + 1].phone == "pau")
            start = position * 3
            y[start] = -1.04
            y[start + 1] = (-0.62 if left_pause and right_pause else
                            -0.80 if left_pause or right_pause else -0.94)
            y[start + 2] = np.nan
        self.boundary_overview.setData(x, y, connect='finite')

    def get_audio(self):
        return self.audio, self.sr


class MiniWaveform(QtWidgets.QWidget):
    """Small, non-interactive waveform drawn from a phrase's real preview."""
    def __init__(self, preview=None, pending="", parent=None):
        super().__init__(parent)
        self.setFixedHeight(42)
        self.setMinimumWidth(100)
        raw = preview[0] if preview else []
        self.samples = np.asarray(raw, np.float32)
        if self.samples.size and not np.all(np.isfinite(self.samples)):
            self.samples = np.nan_to_num(
                self.samples, copy=True, nan=0.0, posinf=0.0,
                neginf=0.0)
        self.pending = str(pending or "")

    def set_pending(self, pending=""):
        pending = str(pending or "")
        if self.pending == pending:
            return
        self.pending = pending
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        background = {"rerender": "#DEDEDC", "generate": "#E1E1DF"}.get(
            self.pending, "#DDE7E1")
        center_color = {"rerender": "#B2B2B0", "generate": "#A9AAA8"}.get(
            self.pending, "#A5B7AD")
        wave_color = {"rerender": "#777777", "generate": "#69717A"}.get(
            self.pending, "#556F84")
        painter.fillRect(self.rect(), QtGui.QColor(background))
        painter.setPen(QtGui.QPen(QtGui.QColor(center_color), 1))
        center = self.height() // 2
        painter.drawLine(0, center, self.width(), center)
        if self.samples.size > 1 and self.width() > 2:
            count = max(2, self.width() - 4)
            edges = np.linspace(0, self.samples.size, count + 1,
                                dtype=np.int64)
            peak = max(1e-6, float(np.max(np.abs(self.samples))))
            scale = (self.height() * .42) / peak
            painter.setPen(QtGui.QPen(QtGui.QColor(wave_color), 1))
            for column in range(count):
                chunk = self.samples[edges[column]:edges[column + 1]]
                if chunk.size:
                    low = int(round(float(np.min(chunk)) * scale))
                    high = int(round(float(np.max(chunk)) * scale))
                    x = column + 2
                    painter.drawLine(x, center - high, x, center - low)
        painter.end()


class SpeakerBadge(QtWidgets.QWidget):
    """Compact bordered portrait with deterministic initials fallback."""
    clicked = QtCore.pyqtSignal(object)

    def __init__(self, name="", path="", size=42, parent=None):
        super().__init__(parent)
        self.name = str(name or "Voice")
        self.pixmap = QtGui.QPixmap(str(path or ""))
        self.setFixedSize(int(size), int(size))
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Choose speaker")

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        box = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QtGui.QPen(QtGui.QColor("#7B746A"), 1))
        painter.setBrush(QtGui.QColor("#EEEBDD"))
        painter.drawRoundedRect(box, 3, 3)
        if not self.pixmap.isNull():
            image = self.pixmap.scaled(
                box.size() - QtCore.QSize(4, 4), Qt.KeepAspectRatio,
                Qt.SmoothTransformation)
            target = QtCore.QRect(QtCore.QPoint(), image.size())
            target.moveCenter(box.center())
            painter.drawPixmap(target, image)
        else:
            clean = "".join(ch for ch in self.name if ch.isalnum()) or "V"
            initials = (clean[0] + clean[-1]).upper()
            font = painter.font()
            font.setBold(True)
            font.setPointSize(max(7, self.height() // 4))
            painter.setFont(font)
            painter.setPen(QtGui.QColor("#4B4942"))
            painter.drawText(box, Qt.AlignCenter, initials)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.globalPos())
            event.accept()
            return
        super().mousePressEvent(event)


class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(object)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.globalPos())
            event.accept()
            return
        super().mousePressEvent(event)


class SentenceTextEdit(QtWidgets.QPlainTextEdit):
    """Compact multiline editor which grows to a sensible text height."""
    focused = QtCore.pyqtSignal()
    submitRequested = QtCore.pyqtSignal()

    MINIMUM_HEIGHT = 48
    MAXIMUM_HEIGHT = 220

    def __init__(self, text="", parent=None):
        super().__init__(str(text), parent)
        self.setObjectName("sentenceText")
        self.setTabChangesFocus(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(self.MINIMUM_HEIGHT)
        self.setMaximumHeight(self.MAXIMUM_HEIGHT)
        self.document().documentLayout().documentSizeChanged.connect(
            self._fit_height)
        self.textChanged.connect(self._fit_height)
        self._fit_height()

    def _fit_height(self, *_args):
        margins = self.contentsMargins()
        document_height = int(np.ceil(self.document().size().height()))
        line_height = QtGui.QFontMetrics(self.font()).lineSpacing()
        logical_lines = max(
            1, self.document().blockCount(),
            self.toPlainText().count("\n") + 1)
        block_height = logical_lines * line_height
        wanted = max(document_height, block_height) + \
            margins.top() + margins.bottom() + 14
        self.setFixedHeight(max(
            self.MINIMUM_HEIGHT, min(self.MAXIMUM_HEIGHT, wanted)))

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.submitRequested.emit()
                event.accept()
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event):
        self.focused.emit()
        super().focusInEvent(event)


def translucent_drag_pixmap(widget, opacity=0.66):
    source = widget.grab()
    preview = QtGui.QPixmap(source.size())
    preview.fill(Qt.transparent)
    painter = QtGui.QPainter(preview)
    painter.setOpacity(float(opacity))
    painter.drawPixmap(0, 0, source)
    painter.end()
    return preview


class PhraseChip(QtWidgets.QFrame):
    """Draggable phrase control with direct audition and context actions."""
    playRequested = QtCore.pyqtSignal(int)
    openRequested = QtCore.pyqtSignal(int)
    contextRequested = QtCore.pyqtSignal(int, object)
    clicked = QtCore.pyqtSignal(int, object)

    def __init__(self, index, phrase, parent=None):
        super().__init__(parent)
        self.index = int(index)
        self.phrase_id = str(phrase.get("id") or index)
        self._press_pos = None
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setObjectName("phraseChip")
        self.setProperty("selected", False)
        self.setProperty("playing", False)
        self.setProperty("pending", str(phrase.get("_pending") or ""))
        self._pending = str(phrase.get("_pending") or "")
        text = str(phrase.get("text") or "")
        preview = phrase.get("_preview")
        preview_seconds = (len(preview[0]) / float(max(1, preview[1]))
                           if preview else 0.0)
        self._preferred_width = int(max(
            170, min(430, 150 + min(180, len(text) * 3.2) +
                         min(100, preview_seconds * 18.0))))
        self.setMinimumWidth(150)
        self.setMaximumWidth(520)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Preferred)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setSpacing(3)
        header = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(str(phrase.get("text") or "(empty phrase)"))
        label.setWordWrap(True)
        label.setMinimumHeight(30)
        header.addWidget(label, 1)
        play = QtWidgets.QToolButton()
        play.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        play.setToolTip("Play phrase")
        play.clicked.connect(lambda: self.playRequested.emit(self.index))
        header.addWidget(play)
        layout.addLayout(header)
        self.mini_waveform = None
        if phrase.get("_preview"):
            self.mini_waveform = MiniWaveform(
                phrase["_preview"], self._pending, self)
            layout.addWidget(self.mini_waveform)
        details = []
        if phrase.get("speaker"):
            details.append(str(phrase["speaker"]))
        faults = [key for key, enabled in
                  dict(phrase.get("fault_mode") or {}).items()
                  if enabled and key != "bit_depth"]
        if phrase.get("fault_mode", {}).get("bit_depth"):
            faults.append("%d-bit" % phrase["fault_mode"]["bit_depth"])
        if faults:
            details.append("faults: %d" % len(faults))
        meta = QtWidgets.QLabel(" | ".join(details) or "default routing")
        meta.setStyleSheet("color:#555; font-size:8pt;")
        layout.addWidget(meta)

    def sizeHint(self):
        hint = super().sizeHint()
        return QtCore.QSize(self._preferred_width, max(82, hint.height()))

    def set_selected(self, selected):
        self.setProperty("selected", bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_playing(self, playing):
        self.setProperty("playing", bool(playing))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_pending(self, pending=""):
        pending = str(pending or "")
        if self._pending == pending:
            return
        self._pending = pending
        self.setProperty("pending", self._pending)
        self.style().unpolish(self)
        self.style().polish(self)
        if self.mini_waveform is not None:
            self.mini_waveform.set_pending(self._pending)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press_pos = event.pos()
            self.clicked.emit(self.index, event.modifiers())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self._press_pos is not None and
                event.buttons() & Qt.LeftButton and
                (event.pos() - self._press_pos).manhattanLength() >=
                QtWidgets.QApplication.startDragDistance()):
            drag = QtGui.QDrag(self)
            mime = QtCore.QMimeData()
            mime.setData("application/x-festvox-phrase",
                         self.phrase_id.encode("utf-8"))
            drag.setMimeData(mime)
            preview = translucent_drag_pixmap(self)
            drag.setPixmap(preview)
            drag.setHotSpot(event.pos())
            drag.exec_(Qt.MoveAction)
            self._press_pos = None
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.openRequested.emit(self.index)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class VerticalResizeHandle(QtWidgets.QWidget):
    """Small drag handle that resizes one sidebar widget vertically."""

    resized = QtCore.pyqtSignal(int)

    def __init__(self, target, minimum=48, maximum=280, parent=None):
        super().__init__(parent)
        self.target = target
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self._start_y = None
        self._start_height = None
        self.setFixedHeight(7)
        self.setCursor(Qt.SplitVCursor)
        self.setToolTip("Drag to resize the voicebank list")

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor("#8C897F"))
        y = self.height() // 2
        painter.drawLine(18, y, max(18, self.width() - 18), y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._start_y = event.globalY()
            self._start_height = self.target.height()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._start_y is None:
            return
        height = max(self.minimum, min(
            self.maximum,
            self._start_height + event.globalY() - self._start_y))
        self.target.setFixedHeight(height)
        self.resized.emit(height)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._start_y = None
        self._start_height = None
        event.accept()

    def contextMenuEvent(self, event):
        self.contextRequested.emit(self.index, event.globalPos())
        event.accept()


# Adapted from Qt for Python's BSD-3-Clause Flow Layout example:
# https://doc.qt.io/qtforpython-6/examples/example_widgets_layouts_flowlayout.html
class FlowLayout(QtWidgets.QLayout):
    """Height-for-width layout that wraps phrase controls into rows."""
    def __init__(self, parent=None, margin=0, spacing=4):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def __del__(self):
        while self.takeAt(0):
            pass

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QtCore.QSize(margins.left() + margins.right(),
                                   margins.top() + margins.bottom())

    def _do_layout(self, rect, test_only):
        spacing = max(0, self.spacing())
        rows, current, used = [], [], 0
        for item in self._items:
            hint = item.sizeHint()
            wanted = min(max(item.minimumSize().width(), hint.width()),
                         max(1, rect.width()))
            next_used = used + (spacing if current else 0) + wanted
            if current and next_used > rect.width():
                rows.append(current)
                current, used = [], 0
            current.append((item, wanted, hint.height()))
            used += (spacing if len(current) > 1 else 0) + wanted
        if current:
            rows.append(current)

        y = rect.y()
        for row in rows:
            available = max(1, rect.width() - spacing * (len(row) - 1))
            widths = [wanted for _item, wanted, _height in row]
            extra = max(0, available - sum(widths))
            growable = set(range(len(row)))
            while extra > 0 and growable:
                weight = sum(max(1, widths[index]) for index in growable)
                consumed = 0
                for index in list(growable):
                    item = row[index][0]
                    maximum = item.maximumSize().width()
                    share = max(1, int(round(
                        extra * max(1, widths[index]) / float(weight))))
                    growth = min(share, max(0, maximum - widths[index]))
                    widths[index] += growth
                    consumed += growth
                    if widths[index] >= maximum:
                        growable.discard(index)
                if consumed <= 0:
                    break
                extra -= consumed
            line_height = max(height for _item, _wanted, height in row)
            x = rect.x()
            for (item, _wanted, _height), width in zip(row, widths):
                if not test_only:
                    item.setGeometry(QtCore.QRect(
                        x, y, max(1, width), line_height))
                x += width + spacing
            y += line_height + spacing
        return max(0, y - rect.y() - (spacing if rows else 0))


class PhraseBoard(QtWidgets.QWidget):
    orderChanged = QtCore.pyqtSignal(object)
    playRequested = QtCore.pyqtSignal(int)
    openRequested = QtCore.pyqtSignal(int)
    contextRequested = QtCore.pyqtSignal(int, object)
    phraseClicked = QtCore.pyqtSignal(int, object)
    selectionDragStarted = QtCore.pyqtSignal(object, object)
    selectionDragMoved = QtCore.pyqtSignal(object)
    selectionDragFinished = QtCore.pyqtSignal(object)

    def __init__(self, phrases=None, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._phrases = []
        self._chips = []
        self._selected = set()
        self._selection_anchor = None
        self._rubber_origin = None
        self._drop_index = None
        self._rubber = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Rectangle, self)
        self.layout_ = FlowLayout(self, 0, 4)
        policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.set_phrases(phrases or [])

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.layout_.heightForWidth(max(1, int(width)))

    def sizeHint(self):
        width = max(280, self.width())
        return QtCore.QSize(width, self.heightForWidth(width))

    def resizeEvent(self, event):
        wanted = max(0, self.heightForWidth(event.size().width()))
        if self.minimumHeight() != wanted:
            self.setMinimumHeight(wanted)
            self.updateGeometry()
        super().resizeEvent(event)

    def set_phrases(self, phrases):
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        self._phrases = [dict(phrase) for phrase in phrases]
        self._chips = []
        self._selected = set()
        for index, phrase in enumerate(self._phrases):
            chip = PhraseChip(index, phrase, self)
            chip.playRequested.connect(self.playRequested)
            chip.openRequested.connect(self.openRequested)
            chip.contextRequested.connect(self._context_requested)
            chip.clicked.connect(self._chip_clicked)
            self.layout_.addWidget(chip)
            self._chips.append(chip)
        self.updateGeometry()

    def selected_indices(self):
        return sorted(self._selected)

    def set_selected_indices(self, indices):
        self._selected = {int(index) for index in indices
                          if 0 <= int(index) < len(self._chips)}
        self._refresh_selection()

    def set_playing(self, phrase_index=None):
        for index, chip in enumerate(self._chips):
            chip.set_playing(phrase_index is not None and
                             index == int(phrase_index))

    def _chip_clicked(self, index, modifiers):
        self._select(index, modifiers)
        self.phraseClicked.emit(int(index), modifiers)

    def _select(self, index, modifiers):
        if modifiers & Qt.ShiftModifier and self._selection_anchor is not None:
            first, last = sorted((self._selection_anchor, int(index)))
            selected = set(range(first, last + 1))
            self._selected = (self._selected | selected
                              if modifiers & (Qt.ControlModifier |
                                              Qt.AltModifier) else selected)
        elif modifiers & (Qt.ControlModifier | Qt.AltModifier):
            if index in self._selected:
                self._selected.remove(index)
            else:
                self._selected.add(index)
        else:
            self._selected = {index}
        self._selection_anchor = int(index)
        self._refresh_selection()

    def _refresh_selection(self):
        for number, chip in enumerate(self._chips):
            chip.set_selected(number in self._selected)

    def _context_requested(self, index, pos):
        if int(index) not in self._selected:
            self._select(int(index), Qt.NoModifier)
        self.contextRequested.emit(int(index), pos)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._rubber_origin = event.pos()
            self._rubber.setGeometry(QtCore.QRect(self._rubber_origin,
                                                  QtCore.QSize()))
            self._rubber.show()
            self.selectionDragStarted.emit(
                event.globalPos(), event.modifiers())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._rubber_origin is not None:
            rect = QtCore.QRect(self._rubber_origin, event.pos()).normalized()
            self._rubber.setGeometry(rect)
            touched = {index for index, chip in enumerate(self._chips)
                       if rect.intersects(chip.geometry())}
            self._selected = touched
            self._refresh_selection()
            self.selectionDragMoved.emit(event.globalPos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._rubber_origin is not None:
            self._rubber_origin = None
            self._rubber.hide()
            self.selectionDragFinished.emit(event.globalPos())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-festvox-phrase"):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        self._drop_index = self._drop_target(event.pos())
        self.update()
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._drop_index = None
        self.update()
        super().dragLeaveEvent(event)

    def _drop_target(self, point):
        target = len(self._chips)
        for index, chip in enumerate(self._chips):
            center = chip.geometry().center()
            same_row = (chip.geometry().top() <= point.y() <=
                        chip.geometry().bottom())
            if (point.y() < center.y() or
                    (same_row and point.x() < center.x())):
                target = index
                break
        return target

    def dropEvent(self, event):
        raw = bytes(event.mimeData().data(
            "application/x-festvox-phrase")).decode("utf-8", "replace")
        ids = [str(phrase.get("id") or index)
               for index, phrase in enumerate(self._phrases)]
        if raw not in ids:
            return
        source = ids.index(raw)
        target = self._drop_target(event.pos())
        moving = (self.selected_indices()
                  if source in self._selected else [source])
        moving = sorted(set(index for index in moving
                            if 0 <= index < len(ids)))
        moving_ids = [ids[index] for index in moving]
        remaining = [phrase_id for index, phrase_id in enumerate(ids)
                     if index not in moving]
        target -= sum(index < target for index in moving)
        target = max(0, min(len(remaining), target))
        self.orderChanged.emit(
            remaining[:target] + moving_ids + remaining[target:])
        self._drop_index = None
        self.update()
        event.acceptProposedAction()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_index is None or not self._chips:
            return
        index = int(self._drop_index)
        if index >= len(self._chips):
            chip = self._chips[-1]
            x = chip.geometry().right() + 3
            top, bottom = chip.geometry().top(), chip.geometry().bottom()
        else:
            chip = self._chips[index]
            x = chip.geometry().left() - 3
            top, bottom = chip.geometry().top(), chip.geometry().bottom()
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtGui.QColor("#1E6FE0"), 3))
        painter.drawLine(x, top, x, bottom)
        painter.end()


class SentenceDragHandle(QtWidgets.QWidget):
    """Compact six-dot handle for dragging selected sentence rows."""
    def __init__(self, index, selected_provider, parent=None):
        super().__init__(parent)
        self.index = int(index)
        self.selected_provider = selected_provider
        self._press = None
        self.setFixedSize(22, 42)
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("Move selected sentences")

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QtGui.QColor("#77736D"))
        for row in range(3):
            for column in range(2):
                painter.drawEllipse(QtCore.QPointF(
                    8 + column * 7, 13 + row * 7), 2, 2)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self._press is not None and event.buttons() & Qt.LeftButton and
                (event.pos() - self._press).manhattanLength() >=
                QtWidgets.QApplication.startDragDistance()):
            selected = list(self.selected_provider() or [])
            if self.index not in selected:
                selected = [self.index]
            drag = QtGui.QDrag(self)
            mime = QtCore.QMimeData()
            mime.setData("application/x-festvox-sentences",
                         json.dumps(sorted(selected)).encode("ascii"))
            drag.setMimeData(mime)
            row = self.parentWidget()
            preview = translucent_drag_pixmap(row)
            drag.setPixmap(preview)
            drag.setHotSpot(row.mapFromGlobal(event.globalPos()))
            drag.exec_(Qt.MoveAction)
            self._press = None
            self.setCursor(Qt.OpenHandCursor)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press = None
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)


class SentenceRow(QtWidgets.QFrame):
    moveRequested = QtCore.pyqtSignal(int, int)
    playRequested = QtCore.pyqtSignal(int)
    generateRequested = QtCore.pyqtSignal(int)
    openRequested = QtCore.pyqtSignal(int, str)
    speakerRequested = QtCore.pyqtSignal(int, object)
    textEdited = QtCore.pyqtSignal(int, str)
    phraseOrderChanged = QtCore.pyqtSignal(int, object)
    phrasePlayRequested = QtCore.pyqtSignal(int, int)
    phraseOpenRequested = QtCore.pyqtSignal(int, int)
    phraseContextRequested = QtCore.pyqtSignal(int, int, object)
    clicked = QtCore.pyqtSignal(int, object)
    contextRequested = QtCore.pyqtSignal(int, object)
    selectionDragStarted = QtCore.pyqtSignal(object, object)
    selectionDragMoved = QtCore.pyqtSignal(object)
    selectionDragFinished = QtCore.pyqtSignal(object)
    dropRequested = QtCore.pyqtSignal(object, int)

    def __init__(self, index, state, selected_provider=None, parent=None):
        super().__init__(parent)
        self.index = int(index)
        self._selection_press = None
        self._selection_dragging = False
        self._selection_modifiers = Qt.NoModifier
        self._drop_edge = None
        self.setAcceptDrops(True)
        self.setObjectName("sentenceRow")
        self.setProperty("selected", False)
        self.setProperty("playing", False)
        pending = ("generate" if state.get("needs_generate") else
                   "rerender" if state.get("needs_rerender") else "")
        self.setProperty("pending", pending)
        self._pending = None
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 7, 6, 8)
        root.setSpacing(5)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(SentenceDragHandle(
            self.index, selected_provider or (lambda: [self.index]), self))
        speaker_name = str(state.get("_speaker_name") or
                           state.get("voicebank") or "Default")
        badge = SpeakerBadge(
            speaker_name, state.get("_speaker_icon") or "", 46, self)
        badge.clicked.connect(
            lambda pos: self.speakerRequested.emit(self.index, pos))
        header.addWidget(badge)
        title_box = QtWidgets.QVBoxLayout()
        sentence_text = SentenceTextEdit(state.get("text") or "", self)
        sentence_text.setAccessibleName("Sentence %d text" % (index + 1))
        sentence_text.focused.connect(
            lambda: self.clicked.emit(self.index, Qt.NoModifier))
        title_box.addWidget(sentence_text)
        speaker = ClickableLabel(speaker_name)
        speaker.setCursor(Qt.PointingHandCursor)
        speaker.setToolTip("Choose speaker for this sentence")
        speaker.setStyleSheet("color:#315D91; padding:2px 1px;")
        speaker.clicked.connect(
            lambda pos: self.speakerRequested.emit(self.index, pos))
        title_box.addWidget(speaker)
        header.addLayout(title_box, 1)
        self.pending_badge = QtWidgets.QLabel()
        self.pending_badge.setObjectName("pendingBadge")
        header.addWidget(self.pending_badge)
        for direction, icon in ((-1, QtWidgets.QStyle.SP_ArrowUp),
                                (1, QtWidgets.QStyle.SP_ArrowDown)):
            button = QtWidgets.QToolButton()
            button.setIcon(self.style().standardIcon(icon))
            button.setToolTip("Move sentence %s" %
                              ("up" if direction < 0 else "down"))
            button.clicked.connect(
                lambda _checked=False, d=direction:
                self.moveRequested.emit(self.index, d))
            header.addWidget(button)
        generate = QtWidgets.QPushButton("Generate")
        generate.setProperty("sentenceGenerate", True)
        generate.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_BrowserReload))
        generate.clicked.connect(lambda: self.generateRequested.emit(self.index))
        self.generate_button = generate
        header.addWidget(generate)
        play = QtWidgets.QPushButton("Play")
        play.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        play.clicked.connect(lambda: self.playRequested.emit(self.index))
        play.setEnabled(bool(state.get("rendered") and
                             np.asarray(state.get("preview_audio") if
                                        state.get("preview_audio") is not None
                                        else []).size > 1))
        self.play_button = play
        header.addWidget(play)
        speech = QtWidgets.QPushButton("Edit")
        speech.clicked.connect(
            lambda: self.openRequested.emit(self.index, "speech"))
        header.addWidget(speech)
        sentence_text.textChanged.connect(
            lambda: self.textEdited.emit(
                self.index, sentence_text.toPlainText()))
        sentence_text.submitRequested.connect(
            lambda: self.generateRequested.emit(self.index))
        root.addLayout(header)
        previews = state.get("phrase_previews") or {}
        phrase_rows = []
        for phrase in state.get("phrases") or []:
            row = dict(phrase)
            row["_preview"] = previews.get(phrase.get("id"))
            row["_pending"] = pending
            phrase_rows.append(row)
        self.board = PhraseBoard(phrase_rows, self)
        self.board.orderChanged.connect(
            lambda order: self.phraseOrderChanged.emit(self.index, order))
        self.board.playRequested.connect(
            lambda phrase: self.phrasePlayRequested.emit(self.index, phrase))
        self.board.openRequested.connect(
            lambda phrase: self.phraseOpenRequested.emit(self.index, phrase))
        self.board.contextRequested.connect(
            lambda phrase, pos: self.phraseContextRequested.emit(
                self.index, phrase, pos))
        root.addWidget(self.board)
        self.set_pending(pending)

    def set_selected(self, selected):
        self.setProperty("selected", bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_pending(self, pending=""):
        pending = str(pending or "")
        if self._pending == pending:
            return
        self._pending = pending
        self.setProperty("pending", pending)
        self.style().unpolish(self)
        self.style().polish(self)
        labels = {"rerender": "Re-render pending"}
        self.pending_badge.setText(labels.get(pending, ""))
        self.pending_badge.setProperty("pending", pending)
        self.pending_badge.setVisible(pending == "rerender")
        self.pending_badge.style().unpolish(self.pending_badge)
        self.pending_badge.style().polish(self.pending_badge)
        self.generate_button.setText(
            "Re-render" if pending == "rerender" else "Generate")
        self.generate_button.setProperty(
            "renderPending", pending == "rerender")
        self.generate_button.setProperty(
            "generatePending", pending == "generate")
        self.generate_button.style().unpolish(self.generate_button)
        self.generate_button.style().polish(self.generate_button)
        for chip in self.board._chips:
            chip.set_pending(pending)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._selection_press = event.globalPos()
            self._selection_modifiers = event.modifiers()
            self.clicked.emit(self.index, event.modifiers())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self._selection_press is not None and
                event.buttons() & Qt.LeftButton):
            if (not self._selection_dragging and
                    (event.globalPos() - self._selection_press).manhattanLength()
                    >= QtWidgets.QApplication.startDragDistance()):
                self._selection_dragging = True
                self.selectionDragStarted.emit(
                    self._selection_press, self._selection_modifiers)
            if self._selection_dragging:
                self.selectionDragMoved.emit(event.globalPos())
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._selection_dragging:
            self.selectionDragFinished.emit(event.globalPos())
            event.accept()
        self._selection_press = None
        self._selection_dragging = False
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        self.contextRequested.emit(self.index, event.globalPos())
        event.accept()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-festvox-sentences"):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-festvox-sentences"):
            self._drop_edge = (
                "bottom" if event.pos().y() > self.height() / 2 else "top")
            self.update()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._drop_edge = None
        self.update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        try:
            indices = json.loads(bytes(event.mimeData().data(
                "application/x-festvox-sentences")).decode("ascii"))
        except (ValueError, UnicodeError):
            return
        target = self.index + (1 if event.pos().y() > self.height() / 2 else 0)
        self.dropRequested.emit(indices, target)
        self._drop_edge = None
        self.update()
        event.acceptProposedAction()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_edge is None:
            return
        y = 1 if self._drop_edge == "top" else self.height() - 2
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtGui.QColor("#1E6FE0"), 3))
        painter.drawLine(3, y, self.width() - 4, y)
        painter.end()


class SentenceSelectionCanvas(QtWidgets.QWidget):
    """Blank Sentences workspace which can start rectangle selection."""
    selectionDragStarted = QtCore.pyqtSignal(object, object)
    selectionDragMoved = QtCore.pyqtSignal(object)
    selectionDragFinished = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selection_press = None
        self._selection_dragging = False
        self._selection_modifiers = Qt.NoModifier

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._selection_press = event.globalPos()
            self._selection_modifiers = event.modifiers()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selection_press is not None and event.buttons() & Qt.LeftButton:
            if (not self._selection_dragging and
                    (event.globalPos() - self._selection_press).manhattanLength()
                    >= QtWidgets.QApplication.startDragDistance()):
                self._selection_dragging = True
                self.selectionDragStarted.emit(
                    self._selection_press, self._selection_modifiers)
            if self._selection_dragging:
                self.selectionDragMoved.emit(event.globalPos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._selection_dragging:
            self.selectionDragFinished.emit(event.globalPos())
            event.accept()
        self._selection_press = None
        self._selection_dragging = False
        super().mouseReleaseEvent(event)


class SentencesView(QtWidgets.QWidget):
    addRequested = QtCore.pyqtSignal(object)
    importRequested = QtCore.pyqtSignal()
    playAllRequested = QtCore.pyqtSignal()
    stopRequested = QtCore.pyqtSignal()
    generateRequested = QtCore.pyqtSignal(int)
    rerenderAllRequested = QtCore.pyqtSignal()
    clearRequested = QtCore.pyqtSignal()
    moveRequested = QtCore.pyqtSignal(int, int)
    playRequested = QtCore.pyqtSignal(int)
    openRequested = QtCore.pyqtSignal(int, str)
    speakerRequested = QtCore.pyqtSignal(int, object)
    textEdited = QtCore.pyqtSignal(int, str)
    selectionChanged = QtCore.pyqtSignal(object)
    phraseOrderChanged = QtCore.pyqtSignal(int, object)
    phrasePlayRequested = QtCore.pyqtSignal(int, int)
    phraseOpenRequested = QtCore.pyqtSignal(int, int)
    phraseContextRequested = QtCore.pyqtSignal(int, int, object)
    exportSelectedRequested = QtCore.pyqtSignal(object)
    removeSelectedRequested = QtCore.pyqtSignal(object)
    moveGroupRequested = QtCore.pyqtSignal(object, int)
    followChanged = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        toolbar = QtWidgets.QHBoxLayout()
        self.add_sentence = QtWidgets.QPushButton("Add sentence")
        self.add_sentence.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_FileIcon))
        self.add_sentence.clicked.connect(
            lambda: self.addRequested.emit(self.selected_sentence_indices()))
        toolbar.addWidget(self.add_sentence)
        self.remove_sentence = QtWidgets.QPushButton("Remove selected")
        self.remove_sentence.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_TrashIcon))
        self.remove_sentence.setEnabled(False)
        self.remove_sentence.clicked.connect(self.remove_selected)
        toolbar.addWidget(self.remove_sentence)
        toolbar.addSpacing(8)
        self.play_all = QtWidgets.QPushButton("Play all")
        self.play_all.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPlay))
        self.play_all.clicked.connect(self.playAllRequested)
        toolbar.addWidget(self.play_all)
        self.stop = QtWidgets.QToolButton()
        self.stop.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self.stop.setToolTip("Stop playback")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.stopRequested)
        toolbar.addWidget(self.stop)
        self.follow_spoken_sentence = QtWidgets.QCheckBox(
            "Follow spoken sentence")
        self.follow_spoken_sentence.setChecked(True)
        self.follow_spoken_sentence.setToolTip(
            "Scroll only when the sentence being spoken leaves the view")
        self.follow_spoken_sentence.toggled.connect(self.followChanged)
        toolbar.addWidget(self.follow_spoken_sentence)
        load = QtWidgets.QPushButton("Load text file")
        load.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_DialogOpenButton))
        load.clicked.connect(self.importRequested)
        toolbar.addWidget(load)
        rerender = QtWidgets.QPushButton("Re-render all")
        rerender.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_BrowserReload))
        rerender.clicked.connect(self.rerenderAllRequested)
        toolbar.addWidget(rerender)
        clear = QtWidgets.QPushButton("Clear all")
        clear.clicked.connect(self.clearRequested)
        toolbar.addWidget(clear)
        toolbar.addStretch(1)
        root.addLayout(toolbar)
        gain_row = QtWidgets.QHBoxLayout()
        gain_label = QtWidgets.QLabel("All sentences volume:")
        gain_label.setObjectName("hdr")
        gain_row.addWidget(gain_label)
        self.gain = GainControl(0.0, self)
        self.gain.setMinimumWidth(360)
        gain_row.addWidget(self.gain, 1)
        self.selection_notice = QtWidgets.QLabel("Select a sentence to edit")
        self.selection_notice.setStyleSheet(
            "color:#666; background:#E2E0D6; border:1px solid #B4B0A4; "
            "padding:7px;")
        gain_row.addWidget(self.selection_notice)
        root.addLayout(gain_row)
        scroll = QtWidgets.QScrollArea()
        self.scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content = SentenceSelectionCanvas()
        self.rows = QtWidgets.QVBoxLayout(self.content)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setSpacing(0)
        self.row_widgets = []
        self._selected = set()
        self._selected_phrases = set()
        self._phrase_anchor = None
        self._selection_anchor = None
        self._rubber_origin = None
        self._rubber_base = set()
        self._rubber = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Rectangle, self.content)
        self._phrase_rubber = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Rectangle, self.content)
        self._phrase_rubber_origin = None
        self._phrase_rubber_base = set()
        self._playing_sentence = None
        self.content.selectionDragStarted.connect(
            self._begin_selection_drag)
        self.content.selectionDragMoved.connect(
            self._update_selection_drag)
        self.content.selectionDragFinished.connect(
            self._end_selection_drag)
        scroll.setWidget(self.content)
        root.addWidget(scroll, 1)

    def refresh(self, states):
        while self.rows.count():
            item = self.rows.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        self.row_widgets = []
        self._selected = {index for index in self._selected
                          if 0 <= index < len(states)}
        self._selected_phrases = {
            (sentence, phrase) for sentence, phrase in self._selected_phrases
            if 0 <= sentence < len(states) and
            0 <= phrase < len(states[sentence].get("phrases") or [])}
        for index, state in enumerate(states):
            row = SentenceRow(
                index, state, self.selected_sentence_indices, self.content)
            row.moveRequested.connect(self.moveRequested)
            row.playRequested.connect(self.playRequested)
            row.generateRequested.connect(self.generateRequested)
            row.openRequested.connect(self.openRequested)
            row.speakerRequested.connect(self.speakerRequested)
            row.textEdited.connect(self.textEdited)
            row.phraseOrderChanged.connect(self.phraseOrderChanged)
            row.phrasePlayRequested.connect(self.phrasePlayRequested)
            row.phraseOpenRequested.connect(self.phraseOpenRequested)
            row.phraseContextRequested.connect(self.phraseContextRequested)
            row.clicked.connect(self._select_sentence)
            row.contextRequested.connect(self._sentence_context_menu)
            row.selectionDragStarted.connect(self._begin_selection_drag)
            row.selectionDragMoved.connect(self._update_selection_drag)
            row.selectionDragFinished.connect(self._end_selection_drag)
            row.dropRequested.connect(self.moveGroupRequested)
            row.board.phraseClicked.connect(
                lambda phrase, modifiers, sentence=index:
                self._select_phrase(sentence, phrase, modifiers))
            row.board.selectionDragStarted.connect(
                self._begin_phrase_selection_drag)
            row.board.selectionDragMoved.connect(
                self._update_phrase_selection_drag)
            row.board.selectionDragFinished.connect(
                self._end_phrase_selection_drag)
            self.rows.addWidget(row)
            self.row_widgets.append(row)
        self.rows.addStretch(1)
        self._refresh_selection()
        self._refresh_phrase_selection()

    def selected_sentence_indices(self):
        return sorted(self._selected)

    def selected_phrase_keys(self):
        return sorted(self._selected_phrases)

    def set_selected_indices(self, indices):
        self._selected = {int(index) for index in indices
                          if 0 <= int(index) < len(self.row_widgets)}
        self._selection_anchor = (max(self._selected)
                                  if self._selected else None)
        # A programmatic sentence selection replaces phrase selection just as
        # a plain sentence-row click does.  Otherwise the button can say
        # "Play all" while playAllRequested still sees stale phrase keys.
        self._selected_phrases.clear()
        self._phrase_anchor = None
        self._refresh_phrase_selection()
        self._refresh_selection()

    def clear_selection(self):
        self._selected.clear()
        self._selected_phrases.clear()
        self._selection_anchor = None
        self._phrase_anchor = None
        self._rubber_origin = None
        self._phrase_rubber_origin = None
        self._rubber.hide()
        self._phrase_rubber.hide()
        self._refresh_phrase_selection()
        self._refresh_selection()

    def set_selected_phrase_keys(self, keys):
        self._selected_phrases = {
            (int(sentence), int(phrase)) for sentence, phrase in (keys or [])
            if 0 <= int(sentence) < len(self.row_widgets) and
            0 <= int(phrase) < len(self.row_widgets[int(sentence)].board._chips)}
        self._phrase_anchor = (max(self._selected_phrases)
                               if self._selected_phrases else None)
        self._selected = {
            sentence for sentence, _phrase in self._selected_phrases}
        self._refresh_phrase_selection()
        self._refresh_selection()

    def _refresh_selection(self):
        for index, row in enumerate(self.row_widgets):
            row.set_selected(index in self._selected)
        self.remove_sentence.setEnabled(bool(self._selected))
        self.play_all.setText(
            "Play selected" if self._selected else "Play all")
        self.selection_notice.setVisible(not bool(self._selected))
        self.selectionChanged.emit(self.selected_sentence_indices())

    def _refresh_phrase_selection(self):
        for sentence, row in enumerate(self.row_widgets):
            row.board.set_selected_indices(
                [phrase for owner, phrase in self._selected_phrases
                 if owner == sentence])

    def _select_phrase(self, sentence, phrase, modifiers=Qt.NoModifier):
        key = (int(sentence), int(phrase))
        additive = bool(modifiers & (Qt.ControlModifier | Qt.AltModifier))
        if (modifiers & Qt.ShiftModifier and self._phrase_anchor and
                self._phrase_anchor[0] == key[0]):
            first, last = sorted((self._phrase_anchor[1], key[1]))
            selected = {(key[0], index)
                        for index in range(first, last + 1)}
            self._selected_phrases = (
                self._selected_phrases | selected if additive else selected)
        elif additive:
            if key in self._selected_phrases:
                self._selected_phrases.remove(key)
            else:
                self._selected_phrases.add(key)
        else:
            self._selected_phrases = {key}
        self._phrase_anchor = key
        owners = {owner for owner, _phrase in self._selected_phrases}
        self._selected = owners or {key[0]}
        self._selection_anchor = key[0]
        self._refresh_phrase_selection()
        self._refresh_selection()

    def _select_sentence(self, index, modifiers=Qt.NoModifier):
        index = int(index)
        if not (0 <= index < len(self.row_widgets)):
            return
        if modifiers & Qt.ShiftModifier and self._selection_anchor is not None:
            first, last = sorted((self._selection_anchor, index))
            selected = set(range(first, last + 1))
            self._selected = (self._selected | selected
                              if modifiers & Qt.ControlModifier else selected)
        elif modifiers & (Qt.ControlModifier | Qt.AltModifier):
            if index in self._selected:
                self._selected.remove(index)
            else:
                self._selected.add(index)
        else:
            self._selected = {index}
        self._selection_anchor = index
        if not modifiers & (Qt.ControlModifier | Qt.AltModifier |
                            Qt.ShiftModifier):
            self._selected_phrases = set()
            self._refresh_phrase_selection()
        self._refresh_selection()

    def _begin_selection_drag(self, global_pos, modifiers=Qt.NoModifier):
        self._rubber_origin = self.content.mapFromGlobal(global_pos)
        self._rubber_base = (set(self._selected)
                             if modifiers & (Qt.ControlModifier |
                                             Qt.AltModifier) else set())
        self._rubber.setGeometry(QtCore.QRect(
            self._rubber_origin, QtCore.QSize()))
        self._rubber.show()

    def _update_selection_drag(self, global_pos):
        if self._rubber_origin is None:
            return
        end = self.content.mapFromGlobal(global_pos)
        rect = QtCore.QRect(self._rubber_origin, end).normalized()
        self._rubber.setGeometry(rect)
        touched = {index for index, row in enumerate(self.row_widgets)
                   if rect.intersects(row.geometry())}
        self._selected = self._rubber_base | touched
        phrase_touched = self._phrases_in_rect(rect)
        if phrase_touched:
            self._selected_phrases = phrase_touched
            self._refresh_phrase_selection()
        self._refresh_selection()

    def _end_selection_drag(self, global_pos=None):
        if global_pos is not None:
            self._update_selection_drag(global_pos)
        self._rubber_origin = None
        self._rubber.hide()

    def _phrases_in_rect(self, rect):
        touched = set()
        for sentence, row in enumerate(self.row_widgets):
            for phrase, chip in enumerate(row.board._chips):
                origin = chip.mapTo(self.content, QtCore.QPoint(0, 0))
                geometry = QtCore.QRect(origin, chip.size())
                if rect.intersects(geometry):
                    touched.add((sentence, phrase))
        return touched

    def _begin_phrase_selection_drag(self, global_pos,
                                     modifiers=Qt.NoModifier):
        self._phrase_rubber_origin = self.content.mapFromGlobal(global_pos)
        self._phrase_rubber_base = (
            set(self._selected_phrases)
            if modifiers & (Qt.ControlModifier | Qt.AltModifier) else set())
        self._phrase_rubber.setGeometry(QtCore.QRect(
            self._phrase_rubber_origin, QtCore.QSize()))
        self._phrase_rubber.show()

    def _update_phrase_selection_drag(self, global_pos):
        if self._phrase_rubber_origin is None:
            return
        end = self.content.mapFromGlobal(global_pos)
        rect = QtCore.QRect(self._phrase_rubber_origin, end).normalized()
        self._phrase_rubber.setGeometry(rect)
        self._selected_phrases = (
            self._phrase_rubber_base | self._phrases_in_rect(rect))
        self._selected = {
            sentence for sentence, _phrase in self._selected_phrases}
        self._refresh_phrase_selection()
        self._refresh_selection()

    def _end_phrase_selection_drag(self, global_pos=None):
        if global_pos is not None:
            self._update_phrase_selection_drag(global_pos)
        self._phrase_rubber_origin = None
        self._phrase_rubber.hide()

    def _sentence_context_menu(self, index, global_pos):
        index = int(index)
        if index not in self._selected:
            self._select_sentence(index)
        selected = self.selected_sentence_indices()
        menu = QtWidgets.QMenu()
        menu.addAction(
            "Export selected to WAV...",
            lambda: self.exportSelectedRequested.emit(selected))
        menu.addAction(
            "Remove selected",
            lambda: self.removeSelectedRequested.emit(selected))
        move = menu.addMenu("Move selected")
        count = len(self.row_widgets)
        move.addAction(
            "To top", lambda: self.moveGroupRequested.emit(selected, 0))
        move.addAction(
            "Up", lambda: self.moveGroupRequested.emit(
                selected, max(0, min(selected) - 1)))
        move.addAction(
            "Down", lambda: self.moveGroupRequested.emit(
                selected, min(count, max(selected) + 2)))
        move.addAction(
            "To bottom", lambda: self.moveGroupRequested.emit(
                selected, count))
        menu.exec_(global_pos)

    def select_all(self):
        self._selected = set(range(len(self.row_widgets)))
        self._selection_anchor = (len(self.row_widgets) - 1
                                  if self.row_widgets else None)
        self._refresh_selection()

    def remove_selected(self):
        selected = self.selected_sentence_indices()
        if not selected:
            return False
        self.removeSelectedRequested.emit(selected)
        return True

    def set_playing(self, playing):
        self.play_all.setEnabled(not bool(playing))
        self.stop.setEnabled(bool(playing))

    def set_playing_item(self, sentence=None, phrase=None):
        for index, row in enumerate(self.row_widgets):
            row.board.set_playing(
                phrase if sentence is not None and index == int(sentence)
                else None)
            active = sentence is not None and index == int(sentence)
            row.setProperty("playing", active)
            row.style().unpolish(row)
            row.style().polish(row)
            row.update()
        if (sentence is not None and
                self.follow_spoken_sentence.isChecked() and
                int(sentence) != self._playing_sentence and
                0 <= int(sentence) < len(self.row_widgets)):
            self.scroll.ensureWidgetVisible(
                self.row_widgets[int(sentence)], 12, 12)
        self._playing_sentence = (
            None if sentence is None else int(sentence))

    def set_pending(self, sentence, pending=""):
        sentence = int(sentence)
        if 0 <= sentence < len(self.row_widgets):
            self.row_widgets[sentence].set_pending(pending)


class SpeakerPortrait(QtWidgets.QLabel):
    changeRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(160, 160)
        self.setStyleSheet(
            "QLabel { border:1px solid #7B746A; background:#F8F7F2; }")
        self.setCursor(Qt.PointingHandCursor)
        self.portrait_path = ""
        self.speaker_name = "Voice"
        self.fallback_color = ""
        self.set_portrait("", self.speaker_name)

    def set_portrait(self, path, speaker_name=""):
        self.portrait_path = str(path or "")
        self.speaker_name = str(speaker_name or self.speaker_name or "Voice")
        pixmap = (QtGui.QPixmap(self.portrait_path)
                  if self.portrait_path else QtGui.QPixmap())
        if pixmap.isNull():
            palette = ("#AFC7DE", "#D4B99A", "#B4CFAB",
                       "#D4B1C6", "#C4B9DA", "#DBBD91")
            identity = self.speaker_name or "Voice"
            color_index = sum((index + 1) * ord(char)
                              for index, char in enumerate(identity)) \
                % len(palette)
            self.fallback_color = palette[color_index]
            pixmap = QtGui.QPixmap(self.size())
            pixmap.fill(QtGui.QColor("#F8F7F2"))
            painter = QtGui.QPainter(pixmap)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QtGui.QColor(self.fallback_color))
            painter.drawEllipse(self.rect().adjusted(20, 20, -20, -20))
            clean = "".join(ch for ch in self.speaker_name if ch.isalnum()) \
                or "V"
            initials = (clean[0] + clean[-1]).upper()
            font = painter.font()
            font.setBold(True)
            font.setPointSize(28)
            painter.setFont(font)
            painter.setPen(QtGui.QColor("#4B4942"))
            painter.drawText(self.rect(), Qt.AlignCenter, initials)
            painter.end()
        else:
            self.fallback_color = ""
        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation)
        left = max(0, (scaled.width() - self.width()) // 2)
        top = max(0, (scaled.height() - self.height()) // 2)
        self.setPixmap(scaled.copy(left, top, self.width(), self.height()))
        self.setToolTip("Speaker: %s\nClick to choose this speaker's portrait" %
                        self.speaker_name)
        self.setAccessibleName("Selected speaker: %s" % self.speaker_name)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.changeRequested.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class AsaxiMoraGrid(QtWidgets.QWidget):
    """Fixed-width Asaxi mora strip for tone and voicing edits."""

    moraSelected = QtCore.pyqtSignal(int)
    moraSelectionChanged = QtCore.pyqtSignal(object)
    toneEdited = QtCore.pyqtSignal(object, object)

    CELL_WIDTH = 96.0
    CELL_TOP = 34.0
    CELL_HEIGHT = 78.0
    MARKER_HEIGHT = 31.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.moras = []
        self.tone_overrides = {}
        self.pitch_offsets = {}
        self.voicing_predictions = {}
        self.voicing_overrides = {}
        self.selected_mora = -1
        self.selected_moras = set()
        self.edit_mode = "accent"
        self._selection_anchor = -1
        self._word_group_cache = []
        self.setMinimumHeight(114)
        self.setMouseTracking(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed,
        )

    def set_model(
            self, moras, tone_overrides=None, pitch_offsets=None,
            voicing_predictions=None, voicing_overrides=None, selected=None):
        self.moras = [dict(row) for row in (moras or [])]
        self._word_group_cache = []
        self.tone_overrides = {
            int(key): str(value).upper()
            for key, value in dict(tone_overrides or {}).items()
            if str(value).upper() in {"H", "L"}
        }
        self.pitch_offsets = {
            int(key): int(value)
            for key, value in dict(pitch_offsets or {}).items()
        }
        self.voicing_predictions = {
            int(row["mora_index"]): dict(row)
            for row in (voicing_predictions or [])
            if isinstance(row, dict) and row.get("mora_index") is not None
        }
        self.voicing_overrides = {
            int(key): float(value)
            for key, value in dict(voicing_overrides or {}).items()
        }
        valid = {
            int(row.get("mora_index", position))
            for position, row in enumerate(self.moras)
        }
        if selected is not None and int(selected) in valid:
            self.selected_mora = int(selected)
        elif self.selected_mora not in valid:
            self.selected_mora = min(valid) if valid else -1
        self.selected_moras &= valid
        if self.selected_mora >= 0 and not self.selected_moras:
            self.selected_moras = {self.selected_mora}
        self.setFixedWidth(max(1, len(self.moras)) * int(self.CELL_WIDTH))
        self._word_group_cache = self._build_word_groups()
        self.update()

    def set_edit_mode(self, mode):
        self.edit_mode = (
            "voicing" if str(mode) == "mora_voicing" else "accent")
        if self.selected_mora >= 0 and not self.selected_moras:
            self.selected_moras = {self.selected_mora}
        self.update()

    def select_mora(self, mora_index):
        if int(mora_index) not in {
                int(row.get("mora_index", position))
                for position, row in enumerate(self.moras)}:
            return
        self.selected_mora = int(mora_index)
        self.selected_moras = {self.selected_mora}
        self._selection_anchor = self.selected_mora
        self.update()

    def _cell_rect(self, position):
        return QtCore.QRectF(
            position * self.CELL_WIDTH,
            self.CELL_TOP,
            self.CELL_WIDTH,
            self.CELL_HEIGHT,
        )

    def _mora_at(self, point):
        position = int(point.x() // self.CELL_WIDTH)
        if 0 <= position < len(self.moras):
            rect = self._cell_rect(position)
            if rect.adjusted(0, -self.MARKER_HEIGHT, 0, 0).contains(point):
                return self.moras[position]
        return None

    @staticmethod
    def _index(row, fallback=0):
        return int(row.get("mora_index", fallback))

    def _tone(self, row, fallback=0):
        index = self._index(row, fallback)
        return self.tone_overrides.get(
            index, str(row.get("pitch") or "L").upper())

    def _select(self, row, modifiers):
        index = self._index(row)
        if self.edit_mode == "voicing":
            phrase = int(row.get("phrase_index", 0))
            same_phrase = [
                self._index(item, position)
                for position, item in enumerate(self.moras)
                if int(item.get("phrase_index", 0)) == phrase
            ]
            selected = self.selected_moras & set(same_phrase)
            if (modifiers & Qt.ShiftModifier and
                    self._selection_anchor in same_phrase):
                first = same_phrase.index(self._selection_anchor)
                last = same_phrase.index(index)
                selected = set(
                    same_phrase[min(first, last):max(first, last) + 1])
            elif modifiers & Qt.ControlModifier:
                if index in selected and len(selected) > 1:
                    selected.remove(index)
                else:
                    selected.add(index)
                self._selection_anchor = index
            else:
                selected = {index}
                self._selection_anchor = index
            self.selected_moras = selected or {index}
        else:
            self.selected_moras = {index}
            self._selection_anchor = index
        self.selected_mora = index
        self.update()
        self.moraSelected.emit(index)
        self.moraSelectionChanged.emit(sorted(self.selected_moras))

    def _build_word_groups(self):
        groups = []
        start = 0
        while start < len(self.moras):
            row = self.moras[start]
            key = (
                int(row.get("phrase_index", 0)),
                int(row.get("word_index", 0)),
                str(row.get("word") or ""),
            )
            end = start + 1
            while end < len(self.moras):
                candidate = self.moras[end]
                candidate_key = (
                    int(candidate.get("phrase_index", 0)),
                    int(candidate.get("word_index", 0)),
                    str(candidate.get("word") or ""),
                )
                if candidate_key != key:
                    break
                end += 1
            groups.append((start, end - 1, key[2]))
            start = end
        return groups

    def _word_groups(self):
        return list(self._word_group_cache)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        exposed = event.region().boundingRect()
        painter.fillRect(exposed, QtGui.QColor("#F3F1E8"))
        if not self.moras:
            painter.setPen(QtGui.QColor("#6D6961"))
            painter.drawText(
                exposed, Qt.AlignCenter,
                "Generate Asaxi text to show morae")
            return

        first = max(
            0, int(math.floor(
                (float(exposed.left()) - self.CELL_WIDTH) /
                self.CELL_WIDTH)))
        last = min(
            len(self.moras), int(math.ceil(
                (float(exposed.right()) + self.CELL_WIDTH) /
                self.CELL_WIDTH)))
        for start, end, word in self._word_groups():
            if end < first or start >= last:
                continue
            left = self._cell_rect(start).left() + 3
            right = self._cell_rect(end).right() - 3
            color = QtGui.QColor("#557461")
            painter.setPen(QtGui.QPen(color, 2))
            painter.drawLine(
                QtCore.QPointF(left, 27), QtCore.QPointF(right, 27))
            painter.drawLine(
                QtCore.QPointF(left, 27), QtCore.QPointF(left, 31))
            painter.drawLine(
                QtCore.QPointF(right, 27), QtCore.QPointF(right, 31))
            if right - left >= 36:
                painter.setPen(color)
                painter.drawText(
                    QtCore.QRectF(left, 1, right - left, 20),
                    Qt.AlignCenter, word)

        tone_points = []
        if self.edit_mode == "accent":
            tone_first = max(0, first - 1)
            tone_last = min(len(self.moras), last + 1)
            for position in range(tone_first, tone_last):
                row = self.moras[position]
                if not bool(row.get("accentable", True)):
                    tone_points.append((position, None))
                    continue
                rect = self._cell_rect(position)
                y = rect.top() + (16 if self._tone(row, position) == "H"
                                  else 28)
                tone_points.append((
                    position, QtCore.QPointF(rect.center().x(), y)))
        for position in range(first, last):
            row = self.moras[position]
            index = self._index(row, position)
            selected = (
                index in self.selected_moras
                if self.edit_mode == "voicing"
                else index == self.selected_mora
            )
            rect = self._cell_rect(position)
            painter.setBrush(QtGui.QColor(
                "#DCE9F7" if selected else "#FAF9F4"))
            painter.setPen(QtGui.QPen(
                QtGui.QColor("#316AC5" if selected else "#9A968C"),
                2 if selected else 1,
            ))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))

            text = str(row.get("text") or "?")
            phones = " ".join(str(phone) for phone in row.get("phones") or [])
            if self.edit_mode == "accent":
                tone = self._tone(row, position)
                manual = index in self.tone_overrides
                painter.setPen(QtGui.QColor(
                    "#9A5A16" if manual else "#285F84"))
                cents = int(self.pitch_offsets.get(index, 0))
                painter.drawText(
                    QtCore.QRectF(
                        rect.left() + 5, rect.top() + 3, 20, 17),
                    Qt.AlignLeft | Qt.AlignVCenter, tone)
                if cents:
                    painter.drawText(
                        QtCore.QRectF(
                            rect.right() - 43, rect.top() + 3, 38, 17),
                        Qt.AlignRight | Qt.AlignVCenter, "%+d" % cents)
                painter.setPen(QtGui.QColor("#202020"))
                painter.drawText(
                    rect.adjusted(3, 29, -3, -25), Qt.AlignCenter, text)
                painter.setPen(QtGui.QColor("#56524C"))
                painter.drawText(
                    rect.adjusted(3, 52, -3, -5), Qt.AlignCenter, phones)
            else:
                prediction = self.voicing_predictions.get(index, {})
                painter.setPen(QtGui.QColor("#202020"))
                painter.drawText(
                    rect.adjusted(3, 7, -3, -43), Qt.AlignCenter, text)
                painter.setPen(QtGui.QColor("#56524C"))
                painter.drawText(
                    rect.adjusted(3, 30, -3, -24), Qt.AlignCenter, phones)
                if prediction.get("eligible"):
                    manual = index in self.voicing_overrides
                    value = (
                        self.voicing_overrides[index]
                        if manual else prediction.get(
                            "automatic_effective_voicing",
                            prediction.get("automatic_voicing", 1.0))
                    )
                    painter.setPen(QtGui.QColor(
                        "#9A5A16" if manual else "#285F84"))
                    painter.drawText(
                        rect.adjusted(3, 52, -3, -4), Qt.AlignCenter,
                        ("Manual " if manual else "Auto ")
                        + "%d%%" % round(float(value) * 100.0))
                else:
                    painter.setPen(QtGui.QColor("#8A8882"))
                    painter.drawText(
                        rect.adjusted(3, 52, -3, -4),
                        Qt.AlignCenter, "No vowel")
        if self.edit_mode == "accent":
            painter.setPen(QtGui.QPen(QtGui.QColor("#426A8C"), 2))
            previous = None
            for _position, point in tone_points:
                if point is not None and previous is not None:
                    painter.drawLine(previous, point)
                previous = point
            painter.setPen(Qt.NoPen)
            for position, point in tone_points:
                if point is None:
                    continue
                index = self._index(self.moras[position], position)
                painter.setBrush(QtGui.QColor(
                    "#B4473F" if index in self.tone_overrides
                    else "#426A8C"))
                painter.drawEllipse(point, 4.0, 4.0)

    def mousePressEvent(self, event):
        row = self._mora_at(event.pos())
        if row is not None and event.button() in {
                Qt.LeftButton, Qt.RightButton}:
            self._select(row, event.modifiers())
            if event.button() == Qt.RightButton:
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        row = self._mora_at(event.pos())
        if (self.edit_mode == "accent" and event.button() == Qt.LeftButton
                and row is not None and bool(
                    row.get("accentable", True))):
            index = self._index(row)
            tone = "L" if self._tone(row) == "H" else "H"
            self.toneEdited.emit([index], tone)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        if self.edit_mode != "accent":
            event.accept()
            return
        row = self._mora_at(event.pos())
        if row is None or not bool(row.get("accentable", True)):
            return
        index = self._index(row)
        menu = QtWidgets.QMenu(self)
        high = menu.addAction("Set mora high")
        high.triggered.connect(
            lambda _on=False: self.toneEdited.emit([index], "H"))
        low = menu.addAction("Set mora low")
        low.triggered.connect(
            lambda _on=False: self.toneEdited.emit([index], "L"))
        menu.addSeparator()
        reset = menu.addAction("Use inferred mora tone")
        reset.setEnabled(index in self.tone_overrides)
        reset.triggered.connect(
            lambda _on=False: self.toneEdited.emit([index], None))
        menu.exec_(event.globalPos())


class AsaxiMoraEditorPanel(QtWidgets.QWidget):
    """Compact rendered-mora editor for Asaxi pitch and phonation."""

    editRequested = QtCore.pyqtSignal(str, object, object)
    moraNavigationRequested = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = asaxi_editing.new_edit_state()
        self._moras = []
        self._predictions = {}
        self._updating = False
        self._edit_mode = "accent"
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.summary = QtWidgets.QLabel(
            "Generate Asaxi text to show its inferred morae.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.grid = AsaxiMoraGrid()
        self.grid.moraSelected.connect(self._select_mora)
        self.grid.moraSelectionChanged.connect(self._select_moras)
        self.grid.toneEdited.connect(
            lambda indices, value:
            self.editRequested.emit("tone", indices, value))
        self.grid.setFixedHeight(114)
        self.grid_scroll = QtWidgets.QScrollArea()
        self.grid_scroll.setWidgetResizable(False)
        self.grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.grid_scroll.setFixedHeight(134)
        self.grid_scroll.setWidget(self.grid)
        layout.addWidget(self.grid_scroll)

        controls = QtWidgets.QGridLayout()
        controls.setHorizontalSpacing(6)
        self.tone_label = QtWidgets.QLabel("Mora tone:")
        controls.addWidget(self.tone_label, 0, 0)
        self.tone = ArrowComboBox()
        self.tone.addItem("From inference", None)
        self.tone.addItem("High (H)", "H")
        self.tone.addItem("Low (L)", "L")
        self.tone.currentIndexChanged.connect(self._tone_changed)
        controls.addWidget(self.tone, 0, 1, 1, 2)
        self.reset_tone = QtWidgets.QPushButton("Use inferred tone")
        self.reset_tone.clicked.connect(
            lambda: self._reset_requested("tone"))
        controls.addWidget(self.reset_tone, 0, 3)

        self.mora_pitch_label = QtWidgets.QLabel("Mora pitch:")
        controls.addWidget(self.mora_pitch_label, 1, 0)
        self.mora_pitch = QtWidgets.QSpinBox()
        self.mora_pitch.setRange(
            asaxi_editing.PITCH_OFFSET_MIN_CENTS,
            asaxi_editing.PITCH_OFFSET_MAX_CENTS)
        self.mora_pitch.setSingleStep(10)
        self.mora_pitch.setSuffix(" cents")
        self.mora_pitch.setToolTip(
            "Offset the inferred H/L target for the selected morae. "
            "Continuous Pitch points remain final.")
        self.mora_pitch.valueChanged.connect(self._pitch_changed)
        controls.addWidget(self.mora_pitch, 1, 1, 1, 2)
        self.reset_pitch = QtWidgets.QPushButton("Reset mora pitch")
        self.reset_pitch.clicked.connect(
            lambda: self._reset_requested("pitch"))
        controls.addWidget(self.reset_pitch, 1, 3)
        self.follow_selection = QtWidgets.QCheckBox("Follow selection")
        self.follow_selection.setChecked(True)
        self.follow_selection.setToolTip(
            "Center the waveform when a selected mora is outside the view")
        controls.addWidget(self.follow_selection, 1, 4)

        self.voicing_label = QtWidgets.QLabel("Mora voicing:")
        controls.addWidget(self.voicing_label, 2, 0)
        self.voicing = QtWidgets.QDoubleSpinBox()
        self.voicing.setRange(0.0, 100.0)
        self.voicing.setDecimals(0)
        self.voicing.setSingleStep(5.0)
        self.voicing.setSuffix(" % voiced")
        self.voicing.setToolTip(
            "Set harmonic excitation for every selected vowel mora. "
            "The detailed continuous Voicing curve remains final.")
        self.voicing.valueChanged.connect(self._voicing_changed)
        controls.addWidget(self.voicing, 2, 1, 1, 2)
        self.reset_voicing = QtWidgets.QPushButton("Use automatic voicing")
        self.reset_voicing.clicked.connect(
            lambda: self._reset_requested("voicing"))
        controls.addWidget(self.reset_voicing, 2, 3)
        self.diagnostic = QtWidgets.QLabel("")
        self.diagnostic.setWordWrap(True)
        controls.addWidget(self.diagnostic, 2, 4, 1, 3)
        controls.setColumnStretch(6, 1)
        layout.addLayout(controls)
        self.set_edit_mode("accent")

    def set_edit_mode(self, mode):
        self._edit_mode = (
            "mora_voicing" if str(mode) == "mora_voicing" else "accent")
        voicing = self._edit_mode == "mora_voicing"
        self.grid.set_edit_mode(self._edit_mode)
        for widget in (
                self.tone_label, self.tone, self.reset_tone,
                self.mora_pitch_label, self.mora_pitch, self.reset_pitch):
            widget.setVisible(not voicing)
        for widget in (
                self.voicing_label, self.voicing, self.reset_voicing,
                self.diagnostic):
            widget.setVisible(voicing)
        self._sync_controls()

    def _selected_indices(self):
        selected = set(self.grid.selected_moras)
        if not selected and self.grid.selected_mora >= 0:
            selected = {self.grid.selected_mora}
        return sorted(int(index) for index in selected)

    def _row(self, mora_index):
        return next((
            row for row in self._moras
            if int(row.get("mora_index", -1)) == int(mora_index)
        ), {})

    def _prediction(self, mora_index):
        return self._predictions.get(int(mora_index), {})

    def set_state(self, state):
        selected = self._selected_indices()
        self._updating = True
        try:
            self._state = asaxi_editing.normalize_edit_state(state)
            plan = dict(self._state.get("last_plan") or {})
            self._moras = asaxi_editing.mora_rows(plan)
            self._predictions = {
                int(row["mora_index"]): dict(row)
                for row in plan.get("mora_phonation_predictions") or []
                if isinstance(row, dict) and
                row.get("mora_index") is not None
            }
            tone_overrides = dict(
                self._state.get("mora_tone_overrides") or {})
            pitch_offsets = dict(
                self._state.get("mora_pitch_offsets_cents") or {})
            voice_overrides = dict(
                self._state.get("mora_voicing_overrides") or {})
            wanted = selected[0] if selected else (
                int(self._moras[0].get("mora_index", 0))
                if self._moras else -1
            )
            self.grid.set_model(
                self._moras,
                tone_overrides=tone_overrides,
                pitch_offsets=pitch_offsets,
                voicing_predictions=list(self._predictions.values()),
                voicing_overrides=voice_overrides,
                selected=wanted,
            )
            self.select_mora(wanted, emit=False)
            diagnostics = len(plan.get("diagnostics") or [])
            if self._moras:
                self.summary.setText(
                    "%d morae | dictionary/morphology H-L inference%s" %
                    (len(self._moras),
                     " | %d diagnostic%s" %
                     (diagnostics, "" if diagnostics == 1 else "s")
                     if diagnostics else ""))
            else:
                self.summary.setText(
                    "Generate Asaxi text to show its inferred morae.")
            self._sync_controls()
        finally:
            self._updating = False

    def select_mora(self, mora_index, emit=False):
        position = next((
            position for position, row in enumerate(self._moras)
            if int(row.get("mora_index", -1)) == int(mora_index)
        ), None)
        if position is None:
            return
        self.grid.select_mora(mora_index)
        self.grid_scroll.ensureVisible(
            int((position + 0.5) * self.grid.CELL_WIDTH), 56,
            int(self.grid.CELL_WIDTH * 0.55), 0)
        self._sync_controls()
        if emit:
            self.moraNavigationRequested.emit(int(mora_index))

    def _select_mora(self, mora_index):
        self._sync_controls()
        self.moraNavigationRequested.emit(int(mora_index))

    def _select_moras(self, _mora_indices):
        self._sync_controls()

    def set_timeline(self, _segments, _plan_rows):
        # The block strip is mora-spaced like the Japanese editor. Waveform
        # selection remains the authoritative time alignment.
        return

    def set_view_range(self, _left, _right):
        return

    def set_playhead(self, _value):
        return

    def _sync_controls(self):
        if not hasattr(self, "mora_pitch"):
            return
        indices = self._selected_indices()
        rows = [self._row(index) for index in indices]
        predictions = [self._prediction(index) for index in indices]
        pitchable = [
            index for index, row in zip(indices, rows)
            if bool(row.get("accentable", True))
        ]
        phonatable = [
            index for index, prediction in zip(indices, predictions)
            if bool(prediction.get("eligible"))
        ]
        tone_overrides = dict(
            self._state.get("mora_tone_overrides") or {})
        pitch_offsets = dict(
            self._state.get("mora_pitch_offsets_cents") or {})
        voice_overrides = dict(
            self._state.get("mora_voicing_overrides") or {})
        self.tone.blockSignals(True)
        self.mora_pitch.blockSignals(True)
        self.voicing.blockSignals(True)
        try:
            self.tone.setEnabled(bool(pitchable))
            self.reset_tone.setEnabled(any(
                str(index) in tone_overrides for index in pitchable))
            tone = (
                tone_overrides.get(str(pitchable[0]))
                if pitchable else None)
            self.tone.setCurrentIndex(
                max(0, self.tone.findData(tone)))
            self.mora_pitch.setEnabled(bool(pitchable))
            self.reset_pitch.setEnabled(any(
                str(index) in pitch_offsets for index in pitchable))
            self.mora_pitch.setValue(int(
                pitch_offsets.get(str(pitchable[0]), 0)
                if pitchable else 0))
            self.voicing.setEnabled(bool(phonatable))
            self.reset_voicing.setEnabled(any(
                str(index) in voice_overrides for index in phonatable))
            if phonatable:
                first = phonatable[0]
                self.voicing.setValue(100.0 * float(
                    voice_overrides.get(
                        str(first),
                        self._prediction(first).get(
                            "automatic_effective_voicing",
                            self._prediction(first).get(
                                "automatic_voicing", 1.0)))))
            else:
                self.voicing.setValue(100.0)
            reasons = (
                self._prediction(indices[0]).get("reasons") or []
                if len(indices) == 1 else []
            )
            if not indices:
                message = "Select a mora."
            elif not phonatable:
                message = "No aligned vowel-bearing span is editable."
            elif reasons:
                message = str(reasons[0])
            else:
                message = "%d selected mora%s" % (
                    len(indices), "" if len(indices) == 1 else "e")
            self.diagnostic.setText(message)
        finally:
            self.tone.blockSignals(False)
            self.mora_pitch.blockSignals(False)
            self.voicing.blockSignals(False)

    def _editable_indices(self, kind):
        indices = self._selected_indices()
        if kind in {"tone", "pitch"}:
            return [
                index for index in indices
                if bool(self._row(index).get("accentable", True))
            ]
        return [
            index for index in indices
            if bool(self._prediction(index).get("eligible"))
        ]

    def _tone_changed(self, _index):
        if self._updating:
            return
        indices = self._editable_indices("tone")
        if indices:
            self.editRequested.emit("tone", indices, self.tone.currentData())

    def _pitch_changed(self, value):
        if not self._updating:
            indices = self._editable_indices("pitch")
            if indices:
                self.editRequested.emit("pitch", indices, int(value))

    def _voicing_changed(self, value):
        if not self._updating:
            indices = self._editable_indices("voicing")
            if indices:
                self.editRequested.emit(
                    "voicing", indices, float(value) / 100.0)

    def _reset_requested(self, kind):
        indices = self._editable_indices(kind)
        if indices:
            self.editRequested.emit(str(kind), indices, None)


class JapaneseMoraGrid(QtWidgets.QWidget):
    """Fixed-width mora strip with direct accent and phrase editing."""

    moraSelected = QtCore.pyqtSignal(int)
    moraSelectionChanged = QtCore.pyqtSignal(object)
    accentEdited = QtCore.pyqtSignal(int, object)
    boundariesEdited = QtCore.pyqtSignal(int, object)

    MARKER_HEIGHT = 31.0
    CELL_TOP = 34.0
    CELL_HEIGHT = 72.0
    MIN_SELECTED_WIDTH = 42.0
    CELL_WIDTH = 96.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.utterance = None
        self.offsets = {}
        self.moras = []
        self.selected_mora = -1
        self.selected_moras = set()
        self.edit_mode = "accent"
        self.voicing_predictions = {}
        self.voicing_overrides = {}
        self._selection_anchor = -1
        self._timeline = {}
        self._view_range = (0.0, 1.0)
        self._playhead = 0.0
        self._mora_positions = {}
        self._accent_by_mora = {}
        self._accent_ranges = []
        self._drag = None
        self._drag_target = None
        self._drag_origin = None
        self._drag_moved = False
        self._suppress_context_menu = False
        self.setMinimumHeight(112)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed,
        )

    def set_model(self, utterance, offsets=None, selected=None,
                  voicing_predictions=None, voicing_overrides=None):
        self.utterance = utterance
        self.offsets = {int(key): int(value) for key, value in
                        dict(offsets or {}).items()}
        self.moras = list(utterance.moras) if utterance is not None else []
        self._mora_positions = {
            mora.index: position
            for position, mora in enumerate(self.moras)
        }
        self._accent_by_mora = {}
        self._accent_ranges = []
        if utterance is not None:
            for accent in utterance.accent_phrases:
                positions = []
                for local, mora in enumerate(accent.moras):
                    self._accent_by_mora[mora.index] = (accent, local)
                    position = self._mora_positions.get(mora.index)
                    if position is not None:
                        positions.append(position)
                if positions:
                    self._accent_ranges.append(
                        (min(positions), max(positions), accent))
        if selected is not None and any(
                mora.index == int(selected) for mora in self.moras):
            self.selected_mora = int(selected)
        elif self.moras:
            self.selected_mora = self.moras[0].index
        else:
            self.selected_mora = -1
        valid = {mora.index for mora in self.moras}
        self.selected_moras = {
            int(index) for index in self.selected_moras
            if int(index) in valid
        }
        if self.selected_mora >= 0 and not self.selected_moras:
            self.selected_moras = {self.selected_mora}
        self.voicing_predictions = {
            int(row.get("mora_index")): dict(row)
            for row in (voicing_predictions or [])
            if isinstance(row, dict) and row.get("mora_index") is not None
        }
        self.voicing_overrides = {
            int(key): float(value) for key, value in
            dict(voicing_overrides or {}).items()
        }
        self.setFixedWidth(max(1, len(self.moras)) * int(self.CELL_WIDTH))
        self.update()

    def set_edit_mode(self, mode):
        self.edit_mode = ("voicing" if str(mode) == "mora_voicing"
                          else "accent")
        if self.selected_mora >= 0 and not self.selected_moras:
            self.selected_moras = {self.selected_mora}
        self.update()

    def set_timeline(self, segments, plan_rows):
        timeline = {}
        for position, row in enumerate(plan_rows or []):
            if not isinstance(row, dict) or row.get("mora_index") is None:
                continue
            try:
                index = int(row.get("index", position))
                mora_index = int(row["mora_index"])
                segment = segments[index]
            except (IndexError, TypeError, ValueError):
                continue
            start, end = float(segment.start), float(segment.end)
            if mora_index in timeline:
                start = min(start, timeline[mora_index][0])
                end = max(end, timeline[mora_index][1])
            timeline[mora_index] = (start, end)
        self._timeline = timeline
        self.update()

    def set_view_range(self, left, right):
        left, right = float(left), float(right)
        if right <= left:
            right = left + 1e-6
        value = (left, right)
        if value != self._view_range:
            self._view_range = value
            self.update()

    def set_playhead(self, value):
        value = float(value or 0.0)
        if abs(value - self._playhead) > 1e-9:
            self._playhead = value
            # The authoritative playhead is drawn over the waveform. The mora
            # strip currently has no playhead glyph, so repainting its entire
            # wide backing widget on every 30 ms playback tick only burns
            # frames for long utterances.

    def select_mora(self, mora_index):
        if any(mora.index == int(mora_index) for mora in self.moras):
            self.selected_mora = int(mora_index)
            self.selected_moras = {self.selected_mora}
            self._selection_anchor = self.selected_mora
            self.update()

    def _select_voicing_mora(self, mora, modifiers):
        index = int(mora.index)
        same_phrase = {
            item.index for item in self.moras
            if item.phrase_index == mora.phrase_index
        }
        selected = self.selected_moras & same_phrase
        if modifiers & Qt.ShiftModifier and self._selection_anchor in same_phrase:
            positions = [item.index for item in self.moras
                         if item.phrase_index == mora.phrase_index]
            first = positions.index(self._selection_anchor)
            last = positions.index(index)
            selected = set(positions[min(first, last):max(first, last) + 1])
        elif modifiers & Qt.ControlModifier:
            if index in selected and len(selected) > 1:
                selected.remove(index)
            else:
                selected.add(index)
            self._selection_anchor = index
        else:
            selected = {index}
            self._selection_anchor = index
        self.selected_mora = index
        self.selected_moras = selected or {index}
        self.update()
        self.moraSelected.emit(index)
        self.moraSelectionChanged.emit(sorted(self.selected_moras))

    def _time_to_x(self, value):
        left, right = self._view_range
        return (float(value) - left) / max(1e-9, right - left) * self.width()

    def _actual_cell_rect(self, position):
        return QtCore.QRectF(
            position * self.CELL_WIDTH, self.CELL_TOP,
            self.CELL_WIDTH, self.CELL_HEIGHT)

    def _cell_rect(self, position, interactive=False):
        rect = self._actual_cell_rect(position)
        if interactive and rect.width() < self.MIN_SELECTED_WIDTH:
            rect.setLeft(rect.center().x() - self.MIN_SELECTED_WIDTH / 2.0)
            rect.setWidth(self.MIN_SELECTED_WIDTH)
        return rect

    def _mora_at(self, point):
        candidates = []
        for position, mora in enumerate(self.moras):
            rect = self._cell_rect(position, interactive=True)
            if rect.adjusted(0, -self.MARKER_HEIGHT, 0, 0).contains(point):
                candidates.append((abs(rect.center().x() - point.x()), mora))
        return min(candidates, default=(None, None))[1]

    def _accent_for_mora(self, mora_index):
        if self.utterance is None:
            return None
        mora = next((item for item in self.moras
                     if item.index == int(mora_index)), None)
        if mora is None:
            return None
        return next((item for item in self.utterance.accent_phrases
                     if item.index == mora.accent_phrase_index), None)

    def _phrase_for_mora(self, mora_index):
        if self.utterance is None:
            return None
        mora = next((item for item in self.moras
                     if item.index == int(mora_index)), None)
        if mora is None:
            return None
        return next((item for item in self.utterance.phrases
                     if item.index == mora.phrase_index), None)

    def _phrase_boundaries(self, phrase):
        return [accent.moras[0].index for accent in phrase.accent_phrases[1:]
                if accent.moras]

    def _boundary_hit(self, point):
        if point.y() > self.MARKER_HEIGHT or self.utterance is None:
            return None
        for phrase in self.utterance.phrases:
            for mora_index in self._phrase_boundaries(phrase):
                position = next((i for i, mora in enumerate(self.moras)
                                 if mora.index == mora_index), None)
                if position is None:
                    continue
                x = self._actual_cell_rect(position).left()
                if abs(point.x() - x) <= 7.0:
                    return phrase, mora_index
        return None

    def _nucleus_marker_hit(self, point):
        """Return the accented mora only when the triangle itself is hit."""
        if self.utterance is None:
            return None
        for position, mora in enumerate(self.moras):
            accent = self._accent_for_mora(mora.index)
            if (accent is None or accent.accent_state != "accented" or
                    accent.accent_nucleus is None):
                continue
            local = next((index for index, item in enumerate(accent.moras)
                          if item.index == mora.index), None)
            if local != accent.accent_nucleus:
                continue
            rect = self._cell_rect(position, interactive=True)
            hit = QtCore.QRectF(
                rect.center().x() - 10.0,
                rect.top(),
                20.0,
                17.0,
            )
            if hit.contains(point):
                return accent, mora
        return None

    def _nearest_internal_mora(self, phrase, x):
        candidates = []
        for mora in phrase.moras[1:]:
            position = next((i for i, item in enumerate(self.moras)
                             if item.index == mora.index), None)
            if position is not None:
                candidates.append((
                    abs(self._actual_cell_rect(position).left() - x),
                    mora.index,
                ))
        return min(candidates, default=(None, None))[1]

    def _emit_boundaries(self, phrase, boundaries):
        self.boundariesEdited.emit(
            phrase.index, sorted({int(value) for value in boundaries}))

    def _set_nucleus(self, mora):
        accent = self._accent_for_mora(mora.index)
        if accent is None:
            return
        local = next((i for i, item in enumerate(accent.moras)
                      if item.index == mora.index), None)
        if local is not None:
            self.accentEdited.emit(accent.index, {
                "accent_state": "accented",
                "accent_nucleus": local,
            })

    def _set_unaccented(self, mora):
        accent = self._accent_for_mora(mora.index)
        if accent is not None:
            self.accentEdited.emit(accent.index, {
                "accent_state": "unaccented",
                "accent_nucleus": None,
            })

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        exposed = event.region().boundingRect()
        painter.fillRect(exposed, QtGui.QColor("#F3F1E8"))
        if not self.moras or self.utterance is None:
            painter.setPen(QtGui.QColor("#6D6961"))
            painter.drawText(exposed, Qt.AlignCenter,
                             "Generate Japanese text to show morae")
            return
        first = max(
            0, int(math.floor(
                (float(exposed.left()) - self.CELL_WIDTH) /
                self.CELL_WIDTH)))
        last = min(
            len(self.moras), int(math.ceil(
                (float(exposed.right()) + self.CELL_WIDTH) /
                self.CELL_WIDTH)))
        for start_position, end_position, accent in self._accent_ranges:
            if end_position < first or start_position >= last:
                continue
            start = self._actual_cell_rect(start_position).left() + 3
            end = self._actual_cell_rect(end_position).right() - 3
            y = 24.0
            color = ("#426A8C" if accent.accent_state == "accented" else
                     "#557461" if accent.accent_state == "unaccented" else
                     "#77736B")
            pen = QtGui.QPen(QtGui.QColor(color), 2)
            if accent.accent_state in {"unknown", "unavailable"}:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(start, y),
                             QtCore.QPointF(end, y))
            painter.drawLine(QtCore.QPointF(start, y),
                             QtCore.QPointF(start, y + 6))
            painter.drawLine(QtCore.QPointF(end, y),
                             QtCore.QPointF(end, y + 6))
            if accent.accent_state != "accented" and end - start >= 64:
                painter.setPen(QtGui.QColor(color))
                painter.drawText(
                    QtCore.QRectF(start, 1, end - start, 18),
                    Qt.AlignCenter,
                    "unaccented" if accent.accent_state == "unaccented"
                    else "unknown",
                )
        detailed = self.CELL_WIDTH >= 25.0
        for position in range(first, last):
            mora = self.moras[position]
            selected = (mora.index in self.selected_moras
                        if self.edit_mode == "voicing" else
                        mora.index == self.selected_mora)
            rect = self._cell_rect(position, interactive=selected)
            painter.setBrush(QtGui.QColor(
                "#DCE9F7" if selected else "#FAF9F4"))
            painter.setPen(QtGui.QPen(QtGui.QColor(
                "#316AC5" if selected else "#9A968C"),
                2 if selected else 1))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))
            accent, local = self._accent_by_mora.get(
                mora.index, (None, -1))
            if (accent is not None and accent.accent_state == "accented" and
                    accent.accent_nucleus == local):
                triangle = QtGui.QPolygonF([
                    QtCore.QPointF(rect.center().x() - 5, rect.top() + 4),
                    QtCore.QPointF(rect.center().x() + 5, rect.top() + 4),
                    QtCore.QPointF(rect.center().x(), rect.top() + 11),
                ])
                painter.setPen(Qt.NoPen)
                painter.setBrush(QtGui.QColor("#B4473F"))
                painter.drawPolygon(triangle)
            if detailed or selected:
                painter.setPen(QtGui.QColor("#202020"))
                label = mora.surface or mora.reading or "?"
                painter.drawText(rect.adjusted(3, 14, -3, -26),
                                 Qt.AlignCenter | Qt.TextWordWrap, label)
                painter.setPen(QtGui.QColor("#56524C"))
                phone_text = " ".join(phone.symbol for phone in mora.phones)
                painter.drawText(rect.adjusted(3, 42, -3, -8),
                                 Qt.AlignCenter, phone_text)
                if self.edit_mode == "voicing":
                    row = self.voicing_predictions.get(mora.index, {})
                    if row.get("eligible"):
                        manual = mora.index in self.voicing_overrides
                        degree = float(
                            self.voicing_overrides[mora.index]
                            if manual else row.get(
                                "automatic_voicing",
                                row.get("final_voicing", 1.0)))
                        painter.setPen(QtGui.QColor(
                            "#9A5A16" if manual else "#285F84"))
                        painter.drawText(
                            rect.adjusted(3, 57, -3, -2), Qt.AlignCenter,
                            ("Manual " if manual else "Auto ")
                            + "%d%%" % round(degree * 100.0))
                else:
                    cents = int(self.offsets.get(mora.index, 0))
                    if cents:
                        painter.setPen(QtGui.QColor("#285F84"))
                        painter.drawText(rect.adjusted(3, 57, -3, -2),
                                         Qt.AlignCenter, "%+d ct" % cents)
        if self._drag_target is not None:
            position = self._mora_positions.get(self._drag_target)
            if position is not None:
                x = self._actual_cell_rect(position).center().x()
                painter.setPen(QtGui.QPen(QtGui.QColor("#B4473F"), 2))
                painter.drawLine(QtCore.QPointF(x, 0),
                                 QtCore.QPointF(x, self.MARKER_HEIGHT))

    def mousePressEvent(self, event):
        mora = self._mora_at(event.pos())
        if (self.edit_mode == "voicing" and mora is not None and
                event.button() in {Qt.LeftButton, Qt.RightButton}):
            self._select_voicing_mora(mora, event.modifiers())
            event.accept()
            return
        if event.button() == Qt.RightButton and mora is not None:
            self._suppress_context_menu = True
            self.selected_mora = mora.index
            self.moraSelected.emit(mora.index)
            self._set_unaccented(mora)
            self.update()
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            boundary = self._boundary_hit(event.pos())
            if boundary is not None:
                phrase, mora_index = boundary
                self._drag = ("boundary", phrase.index, mora_index)
                self._drag_target = mora_index
                self._drag_origin = QtCore.QPoint(event.pos())
                self._drag_moved = False
                event.accept()
                return
            if mora is not None:
                self.selected_mora = mora.index
                self.update()
                self.moraSelected.emit(mora.index)
                marker = self._nucleus_marker_hit(event.pos())
                if marker is not None:
                    accent, marker_mora = marker
                    self._drag = (
                        "nucleus", accent.index, marker_mora.index)
                    self._drag_target = marker_mora.index
                    self._drag_origin = QtCore.QPoint(event.pos())
                    self._drag_moved = False
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.edit_mode == "voicing":
            mora = self._mora_at(event.pos())
            if event.button() == Qt.LeftButton and mora is not None:
                self._select_voicing_mora(mora, event.modifiers())
                event.accept()
                return
        if event.button() == Qt.LeftButton:
            mora = self._mora_at(event.pos())
            if mora is not None:
                self.selected_mora = mora.index
                self.moraSelected.emit(mora.index)
                self._set_nucleus(mora)
                self.update()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        kind, owner, _start = self._drag
        if (not self._drag_moved and self._drag_origin is not None and
                (event.pos() - self._drag_origin).manhattanLength() <
                QtWidgets.QApplication.startDragDistance()):
            event.accept()
            return
        self._drag_moved = True
        if kind == "nucleus":
            mora = self._mora_at(event.pos())
            accent = self._accent_for_mora(mora.index) if mora else None
            self._drag_target = (mora.index if accent is not None
                                 and accent.index == owner else None)
        else:
            phrase = next((item for item in self.utterance.phrases
                           if item.index == owner), None)
            self._drag_target = (
                self._nearest_internal_mora(phrase, event.pos().x())
                if phrase is not None else None)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        kind, owner, start = self._drag
        target = self._drag_target
        moved = self._drag_moved
        self._drag = None
        self._drag_target = None
        self._drag_origin = None
        self._drag_moved = False
        if not moved:
            self.update()
            event.accept()
            return
        if kind == "nucleus":
            mora = next((item for item in self.moras
                         if item.index == target), None)
            if mora is not None:
                self._set_nucleus(mora)
            else:
                accent = next((item for item in self.utterance.accent_phrases
                               if item.index == owner), None)
                if accent is not None:
                    self.accentEdited.emit(accent.index, {
                        "accent_state": "unaccented",
                        "accent_nucleus": None,
                    })
        else:
            phrase = next((item for item in self.utterance.phrases
                           if item.index == owner), None)
            if phrase is not None and target is not None:
                boundaries = self._phrase_boundaries(phrase)
                boundaries = [target if value == start else value
                              for value in boundaries]
                self._emit_boundaries(phrase, boundaries)
        self.update()
        event.accept()

    def contextMenuEvent(self, event):
        if self.edit_mode == "voicing":
            event.accept()
            return
        if self._suppress_context_menu:
            self._suppress_context_menu = False
            event.accept()
            return
        mora = self._mora_at(event.pos())
        if mora is None:
            return
        self.selected_mora = mora.index
        self.moraSelected.emit(mora.index)
        phrase = self._phrase_for_mora(mora.index)
        accent = self._accent_for_mora(mora.index)
        if phrase is None or accent is None:
            return
        boundaries = self._phrase_boundaries(phrase)
        phrase_moras = list(phrase.moras)
        position = phrase_moras.index(mora)
        menu = QtWidgets.QMenu()
        if position > 0 and mora.index not in boundaries:
            action = menu.addAction("Split accent phrase before this mora")
            action.triggered.connect(
                lambda _on=False, p=phrase, values=boundaries + [mora.index]:
                self._emit_boundaries(p, values))
        if mora.index in boundaries:
            action = menu.addAction("Merge with previous accent phrase")
            action.triggered.connect(
                lambda _on=False, p=phrase,
                values=[value for value in boundaries
                        if value != mora.index]:
                self._emit_boundaries(p, values))
        next_boundary = next((value for value in boundaries
                              if value > mora.index), None)
        if next_boundary is not None:
            action = menu.addAction("Merge with next accent phrase")
            action.triggered.connect(
                lambda _on=False, p=phrase, target=next_boundary,
                values=boundaries:
                self._emit_boundaries(
                    p, [value for value in values if value != target]))
        if menu.actions():
            menu.addSeparator()
        unaccented = menu.addAction("Mark accent phrase unaccented")
        unaccented.triggered.connect(
            lambda _on=False, item=mora: self._set_unaccented(item))
        menu.exec_(event.globalPos())


class JapaneseEditorPanel(QtWidgets.QWidget):
    editRequested = QtCore.pyqtSignal(str, int, object)
    moraNavigationRequested = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = je.new_edit_state()
        self._utterance = None
        self._selected_mora = -1
        self._selected_moras = set()
        self._edit_mode = "accent"
        self._runtime = {}
        self._updating = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.summary = QtWidgets.QLabel(
            "Generate Japanese text to analyze its mora and accent structure.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.grid = JapaneseMoraGrid()
        self.grid.moraSelected.connect(self._select_mora)
        self.grid.moraSelectionChanged.connect(self._select_moras)
        self.grid.accentEdited.connect(
            lambda index, value:
            self.editRequested.emit("accent", index, value))
        self.grid.boundariesEdited.connect(
            lambda index, value:
            self.editRequested.emit("accent_structure", index, value))
        self.grid.setFixedHeight(112)
        self.grid_scroll = QtWidgets.QScrollArea()
        self.grid_scroll.setWidgetResizable(False)
        self.grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.grid_scroll.setFixedHeight(132)
        self.grid_scroll.setWidget(self.grid)
        layout.addWidget(self.grid_scroll)

        controls = QtWidgets.QGridLayout()
        controls.setHorizontalSpacing(6)
        self.accent_label = QtWidgets.QLabel("Accent phrase:")
        controls.addWidget(self.accent_label, 0, 0)
        self.accent = ArrowComboBox()
        self.accent.currentIndexChanged.connect(self._accent_changed)
        controls.addWidget(self.accent, 0, 1, 1, 2)
        self.reset_accent = QtWidgets.QPushButton("Use analyzed accent")
        self.reset_accent.clicked.connect(self._reset_accent)
        controls.addWidget(self.reset_accent, 0, 3)
        self.split_phrase = QtWidgets.QPushButton("Split before mora")
        self.split_phrase.clicked.connect(self._split_before_mora)
        controls.addWidget(self.split_phrase, 0, 4)
        self.merge_phrase = QtWidgets.QPushButton("Merge previous")
        self.merge_phrase.clicked.connect(self._merge_previous)
        controls.addWidget(self.merge_phrase, 0, 5)

        self.mora_label = QtWidgets.QLabel("Mora:")
        controls.addWidget(self.mora_label, 1, 0)
        self.mora_pitch = QtWidgets.QSpinBox()
        self.mora_pitch.setRange(
            je.PITCH_OFFSET_MIN_CENTS, je.PITCH_OFFSET_MAX_CENTS)
        self.mora_pitch.setSuffix(" cents")
        self.mora_pitch.setSingleStep(10)
        self.mora_pitch.setToolTip(
            "Offset this mora's generated baseline. Continuous pitch points "
            "remain final.")
        self.mora_pitch.valueChanged.connect(self._mora_pitch_changed)
        controls.addWidget(self.mora_pitch, 1, 1)
        self.reset_mora = QtWidgets.QPushButton("Reset mora pitch")
        self.reset_mora.clicked.connect(lambda: self.mora_pitch.setValue(0))
        controls.addWidget(self.reset_mora, 1, 2)
        self.rebuild = QtWidgets.QLabel("")
        self.rebuild.setStyleSheet("color:#7A4D00; font-weight:bold;")
        controls.addWidget(self.rebuild, 1, 3, 1, 3)

        self.baseline_label = QtWidgets.QLabel("Baseline:")
        controls.addWidget(self.baseline_label, 2, 0)
        self.baseline = ArrowComboBox()
        self.baseline.addItem("Structural", "structural")
        self.baseline.addItem("Open JTalk labels", "openjtalk_labels")
        self.baseline.addItem("External HTS trajectory", "external_hts")
        self.baseline.setMinimumWidth(170)
        self.baseline.currentIndexChanged.connect(self._baseline_changed)
        controls.addWidget(self.baseline, 2, 1, 1, 2)
        self.choose_trajectory = QtWidgets.QPushButton("Choose trajectory...")
        self.choose_trajectory.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_DialogOpenButton))
        self.choose_trajectory.clicked.connect(self._choose_trajectory)
        controls.addWidget(self.choose_trajectory, 2, 3, 1, 2)
        self.follow_selection = QtWidgets.QCheckBox("Follow selection")
        self.follow_selection.setChecked(True)
        self.follow_selection.setToolTip(
            "Center the waveform when a selected mora is outside the view")
        controls.addWidget(self.follow_selection, 2, 5)

        self.mora_voicing_label = QtWidgets.QLabel("Mora voicing:")
        controls.addWidget(self.mora_voicing_label, 3, 0)
        self.mora_voicing = QtWidgets.QDoubleSpinBox()
        self.mora_voicing.setRange(0.0, 100.0)
        self.mora_voicing.setDecimals(0)
        self.mora_voicing.setSingleStep(5.0)
        self.mora_voicing.setSuffix(" % voiced")
        self.mora_voicing.setToolTip(
            "Set the voiced-excitation degree for every selected mora. "
            "The detailed continuous Voicing curve remains final.")
        self.mora_voicing.valueChanged.connect(
            self._mora_voicing_changed)
        controls.addWidget(self.mora_voicing, 3, 1, 1, 2)
        self.reset_mora_voicing = QtWidgets.QPushButton(
            "Use automatic voicing")
        self.reset_mora_voicing.clicked.connect(
            self._reset_mora_voicing)
        controls.addWidget(self.reset_mora_voicing, 3, 3)
        self.mora_voicing_diagnostic = QtWidgets.QLabel("")
        self.mora_voicing_diagnostic.setWordWrap(True)
        controls.addWidget(self.mora_voicing_diagnostic, 3, 4, 1, 3)
        controls.setColumnStretch(6, 1)
        layout.addLayout(controls)
        self.set_edit_mode("accent")

    def set_edit_mode(self, mode):
        self._edit_mode = ("mora_voicing" if str(mode) == "mora_voicing"
                           else "accent")
        voicing = self._edit_mode == "mora_voicing"
        self.grid.set_edit_mode(self._edit_mode)
        accent_widgets = (
            self.accent_label, self.accent, self.reset_accent,
            self.split_phrase, self.merge_phrase, self.mora_label,
            self.mora_pitch, self.reset_mora, self.rebuild,
            self.baseline_label, self.baseline, self.choose_trajectory,
        )
        for widget in accent_widgets:
            widget.setVisible(not voicing)
        for widget in (
            self.mora_voicing_label, self.mora_voicing,
            self.reset_mora_voicing, self.mora_voicing_diagnostic,
        ):
            widget.setVisible(voicing)
        self._sync_controls()

    def set_runtime_metadata(self, runtime):
        self._runtime = dict(runtime or {})

    def set_timeline(self, segments, plan_rows):
        self.grid.set_timeline(segments, plan_rows)

    def set_view_range(self, left, right):
        self.grid.set_view_range(left, right)

    def set_playhead(self, value):
        self.grid.set_playhead(value)

    def select_mora(self, mora_index):
        self._selected_mora = int(mora_index)
        self.grid.select_mora(mora_index)
        position = next((i for i, mora in enumerate(self.grid.moras)
                         if mora.index == int(mora_index)), None)
        if position is not None:
            self.grid_scroll.ensureVisible(
                int((position + 0.5) * self.grid.CELL_WIDTH), 56,
                int(self.grid.CELL_WIDTH * 0.55), 0)
        self._updating = True
        try:
            self._sync_controls()
        finally:
            self._updating = False

    def set_state(self, state):
        self._updating = True
        try:
            self._state = je.normalize_edit_state(state)
            raw = self._state.get("utterance")
            try:
                base = je.utterance_from_dict(raw) if raw else None
                self._utterance = (
                    je.apply_linguistic_edits(base, self._state)
                    if base is not None else None)
            except (TypeError, ValueError, KeyError):
                self._utterance = None
            offsets = self._state.get("mora_pitch_offsets_cents") or {}
            plan = dict(self._state.get("last_plan") or {})
            voicing_rows = list(plan.get(
                "mora_voicing_predictions") or [])
            voicing_overrides = dict(
                self._state.get("mora_voicing_overrides") or {})
            self.grid.set_model(
                self._utterance, offsets, self._selected_mora,
                voicing_rows, voicing_overrides)
            self._selected_mora = self.grid.selected_mora
            self._selected_moras = set(self.grid.selected_moras)
            if self._utterance is None:
                self.summary.setText(
                    "Generate Japanese text to analyze its mora and accent "
                    "structure.")
            else:
                diagnostics = len(self._utterance.diagnostics)
                self.summary.setText(
                    "%s frontend  |  %d morae  |  %d accent phrases%s" %
                    (self._utterance.frontend_name,
                     len(self._utterance.moras),
                     len(self._utterance.accent_phrases),
                     "  |  %d diagnostic%s" %
                     (diagnostics, "" if diagnostics == 1 else "s")
                     if diagnostics else ""))
                if any(phrase.interrogative
                       for phrase in self._utterance.phrases):
                    self.summary.setText(
                        self.summary.text() + "  |  interrogative analyzed")
            self.rebuild.setText(
                "Voice rebuild required" if self._state.get(
                    "needs_voice_rebuild") else "")
            self._sync_refinement_controls()
            self._sync_controls()
        finally:
            self._updating = False

    def _selected_objects(self):
        if self._utterance is None:
            return None, None, None
        mora = next((item for item in self._utterance.moras
                     if item.index == self._selected_mora), None)
        if mora is None:
            return None, None, None
        accent = next((item for item in self._utterance.accent_phrases
                       if item.index == mora.accent_phrase_index), None)
        phrase = next((item for item in self._utterance.phrases
                       if item.index == mora.phrase_index), None)
        return mora, accent, phrase

    def _select_mora(self, mora_index):
        self._selected_mora = int(mora_index)
        self._selected_moras = set(self.grid.selected_moras) or {
            self._selected_mora}
        self._updating = True
        try:
            self._sync_controls()
        finally:
            self._updating = False
        self.moraNavigationRequested.emit(self._selected_mora)

    def _select_moras(self, mora_indices):
        self._selected_moras = {int(index) for index in (mora_indices or [])}
        self._updating = True
        try:
            self._sync_controls()
        finally:
            self._updating = False

    def _sync_controls(self):
        mora, accent, phrase = self._selected_objects()
        enabled = mora is not None and accent is not None and phrase is not None
        for widget in (self.accent, self.reset_accent, self.mora_pitch,
                       self.reset_mora, self.split_phrase,
                       self.merge_phrase):
            widget.setEnabled(enabled)
        self.accent.blockSignals(True)
        self.accent.clear()
        if enabled:
            self.accent.addItem("From analysis", ("inherit", None))
            self.accent.addItem("Unaccented", ("unaccented", None))
            for local, item in enumerate(accent.moras):
                label = item.surface or item.reading or str(local + 1)
                self.accent.addItem(
                    "Accent %d: %s" % (local + 1, label),
                    ("accented", local))
            raw_override = dict((self._state.get("accent_overrides") or {})
                                .get(str(accent.index)) or {})
            wanted = 0
            if raw_override:
                if raw_override.get("accent_state") == "unaccented":
                    wanted = 1
                elif raw_override.get("accent_state") == "accented":
                    wanted = 2 + int(raw_override.get("accent_nucleus") or 0)
            self.accent.setCurrentIndex(max(
                0, min(self.accent.count() - 1, wanted)))
            cents = int((self._state.get("mora_pitch_offsets_cents") or {})
                        .get(str(mora.index), 0))
            self.mora_pitch.setValue(cents)
            self.mora_label.setText(
                "Mora: %s" % (mora.surface or mora.reading or mora.index))
            phrase_moras = list(phrase.moras)
            position = phrase_moras.index(mora)
            boundaries = self._current_boundaries(phrase)
            self.split_phrase.setEnabled(
                position > 0 and mora.index not in boundaries)
            self.merge_phrase.setEnabled(mora.index in boundaries)
        else:
            self.accent.addItem("No Japanese analysis", None)
            self.mora_label.setText("Mora:")
            self.mora_pitch.setValue(0)
        self.accent.blockSignals(False)
        self._sync_mora_voicing_controls()

    def _voicing_prediction(self, mora_index):
        return self.grid.voicing_predictions.get(int(mora_index), {})

    def _eligible_selected_moras(self):
        selected = self._selected_moras or (
            {self._selected_mora} if self._selected_mora >= 0 else set())
        return sorted(index for index in selected
                      if self._voicing_prediction(index).get("eligible"))

    def _sync_mora_voicing_controls(self):
        selected = self._eligible_selected_moras()
        overrides = {
            int(key): float(value) for key, value in
            dict(self._state.get("mora_voicing_overrides") or {}).items()
        }
        self.mora_voicing.blockSignals(True)
        try:
            if not selected:
                self.mora_voicing.setValue(100.0)
                self.mora_voicing.setEnabled(False)
                self.reset_mora_voicing.setEnabled(False)
                self.mora_voicing_diagnostic.setText(
                    "No editable vowel in the selected mora.")
                return
            automatic = [float(self._voicing_prediction(index).get(
                "automatic_voicing", 1.0)) for index in selected]
            final = [overrides.get(index, automatic[position])
                     for position, index in enumerate(selected)]
            self.mora_voicing.setValue(final[0] * 100.0)
            self.mora_voicing.setEnabled(True)
            self.reset_mora_voicing.setEnabled(
                any(index in overrides for index in selected))
            auto_text = ("%d%%" % round(automatic[0] * 100.0)
                         if max(automatic) - min(automatic) < 0.005 else
                         "%d-%d%%" % (round(min(automatic) * 100.0),
                                      round(max(automatic) * 100.0)))
            manual_count = sum(index in overrides for index in selected)
            label = "%d mora%s | automatic %s" % (
                len(selected), "" if len(selected) == 1 else "e", auto_text)
            if manual_count:
                label += " | %d overridden" % manual_count
            reasons = self._voicing_prediction(selected[0]).get(
                "reasons") or []
            if len(selected) == 1 and reasons:
                label += " | " + str(reasons[0])
            self.mora_voicing_diagnostic.setText(label)
        finally:
            self.mora_voicing.blockSignals(False)

    def _mora_voicing_changed(self, value):
        if self._updating:
            return
        indices = self._eligible_selected_moras()
        if indices:
            self.editRequested.emit("mora_voicing", -1, {
                "mora_indices": indices,
                "value": max(0.0, min(1.0, float(value) / 100.0)),
            })

    def _reset_mora_voicing(self):
        indices = self._eligible_selected_moras()
        if indices:
            self.editRequested.emit("mora_voicing", -1, {
                "mora_indices": indices,
                "value": None,
            })

    def _sync_refinement_controls(self):
        widgets = (self.baseline,)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            baseline = str(self._state.get("baseline_provider") or
                           "structural")
            index = self.baseline.findData(baseline)
            self.baseline.setCurrentIndex(max(0, index))
            external = baseline == "external_hts"
            self.choose_trajectory.setVisible(external)
            path = str(self._state.get("external_hts_trajectory") or "")
            self.choose_trajectory.setToolTip(
                path or "Select a Japanese trajectory JSON file")
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def _accent_changed(self, _index):
        if self._updating:
            return
        _mora, accent, _phrase = self._selected_objects()
        if accent is None:
            return
        data = self.accent.currentData()
        if not data or data[0] == "inherit":
            value = None
        else:
            value = {"accent_state": data[0],
                     "accent_nucleus": data[1]}
        self.editRequested.emit("accent", accent.index, value)

    def _reset_accent(self):
        _mora, accent, _phrase = self._selected_objects()
        if accent is not None:
            self.editRequested.emit("accent", accent.index, None)

    def _current_boundaries(self, phrase):
        values = (self._state.get("accent_phrase_boundaries") or {}).get(
            str(phrase.index))
        if values is not None:
            return sorted({int(value) for value in values})
        return [accent.moras[0].index
                for accent in phrase.accent_phrases[1:] if accent.moras]

    def _split_before_mora(self):
        mora, _accent, phrase = self._selected_objects()
        if mora is None or phrase is None or mora == phrase.moras[0]:
            return
        boundaries = self._current_boundaries(phrase)
        if mora.index not in boundaries:
            boundaries.append(mora.index)
            self.editRequested.emit(
                "accent_structure", phrase.index, sorted(boundaries))

    def _merge_previous(self):
        mora, _accent, phrase = self._selected_objects()
        if mora is None or phrase is None:
            return
        boundaries = self._current_boundaries(phrase)
        if mora.index in boundaries:
            boundaries.remove(mora.index)
            self.editRequested.emit(
                "accent_structure", phrase.index, boundaries)

    def _mora_pitch_changed(self, value):
        if not self._updating and self._selected_mora >= 0:
            self.editRequested.emit(
                "mora_pitch", self._selected_mora, int(value))

    def _baseline_changed(self, _index):
        if self._updating:
            return
        mode = str(self.baseline.currentData() or "structural")
        self.choose_trajectory.setVisible(mode == "external_hts")
        self.editRequested.emit("baseline_provider", -1, mode)

    def _choose_trajectory(self):
        start = str(self._state.get("external_hts_trajectory") or "")
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose Japanese HTS trajectory", start,
            "JSON trajectory (*.json);;All files (*)")
        if path:
            self.editRequested.emit("external_hts_trajectory", -1, path)

class JapaneseBankAnalysisDialog(QtWidgets.QDialog):
    """Read-only bank preview with explicit, external profile overrides."""

    def __init__(self, analysis, suggested_profile="", parent=None):
        super().__init__(parent)
        self.analysis = analysis
        self.profile = analysis.profile
        self.profile_changed = False
        self.profile_path = str(suggested_profile or "")
        self.setWindowTitle("Japanese UTAU bank analysis")
        self.resize(820, 540)
        layout = QtWidgets.QVBoxLayout(self)
        self.source = QtWidgets.QLabel(
            "Source (read-only): " + analysis.source_path)
        self.source.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.source)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Bank type:"))
        self.configuration = ArrowComboBox()
        for value in japanese_profiles.BANK_CONFIGURATIONS:
            self.configuration.addItem(value.upper(), value)
        wanted = self.configuration.findData(self.profile.bank_configuration)
        self.configuration.setCurrentIndex(max(0, wanted))
        self.configuration.currentIndexChanged.connect(
            self._configuration_changed)
        row.addWidget(self.configuration)
        self.summary = QtWidgets.QLabel()
        row.addWidget(self.summary, 1)
        layout.addLayout(row)
        self.coverage = QtWidgets.QLabel()
        self.coverage.setWordWrap(True)
        layout.addWidget(self.coverage)
        layout.addWidget(QtWidgets.QLabel(
            "Unresolved aliases are preserved. Resolve only entries whose "
            "linguistic role you know; applying profile changes requires a "
            "voice rebuild."))
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Alias", "OTO source", "Reason", "Role", "Resolution"])
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.doubleClicked.connect(self._resolve_selected)
        layout.addWidget(self.table, 1)
        commands = QtWidgets.QHBoxLayout()
        self.inspect = QtWidgets.QPushButton("Inspect source entry")
        self.inspect.clicked.connect(self._inspect_selected)
        commands.addWidget(self.inspect)
        self.resolve = QtWidgets.QPushButton("Resolve selected alias...")
        self.resolve.clicked.connect(self._resolve_selected)
        commands.addWidget(self.resolve)
        self.reanalyze = QtWidgets.QPushButton("Re-analyze with profile")
        self.reanalyze.clicked.connect(self._reanalyze)
        commands.addWidget(self.reanalyze)
        commands.addStretch(1)
        self.save_profile = QtWidgets.QPushButton("Save profile...")
        self.save_profile.clicked.connect(self._save_profile)
        commands.addWidget(self.save_profile)
        self.apply = QtWidgets.QPushButton("Apply to project")
        self.apply.clicked.connect(self._apply)
        commands.addWidget(self.apply)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        commands.addWidget(close)
        layout.addLayout(commands)
        self._refresh()

    def _refresh(self):
        report = self.analysis.graph.coverage
        self.summary.setText(
            "%s inferred (%.0f%% confidence)" %
            (self.profile.effective_configuration.upper(),
             self.profile.inference_confidence * 100.0))
        self.coverage.setText(
            "%d source entries  |  %d candidates  |  %d unresolved (%.1f%%)"
            "  |  traceable: %s" %
            (report.source_entry_count, report.candidate_count,
             report.unresolved_count, report.unresolved_rate * 100.0,
             "yes" if report.all_entries_traceable else "NO"))
        rows = list(self.analysis.unresolved)
        self.table.setRowCount(len(rows))
        for row, candidate in enumerate(rows):
            key = "%s:%d" % (candidate.source.oto_path,
                              candidate.source.line)
            override = self.profile.alias_overrides.get(key)
            values = (
                candidate.source.alias_raw,
                "%s:%d" % (candidate.source.oto_path,
                             candidate.source.line),
                "; ".join(candidate.reasons[:2]),
                override.role if override else candidate.role,
                "profile override" if override else "unresolved",
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, candidate.candidate_id)
                self.table.setItem(row, column, item)
        available = bool(rows)
        self.inspect.setEnabled(available)
        self.resolve.setEnabled(available)

    def _selected_candidate(self):
        row = self.table.currentRow()
        if row < 0 or self.table.item(row, 0) is None:
            return None
        candidate_id = self.table.item(row, 0).data(Qt.UserRole)
        return next((item for item in self.analysis.graph.candidates
                     if item.candidate_id == candidate_id), None)

    def _configuration_changed(self, _index):
        value = str(self.configuration.currentData() or "auto")
        if value != self.profile.bank_configuration:
            self.profile = replace(self.profile, bank_configuration=value)
            self.profile_changed = True
            self.summary.setText(value.upper() + " override - re-analyze")

    def _inspect_selected(self):
        candidate = self._selected_candidate()
        if candidate is None:
            return
        QtWidgets.QMessageBox.information(
            self, "Source OTO entry",
            "Alias: %s\nOTO: %s:%d\nWAV: %s\nEncoding: %s\n\n"
            "Current role: %s\nReasons: %s" %
            (candidate.source.alias_raw, candidate.source.oto_path,
             candidate.source.line, candidate.source.wav_path or
             candidate.source.wav_raw, candidate.source.oto_encoding,
             candidate.role, "; ".join(candidate.reasons) or "none"))

    def _resolve_selected(self, *_args):
        candidate = self._selected_candidate()
        if candidate is None:
            return
        roles = [role for role in japanese_profiles.PROFILE_ROLES
                 if role != "unresolved"]
        role, ok = QtWidgets.QInputDialog.getItem(
            self, "Resolve alias", "Linguistic role:", roles, 0, False)
        if not ok:
            return
        target, ok = QtWidgets.QInputDialog.getText(
            self, "Canonical target",
            "Mora or context phone (leave blank only for breath/extra):")
        if not ok:
            return
        role = str(role)
        family = ("cv" if role in {
                      "mora_cv", "phrase_start_cv", "vowel_blend"
                  } else
                  "vcv" if role == "vcv_mora" else
                  "cvvc" if role in {"vc_transition", "release"} else
                  "extra")
        values = {"role": role, "family": family}
        target = str(target).strip() or None
        if role in {"mora_cv", "phrase_start_cv", "vowel_blend", "vcv_mora",
                    "special_mora"}:
            values["mora"] = target
        elif role == "vc_transition":
            values["left_context"] = target
        elif role == "release":
            values["right_context"] = target
        try:
            override = japanese_profiles.JapaneseAliasOverride(**values)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Alias override", str(error))
            return
        key = "%s:%d" % (candidate.source.oto_path,
                          candidate.source.line)
        self.profile = je.profile_with_override(self.profile, key, override)
        self.profile_changed = True
        self._refresh()

    def _reanalyze(self):
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.analysis = je.analyze_bank(
                self.analysis.source_path, profile=self.profile)
            self.profile = self.analysis.profile
        except Exception as error:
            QtWidgets.QMessageBox.critical(
                self, "Japanese bank analysis", str(error))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._refresh()

    def _save_profile(self):
        start = self.profile_path or "japanese-bank-profile.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Japanese bank profile", start, "JSON (*.json)")
        if not path:
            return
        try:
            japanese_profiles.write_profile(
                self.profile, Path(path),
                source_root=self.analysis.graph._bank_root)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.critical(self, "Save profile", str(error))
            return
        self.profile_path = str(Path(path).resolve())

    def _apply(self):
        try:
            self.analysis = je.analyze_bank(
                self.analysis.source_path, profile=self.profile)
            self.profile = self.analysis.profile
        except Exception as error:
            QtWidgets.QMessageBox.critical(
                self, "Japanese bank analysis", str(error))
            return
        self.accept()


class PhrasePauseDialog(QtWidgets.QDialog):
    """Semantic phrase-break durations shared by every text frontend."""

    LABELS = (
        ("minor", "Minor phrase pause:"),
        ("major", "Major phrase pause:"),
        ("sentence", "Sentence pause:"),
    )

    def __init__(self, values=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phrase pauses")
        layout = QtWidgets.QFormLayout(self)
        normalized = fc.normalize_phrase_pauses_ms(values)
        self.spins = {}
        for key, label in self.LABELS:
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 2000)
            spin.setSingleStep(10)
            spin.setSuffix(" ms")
            spin.setValue(normalized[key])
            spin.setAccessibleName(label.rstrip(":"))
            self.spins[key] = spin
            layout.addRow(label, spin)
        note = QtWidgets.QLabel(
            "These are total linguistic pause durations. Generated Festival "
            "text can apply changes with Re-render; protected unit-edge "
            "segments remain an internal detail.")
        note.setWordWrap(True)
        layout.addRow(note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok |
            QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self):
        return {key: spin.value() for key, spin in self.spins.items()}


# --------------------------------------------------------------------- main win
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.cfg = cfg
        self.player = Player()
        self.undo_stack = QtWidgets.QUndoStack(self)
        try:
            self.undo_stack.setUndoLimit(max(
                1, int(self.cfg.get("undo_limit", 64))))
        except (TypeError, ValueError):
            self.undo_stack.setUndoLimit(64)
        self._applying_undo = False
        self.shortcuts = dict(DEFAULT_SHORTCUTS)
        self.shortcuts.update({
            str(key): str(value) for key, value in
            dict(self.cfg.get("shortcuts") or {}).items()
            if key in DEFAULT_SHORTCUTS})
        self._shortcut_lookup = {}
        self._shortcut_menu_actions = {}
        self._shortcut_hover_context = ""
        self._playback_timer = QtCore.QTimer(self)
        self._playback_timer.setInterval(30)
        self._playback_timer.timeout.connect(self._advance_playhead)
        self._playback_finish_timer = QtCore.QTimer(self)
        self._playback_finish_timer.setSingleShot(True)
        self._playback_finish_timer.timeout.connect(
            self._finish_scheduled_playback)
        self._playback_finish_token = 0
        self._playback_elapsed = QtCore.QElapsedTimer()
        self._playback_timeline_start = None
        self._playback_highlights = []
        self._playback_token = 0
        self._playback_active = False
        self._batch_active = False
        self._batch_cancel_requested = False
        self._synthesis_busy = False
        self._synthesis_thread = None
        self._synthesis_worker = None
        self._synthesis_busy_states = []
        self._close_requested = False
        self._shutdown_complete = False
        self._generation_progress_labels = []
        self.current = None            # fc.Synthesis of the last render
        self.sentences = []
        self._active_sentence_index = -1
        self._editor_sentence_state = None
        self._switching_sentence = False
        self._sentence_sidebar_syncing = False
        self._mode_tab_index = 0
        self._pitch_fault_target = None
        self.user_dicts = {}           # lang_code -> {word: [phones]} overrides
        self._variant_cache = OrderedDict()  # (engine, voice, token) -> takes
        self._speech_clipboard = None
        self._project_clipboard = None
        self._project_root = ""
        self._voice_dict_loaded = None
        self._pitch_user_edited = False
        self._syncing_japanese_selection = False
        self._voice_root_watcher = QtCore.QFileSystemWatcher(self)
        self._voice_root_watcher.directoryChanged.connect(
            self._schedule_voice_root_refresh)
        self._voice_root_refresh_timer = QtCore.QTimer(self)
        self._voice_root_refresh_timer.setSingleShot(True)
        self._voice_root_refresh_timer.setInterval(350)
        self._voice_root_refresh_timer.timeout.connect(
            self._refresh_watched_voice_roots)
        self.backend = None            # diphone engine (synth_diphone.py)
        self.backend_err = None
        self.fest = fc.FestivalWSLBackend(cfg)  # real Festival inside WSL
        self.setWindowTitle(
            "FestVox Speech GUI v3.0 -- diphone + Festival/WSL")
        self.resize(1024, 680)
        self._init_backend()
        self._build_menu()
        self._build_body()
        self.waveform.set_join_overlay_visible(
            self.action_show_rendered_joins.isChecked())
        self._allow_output_clipping = bool(
            self.cfg.get("allow_output_clipping", False))
        self.speech_gain.set_allow_clipping(
            self._allow_output_clipping, emit=False)
        self.sentences_view.gain.set_allow_clipping(
            self._allow_output_clipping, emit=False)
        self.shortcut_hint = QtWidgets.QLabel()
        self.shortcut_hint.setStyleSheet("color:#4F4B45; padding-right:4px;")
        self.generation_progress = QtWidgets.QProgressBar()
        self.generation_progress.setObjectName("generationProgress")
        self.generation_progress.setTextVisible(True)
        self.generation_progress.setFixedWidth(230)
        self.generation_progress.setAccessibleName(
            "Current generation progress")
        self.generation_progress.hide()
        self.batch_progress = QtWidgets.QProgressBar()
        self.batch_progress.setObjectName("batchProgress")
        self.batch_progress.setTextVisible(True)
        self.batch_progress.setFixedWidth(230)
        self.batch_progress.setAccessibleName("Total generation progress")
        self.batch_progress.hide()
        self.synthesis_progress_stack = QtWidgets.QWidget()
        self.synthesis_progress_stack.setObjectName(
            "synthesisProgressStack")
        self._synthesis_progress_height = max(
            18, int(self.generation_progress.sizeHint().height()))
        if self._synthesis_progress_height % 2:
            self._synthesis_progress_height += 1
        self.synthesis_progress_stack.setFixedHeight(
            self._synthesis_progress_height)
        progress_layout = QtWidgets.QVBoxLayout(
            self.synthesis_progress_stack)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(0)
        progress_layout.addWidget(self.generation_progress)
        progress_layout.addWidget(self.batch_progress)
        self.synthesis_progress_stack.hide()
        self.batch_cancel = QtWidgets.QToolButton()
        self.batch_cancel.setObjectName("batchCancel")
        self.batch_cancel.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_BrowserStop))
        self.batch_cancel.setToolTip("Stop after the current sentence")
        self.batch_cancel.clicked.connect(self._request_batch_cancel)
        self.batch_cancel.hide()
        self.statusBar().addPermanentWidget(self.synthesis_progress_stack)
        self.statusBar().addPermanentWidget(self.batch_cancel)
        self.statusBar().addPermanentWidget(self.shortcut_hint)
        self._rebuild_shortcut_lookup()
        self._register_shortcut_contexts()
        self._populate_from_backend()
        self._configure_voice_root_watcher()
        self._load_dicts_from_cfg()
        self._init_sentence_collection()
        self._refresh_gain_controls()
        self._refresh_pending_ui()
        self._install_shortcuts()
        self._update_shortcut_hints()
        self.statusBar().showMessage("Status: " + (
            "Ready" if (self.backend or self._engine() == "festival_wsl") else
            "diphone engine not loaded -- " + str(self.backend_err)))

    # -- backends ---------------------------------------------------------------
    def _init_backend(self):
        try:
            self.backend = fc.DiphoneBackend(self.cfg)
            self.backend_err = None
        except fc.BackendError as e:
            self.backend = None
            self.backend_err = e

    def _engine(self) -> str:
        if hasattr(self, "engine") and self.engine.currentIndex() >= 0:
            return self.engine.currentData() or "diphone"
        return str(self.cfg.get("engine") or "diphone")

    def _ab(self):
        """Backend for the currently selected engine."""
        return self.fest if self._engine() == "festival_wsl" else self.backend

    def _backend_for_engine(self, engine):
        return self.fest if str(engine) == "festival_wsl" else self.backend

    def _need_backend(self) -> bool:
        if self._engine() == "festival_wsl":
            return True   # WSL problems surface per-call with full details
        if self.backend:
            return True
        QtWidgets.QMessageBox.critical(
            self, "Engine not loaded",
            str(self.backend_err or "synth_diphone.py is not loaded.") +
            "\n\nUse Options > Locate synth_diphone.py...")
        return False

    def _synthesis_lock_objects(self):
        """Controls whose mutation would invalidate an in-flight request."""
        names = [
            "engine", "lang", "voicebank", "speed", "speed_val",
            "speech_gain", "pitch", "fall",
            "sentence_add", "sentence_duplicate", "sentence_remove",
            "btn_gen", "btn_gen_all",
            "btn_rerender", "btn_rerender_all",
        ]
        if not self._batch_active:
            names.extend(("sentence_select", "input_mode", "text"))
        objects = [getattr(self, name, None) for name in names]
        objects.extend(
            getattr(self, name, None)
            for name in ("action_generate", "action_rerender")
        )
        result = []
        seen = set()
        for obj in objects:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            result.append(obj)
        return result

    @staticmethod
    def _control_explicitly_enabled(control):
        """Return local state, ignoring a disabled ancestor widget.

        QWidget.isEnabled() reports the effective inherited state. Capturing
        that while the sentence sidebar has no selection used to turn a
        temporary parent disable into a permanent child disable after a
        synthesis worker completed.
        """
        if isinstance(control, QtWidgets.QWidget):
            return not control.testAttribute(Qt.WA_ForceDisabled)
        return bool(control.isEnabled())

    def _refresh_contextual_control_availability(self):
        if not hasattr(self, "sidebar_editor"):
            return
        sentence_tab = (
            hasattr(self, "mode_tabs") and
            self.mode_tabs.currentIndex() == 1)
        selected = (
            self.sentences_view.selected_sentence_indices()
            if sentence_tab and hasattr(self, "sentences_view") else [])
        self.sidebar_editor.setEnabled(not sentence_tab or bool(selected))
        if hasattr(self, "pitch"):
            self.pitch.setEnabled(self._engine() == "festival_wsl")

    def _set_synthesis_busy(self, busy):
        busy = bool(busy)
        if busy == self._synthesis_busy:
            return
        self._synthesis_busy = busy
        if busy:
            self._synthesis_busy_states = [
                (obj, self._control_explicitly_enabled(obj))
                for obj in self._synthesis_lock_objects()
            ]
            for obj, _enabled in self._synthesis_busy_states:
                obj.setEnabled(False)
        else:
            for obj, enabled in self._synthesis_busy_states:
                try:
                    obj.setEnabled(enabled)
                except RuntimeError:
                    pass
            self._synthesis_busy_states = []
            self._refresh_contextual_control_availability()
            self._update_parameter_availability()
            self._update_fault_availability()
            self._refresh_gain_controls()
            self._refresh_pending_ui()

    def _refresh_synthesis_progress_visibility(self):
        current_visible = not self.generation_progress.isHidden()
        total_visible = not self.batch_progress.isHidden()
        visible = current_visible or total_visible
        if current_visible and total_visible:
            half = self._synthesis_progress_height // 2
            self.generation_progress.setFixedHeight(half)
            self.batch_progress.setFixedHeight(half)
            # Native progress-bar text is clipped at half height. The status
            # message and accessible names retain the same information.
            self.generation_progress.setTextVisible(False)
            self.batch_progress.setTextVisible(False)
        else:
            if current_visible:
                self.generation_progress.setFixedHeight(
                    self._synthesis_progress_height)
            if total_visible:
                self.batch_progress.setFixedHeight(
                    self._synthesis_progress_height)
            self.generation_progress.setTextVisible(True)
            self.batch_progress.setTextVisible(True)
        self.synthesis_progress_stack.setVisible(visible)

    def _begin_generation_progress(self, label):
        label = str(label or "Generating sentence...")
        self._generation_progress_labels.append(label)
        self.generation_progress.setRange(0, 0)
        self.generation_progress.setFormat(label)
        self.generation_progress.setToolTip(label)
        self.generation_progress.show()
        self._refresh_synthesis_progress_visibility()
        QtWidgets.QApplication.processEvents()

    def _end_generation_progress(self):
        if self._generation_progress_labels:
            self._generation_progress_labels.pop()
        if self._generation_progress_labels:
            self.generation_progress.setFormat(
                self._generation_progress_labels[-1])
        else:
            self.generation_progress.hide()
        self._refresh_synthesis_progress_visibility()

    def _run_synthesis_task(self, callback):
        """Run blocking synthesis while a nested Qt loop keeps UI painting."""
        app = QtWidgets.QApplication.instance()
        if (app is None or QtCore.QThread.currentThread() is not app.thread()):
            return callback()
        if self._synthesis_busy:
            return callback()

        thread = QtCore.QThread()
        worker = _SynthesisTask(callback)
        worker.moveToThread(thread)
        event_loop = QtCore.QEventLoop(self)
        result, failure = [], []

        worker.succeeded.connect(
            lambda value: result.append(value), Qt.DirectConnection)
        worker.failed.connect(
            lambda error: failure.append(error), Qt.DirectConnection)
        worker.succeeded.connect(thread.quit, Qt.DirectConnection)
        worker.failed.connect(thread.quit, Qt.DirectConnection)
        worker.succeeded.connect(event_loop.quit, Qt.QueuedConnection)
        worker.failed.connect(event_loop.quit, Qt.QueuedConnection)
        thread.finished.connect(event_loop.quit, Qt.QueuedConnection)
        thread.started.connect(worker.run)

        self._synthesis_thread = thread
        self._synthesis_worker = worker
        self._set_synthesis_busy(True)
        try:
            thread.start()
            event_loop.exec_()
            thread.quit()
            thread.wait()
        finally:
            self._synthesis_worker = None
            self._synthesis_thread = None
            self._set_synthesis_busy(False)
            thread.deleteLater()
            if self._close_requested:
                QtCore.QTimer.singleShot(0, self.close)
        if failure:
            raise failure[0]
        if not result:
            raise RuntimeError("The synthesis worker exited without a result.")
        return result[0]

    def _call_synthesis_backend(self, method, *args, **kwargs):
        return self._run_synthesis_task(
            lambda: method(*args, **kwargs))

    # -- pronunciation dictionaries ---------------------------------------------
    def _active_dict(self):
        """The loaded grapheme->phoneme override dict for the current
        language, or None."""
        return self.user_dicts.get(self._current_lang_code()) or None

    def _expand_inline_phones(self, text):
        """Replace [f ow n z] bracket groups in the text with pseudo-words whose
        pronunciation is those literal phones (injected as Festival lexicon
        addenda), so exact phones -- or [pau] for a forced pause -- can be
        dropped into normal text (handy for rare names). Returns
        (text_with_pseudowords, {pseudoword: [phones]})."""
        import re
        extra, n = {}, [0]

        def repl(m):
            phones = [p for p in m.group(1).split() if p]
            if not phones:
                return " "
            tok = "qphon%dx" % n[0]
            n[0] += 1
            extra[tok] = phones
            return " " + tok + " "

        return re.sub(r"\[([^\]]*)\]", repl, text), extra

    def _supported_inline_phones(self, voicebank, code, backend=None):
        """Return the selected voice's declared literal-phone inventory.

        Kal's ``radio`` phoneset remains the compatibility fallback for old or
        built-in English voices. Generated voices are authoritative about
        their own superset, so integrated phones such as ``q`` must not be
        discarded merely because Kal supplies the English text frontend.
        """
        backend = backend if backend is not None else self._ab()
        read_metadata = getattr(backend, "voice_metadata", None)
        if callable(read_metadata):
            try:
                metadata = read_metadata(str(voicebank or ""))
            except (fc.BackendError, OSError, TypeError, ValueError):
                metadata = {}
            declared = frozenset(
                str(phone).strip() for phone in
                (dict(metadata or {}).get("phones") or ())
                if str(phone).strip()
            )
            if declared:
                return declared
        return EN_PHONES if str(code).casefold() == "en" else None

    def _prepare_inline_phones(self, text, code, voicebank, backend=None):
        """Expand inline phones and remove only symbols the voice rejects."""
        expanded, extra = self._expand_inline_phones(text)
        allowed = self._supported_inline_phones(
            voicebank, code, backend=backend)
        dropped = set()
        if allowed is None:
            return expanded, extra, dropped
        for token in list(extra):
            keep = [phone for phone in extra[token] if phone in allowed]
            dropped.update(
                phone for phone in extra[token] if phone not in allowed)
            if keep:
                extra[token] = keep
            else:
                del extra[token]
                expanded = expanded.replace(token, " ")
        return expanded, extra, dropped

    def _fest_dict_phones(self, text, code, vb, udict):
        """Text -> phone list using the user dictionary for covered words and
        Festival's own lexicon/LTS for the rest. Synthesising these phones
        directly makes the dictionary pronunciations VERBATIM -- the phone path
        skips Festival's postlexical function-word reduction (which was turning
        'this' dh ih s back into dh ax s)."""
        import re
        words = re.findall(r"[A-Za-z0-9’'\-]+", text)
        keys = [w.lower().strip("'’-.") for w in words]
        oov = [k for k in dict.fromkeys(keys) if k and k not in udict]
        pron = self.fest.phonemize(oov, vb, code) if oov else {}
        phones = []
        for k in keys:
            if k:
                phones.extend(udict.get(k) or pron.get(k) or [])
        return phones

    def _load_dicts_from_cfg(self):
        for code, path in dict(self.cfg.get("user_dicts") or {}).items():
            try:
                d = fc.parse_utau_dict(path)
                if d:
                    self.user_dicts[code] = d
            except Exception:
                pass   # a missing/broken saved path must not block startup

    def on_load_dict(self):
        lang = self.lang.currentText()
        code = self._current_lang_code()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load pronunciation dictionary for %s" % lang, "",
            "Dictionaries (*.yaml *.yml *.dict *.dic *.txt);;All files (*)")
        if not path:
            return
        self.statusBar().showMessage("Status: parsing dictionary (large "
                                     "files take a moment)...")
        QtWidgets.QApplication.processEvents()
        try:
            d = fc.parse_utau_dict(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Dictionary", str(e))
            return
        if not d:
            QtWidgets.QMessageBox.warning(
                self, "Dictionary", "No grapheme->phoneme entries found "
                "in that file (expected an OpenUTAU .yaml 'entries:' section "
                "or a 'word ph ph' text list).")
            return
        self.user_dicts[code] = d
        self.cfg.setdefault("user_dicts", {})[code] = path
        try:
            fc.save_config(self.cfg, CONFIG_PATH)
        except Exception:
            pass
        QtWidgets.QMessageBox.information(
            self, "Dictionary",
            "Loaded %d pronunciations for %s.\n\nWords you synthesize that are "
            "in this dictionary now use its pronunciation (e.g. English "
            "'in' -> ih n); other words use the engine's own g2p.\n\n"
            "The UTAU phoneme/timing definitions in the file are ignored."
            % (len(d), lang))
        self.statusBar().showMessage(
            "Status: dictionary loaded for %s -- %d words" % (lang, len(d)))

    def on_clear_dict(self):
        code = self._current_lang_code()
        self.user_dicts.pop(code, None)
        key = self._voice_dictionary_key()
        if key:
            self.cfg.setdefault("voice_dictionaries", {}).pop(key, None)
            self._voice_dict_loaded = None
        if isinstance(self.cfg.get("user_dicts"), dict):
            self.cfg["user_dicts"].pop(code, None)
            try:
                fc.save_config(self.cfg, CONFIG_PATH)
            except Exception:
                pass
        self.statusBar().showMessage(
            "Status: dictionary cleared for %s" % self.lang.currentText())

    def _voice_dictionary_key(self):
        voice = self._current_voicebank()
        if not voice:
            return ""
        return "%s|%s|%s" % (self._engine(), voice,
                              self._current_lang_code())

    def on_install_voice_dictionary(self):
        voice = self._current_voicebank()
        if not voice:
            QtWidgets.QMessageBox.information(
                self, "Install dictionary", "Select a voicebank first.")
            return
        source, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Install pronunciation dictionary", "",
            "Dictionaries (*.yaml *.yml *.dict *.dic *.txt);;All files (*)")
        if not source:
            return
        try:
            entries = fc.parse_utau_dict(source)
            if not entries:
                raise fc.BackendError(
                    "No grapheme-to-phoneme entries were found.")
            installed = self._ab().install_dictionary(
                voice, os.path.basename(source), entries)
        except (fc.BackendError, OSError, ValueError, AttributeError) as error:
            QtWidgets.QMessageBox.critical(
                self, "Install dictionary", str(error))
            return
        key = self._voice_dictionary_key()
        self.cfg.setdefault("voice_dictionaries", {})[key] = installed
        code = self._current_lang_code()
        self.user_dicts[code] = entries
        self._voice_dict_loaded = key
        self._persist_config()
        self.statusBar().showMessage(
            "Status: installed %d cleaned entries in %s" %
            (len(entries), installed))

    def _speaker_portrait_path(self, engine, voice):
        portraits = dict(self.cfg.get("voice_portraits") or {})
        key = "%s|%s" % (str(engine or ""), str(voice or ""))
        if key in portraits:
            return str(portraits.get(key) or "")
        # Preserve pre-per-voice configs, but never let one mapped speaker's
        # legacy image leak onto every other speaker.
        return str(self.cfg.get("speaker_portrait") or "") if not portraits \
            else ""

    def _sync_speaker_control(self):
        if not hasattr(self, "speaker_portrait"):
            return
        item = self.voicebank.currentItem() if hasattr(
            self, "voicebank") else None
        if (item is not None and
                item.data(Qt.UserRole) == MIXED_SELECTION_DATA):
            self.speaker_portrait.set_portrait("", "Multiple voices")
            return
        voice = self._current_voicebank() or ""
        portrait = self._speaker_portrait_path(self._engine(), voice)
        self.speaker_portrait.set_portrait(portrait, voice or "Voice")

    def _on_voicebank_changed(self, *_args):
        if not hasattr(self, "voicebank"):
            return
        if self._sentence_sidebar_syncing:
            return
        # A builder may replace metadata in-place while the registration and
        # selected list item remain unchanged. Re-read compatibility whenever
        # selection is refreshed so newly enabled Japanese support immediately
        # unlocks the sentence-scoped accent editor.
        backend = self._ab()
        invalidate = getattr(backend, "invalidate_voice_metadata", None)
        if invalidate is not None:
            invalidate(self._current_voicebank() or "")
        if self._switching_sentence:
            self._sync_speaker_control()
            self._update_parameter_availability()
            self._refresh_japanese_runtime_controls()
            return
        selected_voice = self._current_voicebank()
        voice_selection_event = self.sender() is self.voicebank
        targets = self._sidebar_sentence_targets()
        if voice_selection_event and selected_voice:
            for index in targets:
                state = self.sentences[index]
                if str(state.get("voicebank") or "") == selected_voice:
                    continue
                state["voicebank"] = selected_voice
                state["rendered"] = False
                state["applied_gain_db"] = None
                state["pre_gain_peak"] = 0.0
                self._set_state_pending(
                    state, "generate", "Voicebank changed")
            self._refresh_pending_ui()
        self._voice_dict_loaded = None
        voice = selected_voice or ""
        self._sync_speaker_control()
        self._apply_voice_language_compatibility(
            auto_select=(self.sender() is self.voicebank)
        )
        if not self._pitch_user_edited:
            try:
                average = self._ab().voice_pitch_hz(voice)
            except (fc.BackendError, OSError, ValueError, AttributeError):
                average = None
            if average is not None:
                self.pitch.setValue(float(average))
                for index in targets:
                    self.sentences[index]["pitch_hz"] = float(average)
        self._refresh_japanese_runtime_controls()
        if voice_selection_event and targets:
            self._refresh_sentences_view_preserving_focus(targets)
            self._sync_sentence_sidebar_values(targets)
        key = self._voice_dictionary_key()
        path = (self.cfg.get("voice_dictionaries") or {}).get(key)
        if not key or not path:
            return
        try:
            entries = self._ab().read_installed_dictionary(path)
        except (fc.BackendError, OSError, ValueError, AttributeError):
            return
        if entries:
            self.user_dicts[self._current_lang_code()] = entries
            self._voice_dict_loaded = key
            self.statusBar().showMessage(
                "Status: loaded %d voicebank dictionary entries" %
                len(entries))

    def on_show_dicts(self):
        if not self.user_dicts:
            QtWidgets.QMessageBox.information(
                self, "Dictionaries", "No pronunciation dictionaries loaded.\n"
                "Dictionary > Load pronunciation dictionary...")
            return
        lines = ["%s:  %d words" % (c, len(d))
                 for c, d in sorted(self.user_dicts.items())]
        QtWidgets.QMessageBox.information(
            self, "Loaded dictionaries",
            "Active per language:\n\n" + "\n".join(lines))

    # -- menu -------------------------------------------------------------------
    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        self.action_new = m_file.addAction("New Sentence", self.on_add_sentence)
        self._shortcut_menu_actions["new_sentence"] = (
            self.action_new, "New Sentence")
        self.action_open = m_file.addAction(
            "Open Project JSON...", self.on_open_project)
        self._shortcut_menu_actions["open_project"] = (
            self.action_open, "Open Project JSON...")
        self.action_save = m_file.addAction("Save Project...", self.on_save_project)
        self._shortcut_menu_actions["save_project"] = (
            self.action_save, "Save Project...")
        m_file.addAction("Import Text File as Sentences...",
                         self.on_import_text_file)
        self.action_export = m_file.addAction(
            "Export Audio (WAV)...", self.on_export)
        self._shortcut_menu_actions["export_audio"] = (
            self.action_export, "Export Audio (WAV)...")
        m_file.addAction("Export Batch (WAV folder)...", self.on_export_batch)
        m_file.addSeparator()
        m_file.addAction("Exit", self.close)

        m_edit = mb.addMenu("&Edit")
        self.action_undo = self.undo_stack.createUndoAction(self, "Undo")
        self.action_redo = self.undo_stack.createRedoAction(self, "Redo")
        m_edit.addAction(self.action_undo)
        m_edit.addAction(self.action_redo)
        self._shortcut_menu_actions["undo"] = (self.action_undo, "Undo")
        self._shortcut_menu_actions["redo"] = (self.action_redo, "Redo")
        m_edit.addSeparator()
        for key, label in (("cut", "Cut"), ("copy", "Copy"),
                           ("paste", "Paste"), ("duplicate", "Duplicate"),
                           ("select_all", "Select All"),
                           ("delete", "Delete")):
            action = m_edit.addAction(
                label, lambda _checked=False, command=key:
                self._dispatch_shortcut(command))
            self._shortcut_menu_actions[key] = (action, label)

        m_view = mb.addMenu("&View")
        self.action_show_rendered_joins = m_view.addAction(
            "Rendered joins in waveform")
        self.action_show_rendered_joins.setCheckable(True)
        self.action_show_rendered_joins.setChecked(bool(
            self.cfg.get("show_rendered_joins", False)))
        self.action_show_rendered_joins.setStatusTip(
            "Show each rendered unit handoff and its effective crossover "
            "span in the Speech waveform")
        self.action_show_rendered_joins.toggled.connect(
            self._set_join_overlay_visible)

        m_gen = mb.addMenu("&Generate")
        self.action_generate = m_gen.addAction("Generate Audio", self.on_generate)
        self._shortcut_menu_actions["generate"] = (
            self.action_generate, "Generate Audio")
        m_gen.addAction("Generate All Sentences", self.on_generate_all)
        self.action_rerender = m_gen.addAction(
            "Re-render edited phonemes", self.on_rerender)
        self._shortcut_menu_actions["rerender"] = (
            self.action_rerender, "Re-render edited phonemes")
        self.action_play = m_gen.addAction("Play", self.on_play)
        self._shortcut_menu_actions["play"] = (self.action_play, "Play")
        self.action_stop = m_gen.addAction("Stop", self.on_stop)
        self._shortcut_menu_actions["stop"] = (self.action_stop, "Stop")
        m_gen.addAction("Re-render All Sentences", self.on_rerender_all)
        m_gen.addSeparator()
        m_gen.addAction("Last render details...", self.on_render_details)
        m_gen.addAction("Inspect joins and UniSyn windows...",
                        self.on_join_loudness_diagnostic)
        m_gen.addAction("Inspect rendered formants...",
                        self.on_rendered_formant_diagnostic)

        m_vb = mb.addMenu("&Voicebank")
        m_vb.addAction("Voicebank manager...", self.on_voicebank_manager)
        m_vb.addSeparator()
        m_vb.addAction("Add diphone DB folder...", self.on_add_voice_folder)
        m_vb.addAction("Analyze Japanese UTAU bank...",
                       self.on_analyze_japanese_bank)
        m_vb.addSeparator()
        m_vb.addAction("Add Festival voice folder...", self.on_add_fest_voice_folder)
        m_vb.addAction("Scan Festival voices (WSL)", self.on_scan_fest_voices)
        m_vb.addSeparator()
        m_vb.addAction("Replace selected speaker icon...",
                       self._choose_speaker_portrait)
        m_vb.addAction("Remove selected speaker icon...",
                       self._remove_speaker_portrait)
        m_vb.addSeparator()
        m_vb.addAction("Uninstall selected voicebank...",
                       self.on_uninstall_voicebank)
        m_vb.addSeparator()
        m_vb.addAction("Set festvox.json...", self.on_set_festvox_config)
        m_vb.addAction("Reload voicebanks", self.on_reload_voicebanks)

        m_dict = mb.addMenu("&Dictionary")
        m_dict.addAction("Load pronunciation dictionary (current language)...",
                         self.on_load_dict)
        m_dict.addAction("Install dictionary into selected voicebank...",
                         self.on_install_voice_dictionary)
        m_dict.addAction("Clear dictionary (current language)",
                         self.on_clear_dict)
        m_dict.addSeparator()
        m_dict.addAction("Loaded dictionaries...", self.on_show_dicts)

        m_opt = mb.addMenu("&Options")
        m_opt.addAction("Locate synth_diphone.py...", self.on_locate_engine)
        m_opt.addAction("WSL / Festival settings...", self.on_wsl_settings)
        self.action_phrase_pauses = m_opt.addAction(
            "Phrase pauses...", self.on_phrase_pauses)
        duration_menu = m_opt.addMenu("Japanese duration model")
        self.japanese_duration_actions = {}
        duration_group = QtWidgets.QActionGroup(self)
        duration_group.setExclusive(True)
        selected_duration = str(self.cfg.get(
            "japanese_duration_model", "contextual"))
        for mode, label in (("contextual", "Contextual (source-based)"),
                            ("legacy", "Legacy mora timing")):
            action = duration_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(mode == selected_duration)
            action.triggered.connect(
                lambda _checked=False, value=mode:
                self._set_japanese_duration_model(value))
            duration_group.addAction(action)
            self.japanese_duration_actions[mode] = action
        devoicing_menu = m_opt.addMenu("Japanese vowel devoicing")
        self.japanese_devoicing_actions = {}
        devoicing_group = QtWidgets.QActionGroup(self)
        devoicing_group.setExclusive(True)
        selected_devoicing = str(self.cfg.get(
            "japanese_vowel_devoicing", "contextual"))
        for mode, label in (("contextual", "Contextual"),
                            ("legacy", "Legacy (duration only)")):
            action = devoicing_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(mode == selected_devoicing)
            action.triggered.connect(
                lambda _checked=False, value=mode:
                self._set_japanese_synthesis_option(
                    "japanese_vowel_devoicing", value))
            devoicing_group.addAction(action)
            self.japanese_devoicing_actions[mode] = action
        renderer_menu = devoicing_menu.addMenu("Renderer")
        self.japanese_devoicing_renderer_actions = {}
        renderer_group = QtWidgets.QActionGroup(self)
        renderer_group.setExclusive(True)
        selected_renderer = str(self.cfg.get(
            "japanese_devoicing_renderer", "auto"))
        if selected_renderer == "mixed_excitation":
            selected_renderer = "source_filter"
            self.cfg["japanese_devoicing_renderer"] = selected_renderer
        for mode, label in (("auto", "Auto (natural / source-filter)"),
                            ("source_filter", "Source-filter residual"),
                            ("shortened_voiced", "Shortened voiced fallback")):
            action = renderer_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(mode == selected_renderer)
            action.triggered.connect(
                lambda _checked=False, value=mode:
                self._set_japanese_synthesis_option(
                    "japanese_devoicing_renderer", value))
            renderer_group.addAction(action)
            self.japanese_devoicing_renderer_actions[mode] = action
        m_opt.addAction("Advanced synthesis settings (diphone)...", self.on_advanced)
        self.cache_menu = m_opt.addMenu("Application caches")
        self.cache_usage_action = self.cache_menu.addAction("Cache usage")
        self.cache_usage_action.setEnabled(False)
        self.cache_actions = {}
        for category, label in (
                ("audio", "Clear audio cache"),
                ("voice", "Clear voice cache"),
                ("model", "Clear model cache")):
            action = self.cache_menu.addAction(
                label,
                lambda _checked=False, value=category:
                self._clear_application_caches(value))
            self.cache_actions[category] = action
        self.cache_menu.addSeparator()
        self.cache_actions["all"] = self.cache_menu.addAction(
            "Clear all application caches",
            lambda _checked=False: self._clear_application_caches("all"))
        self.cache_menu.aboutToShow.connect(self._refresh_cache_menu)
        m_opt.addSeparator()
        m_opt.addAction("Keyboard Shortcuts...", self._show_shortcut_dialog)

        m_fault = mb.addMenu("&Fault Mode")
        self.fault_menu = m_fault
        self.fault_menu_action = m_fault.menuAction()
        stored = dict(self.cfg.get("fault_mode") or {})
        if "monotone" not in stored:
            stored["monotone"] = bool(self.cfg.get("monotone", False))
        self.fault_actions = {}

        def add_fault(label, key, tip):
            action = m_fault.addAction(label)
            action.setCheckable(True)
            action.setChecked(bool(stored.get(key, False)))
            action.setStatusTip(tip)
            action.toggled.connect(self._on_fault_changed)
            self.fault_actions[key] = action
            return action

        add_fault("No phone timing rules", "disable_phone_timing",
                  "Give phones equal duration instead of class/voice timing.")
        add_fault("No learned prosody", "disable_prosody",
                  "Replace English text prosody with equal phones and Fall only.")
        add_fault("Raw F0 joins", "disable_f0_correction",
                  "Disable contour endpoint correction at voiced boundaries.")
        add_fault("Single phrase pause", "single_pause",
                  "Use the old one-pau phrase boundary for comparison.")
        add_fault("Broken pitch estimate", "pitch_glitch",
                  "Randomly corrupt F0 on one or more phones per render.")
        add_fault("Long stretches, no sustain samples", "no_sustain_stretch",
                  "Stretch only the rendered segment instead of its X-X "
                  "voicebank sustain sample.")
        add_fault("Legacy joins", "legacy_joins",
                  "Use the pre-fix fixed linear fade or legacy UniSyn source "
                  "pitchmarks for comparison.")
        m_fault.addSeparator()
        add_fault("Monotone", "monotone",
                  "Force a flat Festival F0 contour at the Pitch value.")
        bit_menu = m_fault.addMenu("Bit depth")
        self.bit_depth_actions = {}
        bit_group = QtWidgets.QActionGroup(self)
        bit_group.setExclusive(True)
        stored_bits = int(stored.get("bit_depth") or 0)
        for bits, label in ((0, "Full quality"), (8, "8-bit"), (4, "4-bit"),
                            (2, "2-bit"), (1, "1-bit")):
            action = bit_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(bits == stored_bits)
            action.setData(bits)
            action.setStatusTip(
                "Quantize output to %s with compensated volume."
                % ("full precision" if not bits else "%d-bit" % bits))
            action.triggered.connect(self._on_fault_changed)
            bit_group.addAction(action)
            self.bit_depth_actions[bits] = action
        m_fault.aboutToShow.connect(self._update_fault_availability)

        m_help = mb.addMenu("&Help")
        m_help.addAction("About", lambda: QtWidgets.QMessageBox.information(
            self, "About",
            "FestVox Speech GUI v3.0\n\n"
            "PyQt5 + PyQtGraph front-end for the pure-Python diphone\n"
            "renderer and real Festival/UniSyn synthesis through WSL.\n\n"
            "Includes waveform-aligned timing, pitch and intonation\n"
            "editing, visible recorded-unit overrides, sentence/phrase\n"
            "projects, diagnostic faults, and WAV export."))

    @staticmethod
    def _format_cache_bytes(byte_count):
        value = max(0, int(byte_count or 0))
        if value < 1024:
            return "%d B" % value
        if value < 1024 * 1024:
            return "%.1f KiB" % (value / 1024.0)
        return "%.1f MiB" % (value / (1024.0 * 1024.0))

    def _application_cache_info(self):
        categories = {
            name: {"bytes": 0, "entries": 0}
            for name in ("audio", "voice", "model")
        }
        for backend in (self.backend, self.fest):
            if backend is None or not hasattr(backend, "cache_info"):
                continue
            report = backend.cache_info()
            for name in categories:
                row = dict(report.get(name) or {})
                categories[name]["bytes"] += int(row.get("bytes", 0))
                categories[name]["entries"] += sum(
                    int(value) for key, value in row.items()
                    if key != "bytes" and isinstance(value, int)
                )
        shared = fc.shared_model_cache_info()
        categories["model"]["bytes"] += int(shared.get("bytes", 0))
        categories["model"]["entries"] += int(shared.get("entries", 0))
        # Values are shared with backend metadata caches; count only the small
        # GUI key container to avoid both double accounting and a recursive
        # walk over multi-megabyte alternative indexes when opening the menu.
        categories["voice"]["bytes"] += len(self._variant_cache) * 256
        categories["voice"]["entries"] += len(self._variant_cache)
        if hasattr(self, "waveform"):
            display = self.waveform._waveform_cache.cache_info()
            categories["audio"]["bytes"] += int(display.get("bytes", 0))
            categories["audio"]["entries"] += int(
                display.get("entries", 0))
        return {
            "categories": categories,
            "bytes": sum(row["bytes"] for row in categories.values()),
        }

    def _refresh_cache_menu(self):
        report = self._application_cache_info()
        categories = report["categories"]
        self.cache_usage_action.setText(
            "In-memory cache: " + self._format_cache_bytes(report["bytes"]))
        for name, label in (("audio", "Clear audio cache"),
                            ("voice", "Clear voice cache"),
                            ("model", "Clear model cache")):
            self.cache_actions[name].setText(
                "%s (%s)" % (
                    label, self._format_cache_bytes(
                        categories[name]["bytes"])))
        self.cache_actions["all"].setText(
            "Clear all application caches (%s)" %
            self._format_cache_bytes(report["bytes"]))

    def _clear_application_caches(self, category):
        category = str(category).casefold()
        if category not in {"audio", "voice", "model", "all"}:
            raise ValueError("unknown application cache category")
        before = self._application_cache_info()
        for backend in (self.backend, self.fest):
            if backend is not None and hasattr(
                    backend, "clear_application_cache"):
                backend.clear_application_cache(category)
        if category in {"model", "all"}:
            fc.clear_shared_model_caches()
        if category in {"voice", "all"}:
            self._variant_cache.clear()
        if category in {"audio", "all"} and hasattr(self, "waveform"):
            self.waveform.clear_display_cache()
        after = self._application_cache_info()
        removed = max(0, int(before["bytes"]) - int(after["bytes"]))
        self._refresh_cache_menu()
        self.statusBar().showMessage(
            "Status: cleared %s cache (%s released); projects, exports, "
            "and voicebank files were not touched" %
            (category, self._format_cache_bytes(removed)))

    # -- body ---------------------------------------------------------------
    def _build_body(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(10)

        # ---- left panel
        left = QtWidgets.QVBoxLayout()
        # Reserve room for the vertical scrollbar instead of letting it cover
        # the portrait and right edges of controls at short window heights.
        left.setContentsMargins(0, 0, 14, 0)
        left.setSpacing(6)
        self.speaker_portrait = SpeakerPortrait()
        self.speaker_portrait.changeRequested.connect(
            self._choose_speaker_portrait)
        left.addWidget(self.speaker_portrait, 0, Qt.AlignHCenter)
        elbl = QtWidgets.QLabel("Engine:"); elbl.setObjectName("hdr")
        left.addWidget(elbl)
        self.engine = ArrowComboBox()
        self.engine.addItem("Diphone (pure Python)", "diphone")
        self.engine.addItem("Festival via WSL (Multisyn)", "festival_wsl")
        self.engine.setToolTip(
            "Diphone: synth_diphone.py, runs anywhere, no Festival needed.\n"
            "Festival via WSL: the real Festival runtime -- required for\n"
            "Multisyn unit-selection voices (see MULTISYN.md).")
        self.engine.currentIndexChanged.connect(self._on_engine_changed)
        left.addWidget(self.engine)

        lbl = QtWidgets.QLabel("Language:"); lbl.setObjectName("hdr")
        left.addWidget(lbl)
        self.lang = ArrowComboBox()
        self.lang.currentIndexChanged.connect(self._update_fault_availability)
        self.lang.currentIndexChanged.connect(self._on_voicebank_changed)
        self.lang.currentIndexChanged.connect(self._on_language_changed)
        left.addWidget(self.lang)

        self.voicebank_heading = QtWidgets.QToolButton()
        self.voicebank_heading.setText("Voicebank Database")
        self.voicebank_heading.setObjectName("hdr")
        self.voicebank_heading.setCheckable(True)
        self.voicebank_heading.setChecked(True)
        self.voicebank_heading.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.voicebank_heading.setArrowType(Qt.DownArrow)
        self.voicebank_heading.setToolTip("Click to collapse or expand")
        left.addWidget(self.voicebank_heading)
        self.voicebank = QtWidgets.QListWidget()
        self.voicebank.setFixedHeight(int(
            self.cfg.get("voicebank_list_height", 76) or 76))
        self.voicebank.currentItemChanged.connect(self._on_voicebank_changed)
        left.addWidget(self.voicebank)
        self.voicebank_resize = VerticalResizeHandle(self.voicebank)
        self.voicebank_resize.resized.connect(
            lambda height: self.cfg.__setitem__(
                "voicebank_list_height", int(height)))
        left.addWidget(self.voicebank_resize)
        self.voicebank_heading.toggled.connect(
            self._toggle_voicebank_list)

        slbl = QtWidgets.QLabel("Output Speed:"); slbl.setObjectName("hdr")
        left.addWidget(slbl)
        self.speed = SpeedSlider(Qt.Horizontal)
        # 2**(v/100): -200 -> x0.25 (engine minimum), +200 -> x4 (maximum)
        self.speed.setMinimum(-200); self.speed.setMaximum(200); self.speed.setValue(0)
        self.speed.setToolTip("double-click to reset to x1.00")
        self.speed.valueChanged.connect(self._on_speed_slider)
        left.addWidget(self.speed)
        srow = QtWidgets.QHBoxLayout()
        q = QtWidgets.QLabel("speed"); q.setStyleSheet("color:#555")
        srow.addWidget(q)
        self.speed_val = QtWidgets.QDoubleSpinBox()
        self.speed_val.setRange(0.25, 4.0)
        self.speed_val.setDecimals(2)
        self.speed_val.setSingleStep(0.05)
        self.speed_val.setValue(1.0)
        self.speed_val.setPrefix("x")
        self.speed_val.setToolTip("type an exact speed factor")
        self.speed_val.valueChanged.connect(self._on_speed_spin)
        srow.addWidget(self.speed_val, 1)
        left.addLayout(srow)

        self.output_gain_label = QtWidgets.QLabel("Output volume:")
        self.output_gain_label.setObjectName("hdr")
        left.addWidget(self.output_gain_label)
        self.speech_gain = GainControl(
            float(self.cfg.get("output_gain_db", 0.0)))
        self.speech_gain.valueChanged.connect(self._on_output_gain_changed)
        self.speech_gain.clippingChanged.connect(
            self._on_allow_clipping_changed)
        left.addWidget(self.speech_gain)
        # Compatibility aliases for state and older integrations.
        self.output_gain = self.speech_gain.spin
        self.output_gain_slider = self.speech_gain.slider

        plbl = QtWidgets.QLabel("Festival Pitch:"); plbl.setObjectName("hdr")
        self.pitch_header = plbl
        left.addWidget(plbl)
        pitch_form = QtWidgets.QFormLayout()
        pitch_form.setContentsMargins(0, 0, 0, 0)
        pitch_form.setSpacing(4)
        self.pitch = QtWidgets.QDoubleSpinBox()
        self.pitch.setRange(60.0, 500.0)
        self.pitch.setDecimals(0)
        self.pitch.setSingleStep(5.0)
        self.pitch.setSuffix(" Hz")
        self.pitch.setValue(float(self.cfg.get("pitch_hz", 185.0)))
        self.pitch.valueChanged.connect(self._on_pitch_parameter_changed)
        self.pitch.lineEdit().textEdited.connect(
            lambda _text: setattr(self, "_pitch_user_edited", True))
        self.pitch.setToolTip(
            "Base pitch for the Festival engine (PSOLA retargeting): applies "
            "to every\nlanguage and to phoneme input, so Asaxi, English and "
            "Japanese come out at\nthe same pitch. The diphone engine plays "
            "the bank as recorded (no effect).")
        pitch_form.addRow("Pitch:", self.pitch)
        self.pitch_field_label = pitch_form.labelForField(self.pitch)
        self.fall = QtWidgets.QDoubleSpinBox()
        self.fall.setRange(0.0, 40.0)
        self.fall.setDecimals(0)
        self.fall.setSingleStep(1.0)
        self.fall.setMinimumWidth(96)
        self.fall.setSuffix(" %")
        self.fall.setValue(float(self.cfg.get("pitch_fall_pct", 10.0)))
        self.fall.valueChanged.connect(self._on_pitch_parameter_changed)
        self.fall.setToolTip(
            "Declination from 0 through the visible maximum of 40%.\n"
            "(0 = flat monotone; ~10% sounds like neutral speech).\n"
            "English text keeps its full f2b contour, recentered here.")
        pitch_form.addRow("Fall (0-40%):", self.fall)
        self.fall_field_label = pitch_form.labelForField(self.fall)
        left.addLayout(pitch_form)

        self.fault_badge = QtWidgets.QToolButton()
        self.fault_badge.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.fault_badge.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning))
        self.fault_badge.clicked.connect(self._show_fault_menu)
        self.fault_badge.hide()
        left.addWidget(self.fault_badge)
        fault_buttons = QtWidgets.QHBoxLayout()
        self.clear_faults_button = QtWidgets.QPushButton("Clear faults")
        self.clear_faults_button.clicked.connect(self._clear_faults)
        fault_buttons.addWidget(self.clear_faults_button)
        self.project_faults_button = QtWidgets.QPushButton("Project faults")
        self.project_faults_button.setMinimumWidth(112)
        project_fault_menu = QtWidgets.QMenu(self.project_faults_button)
        project_fault_menu.addAction(
            "Apply current faults to all sentences",
            self._apply_faults_to_all_sentences)
        project_fault_menu.addAction(
            "Clear faults from all sentences",
            self._clear_faults_from_all_sentences)
        self.project_faults_button.setMenu(project_fault_menu)
        fault_buttons.addWidget(self.project_faults_button)
        left.addLayout(fault_buttons)

        left.addSpacing(8)
        self.btn_gen = self._toolbtn("Generate Audio", "SP_MediaVolume", self.on_generate)
        self.btn_gen_all = self._toolbtn(
            "Generate All Sentences", "SP_DialogApplyButton", self.on_generate_all)
        self.btn_play = self._toolbtn("Play", "SP_MediaPlay", self.on_play)
        self.btn_stop = self._toolbtn("Stop", "SP_MediaStop", self.on_stop)
        self.btn_stop.setEnabled(False)
        # explicit add/remove phoneme buttons (act on the focused/last-
        # clicked phoneme box; the boxes also have a right-click menu)
        pbrow = QtWidgets.QHBoxLayout()
        self.btn_addph = QtWidgets.QPushButton("+ Phone")
        self.btn_addph.setToolTip(
            "Insert a phoneme after the selected box\n(click a phoneme box "
            "first; right-click a box for before/after)")
        self.btn_addph.clicked.connect(self.on_add_phone)
        self.btn_delph = QtWidgets.QPushButton("- Phone")
        self.btn_delph.setToolTip(
            "Remove the selected phoneme box\n(click a phoneme box first)")
        self.btn_delph.clicked.connect(self.on_del_phone)
        pbrow.addWidget(self.btn_addph)
        pbrow.addWidget(self.btn_delph)
        left.addLayout(pbrow)
        self.btn_rerender = self._toolbtn("Re-render Phonemes", "SP_BrowserReload", self.on_rerender)
        self.btn_rerender.setEnabled(False)
        self.btn_rerender.setToolTip(
            "Re-synthesize with the edited phonemes AND the edited timings.\n"
            "After velocity/boundary adjustments this gives optimal quality\n"
            "(the in-editor stretch is only a preview). On the Festival\n"
            "engine the previous pitch contour is kept.")
        self.btn_rerender_all = self._toolbtn(
            "Re-render All Sentences", "SP_DialogApplyButton",
            self.on_rerender_all)
        self.btn_save = self._toolbtn("Save Project", "SP_DialogSaveButton", self.on_save_project)
        self.btn_export = self._toolbtn("Export Audio (WAV)", "SP_DriveHDIcon", self.on_export)
        self.btn_export_batch = self._toolbtn(
            "Export Batch (WAV)", "SP_DirOpenIcon", self.on_export_batch)
        for b in (self.btn_gen, self.btn_gen_all, self.btn_play, self.btn_stop,
                  self.btn_rerender, self.btn_rerender_all, self.btn_save,
                  self.btn_export,
                  self.btn_export_batch):
            left.addWidget(b)
        left.addStretch(1)

        leftw = QtWidgets.QWidget()
        leftw.setLayout(left)
        self.sidebar_editor = leftw
        self.sidebar = QtWidgets.QScrollArea()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.sidebar.setWidgetResizable(True)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.sidebar.setWidget(leftw)
        self.sidebar.setFixedWidth(242)
        root.addWidget(self.sidebar)

        # ---- right side
        speech_page = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(speech_page); right.setSpacing(6)
        right.setContentsMargins(0, 0, 0, 0)
        trow = QtWidgets.QHBoxLayout()
        self.sentence_select = ArrowComboBox()
        self.sentence_select.setMinimumWidth(170)
        self.sentence_select.setToolTip(
            "Select a sentence; each keeps its own phones and parameters.")
        self.sentence_select.currentIndexChanged.connect(
            self._on_sentence_selected)
        trow.addWidget(self.sentence_select)
        self.sentence_add = QtWidgets.QToolButton()
        self.sentence_add.setText("+")
        self.sentence_add.setToolTip("Add sentence")
        self.sentence_add.clicked.connect(self.on_add_sentence)
        trow.addWidget(self.sentence_add)
        self.sentence_duplicate = QtWidgets.QToolButton()
        self.sentence_duplicate.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
        self.sentence_duplicate.setToolTip("Duplicate sentence")
        self.sentence_duplicate.clicked.connect(self.on_duplicate_sentence)
        trow.addWidget(self.sentence_duplicate)
        self.sentence_remove = QtWidgets.QToolButton()
        self.sentence_remove.setText("-")
        self.sentence_remove.setToolTip("Remove sentence from project")
        self.sentence_remove.clicked.connect(self.on_remove_sentence)
        trow.addWidget(self.sentence_remove)
        self.input_mode = ArrowComboBox()
        self.input_mode.addItem("Text", "text")
        self.input_mode.addItem("Phonemes", "phones")
        self.input_mode.setToolTip(
            "Text: normal g2p / text processing.\n"
            "Phonemes: type space-separated phones and synthesize them "
            "directly,\ne.g.  hh eh l ow pau m ay n ey m ih z l eh m")
        self.input_mode.currentIndexChanged.connect(self._on_input_mode)
        trow.addWidget(self.input_mode)
        self.text = QtWidgets.QLineEdit(self.cfg.get("default_text", ""))
        self.text.setToolTip(
            "Text to synthesize. In Text mode you can drop exact phones inline "
            "in [brackets],\ne.g.  a rare name [n ey m z]  -- and [pau] forces a "
            "pause mid-phrase.")
        self.text.returnPressed.connect(self.on_generate)
        self.text.textEdited.connect(self._on_sentence_text_edited)
        trow.addWidget(self.text, 1)
        self.follow_playhead = QtWidgets.QToolButton()
        self.follow_playhead.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward))
        self.follow_playhead.setIconSize(QtCore.QSize(16, 16))
        self.follow_playhead.setCheckable(True)
        self.follow_playhead.setChecked(bool(
            self.cfg.get("follow_playhead", True)))
        self.follow_playhead.setToolTip(
            "Follow playback when the playhead leaves the visible timeline")
        self.follow_playhead.setAccessibleName("Follow playhead")
        self.follow_playhead.toggled.connect(
            self._on_follow_playhead_toggled)
        trow.addWidget(self.follow_playhead)
        right.addLayout(trow)

        self.wf_group = QtWidgets.QGroupBox("Waveform  /  Phonemes")
        self.alignment_group = self.wf_group
        self.wf_lay = QtWidgets.QVBoxLayout(self.wf_group)
        self.wf_lay.setContentsMargins(6, 4, 6, 6)
        alignment_commands = QtWidgets.QHBoxLayout()
        alignment_commands.addStretch(1)
        self.alignment_collapse = QtWidgets.QToolButton()
        self.alignment_collapse.setText("Collapse alignment")
        self.alignment_collapse.setCheckable(True)
        self.alignment_collapse.toggled.connect(
            self._toggle_alignment_collapsed)
        self.alignment_collapse.hide()
        alignment_commands.addWidget(self.alignment_collapse)
        self.wf_lay.addLayout(alignment_commands)
        self.waveform = WaveformEditor()
        self.waveform.variant_menu_hook = self._add_unit_variant_menu
        self.waveform.audioChanged.connect(self._on_audio_changed)
        self.waveform.phonesEdited.connect(self._on_phones_edited)
        self.waveform.rerenderRequested.connect(self.on_rerender)
        self.waveform.regionCutRequested.connect(self._cut_region_to_sentence)
        self.waveform.faultTargetRequested.connect(self._set_pitch_fault_target)
        self.waveform.timingEditCommitted.connect(
            self._on_waveform_timing_edit)
        self.waveform.structureEditCommitted.connect(
            self._on_waveform_structure_edit)
        self.waveform.joinCrossoverCommitted.connect(
            self._set_join_crossover_override)
        self.waveform.sustain_hook = self._sustain_sample
        self.wf_lay.addWidget(self.waveform)
        self.editor_splitter = QtWidgets.QSplitter(Qt.Vertical)
        self.editor_splitter.setChildrenCollapsible(False)
        self.editor_splitter.addWidget(self.wf_group)

        env_group = QtWidgets.QGroupBox("Parameter Editor")
        self.parameter_group = env_group
        env_lay = QtWidgets.QVBoxLayout(env_group)
        env_lay.setContentsMargins(6, 4, 6, 6)
        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("Parameter:"))
        self.parameter_mode = ArrowComboBox()
        self.parameter_mode.addItem("Timing", "timing")
        self.parameter_mode.addItem("Pitch curve", "pitch")
        self.parameter_mode.addItem("Voicing", "voicing")
        self.parameter_mode.addItem("Vocal tract length", "vocal_tract")
        self.parameter_mode.addItem("Intonation blocks", "intonation")
        self.parameter_mode.addItem("Recordings", "recordings")
        self.parameter_mode.addItem("Pitch accent", "japanese")
        self.parameter_mode.addItem("Mora voicing", "mora_voicing")
        self.parameter_mode.currentIndexChanged.connect(
            self._on_parameter_mode)
        param_row.addWidget(self.parameter_mode)
        self.timing_consonants = QtWidgets.QCheckBox("Consonant velocity")
        self.timing_consonants.setChecked(True)
        self.timing_consonants.toggled.connect(self._on_timing_filter)
        param_row.addWidget(self.timing_consonants)
        self.timing_vowels = QtWidgets.QCheckBox("Vowel length")
        self.timing_vowels.setChecked(True)
        self.timing_vowels.toggled.connect(self._on_timing_filter)
        param_row.addWidget(self.timing_vowels)
        self.curve_unit_overlay = QtWidgets.QCheckBox(
            "Show syllables / morae")
        self.curve_unit_overlay.setChecked(bool(
            self.cfg.get("show_curve_linguistic_units", False)))
        self.curve_unit_overlay.setToolTip(
            "Show the parser's English syllables or Japanese/Asaxi morae "
            "over each continuous parameter curve.\nThis view is diagnostic "
            "only and does not change synthesis.")
        self.curve_unit_overlay.toggled.connect(
            self._on_curve_unit_overlay_toggled)
        param_row.addWidget(self.curve_unit_overlay)
        param_row.addStretch(1)
        self.parameter_collapse = QtWidgets.QToolButton()
        self.parameter_collapse.setText("Collapse parameters")
        self.parameter_collapse.setCheckable(True)
        self.parameter_collapse.toggled.connect(
            self._toggle_parameter_collapsed)
        param_row.addWidget(self.parameter_collapse)
        env_lay.addLayout(param_row)

        self.parameter_stack = QtWidgets.QStackedWidget()
        self.timing = TimingTrack()
        self.timing.setXLink(self.waveform.plot)   # zoom/pan stay aligned
        self.timing.zoom_viewbox = self.waveform.plot.getViewBox()
        self.timing.factorsCommitted.connect(self._on_timing_commit)
        self.timing.factorReset.connect(self._on_timing_reset)
        self.parameter_stack.addWidget(self.timing)
        self.pitch_track = PitchTrack()
        self.pitch_track.setXLink(self.waveform.plot)
        self.pitch_track.zoom_viewbox = self.waveform.plot.getViewBox()
        self.pitch_track.targetsCommitted.connect(self._on_pitch_commit)
        self.pitch_track.overrideCleared.connect(self._on_pitch_clear)
        self.pitch_track.viewChanged.connect(self._sync_pitch_navigator)
        pitch_page = QtWidgets.QWidget()
        pitch_page_lay = QtWidgets.QVBoxLayout(pitch_page)
        pitch_page_lay.setContentsMargins(0, 0, 0, 0)
        pitch_page_lay.setSpacing(3)
        pitch_track_row = QtWidgets.QHBoxLayout()
        pitch_track_row.setContentsMargins(0, 0, 0, 0)
        pitch_track_row.setSpacing(3)
        pitch_track_row.addWidget(self.pitch_track, 1)
        self.pitch_navigator = QtWidgets.QWidget()
        self.pitch_navigator.setObjectName("pitchVerticalNavigator")
        pitch_nav_lay = QtWidgets.QVBoxLayout(self.pitch_navigator)
        pitch_nav_lay.setContentsMargins(0, 0, 0, 0)
        pitch_nav_lay.setSpacing(2)
        self.pitch_zoom_in = QtWidgets.QToolButton()
        self.pitch_zoom_in.setText("+")
        self.pitch_zoom_in.setToolTip("Zoom in on the pitch scale")
        self.pitch_zoom_in.setAccessibleName("Pitch zoom in")
        self.pitch_zoom_in.setFixedSize(22, 22)
        self.pitch_zoom_in.clicked.connect(
            lambda: self._change_pitch_zoom(1))
        pitch_nav_lay.addWidget(self.pitch_zoom_in)
        self.pitch_scroll = QtWidgets.QScrollBar(Qt.Vertical)
        self.pitch_scroll.setObjectName("pitchVerticalScroll")
        self.pitch_scroll.setAccessibleName("Pitch scale position")
        self.pitch_scroll.setRange(
            int(fc.PITCH_MIN_HZ), int(fc.PITCH_MAX_HZ))
        self.pitch_scroll.setSingleStep(2)
        self.pitch_scroll.setPageStep(20)
        self.pitch_scroll.setInvertedAppearance(True)
        self.pitch_scroll.setToolTip(
            "Move the visible pitch range up or down")
        self.pitch_scroll.valueChanged.connect(
            lambda value: self.pitch_track.recenter(float(value)))
        pitch_nav_lay.addWidget(self.pitch_scroll, 1)
        self.pitch_zoom_out = QtWidgets.QToolButton()
        self.pitch_zoom_out.setText("-")
        self.pitch_zoom_out.setToolTip("Zoom out from the pitch scale")
        self.pitch_zoom_out.setAccessibleName("Pitch zoom out")
        self.pitch_zoom_out.setFixedSize(22, 22)
        self.pitch_zoom_out.clicked.connect(
            lambda: self._change_pitch_zoom(-1))
        pitch_nav_lay.addWidget(self.pitch_zoom_out)
        self.pitch_navigator.setFixedWidth(24)
        pitch_track_row.addWidget(self.pitch_navigator)
        pitch_page_lay.addLayout(pitch_track_row, 1)
        pitch_command_row = QtWidgets.QHBoxLayout()
        pitch_command_row.addStretch(1)
        self.pitch_reset = QtWidgets.QPushButton("Reset pitch curve")
        self.pitch_reset.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton))
        self.pitch_reset.setToolTip("Restore the generated F0 contour")
        self.pitch_reset.clicked.connect(self.pitch_track.clear_override)
        pitch_command_row.addWidget(self.pitch_reset)
        self.pitch_recenter = QtWidgets.QPushButton("Recenter view")
        self.pitch_recenter.setToolTip(
            "Center the pitch scale on the current Festival Pitch value")
        self.pitch_recenter.clicked.connect(
            lambda: self.pitch_track.recenter(self.pitch.value()))
        pitch_command_row.addWidget(self.pitch_recenter)
        pitch_page_lay.addLayout(pitch_command_row)
        self._sync_pitch_navigator(
            self.pitch_track.view_center(), self.pitch_track.zoom_level())
        self.parameter_stack.addWidget(pitch_page)
        self.voicing_track = VoicingTrack()
        self.voicing_track.setXLink(self.waveform.plot)
        self.voicing_track.zoom_viewbox = self.waveform.plot.getViewBox()
        self.voicing_track.targetsCommitted.connect(
            self._on_voicing_commit)
        self.voicing_track.overrideCleared.connect(
            self._on_voicing_clear)
        voicing_page = QtWidgets.QWidget()
        voicing_page_lay = QtWidgets.QVBoxLayout(voicing_page)
        voicing_page_lay.setContentsMargins(0, 0, 0, 0)
        voicing_page_lay.setSpacing(3)
        voicing_page_lay.addWidget(self.voicing_track, 1)
        voicing_commands = QtWidgets.QHBoxLayout()
        voicing_commands.addStretch(1)
        self.voicing_reset = QtWidgets.QPushButton("Reset voicing curve")
        self.voicing_reset.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)
        )
        self.voicing_reset.setToolTip(
            "Restore the measured/generated harmonic-noise balance"
        )
        self.voicing_reset.clicked.connect(
            self.voicing_track.clear_override)
        voicing_commands.addWidget(self.voicing_reset)
        voicing_page_lay.addLayout(voicing_commands)
        self.parameter_stack.addWidget(voicing_page)
        self.vocal_tract_track = VocalTractTrack()
        self.vocal_tract_track.setXLink(self.waveform.plot)
        self.vocal_tract_track.zoom_viewbox = self.waveform.plot.getViewBox()
        self.vocal_tract_track.targetsCommitted.connect(
            self._on_vocal_tract_commit)
        self.vocal_tract_track.overrideCleared.connect(
            self._on_vocal_tract_clear)
        vocal_tract_page = QtWidgets.QWidget()
        vocal_tract_layout = QtWidgets.QVBoxLayout(vocal_tract_page)
        vocal_tract_layout.setContentsMargins(0, 0, 0, 0)
        vocal_tract_layout.setSpacing(3)
        vocal_tract_layout.addWidget(self.vocal_tract_track, 1)
        vocal_tract_commands = QtWidgets.QHBoxLayout()
        self.vocal_tract_readout = QtWidgets.QLabel("Original voice: 1.000 x")
        self.vocal_tract_readout.setStyleSheet("color:#4E4E4E")
        vocal_tract_commands.addWidget(self.vocal_tract_readout)
        vocal_tract_commands.addStretch(1)
        vocal_tract_commands.addWidget(QtWidgets.QLabel("Ratio:"))
        self.vocal_tract_value = QtWidgets.QDoubleSpinBox()
        self.vocal_tract_value.setDecimals(3)
        self.vocal_tract_value.setSingleStep(0.01)
        self.vocal_tract_value.setSuffix(" x")
        self.vocal_tract_value.setValue(1.0)
        self.vocal_tract_value.setAccessibleName(
            "Uniform vocal tract length ratio")
        self.vocal_tract_value.setToolTip(
            "Exact target/source tract-length ratio for Set all. Above 1 "
            "lowers resonances; below 1 raises them.")
        vocal_tract_commands.addWidget(self.vocal_tract_value)
        self.vocal_tract_set_all = QtWidgets.QPushButton("Set all")
        self.vocal_tract_set_all.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.vocal_tract_set_all.setToolTip(
            "Set the complete sentence to this exact tract-length ratio")
        self.vocal_tract_set_all.clicked.connect(
            self._set_uniform_vocal_tract_ratio)
        vocal_tract_commands.addWidget(self.vocal_tract_set_all)
        self.vocal_tract_chipmunk = QtWidgets.QCheckBox("Chipmunk range")
        self.vocal_tract_chipmunk.setAccessibleName("Chipmunk range")
        self.vocal_tract_chipmunk.setToolTip(
            "Unlocks the bounded exaggerated resonance range. Existing "
            "in-range curve values do not change.")
        self.vocal_tract_chipmunk.toggled.connect(
            self._on_vocal_tract_range_toggled)
        vocal_tract_commands.addWidget(self.vocal_tract_chipmunk)
        self.vocal_tract_reset = QtWidgets.QPushButton(
            "Reset vocal tract curve")
        self.vocal_tract_reset.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.vocal_tract_reset.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)
        )
        self.vocal_tract_reset.setToolTip(
            "Restore the exact original voice at 1.000 x")
        self.vocal_tract_reset.clicked.connect(
            self.vocal_tract_track.clear_override)
        vocal_tract_commands.addWidget(self.vocal_tract_reset)
        vocal_tract_layout.addLayout(vocal_tract_commands)
        self.parameter_stack.addWidget(vocal_tract_page)
        initial_expanded = bool(self.cfg.get("chipmunk_range", False))
        self.vocal_tract_chipmunk.blockSignals(True)
        self.vocal_tract_chipmunk.setChecked(initial_expanded)
        self.vocal_tract_chipmunk.blockSignals(False)
        self.vocal_tract_track.set_chipmunk_range(initial_expanded)
        self._sync_vocal_tract_value_bounds()
        self.intonation = IntonationTrack()
        self.intonation.setXLink(self.waveform.plot)
        self.intonation.zoom_viewbox = self.waveform.plot.getViewBox()
        self.intonation.blocksCommitted.connect(self._on_intonation_commit)
        self.parameter_stack.addWidget(self.intonation)
        self.recordings = RecordingTrack()
        self.recordings.setXLink(self.waveform.plot)
        self.recordings.zoom_viewbox = self.waveform.plot.getViewBox()
        self.recordings.overrideChanged.connect(self._set_unit_override)
        self.recordings.detailsRequested.connect(self._show_recording_details)
        self.recordings.pitchmarksRequested.connect(
            self._show_unit_pitchmarks)
        self.recordings.joinDiagnosticRequested.connect(
            self._show_join_loudness_for_row)
        self.recordings_page = QtWidgets.QWidget()
        recordings_layout = QtWidgets.QVBoxLayout(self.recordings_page)
        recordings_layout.setContentsMargins(0, 0, 0, 0)
        recordings_layout.setSpacing(3)
        recordings_layout.addWidget(self.recordings, 1)
        recordings_commands = QtWidgets.QHBoxLayout()
        recordings_commands.addStretch(1)
        self.recordings_broadband_audit = QtWidgets.QPushButton(
            "Export Broadband Impulse Join Audit...")
        self.recordings_broadband_audit.setToolTip(
            "Create a waveform/STFT image that marks broadband splice "
            "impulses, handoff spans, and their rendered-phone context")
        self.recordings_broadband_audit.clicked.connect(
            self.on_export_broadband_impulse_join_audit)
        recordings_commands.addWidget(self.recordings_broadband_audit)
        self.recordings_mora_details = QtWidgets.QPushButton(
            "Inspect selected mora contributions...")
        self.recordings_mora_details.setToolTip(
            "Show every UTAU source slice contributing to the selected mora")
        self.recordings_mora_details.clicked.connect(
            self._show_selected_mora_contributions)
        self.recordings_mora_details.setVisible(False)
        recordings_commands.addWidget(self.recordings_mora_details)
        recordings_layout.addLayout(recordings_commands)
        self.parameter_stack.addWidget(self.recordings_page)
        self.japanese_editor = JapaneseEditorPanel()
        self.japanese_editor.editRequested.connect(self._on_japanese_edit)
        self.japanese_editor.moraNavigationRequested.connect(
            self._on_japanese_mora_selected)
        self.waveform.selectionChanged.connect(
            self._on_waveform_selection_changed)
        self.waveform.playheadChanged.connect(
            self.japanese_editor.set_playhead)
        self.waveform.plot.getViewBox().sigXRangeChanged.connect(
            self._on_waveform_view_range_changed)
        self.japanese_page = QtWidgets.QScrollArea()
        self.japanese_page.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.japanese_page.setWidgetResizable(True)
        self.japanese_page.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        self.japanese_page.setWidget(self.japanese_editor)
        self.parameter_stack.addWidget(self.japanese_page)
        self.asaxi_editor = AsaxiMoraEditorPanel()
        self.asaxi_editor.editRequested.connect(self._on_asaxi_mora_edit)
        self.asaxi_editor.moraNavigationRequested.connect(
            self._on_asaxi_mora_selected)
        self.waveform.playheadChanged.connect(
            self.asaxi_editor.set_playhead)
        self.asaxi_page = QtWidgets.QScrollArea()
        self.asaxi_page.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.asaxi_page.setWidgetResizable(True)
        self.asaxi_page.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        self.asaxi_page.setWidget(self.asaxi_editor)
        self.parameter_stack.addWidget(self.asaxi_page)
        env_lay.addWidget(self.parameter_stack)
        self.editor_splitter.addWidget(env_group)
        self.editor_splitter.setStretchFactor(0, 3)
        self.editor_splitter.setStretchFactor(1, 1)
        self.editor_splitter.setSizes([430, 170])
        self._alignment_expanded_height = 430
        self._parameter_expanded_height = 170
        right.addWidget(self.editor_splitter, 1)

        self.mode_tabs = QtWidgets.QTabWidget()
        self.mode_tabs.addTab(speech_page, "Speech")
        self.sentences_view = SentencesView()
        self.sentences_view.follow_spoken_sentence.setChecked(bool(
            self.cfg.get("follow_spoken_sentence", True)))
        self.sentences_view.followChanged.connect(
            self._on_follow_spoken_sentence_toggled)
        self.sentences_view.importRequested.connect(self.on_import_text_file)
        self.sentences_view.addRequested.connect(
            self._add_sentence_from_sentences_view)
        self.sentences_view.playAllRequested.connect(self._play_all_sentences)
        self.sentences_view.stopRequested.connect(self.on_stop)
        self.sentences_view.generateRequested.connect(self._generate_sentence)
        self.sentences_view.rerenderAllRequested.connect(self.on_rerender_all)
        self.sentences_view.clearRequested.connect(self._clear_all_sentences)
        self.sentences_view.moveRequested.connect(self._move_sentence)
        self.sentences_view.playRequested.connect(self._play_sentence)
        self.sentences_view.openRequested.connect(self._open_sentence_mode)
        self.sentences_view.speakerRequested.connect(
            self._show_sentence_speaker_menu)
        self.sentences_view.textEdited.connect(
            self._on_sentences_text_edited)
        self.sentences_view.selectionChanged.connect(
            self._on_sentences_selection_changed)
        self.sentences_view.phraseOrderChanged.connect(
            self._reorder_phrases)
        self.sentences_view.phrasePlayRequested.connect(self._play_phrase)
        self.sentences_view.phraseOpenRequested.connect(
            self._open_phrase_in_speech)
        self.sentences_view.phraseContextRequested.connect(
            self._phrase_context_menu)
        self.sentences_view.exportSelectedRequested.connect(
            self._export_selected_sentences)
        self.sentences_view.removeSelectedRequested.connect(
            self._remove_selected_sentences)
        self.sentences_view.moveGroupRequested.connect(
            self._move_sentence_group)
        self.sentences_view.gain.valueChanged.connect(
            self._on_all_sentences_gain_changed)
        self.sentences_view.gain.clippingChanged.connect(
            self._on_allow_clipping_changed)
        self.mode_tabs.addTab(self.sentences_view, "Sentences")
        self.mode_tabs.currentChanged.connect(self._on_mode_tab_changed)
        self.sidebar_toggle = QtWidgets.QToolButton()
        self.sidebar_toggle.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_ArrowLeft))
        self.sidebar_toggle.setToolTip("Collapse speaker controls")
        self.sidebar_toggle.clicked.connect(self._toggle_sidebar)
        self.mode_tabs.setCornerWidget(self.sidebar_toggle, Qt.TopLeftCorner)
        root.addWidget(self.mode_tabs, 1)
        wanted_parameter = str(self.cfg.get("parameter_mode") or "timing")
        pidx = self.parameter_mode.findData(wanted_parameter)
        self.parameter_mode.setCurrentIndex(max(0, pidx))
        self._on_parameter_mode()

    def _toolbtn(self, text, icon_name, slot):
        b = QtWidgets.QPushButton("  " + text)
        try:
            b.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, icon_name)))
        except Exception:
            pass
        b.setStyleSheet("text-align:left;")
        if slot:
            b.clicked.connect(slot)
        return b

    # -- multi-sentence project state ---------------------------------------
    @staticmethod
    def _sentence_label(index, state):
        text = " ".join(str(state.get("text") or "").split())
        if not text:
            text = "Untitled sentence"
        if len(text) > 34:
            text = text[:31].rstrip() + "..."
        return "%d. %s" % (index + 1, text)

    @staticmethod
    def _pending_action(state):
        if not state:
            return ""
        if state.get("needs_generate"):
            return "generate"
        if state.get("needs_rerender"):
            return "rerender"
        return ""

    @staticmethod
    def _style_flag(widget, name, value):
        if widget is None:
            return
        value = bool(value)
        if bool(widget.property(name)) == value:
            return
        widget.setProperty(name, value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _set_state_pending(self, state, action, reason="",
                           bump_revision=True):
        if state is None:
            return ""
        action = str(action or "")
        active = (self.sentences[self._active_sentence_index]
                  if hasattr(self, "sentences") and
                  0 <= self._active_sentence_index < len(self.sentences)
                  else None)
        has_synthesis = bool(
            state.get("synthesis") is not None or
            (state is active and self.current is not None))
        if action == "rerender" and not has_synthesis:
            action = "generate"
        if action == "generate":
            state["needs_generate"] = True
            state["needs_rerender"] = False
            state["rendered"] = False
        elif action == "rerender":
            if not state.get("needs_generate"):
                state["needs_rerender"] = True
        else:
            state["needs_generate"] = False
            state["needs_rerender"] = False
        state["pending_reason"] = str(reason or "") if action else ""
        if bump_revision:
            state["_edit_revision"] = int(
                state.get("_edit_revision") or 0) + 1
        return self._pending_action(state)

    def _mark_active_pending(self, action, reason=""):
        if (self._switching_sentence or self._active_sentence_index < 0 or
                self._active_sentence_index >= len(self.sentences)):
            return ""
        state = self.sentences[self._active_sentence_index]
        result = self._set_state_pending(state, action, reason)
        self._refresh_pending_ui()
        return result

    def _clear_state_pending(self, state):
        self._set_state_pending(state, "", bump_revision=False)

    def _refresh_pending_ui(self):
        if (getattr(self, "_refreshing_pending_ui", False) or
                not hasattr(self, "btn_gen")):
            return
        self._refreshing_pending_ui = True
        try:
            self._refresh_pending_ui_impl()
        finally:
            self._refreshing_pending_ui = False

    def _refresh_pending_ui_impl(self):
        if not hasattr(self, "btn_gen"):
            return
        active = (self.sentences[self._active_sentence_index]
                  if 0 <= self._active_sentence_index < len(self.sentences)
                  else None)
        action = self._pending_action(active)
        can_rerender = bool(
            action == "rerender" and self.current is not None and
            getattr(self.waveform, "segments", None))
        self.btn_rerender.setEnabled(can_rerender)
        self._style_flag(self.btn_rerender, "renderPending",
                         action == "rerender")
        self._style_flag(self.btn_gen, "generatePending",
                         action == "generate")
        any_generate = any(self._pending_action(state) == "generate"
                           for state in self.sentences)
        any_rerender = any(self._pending_action(state) == "rerender"
                           for state in self.sentences)
        self._style_flag(self.btn_gen_all, "generatePending", any_generate)
        self._style_flag(self.btn_rerender_all, "renderPending", any_rerender)
        waveform_action = action if action == "rerender" else ""
        self.waveform.set_pending_action(
            waveform_action,
            str(active.get("pending_reason") or "")
            if active and waveform_action else "")
        title = "Waveform  /  Phonemes"
        if action == "rerender":
            title += " - Re-render pending"
        self.wf_group.setTitle(title)
        if hasattr(self, "sentences_view"):
            for index, state in enumerate(self.sentences):
                self.sentences_view.set_pending(
                    index, self._pending_action(state))

    def _new_sentence_state(self, text=""):
        return {
            "text": str(text),
            "rendered_text": str(text),
            "input_mode": (self.input_mode.currentData() or "text"),
            "engine": self._engine(),
            "language": self.lang.currentText(),
            "lang_code": self._current_lang_code(),
            "voicebank": self._current_voicebank() or "",
            "speed": self._speed_factor(),
            "pitch_hz": float(self.pitch.value()),
            "rendered_pitch_hz": None,
            "pitch_manual": bool(self._pitch_user_edited),
            "fall_pct": float(self.fall.value()),
            "rendered_fall_pct": None,
            "output_gain_db": float(self.output_gain.value()),
            "applied_gain_db": None,
            "pre_gain_peak": 0.0,
            "vocal_tract_length_ratio": float(
                self.cfg.get("vocal_tract_length_ratio", 1.0)),
            "chipmunk_range": bool(
                self.cfg.get("chipmunk_range", False)),
            "applied_vocal_tract_length_ratio": None,
            "fault_mode": self._fault_mode(),
            # Empty means the generated/built-in voice policy. The effective
            # render setting is retained separately on Synthesis.join_settings.
            "join_settings": {},
            "pitch_fault_target": self._pitch_fault_target,
            "parameter_mode": self.parameter_mode.currentData() or "timing",
            "view_mode": "speech",
            "synthesis": None,
            "editor_segments": [],
            "timing_factors": [],
            "preview_audio": np.zeros(1, np.float32),
            "preview_sr": 16000,
            "needs_rerender": False,
            "needs_generate": False,
            "pending_reason": "",
            "rendered": False,
            "phrases": [],
            "japanese_state": je.new_edit_state(),
            "asaxi_state": asaxi_editing.new_edit_state(str(text)),
            "_edit_revision": 0,
        }

    def _init_sentence_collection(self):
        self.sentences = [self._new_sentence_state(self.text.text())]
        self._active_sentence_index = 0
        self._refresh_sentence_selector(0)

    def _refresh_sentence_selector(self, keep=None):
        keep = self._active_sentence_index if keep is None else int(keep)
        self.sentence_select.blockSignals(True)
        self.sentence_select.clear()
        for i, state in enumerate(self.sentences):
            self.sentence_select.addItem(self._sentence_label(i, state), i)
        if self.sentences:
            keep = max(0, min(len(self.sentences) - 1, keep))
            self.sentence_select.setCurrentIndex(keep)
        self.sentence_select.blockSignals(False)

    def _capture_active_sentence(self):
        if (self._switching_sentence or self._active_sentence_index < 0 or
                self._active_sentence_index >= len(self.sentences)):
            return
        state = self.sentences[self._active_sentence_index]
        stored_synthesis = state.get("synthesis")
        editor_owns_render = (
            self._editor_sentence_state is state and
            self.current is stored_synthesis
        )
        mixed_language = (
            self.lang.currentData() == MIXED_SELECTION_DATA)
        voice_item = self.voicebank.currentItem()
        mixed_voice = bool(
            voice_item and
            voice_item.data(Qt.UserRole) == MIXED_SELECTION_DATA)
        if editor_owns_render and self.current is not None:
            if self.current.vocal_tract_mode == "curve":
                tract_ratio = vocal_tract.ratio_curve_summary(
                    self.vocal_tract_track.targets())
            else:
                tract_ratio = float(getattr(
                    self.current, "vocal_tract_requested_ratio",
                    state.get("vocal_tract_length_ratio", 1.0)))
        else:
            tract_ratio = float(
                state.get("vocal_tract_length_ratio", 1.0))
        state.update({
            "text": self.text.text(),
            "input_mode": self.input_mode.currentData() or "text",
            "engine": self._engine(),
            "language": (state.get("language") if mixed_language else
                         self.lang.currentText()),
            "lang_code": (state.get("lang_code") if mixed_language else
                          self._current_lang_code()),
            "voicebank": (state.get("voicebank") if mixed_voice else
                          self._current_voicebank() or ""),
            "speed": self._speed_factor(),
            "pitch_hz": float(self.pitch.value()),
            "pitch_manual": bool(self._pitch_user_edited),
            "fall_pct": float(self.fall.value()),
            "output_gain_db": float(self.output_gain.value()),
            "vocal_tract_length_ratio": tract_ratio,
            "chipmunk_range": self.vocal_tract_chipmunk.isChecked(),
            "fault_mode": self._fault_mode(),
            "pitch_fault_target": self._pitch_fault_target,
            "parameter_mode": self.parameter_mode.currentData() or "timing",
            "needs_rerender": bool(state.get("needs_rerender")),
            "needs_generate": bool(state.get("needs_generate")),
            "japanese_state": je.normalize_edit_state(
                state.get("japanese_state")),
            "asaxi_state": asaxi_editing.normalize_edit_state(
                state.get("asaxi_state")),
        })
        if hasattr(self, "mode_tabs"):
            state["view_mode"] = ("speech", "sentences")[
                max(0, min(1, self.mode_tabs.currentIndex()))]
        # A background batch render can update ``state`` while Speech still
        # displays an older or blank editor.  Never let that stale editor
        # replace the generated synthesis or its parameter metadata.
        if editor_owns_render and self.current is not None:
            state["editor_segments"] = copy.deepcopy(self.waveform.segments)
            state["timing_factors"] = list(self.waveform.factors())
            state["preview_audio"] = np.asarray(
                self.waveform.audio, np.float32)
            state["preview_sr"] = int(self.waveform.sr)
            self._capture_phrase_snapshots(state)
        self.sentence_select.setItemText(
            self._active_sentence_index,
            self._sentence_label(self._active_sentence_index, state))

    def _restore_sentence(self, index, restore_view=True,
                          hydrate_editor=True):
        if not (0 <= index < len(self.sentences)):
            return
        state = self.sentences[index]
        self._switching_sentence = True
        try:
            wanted_engine = state.get("engine") or "diphone"
            engine_changed = self._engine() != wanted_engine
            engine_index = self.engine.findData(wanted_engine)
            if engine_index >= 0:
                self.engine.setCurrentIndex(engine_index)
            language = str(state.get("language") or "")
            if self.lang.findText(language) >= 0:
                self.lang.setCurrentText(language)
            wanted_voice = str(state.get("voicebank") or "")
            if (engine_changed or
                    not self._select_existing_voicebank(wanted_voice)):
                self._refresh_voicebanks(keep=wanted_voice)
            mode_index = self.input_mode.findData(
                state.get("input_mode") or "text")
            self.input_mode.setCurrentIndex(max(0, mode_index))
            self.text.setText(str(state.get("text") or ""))
            speed = max(0.25, min(4.0, float(state.get("speed") or 1.0)))
            self.speed.setValue(int(round(100 * np.log2(speed))))
            self.pitch.setValue(float(state.get("pitch_hz") or 185.0))
            self._pitch_user_edited = bool(state.get("pitch_manual", False))
            self.fall.setValue(float(state.get("fall_pct") or 0.0))
            self.output_gain.setValue(float(
                state.get("output_gain_db", self.cfg.get(
                    "output_gain_db", 0.0))))
            expanded = bool(state.get("chipmunk_range", False))
            self.vocal_tract_chipmunk.blockSignals(True)
            self.vocal_tract_chipmunk.setChecked(expanded)
            self.vocal_tract_chipmunk.blockSignals(False)
            self.vocal_tract_track.set_chipmunk_range(expanded)
            self._sync_vocal_tract_value_bounds()
            self.vocal_tract_value.setValue(
                self.vocal_tract_track.profile.clamp(
                    float(state.get("vocal_tract_length_ratio", 1.0)),
                    expanded))
            self._pitch_fault_target = state.get("pitch_fault_target")
            faults = dict(state.get("fault_mode") or {})
            for key, action in self.fault_actions.items():
                action.blockSignals(True)
                action.setChecked(bool(faults.get(key, False)))
                action.blockSignals(False)
            bits = int(faults.get("bit_depth") or 0)
            if bits not in self.bit_depth_actions:
                bits = 0
            self.bit_depth_actions[bits].setChecked(True)
            parameter_index = self.parameter_mode.findData(
                state.get("parameter_mode") or "timing")
            self.parameter_mode.setCurrentIndex(max(0, parameter_index))
            self.japanese_editor.set_state(state.get("japanese_state"))
            self.asaxi_editor.set_state(state.get("asaxi_state"))
            self._refresh_japanese_runtime_controls()
            syn = state.get("synthesis")
            if syn is not None:
                if hydrate_editor:
                    editor_segments = state.get("editor_segments") or []
                    raw_audio = state.get("preview_audio")
                    display = copy.copy(syn)
                    display.samples = np.asarray(
                        raw_audio if raw_audio is not None else syn.samples,
                        np.float32)
                    display.sr = int(state.get("preview_sr") or syn.sr)
                    if editor_segments:
                        display.segments = editor_segments
                    self._show_synthesis(
                        syn,
                        display=display,
                        timing_factors=state.get("timing_factors") or [],
                    )
                    self._editor_sentence_state = state
                    self.waveform.set_playhead(0.0)
                else:
                    self.current = syn
            else:
                self.current = None
                if hydrate_editor:
                    raw_audio = state.get("preview_audio")
                    preview = np.asarray(
                        raw_audio if raw_audio is not None else np.zeros(1),
                        np.float32)
                    self.waveform.set_synthesis(fc.Synthesis(
                        preview, int(state.get("preview_sr") or 16000), [],
                        text=state.get("text") or "",
                        lang=state.get("lang_code") or "",
                        voicebank=state.get("voicebank") or ""))
                    self.timing.set_segments([], [])
                    self.pitch_track.set_data([], [], [], [])
                    self.vocal_tract_track.set_data([], [], [], [])
                    self.intonation.set_blocks([])
                    self.recordings.set_data([], {}, {}, {})
                    self._editor_sentence_state = state
            self._update_fault_availability()
            if restore_view and hasattr(self, "mode_tabs"):
                mode = {"speech": 0, "sentences": 1}.get(
                    state.get("view_mode"), 0)
                self.mode_tabs.setCurrentIndex(mode)
                self._mode_tab_index = mode
                self._update_fault_availability()
        finally:
            self._switching_sentence = False
        self._sync_speaker_control()
        self._apply_voice_language_compatibility(auto_select=False)
        self._update_parameter_availability()
        self._refresh_gain_controls()
        self._refresh_pending_ui()
        self.statusBar().showMessage(
            "Status: sentence %d of %d" % (index + 1, len(self.sentences)))

    def _on_sentence_selected(self, index):
        if self._switching_sentence or not (0 <= index < len(self.sentences)):
            return
        self._capture_active_sentence()
        self._active_sentence_index = index
        speech_visible = (
            not hasattr(self, "mode_tabs") or
            self.mode_tabs.currentIndex() == 0)
        self._restore_sentence(
            index,
            restore_view=speech_visible,
            hydrate_editor=speech_visible,
        )

    @staticmethod
    def _sentence_has_preview_audio(state):
        raw_audio = state.get("preview_audio")
        return bool(
            state.get("synthesis") is not None and
            np.asarray(raw_audio if raw_audio is not None else []).size > 1)

    def _apply_sentence_text_edit(self, index, text):
        index = int(index)
        if not (0 <= index < len(self.sentences)):
            return False
        state = self.sentences[index]
        text = str(text)
        if text == state.get("text"):
            return False
        rendered_text = str(state.get("rendered_text", state.get("text", "")))
        if (text != rendered_text and
                "_text_edit_revert" not in state):
            state["_text_edit_revert"] = {
                "phrases": copy.deepcopy(state.get("phrases") or []),
                "phrase_previews": state.get("phrase_previews") or {},
                "pending_action": self._pending_action(state),
                "pending_reason": str(state.get("pending_reason") or ""),
                "rendered": bool(state.get("rendered")),
            }
        state["text"] = text
        if text == rendered_text:
            previous = state.pop("_text_edit_revert", {})
            if previous.get("phrases") is not None:
                state["phrases"] = copy.deepcopy(previous["phrases"])
            if previous.get("phrase_previews") is not None:
                state["phrase_previews"] = previous["phrase_previews"]
            if state.get("pending_reason") == "Text changed":
                old_action = str(previous.get("pending_action") or "")
                self._set_state_pending(
                    state, old_action,
                    str(previous.get("pending_reason") or ""))
                state["rendered"] = bool(
                    previous.get("rendered",
                                 self._sentence_has_preview_audio(state)))
        else:
            state["phrases"] = []
            state["phrase_previews"] = {}
            self._set_state_pending(state, "generate", "Text changed")
        if (hasattr(self, "sentences_view") and
                index < len(self.sentences_view.row_widgets)):
            self.sentences_view.row_widgets[index].play_button.setEnabled(
                bool(state.get("rendered") and
                     self._sentence_has_preview_audio(state)))
        return True

    def _on_sentence_text_edited(self, _text):
        if (self._switching_sentence or self._active_sentence_index < 0 or
                self._active_sentence_index >= len(self.sentences)):
            return
        state = self.sentences[self._active_sentence_index]
        self._apply_sentence_text_edit(
            self._active_sentence_index, str(_text))
        self.sentence_select.setItemText(
            self._active_sentence_index,
            self._sentence_label(self._active_sentence_index, state))
        self._refresh_pending_ui()

    def on_add_sentence(self):
        self._capture_active_sentence()
        self.sentences.append(self._new_sentence_state())
        index = len(self.sentences) - 1
        self._active_sentence_index = index
        self._refresh_sentence_selector(index)
        self._restore_sentence(index)
        self.text.setFocus()

    def _add_sentence_from_sentences_view(self, selected):
        self._capture_active_sentence()
        selected = sorted(int(index) for index in (selected or [])
                          if 0 <= int(index) < len(self.sentences))
        insert_at = (selected[-1] + 1 if selected else len(self.sentences))
        state = self._new_sentence_state()
        state["view_mode"] = "sentences"
        self.sentences.insert(insert_at, state)
        self._active_sentence_index = insert_at
        self._refresh_sentence_selector(insert_at)
        self._restore_sentence(insert_at, restore_view=False)
        self._refresh_sentences_view()
        self.sentences_view.set_selected_indices([insert_at])
        if insert_at < len(self.sentences_view.row_widgets):
            editor = self.sentences_view.row_widgets[
                insert_at].findChild(SentenceTextEdit)
            if editor is not None:
                editor.setFocus()

    def _on_sentences_selection_changed(self, indices):
        if not hasattr(self, "mode_tabs") or self.mode_tabs.currentIndex() != 1:
            return
        selected = sorted(int(index) for index in (indices or [])
                          if 0 <= int(index) < len(self.sentences))
        self._refresh_contextual_control_availability()
        if not selected:
            return
        target = (self._active_sentence_index
                  if self._active_sentence_index in selected else selected[0])
        if target != self._active_sentence_index:
            self._capture_active_sentence()
            self._active_sentence_index = target
            self._refresh_sentence_selector(target)
            # The Speech editor is hidden. Restore sentence controls now and
            # hydrate waveform/parameter graphics only when Speech is opened.
            self._restore_sentence(
                target, restore_view=False, hydrate_editor=False)
        self._sync_sentence_sidebar_values(selected)

    def _on_sentences_text_edited(self, index, text):
        index = int(index)
        if not (0 <= index < len(self.sentences)):
            return
        text = str(text)
        if not self._apply_sentence_text_edit(index, text):
            return
        self._refresh_sentence_selector(self._active_sentence_index)
        if index == self._active_sentence_index:
            self.text.blockSignals(True)
            self.text.setText(text)
            self.text.blockSignals(False)
        self._refresh_gain_controls()
        self._refresh_pending_ui()

    def _show_sentence_speaker_menu(self, index, global_pos):
        index = int(index)
        if not (0 <= index < len(self.sentences)):
            return
        self.sentences_view.set_selected_indices([index])
        self._on_sentences_selection_changed([index])
        menu = QtWidgets.QMenu()
        try:
            voices = self._ab().voicebanks()
        except Exception:
            voices = []
        current = str(self.sentences[index].get("voicebank") or "")
        for voice in voices:
            name = str(voice.get("name") or "")
            if not name:
                continue
            action = menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(
                lambda _checked=False, value=name:
                self._set_sentence_speaker(index, value))
        if not menu.actions():
            empty = menu.addAction("No speakers available")
            empty.setEnabled(False)
        menu.exec_(global_pos)

    def _set_sentence_speaker(self, index, speaker):
        index = int(index)
        if not (0 <= index < len(self.sentences)):
            return
        state = self.sentences[index]
        state["voicebank"] = str(speaker)
        state["rendered"] = False
        self._set_state_pending(state, "generate", "Voicebank changed")
        state["applied_gain_db"] = None
        state["pre_gain_peak"] = 0.0
        if index == self._active_sentence_index:
            self._refresh_voicebanks(keep=speaker)
            self._sync_speaker_control()
        self._refresh_sentences_view()
        self.sentences_view.set_selected_indices([index])
        self._refresh_gain_controls()
        self._refresh_pending_ui()

    def on_duplicate_sentence(self):
        self._capture_active_sentence()
        if not self.sentences:
            return
        duplicate = self._sentence_state_snapshot(
            self.sentences[self._active_sentence_index])
        self.sentences.insert(self._active_sentence_index + 1, duplicate)
        self._active_sentence_index += 1
        self._refresh_sentence_selector(self._active_sentence_index)
        self._restore_sentence(self._active_sentence_index)

    def on_remove_sentence(self):
        if not self.sentences:
            return
        if len(self.sentences) == 1:
            self.sentences[0] = self._new_sentence_state()
            self._active_sentence_index = 0
            self._refresh_sentence_selector(0)
            self._restore_sentence(0)
            return
        if QtWidgets.QMessageBox.question(
                self, "Remove sentence",
                "Remove this sentence and its unsaved parameter edits?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        del self.sentences[self._active_sentence_index]
        self._active_sentence_index = min(
            self._active_sentence_index, len(self.sentences) - 1)
        self._refresh_sentence_selector(self._active_sentence_index)
        self._restore_sentence(self._active_sentence_index)

    @staticmethod
    def _new_phrase_state(text):
        return {"id": uuid.uuid4().hex, "text": str(text),
                "speaker": "", "dictionary": "", "fault_mode": {},
                "phones": [], "timing_factors": [],
                "pitch_override": [], "pitch_mode": "",
                "voicing_override": [], "voicing_mode": ""}

    def _ensure_phrase_states(self, state):
        parts = fc.split_sentence_phrases(state.get("text") or "")
        existing = [dict(phrase) for phrase in (state.get("phrases") or [])]
        if [phrase.get("text") for phrase in existing] == parts:
            state["phrases"] = existing
            return existing
        unused = list(existing)
        phrases = []
        for part in parts:
            match = next((phrase for phrase in unused
                          if phrase.get("text") == part), None)
            if match is not None:
                unused.remove(match)
                phrases.append(match)
            else:
                phrases.append(self._new_phrase_state(part))
        state["phrases"] = phrases
        return phrases

    def _capture_phrase_snapshots(
            self, state, synthesis=None, segments=None, audio=None, sr=None,
            timing_factors=None):
        phrases = self._ensure_phrase_states(state)
        direct = synthesis is not None
        owns_editor = (
            self._editor_sentence_state is state or
            self._editor_sentence_state is None and
            0 <= self._active_sentence_index < len(self.sentences) and
            self.sentences[self._active_sentence_index] is state
        )
        render = synthesis if direct else self.current
        render_segments = list(
            segments if segments is not None else
            (render.segments if direct and render is not None else
             self.waveform.segments))
        render_audio = np.asarray(
            audio if audio is not None else
            (render.samples if direct and render is not None else
             self.waveform.audio), np.float32)
        render_sr = int(
            sr if sr is not None else
            (render.sr if direct and render is not None else self.waveform.sr))
        if (not phrases or render is None or
                (not direct and not owns_editor) or
                not render_segments):
            return
        weights = []
        for phrase in phrases:
            phone_count = len(phrase.get("phones") or [])
            text_weight = len(re.sub(
                r"\s+|\[pau\]", "", str(phrase.get("text") or ""),
                flags=re.IGNORECASE))
            weights.append(max(1, phone_count, text_weight))
        spans = fc.phrase_playback_spans(
            render_segments, weights)
        factors = list(timing_factors or ())
        if len(factors) != len(render_segments):
            factors = (self.waveform.factors() if not direct else
                       [1.0] * len(render_segments))
        active_ids = {str(phrase.get("id") or "") for phrase in phrases}
        previews = {
            phrase_id: preview
            for phrase_id, preview in
            dict(state.get("phrase_previews") or {}).items()
            if str(phrase_id) in active_ids
        }
        for phrase_index, (phrase, span) in enumerate(zip(phrases, spans)):
            first = span.get("spoken_start_index")
            last = span.get("spoken_end_index")
            if first is not None and last is not None:
                spoken_indices = [
                    index for index in range(int(first), int(last) + 1)
                    if render_segments[index].phone != "pau"
                ]
                start = float(span["spoken_start"])
                end = float(span["spoken_end"])
                phrase["phones"] = [
                    render_segments[index].phone
                    for index in spoken_indices]
                phrase["timing_factors"] = [
                    float(factors[index]) for index in spoken_indices]
                phrase["segment_indices"] = [int(first), int(last)]
                phrase["start"] = start
                phrase["end"] = end
                phrase["pitch_mode"] = str(render.pitch_mode or "")
                pitch_source = list(render.pitch_override or [])
                if (not pitch_source and
                        render.pitch_mode == "intonation"):
                    pitch_source = list(render.targets or [])
                phrase["pitch_override"] = [
                    [float(time - start), float(value)]
                    for time, value in pitch_source
                    if start <= time <= end]
                phrase["voicing_mode"] = str(
                    render.voicing_mode or "")
                phrase["voicing_override"] = [
                    [float(time - start), float(value)]
                    for time, value in (render.voicing_override or [])
                    if start <= time <= end
                ]
            a = max(0, int(round(float(span["start"]) *
                                 render_sr)))
            b = min(len(render_audio), int(round(
                float(span["end"]) * render_sr)))
            if phrase_index == 0:
                a = 0
            if phrase_index == len(phrases) - 1:
                b = len(render_audio)
            phrase["playback_start"] = a / float(render_sr)
            phrase["playback_end"] = b / float(render_sr)
            previews[phrase["id"]] = (
                render_audio[a:b], render_sr)
        if (len(phrases) == 1 and
                phrases[0]["id"] not in previews and
                render_audio.size > 1):
            # A sentence row should never appear blank after a successful
            # render merely because a frontend supplied no phrase block.
            previews[phrases[0]["id"]] = (
                render_audio, render_sr)
        state["phrase_previews"] = previews

    def _refresh_sentences_view(self):
        if not hasattr(self, "sentences_view"):
            return
        for state in self.sentences:
            sentence_voice = str(state.get("voicebank") or "Default")
            engine = str(state.get("engine") or self._engine())
            state["_speaker_name"] = sentence_voice
            state["_speaker_icon"] = self._speaker_portrait_path(
                engine, sentence_voice)
            for phrase in self._ensure_phrase_states(state):
                phrase_voice = str(phrase.get("speaker") or sentence_voice)
                phrase["_speaker_name"] = phrase_voice
                phrase["_speaker_icon"] = self._speaker_portrait_path(
                    engine, phrase_voice)
        self.sentences_view.refresh(self.sentences)

    def _refresh_sentences_view_preserving_focus(self, selected=None):
        if not hasattr(self, "sentences_view"):
            return
        focus = QtWidgets.QApplication.focusWidget()
        editor_index = None
        cursor_position = cursor_anchor = None
        for index, row in enumerate(self.sentences_view.row_widgets):
            editor = row.findChild(SentenceTextEdit)
            if editor is focus:
                editor_index = index
                cursor = editor.textCursor()
                cursor_position = cursor.position()
                cursor_anchor = cursor.anchor()
                break
        selected = (self.sentences_view.selected_sentence_indices()
                    if selected is None else list(selected))
        self._refresh_sentences_view()
        self.sentences_view.set_selected_indices(selected)
        if (editor_index is None or
                not (0 <= editor_index < len(
                    self.sentences_view.row_widgets))):
            return
        editor = self.sentences_view.row_widgets[
            editor_index].findChild(SentenceTextEdit)
        if editor is None:
            return
        cursor = editor.textCursor()
        limit = max(0, len(editor.toPlainText()))
        cursor.setPosition(min(int(cursor_anchor or 0), limit))
        cursor.setPosition(
            min(int(cursor_position or 0), limit),
            QtGui.QTextCursor.KeepAnchor)
        editor.setTextCursor(cursor)
        editor.setFocus(Qt.OtherFocusReason)

    def _on_mode_tab_changed(self, index):
        if self._switching_sentence or not self.sentences:
            return
        previous = int(getattr(self, "_mode_tab_index", 0))
        if previous == 0:
            self._capture_active_sentence()
        if previous != int(index):
            self.sentences_view.clear_selection()
        self._mode_tab_index = int(index)
        state = self.sentences[self._active_sentence_index]
        state["view_mode"] = ("speech", "sentences")[
            max(0, min(1, int(index)))]
        self.btn_addph.setVisible(index == 0)
        self.btn_delph.setVisible(index == 0)
        if index == 1:
            self._capture_phrase_snapshots(state)
            self._refresh_sentences_view()
        else:
            # Rehydrate the full editor after generation or edits initiated
            # from Sentences. This keeps boundaries and parameter widgets in
            # sync even when the tab was never visited during rendering.
            self._restore_sentence(
                self._active_sentence_index, restore_view=False)
        self._refresh_contextual_control_availability()
        self._update_fault_availability()
        self._update_shortcut_hints()

    def _toggle_alignment_collapsed(self, collapsed):
        collapsed = bool(collapsed)
        sizes = self.editor_splitter.sizes()
        if collapsed:
            if sizes and sizes[0] > 64:
                self._alignment_expanded_height = sizes[0]
            self.waveform.hide()
            self.alignment_group.setMaximumHeight(64)
            total = max(128, sum(sizes))
            self.editor_splitter.setSizes([64, max(64, total - 64)])
        else:
            self.alignment_group.setMaximumHeight(16777215)
            self.waveform.show()
            total = max(128, sum(sizes))
            wanted = min(max(170, self._alignment_expanded_height),
                         max(64, total - 64))
            self.editor_splitter.setSizes([wanted, max(64, total - wanted)])
        self.alignment_collapse.setText(
            "Expand alignment" if collapsed else "Collapse alignment")

    def _toggle_parameter_collapsed(self, collapsed):
        collapsed = bool(collapsed)
        sizes = self.editor_splitter.sizes()
        if collapsed:
            if len(sizes) > 1 and sizes[1] > 64:
                self._parameter_expanded_height = sizes[1]
            self.parameter_stack.hide()
            self.parameter_group.setMaximumHeight(64)
            total = max(128, sum(sizes))
            self.editor_splitter.setSizes([max(64, total - 64), 64])
        else:
            self.parameter_group.setMaximumHeight(16777215)
            self.parameter_stack.show()
            total = max(128, sum(sizes))
            wanted = min(max(110, self._parameter_expanded_height),
                         max(64, total - 64))
            self.editor_splitter.setSizes([max(64, total - wanted), wanted])
        self.parameter_collapse.setText(
            "Expand parameters" if collapsed else "Collapse parameters")

    def _toggle_sidebar(self):
        visible = self.sidebar.isVisible()
        self.sidebar.setVisible(not visible)
        self.sidebar_toggle.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_ArrowRight if visible
            else QtWidgets.QStyle.SP_ArrowLeft))
        self.sidebar_toggle.setToolTip(
            "Expand speaker controls" if visible else
            "Collapse speaker controls")

    def _choose_speaker_portrait(self):
        voice = self._current_voicebank()
        if not voice:
            QtWidgets.QMessageBox.information(
                self, "Speaker portrait", "Select a voicebank first.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose speaker portrait", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*)")
        if not path:
            return
        try:
            installed = self._ab().install_voice_icon(voice, path)
        except (fc.BackendError, OSError, AttributeError) as error:
            QtWidgets.QMessageBox.critical(
                self, "Speaker portrait", str(error))
            return
        self.cfg["speaker_portrait"] = path
        self.cfg.setdefault("voice_portraits", {})[
            "%s|%s" % (self._engine(), voice)] = path
        self._sync_speaker_control()
        self._persist_config()
        self._refresh_sentences_view()
        self.statusBar().showMessage(
            "Status: installed speaker portrait in %s" % installed)

    def _remove_speaker_portrait(self):
        voice = self._current_voicebank()
        if not voice:
            QtWidgets.QMessageBox.information(
                self, "Speaker portrait", "Select a voicebank first.")
            return
        if QtWidgets.QMessageBox.question(
                self, "Remove speaker icon",
                "Remove the installed icon for '%s'?\n\n"
                "This removes only speaker image files from the generated "
                "voice folder. Voice audio and the UTAU source are not "
                "touched." % voice,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        try:
            self._ab().remove_voice_icon(voice)
        except (fc.BackendError, OSError, AttributeError) as error:
            QtWidgets.QMessageBox.critical(
                self, "Speaker portrait", str(error))
            return
        key = "%s|%s" % (self._engine(), voice)
        old = self.cfg.setdefault("voice_portraits", {}).pop(key, None)
        if old and self.cfg.get("speaker_portrait") == old:
            self.cfg["speaker_portrait"] = ""
        self._sync_speaker_control()
        self._persist_config()
        self._refresh_sentences_view()
        self.statusBar().showMessage(
            "Status: removed speaker portrait for %s" % voice)

    def _move_sentence(self, index, direction):
        index = int(index)
        direction = int(direction)
        final = index + direction
        if not (0 <= index < len(self.sentences) and
                0 <= final < len(self.sentences)):
            return
        target = final if direction < 0 else final + 1
        self._move_sentence_group([int(index)], target)

    def _apply_sentence_collection(self, states, active_state=None,
                                   selected_states=None):
        self.sentences = list(states)
        if not self.sentences:
            self.sentences = [self._new_sentence_state()]
        active = next((index for index, state in enumerate(self.sentences)
                       if state is active_state), 0)
        self._active_sentence_index = max(
            0, min(len(self.sentences) - 1, active))
        self._refresh_sentence_selector(self._active_sentence_index)
        self._restore_sentence(self._active_sentence_index)
        self._refresh_sentences_view()
        selected_states = list(selected_states or [])
        selected = [index for index, state in enumerate(self.sentences)
                    if any(state is wanted for wanted in selected_states)]
        self.sentences_view.set_selected_indices(selected)

    def _move_sentence_group(self, indices, target):
        selected = sorted(set(int(index) for index in indices
                              if 0 <= int(index) < len(self.sentences)))
        if not selected:
            return
        self._capture_active_sentence()
        before = list(self.sentences)
        active = self.sentences[self._active_sentence_index]
        moving = [self.sentences[index] for index in selected]
        remaining = [state for index, state in enumerate(self.sentences)
                     if index not in selected]
        target = max(0, min(len(self.sentences), int(target)))
        target -= sum(index < target for index in selected)
        target = max(0, min(len(remaining), target))
        after = remaining[:target] + moving + remaining[target:]
        if all(left is right for left, right in zip(before, after)):
            return
        self._apply_sentence_collection(after, active, moving)
        self._push_applied_undo(
            "move sentences",
            lambda order=before, current=active, chosen=moving:
            self._apply_sentence_collection(order, current, chosen),
            lambda order=after, current=active, chosen=moving:
            self._apply_sentence_collection(order, current, chosen))

    def _remove_selected_sentences(self, indices, confirm=True):
        selected = sorted(set(int(index) for index in indices
                              if 0 <= int(index) < len(self.sentences)))
        if not selected:
            return False
        count = len(selected)
        if confirm and QtWidgets.QMessageBox.question(
                self, "Remove selected sentences",
                "Remove %d selected sentence%s and all of their parameter "
                "edits?" % (count, "" if count == 1 else "s"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return False
        self._capture_active_sentence()
        before = list(self.sentences)
        active_before = self.sentences[self._active_sentence_index]
        removed = [self.sentences[index] for index in selected]
        after = [state for index, state in enumerate(self.sentences)
                 if index not in selected]
        if not after:
            after = [self._new_sentence_state()]
        active_after = (active_before if any(
            state is active_before for state in after) else
            after[min(selected[0], len(after) - 1)])
        self._apply_sentence_collection(after, active_after)
        self._push_applied_undo(
            "remove sentences",
            lambda order=before, current=active_before, chosen=removed:
            self._apply_sentence_collection(order, current, chosen),
            lambda order=after, current=active_after:
            self._apply_sentence_collection(order, current))
        self.statusBar().showMessage(
            "Status: removed %d sentence%s" %
            (count, "" if count == 1 else "s"))
        return True

    def _export_selected_sentences(self, indices):
        self._capture_active_sentence()
        selected = sorted(set(int(index) for index in indices
                              if 0 <= int(index) < len(self.sentences)))
        if not selected:
            return False
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export selected sentences to WAV",
            os.path.join(self._project_root, "exports")
            if self._project_root else "")
        if not folder:
            return False
        exported = 0
        skipped = []
        for index in selected:
            state = self.sentences[index]
            raw_audio = state.get("preview_audio")
            samples = np.asarray(
                raw_audio if raw_audio is not None else [], np.float32)
            if not state.get("rendered") or samples.size <= 1:
                skipped.append(index + 1)
                continue
            base = "%03d_%s" % (
                index + 1, self._batch_slug(state.get("text")))
            path = os.path.join(folder, base + ".wav")
            suffix = 2
            while os.path.exists(path):
                path = os.path.join(
                    folder, "%s_%d.wav" % (base, suffix))
                suffix += 1
            try:
                fc.write_wav(path, samples,
                             int(state.get("preview_sr") or 16000))
            except Exception as error:
                QtWidgets.QMessageBox.critical(
                    self, "Export selected sentences",
                    "Could not write:\n%s\n\n%s" % (path, error))
                return False
            exported += 1
        message = "Exported %d selected WAV file%s." % (
            exported, "" if exported == 1 else "s")
        self.statusBar().showMessage("Status: " + message.lower())
        if skipped:
            QtWidgets.QMessageBox.information(
                self, "Export selected sentences", message +
                "\n\nGenerate these first: " +
                ", ".join("sentence %d" % index for index in skipped))
        elif not exported:
            QtWidgets.QMessageBox.information(
                self, "Export selected sentences",
                "Generate the selected sentences first.")
        return bool(exported)

    def _play_sentence(self, index):
        if not (0 <= index < len(self.sentences)):
            return
        selected = self.sentences_view.selected_sentence_indices()
        phrase_selected = self.sentences_view.selected_phrase_keys()
        if index in selected and len(selected) > 1 and not phrase_selected:
            return self._play_sentence_indices(selected)
        state = self.sentences[index]
        raw_audio = state.get("preview_audio")
        samples = np.asarray(raw_audio if raw_audio is not None else [],
                             np.float32)
        if samples.size <= 1 or not state.get("rendered"):
            self.statusBar().showMessage(
                "Status: generate sentence %d first" % (index + 1))
            return
        try:
            self._start_playback(
                samples, int(state.get("preview_sr") or 16000),
                highlights=self._sentence_highlights(index))
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "Playback", str(error))

    def _sentence_highlights(self, index, offset=0.0):
        if not (0 <= int(index) < len(self.sentences)):
            return []
        state = self.sentences[int(index)]
        highlights = []
        for phrase_index, phrase in enumerate(
                self._ensure_phrase_states(state)):
            try:
                start = float(phrase.get(
                    "playback_start", phrase.get("start")))
                end = float(phrase.get(
                    "playback_end", phrase.get("end")))
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                highlights.append((offset + start, offset + end,
                                   int(index), phrase_index))
        if highlights:
            return highlights
        raw = state.get("preview_audio")
        samples = np.asarray(raw if raw is not None else [], np.float32)
        sr = int(state.get("preview_sr") or 16000)
        if samples.size > 1:
            return [(offset, offset + samples.size / float(max(1, sr)),
                     int(index), None)]
        return []

    def _play_sentence_indices(self, indices):
        items, highlights = [], []
        offset = 0.0
        for index in sorted(set(int(value) for value in indices)):
            if not (0 <= index < len(self.sentences)):
                continue
            state = self.sentences[index]
            raw = state.get("preview_audio")
            samples = np.asarray(raw if raw is not None else [], np.float32)
            sr = int(state.get("preview_sr") or 16000)
            if not state.get("rendered") or samples.size <= 1:
                continue
            items.append((samples, sr))
            highlights.extend(self._sentence_highlights(index, offset))
            offset += samples.size / float(max(1, sr)) + .25
        if not items:
            self.statusBar().showMessage("Status: generate sentences first")
            return None
        samples, sr = fc.concat_audio(items, gap_s=.25)
        self._start_playback(samples, sr, highlights=highlights)
        return True

    def _generate_sentence(self, index):
        if self._synthesis_busy:
            self.statusBar().showMessage(
                "Status: synthesis is already in progress...")
            return
        if not (0 <= int(index) < len(self.sentences)):
            return
        self.sentences_view.set_selected_indices([int(index)])
        self.sentence_select.setCurrentIndex(int(index))
        state = self.sentences[int(index)]
        if (self._pending_action(state) == "rerender" and
                self.current is not None and self.waveform.segments):
            result = self.on_rerender()
        else:
            result = self._generate_for_sentence_mode(confirm_replace=True)
        if result is not None:
            self._refresh_sentences_view()
            self.sentences_view.set_selected_indices([int(index)])

    def _play_all_sentences(self):
        self._capture_active_sentence()
        phrase_keys = self.sentences_view.selected_phrase_keys()
        if phrase_keys:
            return self._play_selected_phrases(phrase_keys)
        sentence_selection = self.sentences_view.selected_sentence_indices()
        wanted = sentence_selection or list(range(len(self.sentences)))
        missing = []
        for index in wanted:
            state = self.sentences[index]
            raw_audio = state.get("preview_audio")
            samples = np.asarray(
                raw_audio if raw_audio is not None else [], np.float32)
            if not state.get("rendered") or samples.size <= 1:
                missing.append(index)
        if missing:
            answer = QtWidgets.QMessageBox.question(
                self, "Generate missing audio",
                "%d sentence%s need audio for their current speaker. "
                "Generate them now?" %
                (len(missing), "" if len(missing) == 1 else "s"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes)
            if answer == QtWidgets.QMessageBox.Yes:
                self.on_rerender_all(only_indices=missing)
                self._capture_active_sentence()
        return self._play_sentence_indices(wanted)

    def _clear_all_sentences(self):
        if QtWidgets.QMessageBox.question(
                self, "Clear project sentences",
                "Remove every sentence and its parameter edits?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        self.sentences = [self._new_sentence_state()]
        self._active_sentence_index = 0
        self._refresh_sentence_selector(0)
        self._restore_sentence(0)
        self._refresh_sentences_view()

    def _open_sentence_mode(self, index, mode):
        if not (0 <= index < len(self.sentences)):
            return
        self.sentence_select.setCurrentIndex(index)
        self.mode_tabs.setCurrentIndex(0)

    def _play_phrase(self, sentence_index, phrase_index):
        if not (0 <= sentence_index < len(self.sentences)):
            return
        state = self.sentences[sentence_index]
        phrases = self._ensure_phrase_states(state)
        if not (0 <= phrase_index < len(phrases)):
            return
        selected = self.sentences_view.selected_phrase_keys()
        if ((sentence_index, phrase_index) in selected and
                len(selected) > 1):
            return self._play_selected_phrases(selected)
        preview = (state.get("phrase_previews") or {}).get(
            phrases[phrase_index]["id"])
        if not preview:
            self.statusBar().showMessage(
                "Status: generate this sentence before phrase playback")
            return
        samples, sr = preview
        duration = len(samples) / float(max(1, int(sr)))
        self._start_playback(
            samples, sr,
            highlights=[(0.0, duration, sentence_index, phrase_index)])

    def _play_selected_phrases(self, keys):
        items, highlights = [], []
        offset = 0.0
        for sentence_index, phrase_index in sorted(set(keys)):
            if not (0 <= sentence_index < len(self.sentences)):
                continue
            state = self.sentences[sentence_index]
            phrases = self._ensure_phrase_states(state)
            if not (0 <= phrase_index < len(phrases)):
                continue
            preview = (state.get("phrase_previews") or {}).get(
                phrases[phrase_index].get("id"))
            if not preview:
                continue
            samples, sr = preview
            samples = np.asarray(samples, np.float32)
            sr = int(sr)
            if samples.size <= 1:
                continue
            duration = samples.size / float(max(1, sr))
            items.append((samples, sr))
            highlights.append((offset, offset + duration,
                               sentence_index, phrase_index))
            offset += duration + .12
        if not items:
            self.statusBar().showMessage(
                "Status: generate selected phrases first")
            return None
        samples, sr = fc.concat_audio(items, gap_s=.12)
        self._start_playback(samples, sr, highlights=highlights)
        return True

    def _open_phrase_in_speech(self, sentence_index, phrase_index):
        self._open_sentence_mode(sentence_index, "speech")
        phrases = self._ensure_phrase_states(self.sentences[sentence_index])
        if 0 <= phrase_index < len(phrases):
            indices = phrases[phrase_index].get("segment_indices") or []
            if len(indices) == 2 and self.waveform.segments:
                first, last = indices
                if 0 <= first <= last < len(self.waveform.segments):
                    self.waveform._on_selection_drag(
                        self.waveform.segments[first].start,
                        self.waveform.segments[last].end, True)

    def _split_active_phrase_at_playhead(self):
        if (self.current is None or len(self.waveform.segments) < 3 or
                not (0 <= self._active_sentence_index < len(self.sentences))):
            self.statusBar().showMessage(
                "Status: generate and open a sentence before splitting")
            return
        state = self.sentences[self._active_sentence_index]
        self._capture_active_sentence()
        self._capture_phrase_snapshots(state)
        segments = self.waveform.segments
        playhead = self.waveform.playhead_time()
        candidates = [(abs(segment.end - playhead), index)
                      for index, segment in enumerate(segments[:-1])]
        _distance, boundary_index = min(candidates)
        phrases = self._ensure_phrase_states(state)
        phrase_index = next((index for index, phrase in enumerate(phrases)
                             if len(phrase.get("segment_indices") or []) == 2
                             and phrase["segment_indices"][0] <= boundary_index
                             < phrase["segment_indices"][1]), None)
        if phrase_index is None:
            self.statusBar().showMessage(
                "Status: the playhead is already on a phrase boundary")
            return
        phrase = phrases[phrase_index]
        first, last = phrase["segment_indices"]
        left_phones = [segment.phone for segment in
                       segments[first:boundary_index + 1]
                       if segment.phone != "pau"]
        right_phones = [segment.phone for segment in
                        segments[boundary_index + 1:last + 1]
                        if segment.phone != "pau"]
        if not left_phones or not right_phones:
            self.statusBar().showMessage(
                "Status: move the playhead inside the spoken phrase")
            return
        source_text = str(phrase.get("text") or "").strip()
        if state.get("input_mode") == "phones":
            left_text, right_text = " ".join(left_phones), " ".join(right_phones)
        else:
            words = source_text.split()
            if len(words) < 2:
                QtWidgets.QMessageBox.information(
                    self, "Split phrase",
                    "This text phrase has no word boundary to split. "
                    "Move the split to another phrase or use Phonemes mode.")
                return
            ratio = len(left_phones) / float(len(left_phones) + len(right_phones))
            word_cut = max(1, min(len(words) - 1,
                                  int(round(len(words) * ratio))))
            left_text = " ".join(words[:word_cut])
            right_text = " ".join(words[word_cut:])
        split_time = segments[boundary_index].end
        left_phrase = copy.deepcopy(phrase)
        right_phrase = copy.deepcopy(phrase)
        left_phrase.update({"id": uuid.uuid4().hex, "text": left_text,
                            "phones": left_phones,
                            "segment_indices": [first, boundary_index],
                            "start": segments[first].start,
                            "end": split_time})
        right_phrase.update({"id": uuid.uuid4().hex, "text": right_text,
                             "phones": right_phones,
                             "segment_indices": [boundary_index + 1, last],
                             "start": split_time,
                             "end": segments[last].end})
        previews = dict(state.get("phrase_previews") or {})
        a = int(round(segments[first].start * self.waveform.sr))
        cut = int(round(split_time * self.waveform.sr))
        b = int(round(segments[last].end * self.waveform.sr))
        previews.pop(phrase.get("id"), None)
        previews[left_phrase["id"]] = (
            self.waveform.audio[a:cut].copy(), self.waveform.sr)
        previews[right_phrase["id"]] = (
            self.waveform.audio[cut:b].copy(), self.waveform.sr)
        phrases[phrase_index:phrase_index + 1] = [left_phrase, right_phrase]
        state["phrases"] = phrases
        state["phrase_previews"] = previews
        state["text"] = PHRASE_TEXT_SEPARATOR.join(
            item["text"] for item in phrases)
        self._rebuild_phrase_preview(state)
        self._invalidate_phrase_editor(self._active_sentence_index, state)
        self._refresh_sentence_selector(self._active_sentence_index)
        self._refresh_sentences_view()
        self.statusBar().showMessage(
            "Status: split phrase at %.3f s; re-render to refresh phones" %
            split_time)

    def _rebuild_phrase_preview(self, state):
        previews = state.get("phrase_previews") or {}
        items = [previews.get(phrase.get("id"))
                 for phrase in state.get("phrases") or []]
        items = [item for item in items if item]
        if not items:
            state["rendered"] = False
            state["preview_audio"] = np.zeros(1, np.float32)
            return
        samples, sr = fc.concat_audio(items, gap_s=.20)
        state["preview_audio"] = samples
        state["preview_sr"] = sr
        state["rendered"] = True

    def _invalidate_phrase_editor(self, sentence_index, state):
        """Discard a stale full-sentence map after structural phrase edits."""
        state["synthesis"] = None
        state["editor_segments"] = []
        state["timing_factors"] = []
        self._set_state_pending(
            state, "generate", "Phrase structure changed")
        if (sentence_index != self._active_sentence_index or
                self._editor_sentence_state is not state):
            self._refresh_pending_ui()
            return
        self.text.blockSignals(True)
        self.text.setText(str(state.get("text") or ""))
        self.text.blockSignals(False)
        self.current = None
        raw_audio = state.get("preview_audio")
        samples = np.asarray(
            raw_audio if raw_audio is not None else np.zeros(1), np.float32)
        self.waveform.set_synthesis(fc.Synthesis(
            samples, int(state.get("preview_sr") or 16000), [],
            text=state.get("text") or "", lang=state.get("lang_code") or "",
            voicebank=state.get("voicebank") or ""))
        self.timing.set_segments([], [], [])
        self.pitch_track.set_data([], [], [], [])
        self.intonation.set_blocks([])
        self.recordings.set_data([], {}, {}, {})
        self._refresh_pending_ui()

    def _reorder_phrases(self, sentence_index, ordered_ids):
        if not (0 <= sentence_index < len(self.sentences)):
            return
        state = self.sentences[sentence_index]
        before = self._sentence_state_snapshot(state)
        by_id = {str(phrase.get("id")): phrase
                 for phrase in self._ensure_phrase_states(state)}
        phrases = [by_id[phrase_id] for phrase_id in ordered_ids
                   if phrase_id in by_id]
        if len(phrases) != len(by_id):
            return
        state["phrases"] = phrases
        state["text"] = PHRASE_TEXT_SEPARATOR.join(
            phrase["text"] for phrase in phrases)
        self._rebuild_phrase_preview(state)
        self._invalidate_phrase_editor(sentence_index, state)
        self._refresh_sentence_selector(self._active_sentence_index)
        self._refresh_sentences_view()
        after = self._sentence_state_snapshot(state)
        self._push_applied_undo(
            "move phrases",
            lambda row=sentence_index, snapshot=before:
            self._apply_sentence_state_snapshot(row, snapshot),
            lambda row=sentence_index, snapshot=after:
            self._apply_sentence_state_snapshot(row, snapshot))

    def _phrase_context_menu(self, sentence_index, phrase_index, global_pos):
        if not (0 <= sentence_index < len(self.sentences)):
            return
        state = self.sentences[sentence_index]
        phrases = self._ensure_phrase_states(state)
        if not (0 <= phrase_index < len(phrases)):
            return
        selected = [phrase_index]
        if sentence_index < len(self.sentences_view.row_widgets):
            board_selection = self.sentences_view.row_widgets[
                sentence_index].board.selected_indices()
            if board_selection:
                selected = board_selection
        menu = QtWidgets.QMenu()
        menu.addAction("Play phrase",
                       lambda: self._play_phrase(sentence_index, phrase_index))
        menu.addAction("Open in Speech editor",
                       lambda: self._open_phrase_in_speech(
                           sentence_index, phrase_index))
        menu.addSeparator()
        menu.addAction("Cut phrase to new sentence",
                       lambda: self._cut_phrase_to_sentence(
                           sentence_index, phrase_index))
        merge = menu.addAction("Merge selected phrases",
                               lambda: self._merge_phrases(
                                   sentence_index, selected))
        merge.setEnabled(len(selected) > 1 and selected == list(
            range(min(selected), max(selected) + 1)))
        menu.addAction(
            "Export selected phrases to WAV...",
            lambda: self._export_selected_phrases(
                sentence_index, selected))
        menu.addAction(
            "Remove selected phrases",
            lambda: self._remove_selected_phrases(
                sentence_index, selected))
        fault_menu = menu.addMenu("Apply faults to phrase")
        phrase_faults = dict(phrases[phrase_index].get("fault_mode") or {})
        sentence_faults = dict(state.get("fault_mode") or {})
        for key, action in self.fault_actions.items():
            if key == "no_sustain_stretch":
                continue
            local_enabled = bool(phrase_faults.get(key))
            inherited = bool(sentence_faults.get(key)) and not local_enabled
            label = action.text() + (" (inherited)" if inherited else "")
            item = fault_menu.addAction(label)
            item.setCheckable(True)
            item.setChecked(local_enabled or inherited)
            item.setVisible(action.isVisible())
            if inherited:
                item.setEnabled(False)
                item.setToolTip(
                    "Enabled by the sentence Fault Mode setting.")
            item.triggered.connect(
                lambda checked=False, k=key: self._set_phrase_fault(
                    sentence_index, phrase_index, k, checked))
        bit_menu = fault_menu.addMenu("Bit depth")
        bit_group = QtWidgets.QActionGroup(bit_menu)
        bit_group.setExclusive(True)
        current_bits = int(phrase_faults.get("bit_depth") or 0)
        for bits, label in ((0, "Full quality"), (8, "8-bit"),
                            (4, "4-bit"), (2, "2-bit"), (1, "1-bit")):
            item = bit_menu.addAction(label)
            item.setCheckable(True)
            item.setChecked(bits == current_bits)
            item.triggered.connect(
                lambda _checked=False, value=bits:
                self._set_phrase_bit_depth(
                    sentence_index, phrase_index, value))
            bit_group.addAction(item)
        fault_menu.addSeparator()
        fault_menu.addAction("Clear phrase faults",
                             lambda: self._clear_phrase_faults(
                                 sentence_index, phrase_index))
        speakers = menu.addMenu("Assign speaker")
        default_speaker = speakers.addAction("Sentence default")
        default_speaker.triggered.connect(
            lambda: self._set_phrase_speaker(sentence_index, phrase_index, ""))
        try:
            voices = self._ab().voicebanks()
        except Exception:
            voices = []
        for voice in voices:
            name = str(voice.get("name") or "")
            if not name:
                continue
            item = speakers.addAction(name)
            item.setCheckable(True)
            item.setChecked(phrases[phrase_index].get("speaker") == name)
            item.triggered.connect(
                lambda _checked=False, value=name:
                self._set_phrase_speaker(
                    sentence_index, phrase_index, value))
        dictionaries = menu.addMenu("Assign dictionary")
        dictionaries.addAction(
            "Sentence default", lambda: self._set_phrase_dictionary(
                sentence_index, phrase_index, ""))
        for key, path in sorted((self.cfg.get("voice_dictionaries") or {}).items()):
            item = dictionaries.addAction(os.path.basename(str(path)))
            item.setData(key)
            item.triggered.connect(
                lambda _checked=False, value=key:
                self._set_phrase_dictionary(
                    sentence_index, phrase_index, value))
        menu.exec_(global_pos)

    def _apply_sentence_state_snapshot(self, index, snapshot):
        if not (0 <= int(index) < len(self.sentences)):
            return
        self.sentences[int(index)] = self._sentence_state_snapshot(snapshot)
        self._refresh_sentence_selector(self._active_sentence_index)
        if int(index) == self._active_sentence_index:
            self._restore_sentence(int(index))
        self._refresh_sentences_view()

    def _export_selected_phrases(self, sentence_index, selected):
        if not (0 <= int(sentence_index) < len(self.sentences)):
            return False
        state = self.sentences[int(sentence_index)]
        phrases = self._ensure_phrase_states(state)
        selected = sorted(set(int(index) for index in selected
                              if 0 <= int(index) < len(phrases)))
        if not selected:
            return False
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export selected phrases to WAV",
            os.path.join(self._project_root, "exports")
            if self._project_root else "")
        if not folder:
            return False
        previews = state.get("phrase_previews") or {}
        exported = 0
        skipped = []
        for phrase_index in selected:
            phrase = phrases[phrase_index]
            preview = previews.get(phrase.get("id"))
            if not preview:
                skipped.append(phrase_index + 1)
                continue
            samples, sr = preview
            samples = np.asarray(samples, np.float32)
            if samples.size <= 1:
                skipped.append(phrase_index + 1)
                continue
            base = "%03d_%03d_%s" % (
                sentence_index + 1, phrase_index + 1,
                self._batch_slug(phrase.get("text")))
            path = os.path.join(folder, base + ".wav")
            suffix = 2
            while os.path.exists(path):
                path = os.path.join(
                    folder, "%s_%d.wav" % (base, suffix))
                suffix += 1
            try:
                fc.write_wav(path, samples, int(sr))
            except Exception as error:
                QtWidgets.QMessageBox.critical(
                    self, "Export selected phrases",
                    "Could not write:\n%s\n\n%s" % (path, error))
                return False
            exported += 1
        self.statusBar().showMessage(
            "Status: exported %d selected phrase WAV%s" %
            (exported, "" if exported == 1 else "s"))
        if skipped:
            QtWidgets.QMessageBox.information(
                self, "Export selected phrases",
                "Exported %d phrase%s. Generate these phrases first: %s" %
                (exported, "" if exported == 1 else "s",
                 ", ".join("phrase %d" % index for index in skipped)))
        elif not exported:
            QtWidgets.QMessageBox.information(
                self, "Export selected phrases",
                "Generate the selected phrases first.")
        return bool(exported)

    def _remove_selected_phrases(self, sentence_index, selected,
                                 confirm=True):
        if not (0 <= int(sentence_index) < len(self.sentences)):
            return False
        state = self.sentences[int(sentence_index)]
        phrases = self._ensure_phrase_states(state)
        selected = sorted(set(int(index) for index in selected
                              if 0 <= int(index) < len(phrases)))
        if not selected:
            return False
        count = len(selected)
        if confirm and QtWidgets.QMessageBox.question(
                self, "Remove selected phrases",
                "Remove %d selected phrase%s and their parameter edits?" %
                (count, "" if count == 1 else "s"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return False
        before = self._sentence_state_snapshot(state)
        removed_ids = {str(phrases[index].get("id")) for index in selected}
        state["phrases"] = [phrase for index, phrase in enumerate(phrases)
                            if index not in selected]
        previews = dict(state.get("phrase_previews") or {})
        state["phrase_previews"] = {
            phrase_id: preview for phrase_id, preview in previews.items()
            if str(phrase_id) not in removed_ids}
        state["text"] = PHRASE_TEXT_SEPARATOR.join(
            phrase["text"] for phrase in state["phrases"])
        self._rebuild_phrase_preview(state)
        self._invalidate_phrase_editor(int(sentence_index), state)
        self._refresh_sentence_selector(self._active_sentence_index)
        self._refresh_sentences_view()
        after = self._sentence_state_snapshot(state)
        self._push_applied_undo(
            "remove phrases",
            lambda row=int(sentence_index), snapshot=before:
            self._apply_sentence_state_snapshot(row, snapshot),
            lambda row=int(sentence_index), snapshot=after:
            self._apply_sentence_state_snapshot(row, snapshot))
        self.statusBar().showMessage(
            "Status: removed %d phrase%s" %
            (count, "" if count == 1 else "s"))
        return True

    def _cut_phrase_to_sentence(self, sentence_index, phrase_index):
        state = self.sentences[sentence_index]
        phrases = self._ensure_phrase_states(state)
        if not (0 <= phrase_index < len(phrases)):
            return
        phrase = phrases.pop(phrase_index)
        new_state = self._sentence_state_snapshot(state)
        new_state["text"] = phrase["text"]
        new_state["phrases"] = [phrase]
        new_state["synthesis"] = None
        new_state["editor_segments"] = []
        new_state["timing_factors"] = []
        self._set_state_pending(
            new_state, "generate", "Phrase moved to a new sentence")
        preview = (state.get("phrase_previews") or {}).get(phrase["id"])
        new_state["phrase_previews"] = ({phrase["id"]: preview}
                                         if preview else {})
        self._rebuild_phrase_preview(new_state)
        state["text"] = PHRASE_TEXT_SEPARATOR.join(
            item["text"] for item in phrases)
        self._rebuild_phrase_preview(state)
        self._invalidate_phrase_editor(sentence_index, state)
        if self._active_sentence_index > sentence_index:
            self._active_sentence_index += 1
        self.sentences.insert(sentence_index + 1, new_state)
        self._refresh_sentence_selector(self._active_sentence_index)
        self._refresh_sentences_view()

    def _merge_phrases(self, sentence_index, selected):
        selected = sorted(set(int(index) for index in selected))
        state = self.sentences[sentence_index]
        phrases = self._ensure_phrase_states(state)
        if (len(selected) < 2 or selected != list(
                range(selected[0], selected[-1] + 1))):
            return
        chosen = [phrases[index] for index in selected]
        merged = copy.deepcopy(chosen[0])
        merged["id"] = uuid.uuid4().hex
        merged["text"] = " ".join(phrase["text"] for phrase in chosen)
        merged["phones"] = [phone for phrase in chosen
                            for phone in phrase.get("phones") or []]
        merged["timing_factors"] = [factor for phrase in chosen
                                    for factor in
                                    phrase.get("timing_factors") or []]
        previews = state.get("phrase_previews") or {}
        audio = [previews.get(phrase["id"]) for phrase in chosen]
        audio = [item for item in audio if item]
        if audio:
            previews[merged["id"]] = fc.concat_audio(audio, gap_s=0.0)
        phrases[selected[0]:selected[-1] + 1] = [merged]
        state["phrases"] = phrases
        state["text"] = PHRASE_TEXT_SEPARATOR.join(
            phrase["text"] for phrase in phrases)
        self._rebuild_phrase_preview(state)
        self._invalidate_phrase_editor(sentence_index, state)
        self._refresh_sentence_selector(self._active_sentence_index)
        self._refresh_sentences_view()

    def _set_phrase_fault(self, sentence_index, phrase_index, key, enabled):
        state = self.sentences[sentence_index]
        phrase = state["phrases"][phrase_index]
        local_faults = phrase.setdefault("fault_mode", {})
        if enabled:
            local_faults[key] = True
        else:
            # An unchecked phrase action inherits the sentence value.  Keeping
            # an explicit False here would silently defeat a later sentence-
            # wide or project-wide fault activation.
            local_faults.pop(key, None)
        self._set_state_pending(state, "generate", "Phrase fault changed")
        self._refresh_sentences_view()
        self._refresh_pending_ui()

    def _clear_phrase_faults(self, sentence_index, phrase_index):
        state = self.sentences[sentence_index]
        state["phrases"][phrase_index]["fault_mode"] = {}
        self._set_state_pending(state, "generate", "Phrase faults cleared")
        self._refresh_sentences_view()
        self._refresh_pending_ui()

    def _set_phrase_bit_depth(self, sentence_index, phrase_index, bits):
        state = self.sentences[sentence_index]
        phrase = state["phrases"][phrase_index]
        local_faults = phrase.setdefault("fault_mode", {})
        bits = int(bits or 0)
        if bits:
            local_faults["bit_depth"] = bits
        else:
            local_faults.pop("bit_depth", None)
        self._set_state_pending(state, "generate", "Phrase bit depth changed")
        self._refresh_sentences_view()
        self._refresh_pending_ui()

    def _set_phrase_speaker(self, sentence_index, phrase_index, speaker):
        state = self.sentences[sentence_index]
        state["phrases"][phrase_index]["speaker"] = str(speaker)
        state["rendered"] = False
        self._set_state_pending(state, "generate", "Phrase speaker changed")
        self._refresh_sentences_view()
        self._refresh_pending_ui()

    def _set_phrase_dictionary(self, sentence_index, phrase_index, key):
        state = self.sentences[sentence_index]
        state["phrases"][phrase_index]["dictionary"] = str(key)
        self._set_state_pending(
            state, "generate", "Phrase dictionary changed")
        self._refresh_sentences_view()
        self._refresh_pending_ui()

    @staticmethod
    def _portable_shortcut(value):
        return QtGui.QKeySequence(str(value or "")).toString(
            QtGui.QKeySequence.PortableText)

    def _rebuild_shortcut_lookup(self):
        lookup = {}
        for key, _label, default in SHORTCUT_SPECS:
            value = self._portable_shortcut(self.shortcuts.get(key, default))
            if value and value not in lookup:
                lookup[value] = key
            elif value:
                value = ""
            self.shortcuts[key] = value
        # Conventional alternate redo remains available unless the user has
        # assigned it to another command.
        lookup.setdefault("Ctrl+Shift+Z", "redo")
        self._shortcut_lookup = lookup
        for key, (action, base) in self._shortcut_menu_actions.items():
            shortcut = self.shortcuts.get(key, "")
            action.setText(base + (("\t" + shortcut) if shortcut else ""))
        if hasattr(self, "shortcut_hint"):
            self._update_shortcut_hints()

    def _show_shortcut_dialog(self):
        dialog = ShortcutDialog(self.shortcuts, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self.shortcuts = dialog.values()
        self.cfg["shortcuts"] = dict(self.shortcuts)
        self._rebuild_shortcut_lookup()
        self._persist_config()
        self.statusBar().showMessage("Status: keyboard shortcuts updated")

    def _register_shortcut_contexts(self):
        groups = {
            "waveform": (self.waveform, self.waveform.plot,
                         self.waveform.timeline, self.waveform.fields_host),
            "parameter": (self.parameter_stack, self.timing,
                          self.pitch_track, self.intonation, self.recordings),
            "sentences": (self.sentences_view, self.sentences_view.content),
        }
        for context, widgets in groups.items():
            for widget in widgets:
                widget.setProperty("shortcutContext", context)

    def _default_shortcut_context(self):
        if hasattr(self, "mode_tabs") and self.mode_tabs.currentIndex() == 1:
            return "sentences"
        return "waveform"

    def _update_shortcut_hints(self, context=None):
        if not hasattr(self, "shortcut_hint"):
            return
        context = context or self._shortcut_hover_context or \
            self._default_shortcut_context()
        keys = {
            "waveform": ("play", "generate", "rerender", "undo"),
            "parameter": ("undo", "redo", "rerender", "play"),
            "sentences": ("select_all", "delete", "play", "undo"),
        }.get(context, ("play", "generate", "undo"))
        short_labels = {
            "select_all": "Select all",
            "generate": "Generate", "rerender": "Re-render",
            "play": ("Stop" if self._playback_active else "Play"),
            "undo": "Undo", "redo": "Redo",
            "copy": "Copy", "delete": "Delete",
        }
        parts = []
        for key in keys:
            value = self.shortcuts.get(key, "")
            if value:
                parts.append("%s %s" % (short_labels.get(
                    key, SHORTCUT_LABELS.get(key, key)), value))
        self.shortcut_hint.setText("  |  ".join(parts))

    @staticmethod
    def _event_shortcut(event):
        modifiers = event.modifiers() & (
            Qt.ControlModifier | Qt.AltModifier |
            Qt.ShiftModifier | Qt.MetaModifier)
        sequence = QtGui.QKeySequence(int(modifiers) | int(event.key()))
        return sequence.toString(QtGui.QKeySequence.PortableText)

    def _shortcut_blocked(self):
        focus = QtWidgets.QApplication.focusWidget()
        if (focus is None or
                (focus is not self and not self.isAncestorOf(focus))):
            focus = self.focusWidget()
        return isinstance(focus, (
            QtWidgets.QLineEdit, QtWidgets.QTextEdit,
            QtWidgets.QPlainTextEdit, QtWidgets.QAbstractSpinBox,
            QtWidgets.QComboBox))

    def _install_shortcuts(self):
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def _shortcut_select_all(self):
        if self.mode_tabs.currentIndex() == 1 and hasattr(
                self.sentences_view, "select_all"):
            self.sentences_view.select_all()
            return
        if self.waveform.segments:
            self.waveform._on_selection_drag(
                self.waveform.segments[0].start,
                self.waveform.segments[-1].end, True)

    def _shortcut_copy(self):
        if self.mode_tabs.currentIndex() == 1:
            phrase_keys = self.sentences_view.selected_phrase_keys()
            if phrase_keys:
                items = []
                for sentence, phrase_index in phrase_keys:
                    if not (0 <= sentence < len(self.sentences)):
                        continue
                    state = self.sentences[sentence]
                    phrases = self._ensure_phrase_states(state)
                    if not (0 <= phrase_index < len(phrases)):
                        continue
                    phrase = copy.deepcopy(phrases[phrase_index])
                    preview = (state.get("phrase_previews") or {}).get(
                        phrase.get("id"))
                    items.append({
                        "phrase": phrase,
                        "preview": preview,
                    })
                if not items:
                    return False
                self._project_clipboard = {
                    "kind": "phrases", "items": items}
                self.statusBar().showMessage(
                    "Status: copied %d phrase%s" %
                    (len(items), "" if len(items) == 1 else "s"))
                return True
            selected = self.sentences_view.selected_sentence_indices()
            if not selected:
                return False
            self._capture_active_sentence()
            self._project_clipboard = {
                "kind": "sentences",
                "items": [self._sentence_state_snapshot(
                    self.sentences[index])
                          for index in selected],
            }
            self.statusBar().showMessage(
                "Status: copied %d sentence%s" %
                (len(selected), "" if len(selected) == 1 else "s"))
            return True
        payload = self.waveform.copy_selection_payload()
        if payload is None:
            return False
        self._speech_clipboard = payload
        count = len(payload.get("segments") or [])
        self.statusBar().showMessage(
            "Status: copied %d phoneme%s" %
            (count, "" if count == 1 else "s"))
        return True

    def _shortcut_paste(self):
        if self.mode_tabs.currentIndex() == 1:
            payload = dict(self._project_clipboard or {})
            kind = payload.get("kind")
            if kind == "phrases":
                selected_phrases = self.sentences_view.selected_phrase_keys()
                selected_sentences = self.sentences_view.selected_sentence_indices()
                if selected_phrases:
                    target = selected_phrases[-1][0]
                    same = [phrase for sentence, phrase in selected_phrases
                            if sentence == target]
                    insert_at = max(same) + 1 if same else 0
                else:
                    target = (selected_sentences[-1]
                              if selected_sentences else
                              self._active_sentence_index)
                if not (0 <= target < len(self.sentences)):
                    return False
                if not selected_phrases:
                    insert_at = len(self._ensure_phrase_states(
                        self.sentences[target]))
                state = self.sentences[target]
                phrases = self._ensure_phrase_states(state)
                previews = dict(state.get("phrase_previews") or {})
                pasted = []
                for item in payload.get("items") or []:
                    phrase = copy.deepcopy(item.get("phrase") or {})
                    if not phrase:
                        continue
                    old_id = str(phrase.get("id") or "")
                    phrase["id"] = uuid.uuid4().hex
                    for key in list(phrase):
                        if str(key).startswith("_"):
                            phrase.pop(key, None)
                    preview = item.get("preview")
                    if preview:
                        previews[phrase["id"]] = preview
                    pasted.append(phrase)
                if not pasted:
                    return False
                phrases[insert_at:insert_at] = pasted
                state["phrases"] = phrases
                state["phrase_previews"] = previews
                state["text"] = PHRASE_TEXT_SEPARATOR.join(
                    phrase.get("text") or "" for phrase in phrases)
                self._rebuild_phrase_preview(state)
                self._invalidate_phrase_editor(target, state)
                self._refresh_sentence_selector(self._active_sentence_index)
                self._refresh_sentences_view()
                self.sentences_view.set_selected_phrase_keys(
                    [(target, index) for index in
                     range(insert_at, insert_at + len(pasted))])
                self.statusBar().showMessage(
                    "Status: pasted %d phrase%s" %
                    (len(pasted), "" if len(pasted) == 1 else "s"))
                return True
            if kind == "sentences":
                items = payload.get("items") or []
                if not items:
                    return False
                self._capture_active_sentence()
                selected = self.sentences_view.selected_sentence_indices()
                insert_at = (max(selected) + 1 if selected else
                             self._active_sentence_index + 1)
                copies = [self._fresh_sentence_copy(item) for item in items]
                self.sentences[insert_at:insert_at] = copies
                self._active_sentence_index = insert_at
                self._refresh_sentence_selector(insert_at)
                self._restore_sentence(insert_at, restore_view=False)
                self._refresh_sentences_view()
                indices = list(range(insert_at, insert_at + len(copies)))
                self.sentences_view.set_selected_indices(indices)
                self.statusBar().showMessage(
                    "Status: pasted %d sentence%s" %
                    (len(copies), "" if len(copies) == 1 else "s"))
                return True
            return False
        payload = self._speech_clipboard
        if not payload:
            return False
        before = self.waveform.structure_snapshot()
        if not self.waveform.paste_region_payload(copy.deepcopy(payload)):
            return False
        after = self.waveform.structure_snapshot()
        self._on_waveform_structure_edit(
            before, after, "paste phoneme region")
        return True

    def _fresh_sentence_copy(self, source):
        state = self._sentence_state_snapshot(source)
        segment_ids = {}
        segment_groups = []
        synthesis = state.get("synthesis")
        if synthesis is not None:
            segment_groups.append(synthesis.segments)
        segment_groups.append(state.get("editor_segments") or [])
        for segments in segment_groups:
            for segment in segments:
                old_id = str(segment.uid or "")
                segment.uid = segment_ids.setdefault(
                    old_id, uuid.uuid4().hex)
        previews = dict(state.get("phrase_previews") or {})
        refreshed = {}
        for phrase in state.get("phrases") or []:
            old_id = str(phrase.get("id") or "")
            phrase["id"] = uuid.uuid4().hex
            if old_id in previews:
                refreshed[phrase["id"]] = previews[old_id]
            for key in list(phrase):
                if str(key).startswith("_"):
                    phrase.pop(key, None)
        state["phrase_previews"] = refreshed
        state.pop("_speaker_name", None)
        state.pop("_speaker_icon", None)
        return state

    def _shortcut_duplicate(self):
        if self.mode_tabs.currentIndex() == 1:
            previous = self._project_clipboard
            if not self._shortcut_copy():
                return False
            payload = self._project_clipboard
            result = self._shortcut_paste()
            self._project_clipboard = previous
            return result
        payload = self.waveform.copy_selection_payload()
        if payload is None:
            return False
        before = self.waveform.structure_snapshot()
        if not self.waveform.paste_region_payload(payload):
            return False
        self._on_waveform_structure_edit(
            before, self.waveform.structure_snapshot(),
            "duplicate phoneme region")
        return True

    def _shortcut_delete(self, confirm=True):
        if self.mode_tabs.currentIndex() == 1:
            phrase_keys = self.sentences_view.selected_phrase_keys()
            if phrase_keys:
                grouped = {}
                for sentence, phrase in phrase_keys:
                    grouped.setdefault(sentence, []).append(phrase)
                removed = False
                for sentence in sorted(grouped, reverse=True):
                    removed = self._remove_selected_phrases(
                        sentence, grouped[sentence], confirm=confirm) or removed
                    confirm = False
                return removed
            selected = self.sentences_view.selected_sentence_indices()
            return self._remove_selected_sentences(
                selected, confirm=confirm) if selected else False
        if not self.waveform.selected_indices():
            return False
        before = self.waveform.structure_snapshot()
        if not self.waveform.delete_selection():
            return False
        self._on_waveform_structure_edit(
            before, self.waveform.structure_snapshot(),
            "delete phoneme region")
        return True

    def _dispatch_shortcut(self, command):
        if command == "undo":
            self.undo_stack.undo()
        elif command == "redo":
            self.undo_stack.redo()
        elif command == "new_sentence":
            self.on_add_sentence()
        elif command == "open_project":
            self.on_open_project()
        elif command == "save_project":
            self.on_save_project()
        elif command == "export_audio":
            self.on_export()
        elif command == "generate":
            if (hasattr(self, "mode_tabs") and
                    self.mode_tabs.currentIndex() == 1):
                selected = self.sentences_view.selected_sentence_indices()
                self.on_generate_all(
                    only_indices=selected or
                    list(range(len(self.sentences))))
            else:
                self.on_generate()
        elif command == "rerender":
            self.on_rerender()
        elif command == "play":
            self.on_play()
        elif command == "stop":
            self.on_stop()
            self.waveform.clear_join_selection()
        elif command == "select_all":
            self._shortcut_select_all()
        elif command == "copy":
            self._shortcut_copy()
        elif command == "cut":
            if self._shortcut_copy():
                self._shortcut_delete(confirm=False)
        elif command == "paste":
            self._shortcut_paste()
        elif command == "duplicate":
            self._shortcut_duplicate()
        elif command == "delete":
            self._shortcut_delete()

    def _push_applied_undo(self, label, undo, redo):
        if self._applying_undo:
            return

        def guarded(callback):
            def run():
                self._applying_undo = True
                try:
                    callback()
                finally:
                    self._applying_undo = False
            return run

        self.undo_stack.push(AppliedUndoCommand(
            label, guarded(undo), guarded(redo)))

    def eventFilter(self, obj, event):
        context = obj.property("shortcutContext") \
            if hasattr(obj, "property") else None
        if event.type() == QtCore.QEvent.Enter and context:
            self._shortcut_hover_context = str(context)
            self._update_shortcut_hints(self._shortcut_hover_context)
        elif (event.type() == QtCore.QEvent.Leave and context and
              self._shortcut_hover_context == str(context)):
            self._shortcut_hover_context = ""
            QtCore.QTimer.singleShot(0, self._update_shortcut_hints)
        if (event.type() == QtCore.QEvent.KeyPress and
                not event.isAutoRepeat() and
                QtWidgets.QApplication.activeModalWidget() is None):
            command = self._shortcut_lookup.get(self._event_shortcut(event))
            if command:
                global_commands = {
                    "generate", "new_sentence", "open_project",
                    "save_project", "export_audio", "stop",
                }
                if command == "play" and self._playback_active:
                    global_commands.add("play")
                if command not in global_commands and self._shortcut_blocked():
                    return super().eventFilter(obj, event)
                self._dispatch_shortcut(command)
                return True
        return super().eventFilter(obj, event)

    # -- populate from config/backend -----------------------------------------
    def _populate_from_backend(self):
        eng = str(self.cfg.get("engine") or "diphone")
        ix = self.engine.findData(eng)
        self.engine.blockSignals(True)
        self.engine.setCurrentIndex(ix if ix >= 0 else 0)
        self.engine.blockSignals(False)

        langs = self.cfg.get("languages") or {"Asaxi": "asaxi"}
        self.lang.clear()
        for label in langs:
            self.lang.addItem(label)
        want = self.cfg.get("default_language")
        if self.backend and not want:
            code = self.backend.default_lang_code()
            want = next((L for L, c in langs.items() if c == code), None)
        if want and self.lang.findText(want) >= 0:
            self.lang.setCurrentText(want)

        if eng == "festival_wsl":
            self._refresh_configured_voice_roots()
        self._refresh_voicebanks()

        spd = self.backend.default_speed() if self.backend else \
            float(self.cfg.get("synth_speed", 1.0) or 1.0)
        try:
            self.speed.setValue(int(round(100 * np.log2(max(0.25, min(4.0, spd))))))
        except Exception:
            self.speed.setValue(0)
        self._on_speed_slider()
        fest = eng == "festival_wsl"
        self.pitch.setEnabled(fest)
        self._update_fault_availability()
        self._update_parameter_availability()

    def _refresh_voicebanks(self, keep=None):
        keep = keep or self._current_voicebank()
        self.voicebank.clear()
        ab = self._ab()
        try:
            vbs = ab.voicebanks() if ab else []
        except Exception:
            vbs = []
        if not vbs:
            it = QtWidgets.QListWidgetItem("(no voicebanks)")
            it.setFlags(Qt.NoItemFlags)
            it.setToolTip(
                "No Festival voices registered.\nVoicebank > Scan Festival "
                "voices (WSL), or Add Festival voice folder..."
                if self._engine() == "festival_wsl" else
                "No voices in festvox.json and none added.\n"
                "Voicebank > Add diphone DB folder...")
            self.voicebank.addItem(it)
            self._sync_speaker_control()
            return
        default = ab.default_voicebank() if ab else None
        for v in vbs:
            metadata_status = str(v.get("metadata_status") or "")
            suffix = (
                "  (legacy metadata)" if metadata_status == "legacy"
                else "  (metadata unknown)"
                if metadata_status == "unknown" else ""
            )
            it = QtWidgets.QListWidgetItem(
                (("%s" if v["ok"] else "%s  (missing)") % v["name"])
                + suffix)
            it.setData(Qt.UserRole, v["name"])
            tip = "%s\nfrom %s" % (v["dir"] or "(no path)", v["source"])
            if v.get("note"):
                tip += "\n" + v["note"]
            if not v["ok"]:
                it.setForeground(QtGui.QBrush(QtGui.QColor("#A00000")))
                if self._engine() != "festival_wsl":
                    tip += "\nNOT FOUND: needs dic/diphone_index.json"
            it.setToolTip(tip)
            self.voicebank.addItem(it)
            if v["name"] == (keep or default):
                self.voicebank.setCurrentItem(it)
        if self.voicebank.currentRow() < 0:
            self.voicebank.setCurrentRow(0)
        self._sync_speaker_control()

    def _select_existing_voicebank(self, name):
        """Select from the populated list without rescanning voice metadata."""
        name = str(name or "")
        if not name:
            return self._current_voicebank() is None
        for row in range(self.voicebank.count()):
            item = self.voicebank.item(row)
            if str(item.data(Qt.UserRole) or "") != name:
                continue
            self.voicebank.blockSignals(True)
            try:
                self.voicebank.setCurrentItem(item)
            finally:
                self.voicebank.blockSignals(False)
            self._sync_speaker_control()
            return True
        return False

    def _toggle_voicebank_list(self, expanded):
        self.voicebank.setVisible(bool(expanded))
        self.voicebank_resize.setVisible(bool(expanded))
        self.voicebank_heading.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow)

    def _refresh_configured_voice_roots(self, show_errors=False):
        try:
            report = self.fest.refresh_voice_roots(install_kal=True)
        except (fc.BackendError, OSError, ValueError) as exc:
            if show_errors:
                QtWidgets.QMessageBox.warning(
                    self, "Voicebank scan", str(exc))
            return None
        self._variant_cache.clear()
        self._persist_config()
        warnings = list(report.get("warnings") or [])
        if warnings and show_errors:
            QtWidgets.QMessageBox.warning(
                self, "Voicebank scan",
                "The scan completed with warnings:\n\n" +
                "\n".join(warnings[:12]))
        self._configure_voice_root_watcher()
        return report

    def _configure_voice_root_watcher(self):
        if not hasattr(self, "_voice_root_watcher"):
            return
        existing = self._voice_root_watcher.directories()
        if existing:
            self._voice_root_watcher.removePaths(existing)
        try:
            root = self.fest.generated_voice_root()
        except (fc.BackendError, OSError, ValueError):
            return
        if os.path.isdir(root):
            self._voice_root_watcher.addPath(root)

    def _schedule_voice_root_refresh(self, _path=""):
        self._voice_root_refresh_timer.start()

    def _refresh_watched_voice_roots(self):
        if self._engine() != "festival_wsl":
            return
        keep = self._current_voicebank()
        self._refresh_configured_voice_roots()
        self._refresh_voicebanks(keep=keep)
        self._apply_voice_language_compatibility(auto_select=True)

    def _current_voicebank(self):
        it = self.voicebank.currentItem() if hasattr(self, "voicebank") else None
        value = it.data(Qt.UserRole) if it else None
        return value if value and value != MIXED_SELECTION_DATA else None

    def _current_lang_code(self):
        if (hasattr(self, "lang") and
                self.lang.currentData() == MIXED_SELECTION_DATA):
            return ""
        langs = self.cfg.get("languages") or {}
        return langs.get(self.lang.currentText(), "asaxi")

    def _sidebar_sentence_targets(self):
        if (hasattr(self, "mode_tabs") and
                self.mode_tabs.currentIndex() == 1 and
                hasattr(self, "sentences_view")):
            selected = self.sentences_view.selected_sentence_indices()
            if selected:
                return selected
        if 0 <= self._active_sentence_index < len(self.sentences):
            return [self._active_sentence_index]
        return []

    def _sync_sentence_sidebar_values(self, indices=None):
        if not hasattr(self, "lang") or not hasattr(self, "voicebank"):
            return
        indices = list(self._sidebar_sentence_targets()
                       if indices is None else indices)
        states = [self.sentences[int(index)] for index in indices
                  if 0 <= int(index) < len(self.sentences)]
        if not states:
            return
        languages = {str(state.get("language") or "") for state in states}
        voices = {str(state.get("voicebank") or "") for state in states}
        self._sentence_sidebar_syncing = True
        self.lang.blockSignals(True)
        self.voicebank.blockSignals(True)
        try:
            mixed_language = self.lang.findData(MIXED_SELECTION_DATA)
            if len(languages) > 1:
                if mixed_language < 0:
                    self.lang.insertItem(0, "-", MIXED_SELECTION_DATA)
                    mixed_language = 0
                self.lang.setCurrentIndex(mixed_language)
            else:
                language = next(iter(languages), "")
                if mixed_language >= 0:
                    self.lang.removeItem(mixed_language)
                row = self.lang.findText(language)
                if row >= 0:
                    self.lang.setCurrentIndex(row)

            for row in range(self.voicebank.count() - 1, -1, -1):
                if (self.voicebank.item(row).data(Qt.UserRole) ==
                        MIXED_SELECTION_DATA):
                    self.voicebank.takeItem(row)
            if len(voices) > 1:
                item = QtWidgets.QListWidgetItem("-")
                item.setData(Qt.UserRole, MIXED_SELECTION_DATA)
                item.setToolTip(
                    "Selected sentences use different voicebanks")
                self.voicebank.insertItem(0, item)
                self.voicebank.setCurrentItem(item)
            else:
                voice = next(iter(voices), "")
                for row in range(self.voicebank.count()):
                    item = self.voicebank.item(row)
                    if str(item.data(Qt.UserRole) or "") == voice:
                        self.voicebank.setCurrentItem(item)
                        break
        finally:
            self.voicebank.blockSignals(False)
            self.lang.blockSignals(False)
            self._sentence_sidebar_syncing = False
        self._sync_speaker_control()
        self._apply_voice_language_compatibility(auto_select=False)
        self._update_parameter_availability()

    def _apply_voice_language_compatibility(self, auto_select=False):
        """Constrain language choices to the selected generated manifest."""
        if not hasattr(self, "lang"):
            return
        compatibility = None
        backend = self._ab()
        voice = self._current_voicebank() or ""
        if (self._engine() == "festival_wsl" and backend and voice and
                hasattr(backend, "voice_compatibility")):
            try:
                compatibility = backend.voice_compatibility(voice)
            except (fc.BackendError, OSError, ValueError, AttributeError):
                compatibility = None
        supported = set(
            getattr(compatibility, "supported_languages", ()) or ()
        )
        sentence_scope = bool(
            hasattr(self, "mode_tabs") and
            self.mode_tabs.currentIndex() == 1 and
            self._sidebar_sentence_targets()
        )
        model = self.lang.model()
        languages = self.cfg.get("languages") or {}
        for row in range(self.lang.count()):
            item = model.item(row)
            if item is None:
                continue
            code = str(languages.get(self.lang.itemText(row), ""))
            # A sentence's language must remain editable independently of its
            # current speaker. The generated-voice manifest is still enforced
            # by the backend when rendering; users can select a compatible
            # speaker after changing one or several sentence rows.
            item.setEnabled(
                sentence_scope or not supported or code in supported)
        primary = getattr(compatibility, "primary_language", None)
        current = self._current_lang_code()
        mixed_selection_supported = False
        if (self.lang.currentData() == MIXED_SELECTION_DATA and
                self.sentences):
            selected_codes = {
                str(self.sentences[index].get("lang_code") or "")
                for index in self._sidebar_sentence_targets()
            }
            selected_codes.discard("")
            mixed_selection_supported = bool(
                selected_codes and
                (not supported or selected_codes.issubset(supported)))
        if (auto_select and primary and current != primary and
                not mixed_selection_supported):
            label = next((
                label for label, code in languages.items()
                if str(code) == str(primary)
            ), None)
            if label and self.lang.findText(label) >= 0:
                self.lang.blockSignals(True)
                self.lang.setCurrentText(label)
                self.lang.blockSignals(False)
                if not self._switching_sentence and self.sentences:
                    for index in self._sidebar_sentence_targets():
                        state = self.sentences[index]
                        if (state.get("language") == label and
                                state.get("lang_code") == primary):
                            continue
                        state["language"] = label
                        state["lang_code"] = primary
                        state["rendered"] = False
                        self._set_state_pending(
                            state, "generate", "Language changed")

    # -- helpers ------------------------------------------------------------
    def _speed_factor(self):
        return float(2.0 ** (self.speed.value() / 100.0))

    def _on_speed_slider(self):
        self.speed_val.blockSignals(True)
        self.speed_val.setValue(self._speed_factor())
        self.speed_val.blockSignals(False)
        if not self._switching_sentence and self.sentences:
            state = self.sentences[self._active_sentence_index]
            state["speed"] = self._speed_factor()
            self._mark_active_pending(
                "generate", "Output speed changed")

    def _on_speed_spin(self, v):
        self.speed.blockSignals(True)
        self.speed.setValue(int(round(100 * np.log2(max(0.25, min(4.0, v))))))
        self.speed.blockSignals(False)
        if not self._switching_sentence and self.sentences:
            state = self.sentences[self._active_sentence_index]
            state["speed"] = self._speed_factor()
            self._mark_active_pending(
                "generate", "Output speed changed")

    def _on_pitch_parameter_changed(self, _value=None):
        if self._switching_sentence or not self.sentences:
            return
        state = self.sentences[self._active_sentence_index]
        state["pitch_hz"] = float(self.pitch.value())
        state["fall_pct"] = float(self.fall.value())
        self._mark_active_pending(
            "rerender", "Festival pitch or Fall changed")

    def _on_language_changed(self, _index=None):
        if (self._switching_sentence or self._sentence_sidebar_syncing or
                not self.sentences or
                self.lang.currentData() == MIXED_SELECTION_DATA):
            return
        targets = self._sidebar_sentence_targets()
        language = self.lang.currentText()
        code = self._current_lang_code()
        for index in targets:
            state = self.sentences[index]
            state["language"] = language
            state["lang_code"] = code
            state["rendered"] = False
            self._set_state_pending(state, "generate", "Language changed")
        self._update_parameter_availability()
        active = self.sentences[self._active_sentence_index]
        self.japanese_editor.set_state(active.get("japanese_state"))
        self.asaxi_editor.set_state(active.get("asaxi_state"))
        self._refresh_japanese_runtime_controls()
        self._refresh_sentences_view_preserving_focus(targets)
        self._sync_sentence_sidebar_values(targets)
        self._refresh_pending_ui()

    def _pitch(self):
        """(pitch_hz, fall_pct) for the Festival engine, else (None, None)."""
        if self._engine() != "festival_wsl":
            return None, None
        return float(self.pitch.value()), float(self.fall.value())

    def _monotone(self):
        """True when flat-pitch output is requested (Festival engine only)."""
        return (self._engine() == "festival_wsl"
                and bool(self.fault_actions["monotone"].isChecked()))

    def _fault_mode(self):
        out = {key: bool(action.isChecked())
               for key, action in self.fault_actions.items()}
        out["bit_depth"] = self._bit_depth()
        if out.get("pitch_glitch") and self._pitch_fault_target is not None:
            target = self._pitch_fault_target
            if isinstance(target, dict):
                out["pitch_glitch_pins"] = [dict(target)]
            elif isinstance(target, (list, tuple)):
                pins = [dict(event) for event in target
                        if isinstance(event, dict)]
                if pins:
                    out["pitch_glitch_pins"] = pins
            else:
                out["pitch_glitch_segment"] = int(target)
        return out

    def _bit_depth(self):
        for bits, action in getattr(self, "bit_depth_actions", {}).items():
            if action.isChecked():
                return int(bits)
        return 0

    def _on_fault_changed(self, _checked=False):
        if not self.fault_actions["pitch_glitch"].isChecked():
            self._pitch_fault_target = None
        self._update_fault_availability()
        action = self._mark_active_pending(
            "rerender", "Fault settings changed")
        if action:
            self.statusBar().showMessage(
                "Status: fault settings changed -- %s applies them" %
                ("Generate Audio" if action == "generate" else "Re-render"))
        self._capture_active_sentence()
        self._persist_config()

    def _clear_faults(self):
        for action in self.fault_actions.values():
            action.blockSignals(True)
            action.setChecked(False)
            action.blockSignals(False)
        self.bit_depth_actions[0].setChecked(True)
        self._pitch_fault_target = None
        self._on_fault_changed()

    def _apply_faults_to_all_sentences(self):
        self._capture_active_sentence()
        faults = self._fault_mode()
        for state in self.sentences:
            state["fault_mode"] = dict(faults)
            for phrase in self._ensure_phrase_states(state):
                local = phrase.setdefault("fault_mode", {})
                phrase["fault_mode"] = {
                    key: value for key, value in local.items()
                    if (key == "bit_depth" and int(value or 0) > 0)
                    or (key != "bit_depth" and bool(value))
                }
            state["pitch_fault_target"] = self._pitch_fault_target
            self._set_state_pending(
                state, "rerender", "Project fault settings changed")
        self._refresh_pending_ui()
        self.statusBar().showMessage(
            "Status: current fault settings applied to all sentences")

    def _clear_faults_from_all_sentences(self):
        clean = {key: False for key in self.fault_actions}
        clean["bit_depth"] = 0
        for state in self.sentences:
            state["fault_mode"] = dict(clean)
            state["pitch_fault_target"] = None
            for phrase in self._ensure_phrase_states(state):
                phrase["fault_mode"] = {}
            self._set_state_pending(
                state, "rerender", "Project faults cleared")
        self._clear_faults()
        self._refresh_pending_ui()
        self.statusBar().showMessage(
            "Status: faults cleared from all sentences")

    def _set_pitch_fault_target(self, target):
        if target is None:
            normalized = None
        elif isinstance(target, dict):
            normalized = [dict(target)]
        elif isinstance(target, (list, tuple)):
            normalized = [dict(event) for event in target
                          if isinstance(event, dict) and
                          event.get("broken_hz") is not None]
            normalized = normalized or None
        else:
            normalized = int(target)
        self._pitch_fault_target = normalized
        if normalized is not None:
            self.fault_actions["pitch_glitch"].setChecked(True)
        self._mark_active_pending(
            "rerender", "Broken pitch pin changed")
        self._capture_active_sentence()
        if isinstance(normalized, list):
            summary = ", ".join(
                "%s %.1f Hz" % (event.get("phone") or
                                 "phone %s" % event.get("segment", "?"),
                                 float(event["broken_hz"]))
                for event in normalized)
        elif normalized is None:
            summary = "random"
        else:
            summary = "phone %d (legacy location pin)" % int(normalized)
        self.statusBar().showMessage(
            "Status: broken pitch = %s" % summary)

    def _show_fault_menu(self):
        if not hasattr(self, "fault_menu"):
            return
        self.fault_menu.exec_(
            self.fault_badge.mapToGlobal(
                QtCore.QPoint(0, self.fault_badge.height())))

    def _update_fault_indicator(self):
        if not hasattr(self, "fault_badge"):
            return
        faults = self._fault_mode()
        active = [key for key, value in faults.items()
                  if key != "bit_depth" and bool(value)]
        if faults.get("bit_depth"):
            active.append("%d-bit" % faults["bit_depth"])
        count = len(active)
        self.fault_menu_action.setText(
            "&Fault Mode" + (" [%d]" % count if count else ""))
        self.fault_badge.setVisible(bool(count))
        self.fault_badge.setText("Faults active: %d" % count)
        self.fault_badge.setToolTip(
            ", ".join(active) if active else "No active faults")
        self.fault_badge.setStyleSheet(
            "QToolButton { background:#FFF2B2; border:1px solid #B08A20; "
            "padding:3px; }" if count else "")

    def _update_fault_availability(self):
        if not hasattr(self, "fault_actions"):
            return
        fest = self._engine() == "festival_wsl"
        phones = (hasattr(self, "input_mode") and
                  self.input_mode.currentData() == "phones")
        code = self._current_lang_code() if hasattr(self, "lang") else ""
        text_mode = not phones
        self.fault_actions["disable_phone_timing"].setVisible(
            fest and (phones or code in ("asaxi", "ja", "jp")))
        self.fault_actions["disable_prosody"].setVisible(
            fest and text_mode and code == "en")
        self.fault_actions["disable_f0_correction"].setVisible(fest)
        self.fault_actions["single_pause"].setVisible(fest and text_mode)
        self.fault_actions["pitch_glitch"].setVisible(fest)
        self.fault_actions["no_sustain_stretch"].setVisible(True)
        self.fault_actions["legacy_joins"].setVisible(True)
        self.fault_actions["monotone"].setVisible(fest)
        if hasattr(self, "waveform"):
            self.waveform.use_sustain = not self.fault_actions[
                "no_sustain_stretch"].isChecked()
            self.waveform.set_fault_mode_active(
                fest and self.fault_actions["pitch_glitch"].isChecked())
            self._refresh_join_overlay_controls()
        if hasattr(self, "fall"):
            pitch_control_visible = fest
            fall_visible = fest and not self._monotone()
            self.pitch_header.setVisible(fest)
            self.pitch_header.setText("Festival Pitch:")
            self.pitch.setVisible(pitch_control_visible)
            self.pitch_field_label.setVisible(pitch_control_visible)
            self.fall.setVisible(fall_visible)
            self.fall_field_label.setVisible(fall_visible)
        self._update_fault_indicator()

    def _on_parameter_mode(self, _index=None):
        if not hasattr(self, "parameter_stack"):
            return
        idx = max(0, self.parameter_mode.currentIndex())
        mode = self.parameter_mode.currentData()
        if mode in {"japanese", "mora_voicing"}:
            edit_mode = (
                "mora_voicing" if mode == "mora_voicing" else "accent")
            if self._current_lang_code() == "asaxi":
                self.parameter_stack.setCurrentWidget(self.asaxi_page)
                self.asaxi_editor.set_edit_mode(edit_mode)
            else:
                self.parameter_stack.setCurrentWidget(self.japanese_page)
                self.japanese_editor.set_edit_mode(edit_mode)
        else:
            self.parameter_stack.setCurrentIndex(idx)
        pitch = mode == "pitch"
        voicing = mode == "voicing"
        vocal_tract_mode = mode == "vocal_tract"
        timing = mode == "timing"
        self.timing_consonants.setVisible(timing)
        self.timing_vowels.setVisible(timing)
        self.curve_unit_overlay.setVisible(
            mode in {"pitch", "voicing", "vocal_tract"})
        self.pitch_navigator.setVisible(pitch)
        self.pitch_reset.setVisible(pitch)
        self.voicing_reset.setVisible(voicing)
        self.vocal_tract_reset.setVisible(vocal_tract_mode)
        if mode in {"japanese", "mora_voicing"} and hasattr(
                self, "editor_splitter"):
            sizes = self.editor_splitter.sizes()
            total = sum(sizes)
            target = min(280, max(225, int(total * .42)))
            if len(sizes) == 2 and sizes[1] < target and total > target + 180:
                self.editor_splitter.setSizes([total - target, target])
        if self._shortcut_hover_context == "parameter":
            self._update_shortcut_hints("parameter")

    def _on_timing_filter(self, _checked=False):
        if hasattr(self, "timing"):
            self.timing.set_filter(self.timing_consonants.isChecked(),
                                   self.timing_vowels.isChecked())

    def _update_parameter_availability(self):
        if not hasattr(self, "parameter_mode"):
            return
        fest = self._engine() == "festival_wsl"
        model = self.parameter_mode.model()
        for key in ("pitch", "intonation"):
            item = model.item(self.parameter_mode.findData(key))
            if item is not None:
                item.setEnabled(fest)
        mora_items = {
            key: model.item(self.parameter_mode.findData(key))
            for key in ("japanese", "mora_voicing")
        }
        language = self._current_lang_code()
        mora_language = None
        mora_available = False
        if fest and language in ("ja", "jp"):
            try:
                compatibility = self.fest.voice_compatibility(
                    self._current_voicebank() or "")
                mora_available = compatibility.supports("ja")
                mora_language = "Japanese"
            except (fc.BackendError, OSError, ValueError, AttributeError):
                mora_available = False
        elif (fest and language == "asaxi" and
                self.input_mode.currentData() == "text"):
            try:
                compatibility = self.fest.voice_compatibility(
                    self._current_voicebank() or "")
                mora_available = compatibility.supports("asaxi")
                mora_language = "Asaxi"
            except (fc.BackendError, OSError, ValueError, AttributeError):
                mora_available = False
        for key, item in mora_items.items():
            if item is None:
                continue
            item.setEnabled(mora_available)
            if mora_language == "Asaxi" and mora_available:
                tooltip = (
                    "Edit inferred Asaxi H/L blocks and per-mora pitch."
                    if key == "japanese" else
                    "Edit predicted Asaxi mora voicing blocks.")
            elif mora_language == "Japanese" and mora_available:
                tooltip = (
                    "Edit Japanese accent phrases and per-mora pitch."
                    if key == "japanese" else
                    "Edit predicted Japanese mora voicing.")
            else:
                tooltip = (
                    "Select a Japanese sentence or an Asaxi Text sentence "
                    "using a compatible Festival/WSL voice.")
            item.setToolTip(tooltip)
        if (not fest and self.parameter_mode.currentData()
                in ("pitch", "intonation")):
            self.parameter_mode.setCurrentIndex(0)
        if (not mora_available and
                self.parameter_mode.currentData() in
                {"japanese", "mora_voicing"}):
            self.parameter_mode.setCurrentIndex(0)
        self._on_parameter_mode()

    def on_add_phone(self):
        # insert a blank phone after the selected one (or at the end if nothing
        # is selected); the new box is highlighted and focused for typing
        idx = self.waveform.focused_idx
        if idx is None or not (0 <= idx < len(self.waveform.segments)):
            idx = len(self.waveform.segments) - 1
        self.waveform._insert_phone(idx, before=False)
        self.statusBar().showMessage(
            "Status: blank phone inserted -- type a phone into the highlighted "
            "box, then Re-render Phonemes")

    def on_del_phone(self):
        idx = self.waveform.focused_idx
        if idx is None or not (0 <= idx < len(self.waveform.segments)):
            self.statusBar().showMessage(
                "Status: click a phoneme to select it, then - Phone removes it")
            return
        self.waveform._delete_phone(idx)          # pau removable too
        self.statusBar().showMessage(
            "Status: phone removed -- Re-render Phonemes applies it")

    def _sustain_sample(self, phone):
        voicebank = self._current_voicebank()
        backend = self._ab()
        if not voicebank or backend is None:
            return None
        try:
            return backend.sustain_sample(phone, voicebank)
        except (fc.BackendError, AttributeError):
            return None

    def _on_output_gain_changed(self, _value):
        if self._switching_sentence:
            return
        if 0 <= self._active_sentence_index < len(self.sentences):
            state = self.sentences[self._active_sentence_index]
            state["output_gain_db"] = float(_value)
            pending = self._gain_pending(state)
            if pending:
                self._set_state_pending(
                    state, "rerender", "Output volume changed")
            elif state.get("pending_reason") == "Output volume changed":
                self._set_state_pending(state, "")
        else:
            pending = self.current is not None
        self.speech_gain.set_pending(pending)
        if pending:
            self.statusBar().showMessage(
                "Status: output gain changed -- Re-render applies it")
        self._refresh_pending_ui()
        self._capture_active_sentence()

    def _sync_vocal_tract_value_bounds(self):
        if not hasattr(self, "vocal_tract_value"):
            return
        lower, upper = self.vocal_tract_track.bounds()
        current = self.vocal_tract_track.profile.clamp(
            self.vocal_tract_value.value(),
            self.vocal_tract_track.chipmunk_range(),
        )
        self.vocal_tract_value.blockSignals(True)
        self.vocal_tract_value.setRange(lower, upper)
        self.vocal_tract_value.setValue(current)
        self.vocal_tract_value.blockSignals(False)

    def _update_vocal_tract_readout(self):
        if not hasattr(self, "vocal_tract_readout"):
            return
        lower, upper = self.vocal_tract_track.ratio_range()
        if abs(lower - upper) <= 0.0005:
            text = ("Original voice: 1.000 x" if abs(lower - 1.0) <= 0.0005
                    else "Uniform: %.3f x" % lower)
            summary = lower
        else:
            text = "Curve: %.3f-%.3f x" % (lower, upper)
            summary = vocal_tract.ratio_curve_summary(
                self.vocal_tract_track.targets())
        self.vocal_tract_readout.setText(text)
        self.vocal_tract_readout.setToolTip(
            "%s; median resonance shift %+.2f semitones. This is an "
            "acoustic spectral-envelope approximation, not a biological "
            "gender model." % (
                text, vocal_tract.ratio_to_formant_semitones(summary)))
        self.vocal_tract_value.blockSignals(True)
        self.vocal_tract_value.setValue(
            self.vocal_tract_track.profile.clamp(
                summary, self.vocal_tract_track.chipmunk_range()))
        self.vocal_tract_value.blockSignals(False)

    def _set_uniform_vocal_tract_ratio(self):
        ratio = float(self.vocal_tract_value.value())
        if self.current is None:
            if 0 <= self._active_sentence_index < len(self.sentences):
                state = self.sentences[self._active_sentence_index]
                state["vocal_tract_length_ratio"] = ratio
                state["chipmunk_range"] = \
                    self.vocal_tract_chipmunk.isChecked()
            self.statusBar().showMessage(
                "Status: vocal tract ratio %.3f x will apply on Generate" %
                ratio)
            return
        self.vocal_tract_track.set_uniform_ratio(ratio, emit=True)

    def _vocal_tract_is_pending(self):
        if self.current is None:
            return False
        requested = (self.vocal_tract_track.targets()
                     if self.current.vocal_tract_mode == "curve" else [])
        applied = list(self.current.applied_vocal_tract_targets or [])
        return not vocal_tract.ratio_curves_close(requested, applied)

    def _refresh_vocal_tract_pending(self):
        if self.current is None:
            if 0 <= self._active_sentence_index < len(self.sentences):
                self.sentences[self._active_sentence_index][
                    "chipmunk_range"] = \
                    self.vocal_tract_chipmunk.isChecked()
            self._update_vocal_tract_readout()
            self._refresh_pending_ui()
            return False
        pending = self._vocal_tract_is_pending()
        if 0 <= self._active_sentence_index < len(self.sentences):
            state = self.sentences[self._active_sentence_index]
            state["vocal_tract_length_ratio"] = \
                vocal_tract.ratio_curve_summary(
                    self.vocal_tract_track.targets())
            state["chipmunk_range"] = \
                self.vocal_tract_chipmunk.isChecked()
            if pending:
                self._set_state_pending(
                    state, "rerender", "Vocal tract length changed")
            elif state.get("pending_reason") == "Vocal tract length changed":
                self._set_state_pending(state, "")
        self._update_vocal_tract_readout()
        self._refresh_pending_ui()
        return pending

    def _apply_vocal_tract_state(self, mode, targets):
        if self.current is None:
            return
        safe = [
            (float(time), self.vocal_tract_track.profile.clamp(
                float(value), self.vocal_tract_track.chipmunk_range()))
            for time, value in targets
        ]
        self.current.vocal_tract_mode = str(mode or "")
        self.current.vocal_tract_override = safe
        self._sync_timing_track(reset=True)
        self._refresh_vocal_tract_pending()

    def _on_vocal_tract_commit(self, targets):
        if self.current is None:
            return
        old = (self.current.vocal_tract_mode,
               list(self.current.vocal_tract_override))
        values = [float(value) for _time, value in targets]
        identity = values and all(abs(value - 1.0) <= 1e-10
                                  for value in values)
        new = ("" if identity else "curve",
               [] if identity else list(targets))
        self._apply_vocal_tract_state(*new)
        if old != new:
            self._push_applied_undo(
                "vocal tract curve",
                lambda state=old: self._apply_vocal_tract_state(*state),
                lambda state=new: self._apply_vocal_tract_state(*state),
            )
        self.statusBar().showMessage(
            "Status: vocal tract curve edited -- Re-render shifts the "
            "spectral envelope without changing pitch or timing")

    def _on_vocal_tract_clear(self):
        if self.current is None:
            return
        old = (self.current.vocal_tract_mode,
               list(self.current.vocal_tract_override))
        new = ("", [])
        self._apply_vocal_tract_state(*new)
        if old != new:
            self._push_applied_undo(
                "reset vocal tract curve",
                lambda state=old: self._apply_vocal_tract_state(*state),
                lambda state=new: self._apply_vocal_tract_state(*state),
            )
        self.statusBar().showMessage(
            "Status: vocal tract curve reset to the original 1.000 x voice")

    def _on_vocal_tract_range_toggled(self, enabled):
        if self._switching_sentence:
            return
        before = (list(self.vocal_tract_track.targets())
                  if self.current is not None and
                  self.current.vocal_tract_mode == "curve" else [])
        changed = self.vocal_tract_track.set_chipmunk_range(bool(enabled))
        self._sync_vocal_tract_value_bounds()
        if changed and self.current is not None:
            self.current.vocal_tract_override = \
                self.vocal_tract_track.targets()
        if 0 <= self._active_sentence_index < len(self.sentences):
            self.sentences[self._active_sentence_index][
                "chipmunk_range"] = bool(enabled)
        pending = self._refresh_vocal_tract_pending()
        if changed:
            self.statusBar().showMessage(
                "Status: curve clamped to the reference-derived realistic "
                "range -- Re-render applies the clamped values")
        elif before:
            self.statusBar().showMessage(
                "Status: %s range enabled; the current curve is unchanged" %
                ("expanded chipmunk" if enabled else "realistic"))
        if pending:
            self._capture_active_sentence()

    @staticmethod
    def _state_peak_and_applied(state):
        pre_gain = float(state.get("pre_gain_peak") or 0.0)
        applied = state.get("applied_gain_db")
        if pre_gain > 0.0:
            return pre_gain, 0.0
        raw = state.get("preview_audio")
        samples = np.asarray(raw if raw is not None else [], np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        return peak, float(applied or 0.0)

    @staticmethod
    def _gain_pending(state):
        applied = state.get("applied_gain_db")
        return bool(state.get("rendered") and applied is not None and
                    abs(float(state.get("output_gain_db") or 0.0) -
                        float(applied)) > 0.049)

    def _refresh_gain_controls(self):
        if getattr(self, "_refreshing_gain_controls", False):
            return
        self._refreshing_gain_controls = True
        try:
            self._refresh_gain_controls_impl()
        finally:
            self._refreshing_gain_controls = False

    def _refresh_gain_controls_impl(self):
        if not hasattr(self, "speech_gain"):
            return
        active = (self.sentences[self._active_sentence_index]
                  if 0 <= self._active_sentence_index < len(self.sentences)
                  else None)
        if active is None:
            self.speech_gain.set_audio_state(False)
            self.speech_gain.set_pending(False)
        else:
            peak, applied = self._state_peak_and_applied(active)
            available = bool(active.get("rendered") and peak > 0.0)
            self.speech_gain.set_audio_state(available, peak, applied)
            self.speech_gain.set_value(
                float(active.get("output_gain_db") or 0.0), emit=False)
            self.speech_gain.set_pending(self._gain_pending(active))

        rendered = [state for state in self.sentences
                    if state.get("rendered")]
        ceilings = []
        for state in rendered:
            peak, applied = self._state_peak_and_applied(state)
            if peak > 0.0:
                ceiling = safe_gain_ceiling_db(peak, applied)
                ceilings.append(ceiling)
                if (not self._allow_output_clipping and
                        float(state.get("output_gain_db") or 0.0) > ceiling):
                    state["output_gain_db"] = ceiling
        self.sentences_view.gain.set_safe_ceiling(
            bool(ceilings), min(ceilings) if ceilings else 12.0)
        if active is not None:
            self.speech_gain.set_value(
                float(active.get("output_gain_db") or 0.0), emit=False)
            self.speech_gain.set_pending(self._gain_pending(active))
            self.sentences_view.gain.set_value(
                float(active.get("output_gain_db") or 0.0), emit=False)
        self.sentences_view.gain.set_pending(
            any(self._gain_pending(state) for state in rendered))

    def _on_allow_clipping_changed(self, enabled):
        if getattr(self, "_syncing_clipping", False):
            return
        self._syncing_clipping = True
        try:
            self._allow_output_clipping = bool(enabled)
            self.speech_gain.set_allow_clipping(enabled, emit=False)
            self.sentences_view.gain.set_allow_clipping(enabled, emit=False)
            self.cfg["allow_output_clipping"] = bool(enabled)
            self._refresh_gain_controls()
        finally:
            self._syncing_clipping = False

    def _on_all_sentences_gain_changed(self, value):
        if self._switching_sentence:
            return
        value = float(value)
        for state in self.sentences:
            state["output_gain_db"] = value
            if self._gain_pending(state):
                self._set_state_pending(
                    state, "rerender", "Output volume changed")
            elif state.get("pending_reason") == "Output volume changed":
                self._set_state_pending(state, "")
        if 0 <= self._active_sentence_index < len(self.sentences):
            self.speech_gain.set_value(value, emit=False)
        self._refresh_gain_controls()
        self._refresh_pending_ui()
        self.statusBar().showMessage(
            "Status: all sentence volumes changed -- Re-render applies them")

    def _mark_gain_applied(self, syn):
        gain = float(getattr(syn, "applied_gain_db",
                             self.output_gain.value()))
        peak = float(getattr(syn, "pre_gain_peak", 0.0))
        if 0 <= self._active_sentence_index < len(self.sentences):
            state = self.sentences[self._active_sentence_index]
            state["output_gain_db"] = gain
            state["applied_gain_db"] = gain
            state["pre_gain_peak"] = peak
            state["rendered"] = not bool(self._pending_action(state))
        self._refresh_gain_controls()

    def _refresh_sentences_after_render(self):
        if (hasattr(self, "mode_tabs") and
                self.mode_tabs.currentIndex() == 1):
            selected = self.sentences_view.selected_sentence_indices()
            self._refresh_sentences_view()
            self.sentences_view.set_selected_indices(
                selected or [self._active_sentence_index])

    def _cut_region_to_sentence(self, first, last):
        if not (0 <= first <= last < len(self.waveform.segments)):
            return
        if first == 0 and last == len(self.waveform.segments) - 1:
            self.statusBar().showMessage(
                "Status: the selection already contains the whole sentence")
            return
        selected = [segment.phone for segment in
                    self.waveform.segments[first:last + 1]
                    if segment.phone != "pau"]
        if not selected:
            return
        self._capture_active_sentence()
        new_state = self._sentence_state_snapshot(
            self.sentences[self._active_sentence_index])
        new_state.update({
            "text": " ".join(selected), "input_mode": "phones",
            "synthesis": None, "editor_segments": [], "timing_factors": [],
            "preview_audio": np.zeros(1, np.float32),
            "preview_sr": self.waveform.sr, "needs_rerender": False,
            "needs_generate": True,
            "pending_reason": "Region moved to a new sentence",
            "rendered": False, "phrases": [],
        })
        for index in range(last, first - 1, -1):
            self.waveform._delete_phone(index)
        remaining = [phone for phone in self.waveform.phone_list()
                     if phone != "pau"]
        self.input_mode.setCurrentIndex(self.input_mode.findData("phones"))
        self.text.setText(" ".join(remaining))
        self._on_sentence_text_edited(self.text.text())
        insert_at = self._active_sentence_index + 1
        self.sentences.insert(insert_at, new_state)
        self._refresh_sentence_selector(self._active_sentence_index)
        self._refresh_pending_ui()
        self.statusBar().showMessage(
            "Status: selection moved to sentence %d; generate both sentences"
            % (insert_at + 1))

    def _persist_config(self):
        self.cfg["default_language"] = self.lang.currentText()
        self.cfg["default_text"] = self.text.text()
        self.cfg["synth_speed"] = round(self._speed_factor(), 3)
        self.cfg["pitch_hz"] = float(self.pitch.value())
        self.cfg["pitch_fall_pct"] = float(self.fall.value())
        self.cfg["output_gain_db"] = float(self.output_gain.value())
        active_state = (
            self.sentences[self._active_sentence_index]
            if 0 <= self._active_sentence_index < len(self.sentences)
            else {})
        self.cfg["vocal_tract_length_ratio"] = (
            vocal_tract.ratio_curve_summary(
                self.vocal_tract_track.targets())
            if self.current is not None else
            float(active_state.get("vocal_tract_length_ratio", 1.0))
        )
        self.cfg["chipmunk_range"] = \
            self.vocal_tract_chipmunk.isChecked()
        self.cfg["allow_output_clipping"] = bool(
            self._allow_output_clipping)
        self.cfg["fault_mode"] = self._fault_mode()
        self.cfg["monotone"] = self._monotone()
        self.cfg["parameter_mode"] = self.parameter_mode.currentData()
        self.cfg["show_curve_linguistic_units"] = bool(
            self.curve_unit_overlay.isChecked())
        self.cfg["follow_playhead"] = bool(self.follow_playhead.isChecked())
        self.cfg["follow_spoken_sentence"] = bool(
            self.sentences_view.follow_spoken_sentence.isChecked())
        self.cfg["shortcuts"] = dict(self.shortcuts)
        try:
            fc.save_config(self.cfg, CONFIG_PATH)
        except Exception as e:
            self.statusBar().showMessage(
                "Status: could not save config.json (%s)" % e)

    def closeEvent(self, ev):
        if self._synthesis_busy:
            self._close_requested = True
            self._batch_cancel_requested = True
            ev.ignore()
            self.statusBar().showMessage(
                "Status: closing after the current synthesis call finishes")
            return
        self._close_requested = False
        self._shutdown_resources()
        super().closeEvent(ev)

    def _shutdown_resources(self):
        """Release process-owned resources once, including warm WSL children."""
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        self._playback_timer.stop()
        self._playback_finish_timer.stop()
        self._voice_root_refresh_timer.stop()
        self._playback_token += 1
        try:
            self.player.shutdown()
        except Exception:
            pass
        try:
            self.fest.shutdown()
        except Exception:
            pass
        self._persist_config()

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def _output_audio(self):
        if 0 <= self._active_sentence_index < len(self.sentences):
            samples, sr = self._state_audio(
                self.sentences[self._active_sentence_index])
            if samples.size:
                return samples, sr
        return self.waveform.get_audio()

    @staticmethod
    def _state_audio(state):
        raw = (state or {}).get("preview_audio")
        samples = np.asarray(raw if raw is not None else [], np.float32)
        return samples, int((state or {}).get("preview_sr") or 16000)

    @staticmethod
    def _sentence_state_snapshot(state):
        """Deep-copy editor metadata while sharing rendered audio buffers.

        Rendered arrays are replaced, never edited in place. Sharing them
        prevents one sentence waveform from being duplicated into every undo,
        clipboard, phrase-preview, and duplicate-sentence snapshot.
        """
        source = state or {}
        memo = {}

        def share(value):
            if isinstance(value, np.ndarray):
                memo[id(value)] = value

        share(source.get("preview_audio"))
        synthesis = source.get("synthesis")
        if synthesis is not None:
            share(getattr(synthesis, "samples", None))
        for preview in dict(source.get("phrase_previews") or {}).values():
            if isinstance(preview, (tuple, list)) and preview:
                share(preview[0])
        return copy.deepcopy(source, memo)

    def _commit_synthesis_to_state(
            self, state, syn, rendered_text, *, samples=None, sr=None,
            segments=None, timing_factors=None):
        """Commit a render without making its sentence the visible editor."""
        if state is None:
            return
        output = np.asarray(
            syn.samples if samples is None else samples, np.float32)
        output_sr = int(syn.sr if sr is None else sr)
        editor_segments = copy.deepcopy(
            syn.segments if segments is None else segments)
        factors = list(timing_factors or ())
        if len(factors) != len(editor_segments):
            factors = [1.0] * len(editor_segments)
        applied_gain = float(getattr(
            syn, "applied_gain_db", state.get("output_gain_db", 0.0)))
        state.update({
            "synthesis": syn,
            "rendered_text": str(rendered_text),
            "editor_segments": editor_segments,
            "timing_factors": factors,
            "preview_audio": output,
            "preview_sr": output_sr,
            "rendered": True,
            "cache_loaded": False,
            "output_gain_db": applied_gain,
            "applied_gain_db": applied_gain,
            "pre_gain_peak": float(getattr(syn, "pre_gain_peak", 0.0)),
            "rendered_pitch_hz": float(
                state.get("pitch_hz") or 185.0),
            "rendered_fall_pct": float(
                state.get("fall_pct") or 0.0),
            "applied_vocal_tract_length_ratio": float(
                getattr(syn, "vocal_tract_length_ratio", 1.0)),
            "applied_vocal_tract_targets": list(getattr(
                syn, "applied_vocal_tract_targets", []) or []),
        })
        self._clear_state_pending(state)
        state.pop("_text_edit_revert", None)
        self._capture_phrase_snapshots(
            state, synthesis=syn, segments=editor_segments,
            audio=output, sr=output_sr, timing_factors=factors)

    def _commit_rendered_state(self, syn):
        """Make the just-rendered waveform canonical for every GUI surface."""
        if not (0 <= self._active_sentence_index < len(self.sentences)):
            return
        state = self.sentences[self._active_sentence_index]
        samples, sr = self.waveform.get_audio()
        self._commit_synthesis_to_state(
            state, syn, state.get("text", self.text.text()),
            samples=samples, sr=sr, segments=self.waveform.segments,
            timing_factors=self.waveform.factors())

    @staticmethod
    def _pitchmark_pitch_source(synthesis, step_seconds=0.02):
        """Recover a compact rendered F0 contour when Target is unavailable.

        UniSyn's target pitchmarks describe the actual output-period timeline.
        They are preferable to inventing a contour, but are much denser than
        an editor needs, so deterministic 20 ms median bins keep the fallback
        responsive. Pause epochs and implausible periods are excluded.
        """
        if synthesis is None:
            return []
        track = fc.pitchmark_f0_track(
            getattr(synthesis, "target_pitchmarks", ()) or ())
        if not track:
            return []
        spans = sorted(
            (float(segment.start), float(segment.end))
            for segment in (getattr(synthesis, "segments", ()) or ())
            if str(segment.phone) != "pau" and
            float(segment.end) > float(segment.start)
        )
        if not spans:
            return []
        accepted = []
        span_index = 0
        for time, value in track:
            time, value = float(time), float(value)
            if (not np.isfinite(time) or not np.isfinite(value) or
                    value < fc.PITCH_MIN_HZ or value > fc.PITCH_MAX_HZ):
                continue
            while (span_index < len(spans) and
                   time > spans[span_index][1] + 1e-9):
                span_index += 1
            if span_index >= len(spans):
                break
            if spans[span_index][0] - 1e-9 <= time <= \
                    spans[span_index][1] + 1e-9:
                accepted.append((time, value))
        if not accepted:
            return []
        step = max(0.005, float(step_seconds))
        buckets = {}
        origin = accepted[0][0]
        for time, value in accepted:
            key = int(math.floor((time - origin) / step + 1e-9))
            buckets.setdefault(key, []).append((time, value))
        return [
            (float(np.median([time for time, _value in buckets[key]])),
             float(np.median([value for _time, value in buckets[key]])))
            for key in sorted(buckets)
        ]

    def _synthesis_pitch_source(self, synthesis, *, text="", base=160.0,
                                fall=0.0, allow_baseline=True):
        generated = list(
            getattr(synthesis, "generated_targets", ()) or ())
        rendered = list(getattr(synthesis, "targets", ()) or ())
        # Before a manual overlay exists, Festival's returned Target relation
        # is the contour that actually made the current waveform. A pre-fix
        # in-memory render may carry a pre-recenter ``generated_targets``
        # contour; if that stale curve seeds the first edit, the whole
        # sentence shifts.
        # Once an override is active, retain the separate generated baseline
        # underneath it so Reset continues to mean "restore generated F0".
        mode = str(getattr(synthesis, "pitch_mode", "") or "")
        reset_pending = bool(
            getattr(synthesis, "_pitch_reset_pending", False))
        direct = (
            rendered if (
                rendered and not reset_pending and
                mode not in ("curve", "intonation")
            )
            else generated or rendered
        )
        if direct:
            return direct
        recovered = self._pitchmark_pitch_source(synthesis)
        if recovered:
            return recovered
        if not allow_baseline or synthesis is None:
            return []
        blocks = fc.phrase_blocks(
            getattr(synthesis, "segments", ()) or (),
            str(text or getattr(synthesis, "text", "") or ""))
        return fc.intonation_targets(blocks, float(base), float(fall))

    @staticmethod
    def _contiguous_unit_indexes(values, segment_count):
        try:
            indexes = sorted({int(value) for value in values})
        except (TypeError, ValueError):
            return []
        if (not indexes or indexes[0] < 0 or
                indexes[-1] >= int(segment_count)):
            return []
        if indexes != list(range(indexes[0], indexes[-1] + 1)):
            return []
        return indexes

    def _japanese_mora_unit_debug(self):
        rows = self._japanese_plan_rows()
        if not rows:
            return {}
        details = {}
        utterance = dict(self._active_japanese_state().get(
            "utterance") or {})
        for phrase in utterance.get("phrases") or ():
            if not isinstance(phrase, dict):
                continue
            for accent_phrase in phrase.get("accent_phrases") or ():
                if not isinstance(accent_phrase, dict):
                    continue
                for mora in accent_phrase.get("moras") or ():
                    if not isinstance(mora, dict):
                        continue
                    try:
                        details[int(mora.get("index"))] = dict(mora)
                    except (TypeError, ValueError):
                        continue

        grouped = {}
        for row in rows:
            try:
                mora_index = int(row.get("mora_index"))
                segment_index = int(row.get("index"))
            except (TypeError, ValueError):
                continue
            grouped.setdefault(mora_index, []).append(segment_index)

        units = []
        for mora_index in sorted(grouped):
            indexes = self._contiguous_unit_indexes(
                grouped[mora_index], len(self.waveform.segments))
            if not indexes:
                continue
            detail = details.get(mora_index, {})
            surface = str(detail.get("surface") or "").strip()
            reading = str(detail.get("reading") or "").strip()
            phones = [
                str(self.waveform.segments[index].phone)
                for index in indexes
            ]
            units.append({
                "index": mora_index,
                "kind": "mora",
                "phone_start": indexes[0],
                "phone_end": indexes[-1] + 1,
                "phones": phones,
                "display_label": surface or reading or " ".join(phones),
                "confidence": detail.get("confidence", 0.0),
                "tooltip": (
                    "Japanese mora %d\nsurface: %s\nreading: %s\nphones: %s"
                    % (
                        mora_index + 1,
                        surface or "(unavailable)",
                        reading or "(unavailable)",
                        " ".join(phones) or "(none)",
                    )
                ),
            })
        return {"language": "ja", "unit_kind": "mora", "units": units}

    def _asaxi_mora_unit_debug(self):
        units = []
        for row in self._asaxi_mora_rows():
            indexes = self._contiguous_unit_indexes(
                row.get("segment_indices") or (),
                len(self.waveform.segments))
            if not indexes:
                continue
            try:
                mora_index = int(row.get("mora_index"))
            except (TypeError, ValueError):
                mora_index = len(units)
            phones = [
                str(self.waveform.segments[index].phone)
                for index in indexes
            ]
            text = str(row.get("text") or "").strip()
            word = str(row.get("word") or "").strip()
            pitch = str(row.get("pitch") or "").strip()
            units.append({
                "index": mora_index,
                "kind": "mora",
                "phone_start": indexes[0],
                "phone_end": indexes[-1] + 1,
                "phones": phones,
                "display_label": text or " ".join(phones),
                "tooltip": (
                    "Asaxi mora %d\nword: %s\nmora: %s\nphones: %s\n"
                    "pitch: %s" % (
                        mora_index + 1,
                        word or "(unavailable)",
                        text or "(unavailable)",
                        " ".join(phones) or "(none)",
                        pitch or "(unavailable)",
                    )
                ),
            })
        return {
            "language": "asaxi", "unit_kind": "mora", "units": units,
        }

    def _english_profile_nucleus_phones(self):
        """Return vowel nuclei declared by the active generated voice."""

        voice = str(self._current_voicebank() or "")
        if not voice:
            return ()
        try:
            backend = self._ab()
            reader = getattr(backend, "voice_metadata", None)
            if reader is not None:
                metadata = dict(reader(voice) or {})
            else:
                db_reader = getattr(backend, "db", None)
                metadata = dict(
                    getattr(db_reader(voice), "metadata", {}) or {}
                ) if db_reader is not None else {}
        except (fc.BackendError, OSError, ValueError, TypeError):
            return ()
        profile = metadata.get("arpasing_profile")
        symbol_types = (
            profile.get("symbol_types")
            if isinstance(profile, dict) else {}
        )
        return english_syllables.profile_nucleus_phones(
            symbol_types if isinstance(symbol_types, dict) else {})

    def _linguistic_unit_debug(self, phones):
        if (not hasattr(self, "curve_unit_overlay") or
                not self.curve_unit_overlay.isChecked() or
                self.current is None):
            return {}
        language = fc.normalize_language_code(
            self.current.lang or self._current_lang_code())
        if language == "en":
            metadata = dict(
                getattr(self.current, "english_syllabification", {}) or {})
            stored_phones = [
                str(phone) for phone in metadata.get("phones") or ()
            ]
            declared_nuclei = self._english_profile_nucleus_phones()
            stored_nuclei = tuple(sorted(
                str(phone) for phone in
                metadata.get("declared_nucleus_phones") or ()
            ))
            if (
                stored_phones != [str(phone) for phone in phones]
                or str(metadata.get("frontend_version") or "") !=
                english_syllables.FRONTEND_VERSION
                or stored_nuclei != declared_nuclei
            ):
                metadata = english_syllables.syllabify_english(
                    phones,
                    extra_nucleus_phones=declared_nuclei,
                ).to_dict()
                self.current.english_syllabification = metadata
            return metadata
        if language in ("ja", "jp"):
            return self._japanese_mora_unit_debug()
        if language == "asaxi":
            return self._asaxi_mora_unit_debug()
        return {}

    def _sync_linguistic_unit_overlay(self, phones=None):
        if phones is None:
            phones = [
                str(segment.phone) for segment in self.waveform.segments
            ]
        metadata = self._linguistic_unit_debug(phones)
        for name in ("pitch_track", "voicing_track", "vocal_tract_track"):
            track = getattr(self, name, None)
            if track is not None:
                track.set_linguistic_unit_debug(metadata)

    def _on_curve_unit_overlay_toggled(self, enabled):
        self.cfg["show_curve_linguistic_units"] = bool(enabled)
        self._sync_linguistic_unit_overlay()
        self._persist_config()

    def _sync_timing_track(self, reset=False):
        spans = [(s.start, s.end) for s in self.waveform.segments]
        phones = [s.phone for s in self.waveform.segments]
        segment_ids = [str(s.uid or "") for s in self.waveform.segments]
        timing_roles = self._japanese_timing_roles()
        for index, segment in enumerate(self.waveform.segments):
            segment.timing_role = (
                timing_roles[index] if index < len(timing_roles) else ""
            )
        if (self.current is not None and
                len(self.current.segments) == len(timing_roles)):
            for segment, role in zip(self.current.segments, timing_roles):
                segment.timing_role = role
        self.timing.set_segments(
            spans, self.waveform.factors(), phones, timing_roles
        )
        self._on_timing_filter()
        self._sync_japanese_timeline()
        self._sync_asaxi_timeline()
        if not hasattr(self, "pitch_track") or self.current is None:
            self._sync_linguistic_unit_overlay([])
            return
        base = float(self.pitch.value())
        source = self._synthesis_pitch_source(
            self.current, text=self.current.text or self.text.text(),
            base=base, fall=float(self.fall.value()),
            allow_baseline=self._engine() == "festival_wsl")
        ground = fc.remap_targets_aligned(
            source, self.current.segments,
            self.waveform.segments) if source else []
        anchor_edges = not bool(
            self._fault_mode().get("disable_f0_correction"))
        if ground and anchor_edges:
            ground = fc.anchor_phrase_targets(
                [(segment.phone, max(0.01, segment.dur))
                 for segment in self.waveform.segments],
                ground, base)
        kinds = [b.get("kind", ".") for b in
                 (self.current.intonation_blocks or [])]
        blocks = fc.phrase_blocks(self.waveform.segments,
                                  self.text.text(), kinds=kinds)
        if self.current.pitch_mode == "intonation":
            self.current.intonation_blocks = blocks
            override = fc.overlay_intonation_targets(
                ground, blocks, base, float(self.fall.value()))
        elif self.current.pitch_mode == "curve":
            override = self.current.pitch_override
        else:
            override = []
        if reset or self.current.pitch_mode == "intonation":
            self.pitch_track.set_data(
                spans, phones, ground, override, base,
                anchor_edges=anchor_edges, segment_ids=segment_ids)
        else:
            self.pitch_track.update_geometry(
                spans, phones, ground, base, anchor_edges=anchor_edges,
                segment_ids=segment_ids)
            if self.current.pitch_mode == "curve":
                self.current.pitch_override = \
                    self.pitch_track.render_targets()
        voicing_source = (
            self.current.generated_voicing_targets
            or self.current.source_voicing_targets
        )
        voicing_ground = fc.remap_targets_aligned(
            voicing_source, self.current.segments,
            self.waveform.segments,
        ) if voicing_source else []
        voicing_override = (
            self.current.voicing_override
            if self.current.voicing_mode == "curve" else []
        )
        if reset:
            self.voicing_track.set_data(
                spans, phones, voicing_ground, voicing_override, 1.0,
                segment_ids=segment_ids,
            )
        else:
            self.voicing_track.update_geometry(
                spans, phones, voicing_ground, 1.0,
                segment_ids=segment_ids,
            )
            if self.current.voicing_mode == "curve":
                self.current.voicing_override = \
                    self.voicing_track.targets()
        self._sync_linguistic_unit_overlay(phones)
        tract_ground_source = list(
            self.current.generated_vocal_tract_targets or [])
        if not tract_ground_source:
            old_duration = max(
                (float(segment.end) for segment in self.current.segments),
                default=0.0)
            tract_ground_source = [(0.0, 1.0), (old_duration, 1.0)]
        tract_ground = fc.remap_targets_aligned(
            tract_ground_source, self.current.segments,
            self.waveform.segments,
        ) if self.current.segments else tract_ground_source
        tract_override = (
            list(self.current.vocal_tract_override)
            if self.current.vocal_tract_mode == "curve" else []
        )
        if reset:
            self.vocal_tract_track.set_data(
                spans, phones, tract_ground, tract_override, 1.0,
                segment_ids=segment_ids,
            )
        else:
            self.vocal_tract_track.update_geometry(
                spans, phones, tract_ground, 1.0,
                segment_ids=segment_ids,
            )
            if self.current.vocal_tract_mode == "curve":
                self.current.vocal_tract_override = \
                    self.vocal_tract_track.targets()
        self._update_vocal_tract_readout()
        self.intonation.set_blocks(blocks)
        self.recordings.set_data(
            self.waveform.segments, self._unit_alternatives(),
            self.current.selected_units, self.current.unit_overrides,
            self._source_selection_phones())

    def _on_timing_commit(self, changes):
        changes = {int(index): float(value)
                   for index, value in dict(changes).items()
                   if 0 <= int(index) < len(self.waveform.segments)}
        old_factors = self.waveform.factors()
        old = {index: old_factors[index] for index in changes}
        self.waveform.set_factors(changes)
        if old != changes:
            self._push_applied_undo(
                "phoneme timing",
                lambda values=dict(old): self.waveform.set_factors(values),
                lambda values=dict(changes): self.waveform.set_factors(values))
        self._mark_active_pending("rerender", "Phoneme timing changed")
        self.statusBar().showMessage(
            "Status: %d phoneme timing(s) adjusted -- Re-render Phonemes "
            "re-synthesizes at these timings for optimal quality"
            % len(changes))

    def _on_timing_reset(self, idx):
        old = self.waveform.factors()[idx]
        self.waveform.set_factor(idx, 1.0)
        if abs(old - 1.0) > 1e-9:
            self._push_applied_undo(
                "reset phoneme timing",
                lambda index=idx, factor=old:
                self.waveform.set_factor(index, factor),
                lambda index=idx: self.waveform.set_factor(index, 1.0))
        self.statusBar().showMessage(
            "Status: phoneme %d timing reset to the rendered original" % idx)

    def _on_waveform_timing_edit(self, before, after):
        old = {index: float(value) for index, value in enumerate(before)}
        new = {index: float(value) for index, value in enumerate(after)}
        self._push_applied_undo(
            "waveform boundary",
            lambda values=old: self.waveform.set_factors(values),
            lambda values=new: self.waveform.set_factors(values))

    def _on_waveform_structure_edit(self, before, after, label):
        self._push_applied_undo(
            str(label or "edit phoneme region"),
            lambda snapshot=before:
            self.waveform.restore_structure(snapshot),
            lambda snapshot=after:
            self.waveform.restore_structure(snapshot))

    def _apply_pitch_state(self, mode, targets, blocks):
        if self.current is None:
            return
        previous_mode = str(self.current.pitch_mode or "")
        self.current.pitch_mode = str(mode or "")
        self.current.pitch_override = [tuple(point) for point in targets]
        self.current.intonation_blocks = [dict(block) for block in blocks]
        self.current._pitch_reset_pending = bool(
            not self.current.pitch_mode and
            previous_mode in ("curve", "intonation") and
            not self.current.pitch_override
        )
        self._sync_timing_track(reset=True)
        self._mark_active_pending("rerender", "Pitch contour changed")

    def _on_pitch_commit(self, targets):
        if self.current is None:
            return
        old = (self.current.pitch_mode,
               list(self.current.pitch_override),
               [dict(block) for block in self.current.intonation_blocks])
        rendered = self.pitch_track.render_targets()
        new = ("curve", list(rendered or targets), old[2])
        self._apply_pitch_state(*new)
        self._push_applied_undo(
            "pitch curve",
            lambda state=old: self._apply_pitch_state(*state),
            lambda state=new: self._apply_pitch_state(*state))
        self.statusBar().showMessage(
            "Status: pitch curve edited -- Re-render applies it through PSOLA")

    def _on_pitch_clear(self):
        if self.current is None:
            return
        old = (self.current.pitch_mode,
               list(self.current.pitch_override),
               [dict(block) for block in self.current.intonation_blocks])
        mode = "" if old[0] in ("curve", "intonation") else old[0]
        new = (mode, [], old[2])
        self._apply_pitch_state(*new)
        if old != new:
            self._push_applied_undo(
                "reset pitch curve",
                lambda state=old: self._apply_pitch_state(*state),
                lambda state=new: self._apply_pitch_state(*state))
        self.statusBar().showMessage(
            "Status: pitch override cleared -- Re-render restores generated F0")

    def _apply_voicing_state(self, mode, targets):
        if self.current is None:
            return
        self.current.voicing_mode = str(mode or "")
        self.current.voicing_override = [tuple(point) for point in targets]
        self._sync_timing_track(reset=True)
        self._mark_active_pending("rerender", "Voicing curve changed")

    def _on_voicing_commit(self, targets):
        if self.current is None:
            return
        old = (self.current.voicing_mode,
               list(self.current.voicing_override))
        new = ("curve", list(targets))
        self._apply_voicing_state(*new)
        self._push_applied_undo(
            "voicing curve",
            lambda state=old: self._apply_voicing_state(*state),
            lambda state=new: self._apply_voicing_state(*state),
        )
        self.statusBar().showMessage(
            "Status: voicing curve edited -- Re-render remixes measured "
            "harmonic and aperiodic excitation"
        )

    def _on_voicing_clear(self):
        if self.current is None:
            return
        old = (self.current.voicing_mode,
               list(self.current.voicing_override))
        new = ("", [])
        self._apply_voicing_state(*new)
        if old != new:
            self._push_applied_undo(
                "reset voicing curve",
                lambda state=old: self._apply_voicing_state(*state),
                lambda state=new: self._apply_voicing_state(*state),
            )
        self.statusBar().showMessage(
            "Status: voicing override cleared -- Re-render restores the "
            "generated excitation balance"
        )

    def _on_intonation_commit(self, blocks):
        if self.current is None:
            return
        old = (self.current.pitch_mode,
               list(self.current.pitch_override),
               [dict(block) for block in self.current.intonation_blocks])
        new = ("intonation", list(self.current.pitch_override),
               [dict(block) for block in blocks])
        self._apply_pitch_state(*new)
        self._push_applied_undo(
            "intonation block",
            lambda state=old: self._apply_pitch_state(*state),
            lambda state=new: self._apply_pitch_state(*state))
        self.statusBar().showMessage(
            "Status: intonation block changed -- Re-render applies it through PSOLA")

    def _on_phones_edited(self):
        self._mark_active_pending("rerender", "Phoneme sequence changed")
        if self.current is not None:
            self.recordings.set_data(
                self.waveform.segments, self._unit_alternatives(),
                self.current.selected_units, self.current.unit_overrides,
                self._source_selection_phones())

    def _japanese_plan_rows(self):
        state = self._active_japanese_state()
        plan = dict(state.get("last_plan") or {})
        rows = [dict(row) for row in plan.get("segments") or []
                if isinstance(row, dict)]
        if not rows or not self.waveform.segments:
            return rows

        old_phones = [str(row.get("phone") or "") for row in rows]
        new_phones = [str(segment.phone) for segment in self.waveform.segments]
        matches = SequenceMatcher(
            None, old_phones, new_phones, autojunk=False).get_matching_blocks()
        index_map = {}
        for block in matches:
            for offset in range(block.size):
                index_map[block.a + offset] = block.b + offset

        # A mora remains editable only while all of its planned phones are
        # still present.  This avoids attaching stale linguistic data to a
        # neighboring phone after deletion or direct phone-text editing.
        mora_old_indexes = {}
        for position, row in enumerate(rows):
            if row.get("mora_index") is not None:
                mora_old_indexes.setdefault(int(row["mora_index"]), []).append(
                    int(row.get("index", position)))
        complete_moras = {
            mora for mora, indexes in mora_old_indexes.items()
            if indexes and all(index in index_map for index in indexes)
        }
        aligned = []
        for position, row in enumerate(rows):
            old_index = int(row.get("index", position))
            new_index = index_map.get(old_index)
            if new_index is None:
                continue
            if (row.get("mora_index") is not None and
                    int(row["mora_index"]) not in complete_moras):
                continue
            row["index"] = new_index
            aligned.append(row)
        return aligned

    def _asaxi_mora_rows(self):
        state = self._active_asaxi_state()
        plan = dict(state.get("last_plan") or {})
        rows = asaxi_editing.mora_rows(plan)
        old_phones = [
            str(phone) for phone in plan.get("rendered_phones") or []
        ]
        if not rows or not old_phones or not self.waveform.segments:
            return rows
        new_phones = [str(segment.phone) for segment in self.waveform.segments]
        index_map = {}
        for block in SequenceMatcher(
                None, old_phones, new_phones,
                autojunk=False).get_matching_blocks():
            for offset in range(block.size):
                index_map[block.a + offset] = block.b + offset
        aligned = []
        for row in rows:
            old_indices = [
                int(index) for index in row.get("segment_indices") or []
            ]
            if not old_indices or not all(
                    index in index_map for index in old_indices):
                continue
            updated = dict(row)
            new_indices = [index_map[index] for index in old_indices]
            updated["segment_indices"] = new_indices
            updated["start"] = min(
                self.waveform.segments[index].start
                for index in new_indices)
            updated["end"] = max(
                self.waveform.segments[index].end
                for index in new_indices)
            aligned.append(updated)
        return aligned

    def _japanese_timing_roles(self):
        """Align linguistic roles while keeping structural phones universal."""
        roles = [
            str(getattr(segment, "timing_role", "") or "")
            for segment in self.waveform.segments
        ]
        source_phones = self._source_selection_phones()
        for index, segment in enumerate(self.waveform.segments):
            if (
                str(segment.phone) == "cl"
                and index < len(source_phones)
                and str(source_phones[index]) != "cl"
            ):
                # cl owns a normal editable interval in every language. Its
                # acoustic source is the following consonant, but it remains
                # a consonant/VC timing region rather than a hidden extra
                # release or a literal OTO cl sample.
                roles[index] = roles[index] or "structural_vc"
        if self._current_lang_code() not in ("ja", "jp"):
            return roles
        state = self._active_japanese_state()
        special_moras = {}
        utterance = dict(state.get("utterance") or {})
        for phrase in utterance.get("phrases") or []:
            for accent in dict(phrase).get("accent_phrases") or []:
                for mora in dict(accent).get("moras") or []:
                    row = dict(mora)
                    if row.get("index") is not None:
                        special_moras[int(row["index"])] = str(
                            row.get("special_mora") or ""
                        )
        for row in self._japanese_plan_rows():
            index = int(row.get("index", -1))
            if not 0 <= index < len(roles):
                continue
            role = str(row.get("timing_role") or "")
            mora_index = row.get("mora_index")
            if (not role and mora_index is not None and
                    special_moras.get(int(mora_index)) == "moraic_nasal"):
                role = "moraic_nasal"
            roles[index] = role
        return roles

    def _on_waveform_view_range_changed(self, *_args):
        if not hasattr(self, "japanese_editor"):
            return
        left, right = self.waveform.plot.getViewBox().viewRange()[0]
        self.japanese_editor.set_view_range(left, right)
        if hasattr(self, "asaxi_editor"):
            self.asaxi_editor.set_view_range(left, right)

    def _change_pitch_zoom(self, delta):
        if not hasattr(self, "pitch_track"):
            return
        self.pitch_track.set_zoom(
            self.pitch_track.zoom_level() + int(delta))

    def _sync_pitch_navigator(self, center=None, zoom=None):
        if not hasattr(self, "pitch_scroll"):
            return
        center = (self.pitch_track.view_center()
                  if center is None else float(center))
        zoom = (self.pitch_track.zoom_level()
                if zoom is None else int(zoom))
        self.pitch_scroll.blockSignals(True)
        try:
            self.pitch_scroll.setValue(int(round(center)))
            y_range = self.pitch_track.getViewBox().viewRange()[1]
            self.pitch_scroll.setPageStep(max(
                2, int(round((y_range[1] - y_range[0]) * .25))))
        finally:
            self.pitch_scroll.blockSignals(False)
        self.pitch_zoom_in.setEnabled(zoom < 5)
        self.pitch_zoom_out.setEnabled(zoom > 0)

    def _sync_japanese_timeline(self):
        if not hasattr(self, "japanese_editor"):
            return
        rows = self._japanese_plan_rows()
        self.japanese_editor.set_timeline(self.waveform.segments, rows)
        self._on_waveform_view_range_changed()
        self.japanese_editor.set_playhead(self.waveform.playhead_time())
        available = bool(
            rows and self._current_lang_code() in ("ja", "jp"))
        self.recordings_mora_details.setVisible(available)
        selected = self.waveform.selected_indices()
        if selected is not None:
            self._on_waveform_selection_changed(tuple(selected))
        else:
            mora_index = int(getattr(
                self.japanese_editor, "_selected_mora", -1))
            self.recordings_mora_details.setEnabled(
                bool(self._japanese_mora_contributions(mora_index))
                if mora_index >= 0 else False)

    def _sync_asaxi_timeline(self):
        if not hasattr(self, "asaxi_editor"):
            return
        state = self._active_asaxi_state()
        self.asaxi_editor.set_state(state)
        self.asaxi_editor.set_timeline(
            self.waveform.segments, self._asaxi_mora_rows())
        left, right = self.waveform.plot.getViewBox().viewRange()[0]
        self.asaxi_editor.set_view_range(left, right)
        self.asaxi_editor.set_playhead(self.waveform.playhead_time())
        selected = self.waveform.selected_indices()
        if selected is not None and self._current_lang_code() == "asaxi":
            self._on_waveform_selection_changed(tuple(selected))

    def _on_waveform_selection_changed(self, selected):
        if self._syncing_japanese_selection or not selected:
            return
        if self._current_lang_code() == "asaxi":
            first, last = int(selected[0]), int(selected[1])
            mora_index = next((
                int(row["mora_index"])
                for row in self._asaxi_mora_rows()
                if row.get("mora_index") is not None and any(
                    first <= int(index) <= last
                    for index in row.get("segment_indices") or [])
            ), None)
            if mora_index is not None:
                self._syncing_japanese_selection = True
                try:
                    self.asaxi_editor.select_mora(mora_index)
                finally:
                    self._syncing_japanese_selection = False
            return
        rows = self._japanese_plan_rows()
        if not rows:
            return
        first, last = (int(selected[0]), int(selected[1]))
        mora_index = next((
            int(row["mora_index"])
            for row in rows
            if row.get("mora_index") is not None
            and first <= int(row.get("index", -1)) <= last
        ), None)
        if mora_index is None:
            return
        self._syncing_japanese_selection = True
        try:
            self.japanese_editor.select_mora(mora_index)
        finally:
            self._syncing_japanese_selection = False
        self.recordings_mora_details.setEnabled(
            bool(self._japanese_mora_contributions(mora_index)))

    def _on_japanese_mora_selected(self, mora_index):
        if self._syncing_japanese_selection:
            return
        rows = self._japanese_plan_rows()
        indexes = sorted({
            int(row.get("index", position))
            for position, row in enumerate(rows)
            if row.get("mora_index") == int(mora_index)
            and 0 <= int(row.get("index", position))
            < len(self.waveform.segments)
        })
        if not indexes:
            self.recordings_mora_details.setEnabled(False)
            return
        self._syncing_japanese_selection = True
        try:
            self.waveform._set_selected_range(indexes[0], indexes[-1])
        finally:
            self._syncing_japanese_selection = False
        self.recordings_mora_details.setEnabled(
            bool(self._japanese_mora_contributions(mora_index)))
        start = self.waveform.segments[indexes[0]].start
        end = self.waveform.segments[indexes[-1]].end
        left, right = self.waveform.plot.getViewBox().viewRange()[0]
        if self.japanese_editor.follow_selection.isChecked():
            span = max(end - start, right - left)
            center = (start + end) * 0.5
            duration = max(
                (float(segment.end) for segment in self.waveform.segments),
                default=span)
            new_left = max(0.0, min(center - span * 0.5,
                                    max(0.0, duration - span)))
            self.waveform.plot.setXRange(
                new_left, new_left + span, padding=0)

    def _on_asaxi_mora_selected(self, mora_index):
        if self._syncing_japanese_selection:
            return
        row = next((
            item for item in self._asaxi_mora_rows()
            if int(item.get("mora_index", -1)) == int(mora_index)
        ), None)
        indexes = sorted({
            int(index) for index in (row or {}).get("segment_indices") or []
            if 0 <= int(index) < len(self.waveform.segments)
        })
        if not indexes:
            return
        self._syncing_japanese_selection = True
        try:
            self.waveform._set_selected_range(indexes[0], indexes[-1])
        finally:
            self._syncing_japanese_selection = False
        if not self.asaxi_editor.follow_selection.isChecked():
            return
        start = self.waveform.segments[indexes[0]].start
        end = self.waveform.segments[indexes[-1]].end
        left, right = self.waveform.plot.getViewBox().viewRange()[0]
        if start < left or end > right:
            span = max(end - start, right - left)
            center = (start + end) * 0.5
            duration = max(
                (float(segment.end) for segment in self.waveform.segments),
                default=span)
            new_left = max(
                0.0,
                min(center - span * 0.5, max(0.0, duration - span)),
            )
            self.waveform.plot.setXRange(
                new_left, new_left + span, padding=0)

    def _active_japanese_state(self):
        if not (0 <= self._active_sentence_index < len(self.sentences)):
            return je.new_edit_state()
        state = self.sentences[self._active_sentence_index]
        normalized = je.normalize_edit_state(state.get("japanese_state"))
        state["japanese_state"] = normalized
        return normalized

    def _apply_japanese_state_snapshot(self, snapshot, invalidation="rerender"):
        if not (0 <= self._active_sentence_index < len(self.sentences)):
            return
        state = self.sentences[self._active_sentence_index]
        state["japanese_state"] = je.normalize_edit_state(snapshot)
        self.japanese_editor.set_state(state["japanese_state"])
        self._sync_japanese_timeline()
        if invalidation == "rerender":
            self._mark_active_pending(
                "rerender", "Pitch accent or pitch changed")
        elif invalidation == "rebuild":
            self.statusBar().showMessage(
                "Status: Japanese bank profile changed -- rebuild the "
                "generated voice before rendering")
        self._refresh_pending_ui()

    def _on_japanese_edit(self, kind, index, value):
        if self._switching_sentence:
            return
        before = copy.deepcopy(self._active_japanese_state())
        after = copy.deepcopy(before)
        key = str(int(index))
        if kind == "accent":
            values = after.setdefault("accent_overrides", {})
            if value is None:
                values.pop(key, None)
            else:
                values[key] = dict(value)
        elif kind == "accent_structure":
            values = after.setdefault("accent_phrase_boundaries", {})
            values[key] = sorted({int(item) for item in (value or [])})
        elif kind == "phrase":
            values = after.setdefault("phrase_overrides", {})
            if value is None:
                values.pop(key, None)
            else:
                merged = dict(values.get(key) or {})
                merged.update(dict(value))
                values[key] = merged
        elif kind == "mora_pitch":
            values = after.setdefault("mora_pitch_offsets_cents", {})
            cents = int(value or 0)
            if cents:
                values[key] = cents
            else:
                values.pop(key, None)
        elif kind == "mora_voicing":
            values = after.setdefault("mora_voicing_overrides", {})
            row = dict(value or {})
            indexes = sorted({int(item) for item in
                              (row.get("mora_indices") or [])})
            degree = row.get("value")
            for mora_index in indexes:
                mora_key = str(mora_index)
                if degree is None:
                    values.pop(mora_key, None)
                else:
                    values[mora_key] = max(
                        je.MORA_VOICING_MIN,
                        min(je.MORA_VOICING_MAX, float(degree)))
        elif kind == "candidate_override":
            values = after.setdefault("manual_candidate_overrides", {})
            if value:
                values[key] = str(value)
            else:
                values.pop(key, None)
        elif kind in {
                "baseline_provider", "external_hts_trajectory",
                "dynamic_multipitch", "voice_color"}:
            after[kind] = value
        else:
            return
        after = je.normalize_edit_state(after)
        if before == after:
            return
        invalidation = je.invalidation_for_edit(
            kind if kind in {
                "baseline_provider", "external_hts_trajectory",
                "dynamic_multipitch", "voice_color",
                "accent_structure"} else
            "mora_pitch" if kind == "mora_pitch" else
            "mora_voicing" if kind == "mora_voicing" else
            "candidate_override" if kind == "candidate_override" else
            "question" if kind == "phrase" else "accent")
        self._apply_japanese_state_snapshot(after, invalidation)
        self._push_applied_undo(
            "Japanese %s" % kind.replace("_", " "),
            lambda state=before, mode=invalidation:
            self._apply_japanese_state_snapshot(state, mode),
            lambda state=after, mode=invalidation:
            self._apply_japanese_state_snapshot(state, mode))

    def _active_asaxi_state(self):
        if not (0 <= self._active_sentence_index < len(self.sentences)):
            return asaxi_editing.new_edit_state()
        state = self.sentences[self._active_sentence_index]
        normalized = asaxi_editing.normalize_edit_state(
            state.get("asaxi_state"))
        state["asaxi_state"] = normalized
        return normalized

    def _preview_asaxi_mora_voicing(self, sentence_state):
        """Refresh the dashed voicing baseline before audio is re-rendered."""

        syn = (sentence_state or {}).get("synthesis")
        if not isinstance(syn, fc.Synthesis) or syn.lang != "asaxi":
            return
        source_curve = list(syn.source_voicing_targets or [])
        metadata = dict(
            getattr(syn, "asaxi_prosody", None)
            or dict((sentence_state.get("asaxi_state") or {}).get(
                "last_plan") or {})
        )
        if not source_curve or not metadata:
            return
        overlays = self._asaxi_edit_overlays(
            sentence_state, sentence_state.get("text") or "")
        curve, predictions = asaxi_phonation.mora_voicing_curve(
            metadata,
            syn.segments,
            source_curve,
            voicing_overrides=overlays["voicing"],
        )
        if not curve:
            return
        syn.generated_voicing_targets = list(curve)
        metadata["mora_phonation_predictions"] = [
            prediction.to_dict() for prediction in predictions
        ]
        syn.asaxi_prosody = metadata
        sentence_state["asaxi_state"]["last_plan"] = copy.deepcopy(metadata)
        if syn is self.current:
            self._sync_timing_track(reset=True)

    def _apply_asaxi_state_snapshot(self, snapshot):
        if not (0 <= self._active_sentence_index < len(self.sentences)):
            return
        state = self.sentences[self._active_sentence_index]
        state["asaxi_state"] = asaxi_editing.normalize_edit_state(snapshot)
        self._preview_asaxi_mora_voicing(state)
        self.asaxi_editor.set_state(state["asaxi_state"])
        self._mark_active_pending(
            "rerender", "Asaxi mora accent or voicing changed")
        self._refresh_pending_ui()

    def _on_asaxi_mora_edit(self, kind, indices, value):
        if self._switching_sentence:
            return
        before = copy.deepcopy(self._active_asaxi_state())
        try:
            after = asaxi_editing.with_mora_edit(
                before, str(kind), indices, value)
        except (TypeError, ValueError):
            return
        if before == after:
            return
        self._apply_asaxi_state_snapshot(after)
        self._push_applied_undo(
            "Asaxi mora %s" % str(kind),
            lambda state=before: self._apply_asaxi_state_snapshot(state),
            lambda state=after: self._apply_asaxi_state_snapshot(state),
        )

    def _japanese_mora_contributions(self, mora_index):
        state = self._active_japanese_state()
        plan = dict(state.get("last_plan") or {})
        source_plan = dict(plan.get("source_contributions") or {})
        rows = []
        for contribution in source_plan.get("contributions") or []:
            if not isinstance(contribution, dict):
                continue
            indices = contribution.get("mora_indices") or []
            if int(mora_index) not in [int(value) for value in indices
                                       if value is not None]:
                continue
            components = contribution.get("source_components") or []
            if components:
                for component in components:
                    if not isinstance(component, dict):
                        continue
                    merged = dict(contribution)
                    merged.update(component)
                    merged["component_of"] = contribution.get("diphone")
                    rows.append(merged)
            else:
                rows.append(dict(contribution))
        return rows

    def _show_selected_mora_contributions(self):
        mora_index = int(getattr(
            self.japanese_editor, "_selected_mora", -1))
        rows = self._japanese_mora_contributions(mora_index) \
            if mora_index >= 0 else []
        state = self._active_japanese_state()
        raw = state.get("utterance")
        try:
            utterance = je.utterance_from_dict(raw) if raw else None
        except (TypeError, ValueError, KeyError):
            utterance = None
        mora = next((item for item in utterance.moras
                     if item.index == int(mora_index)), None) \
            if utterance is not None else None
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "Japanese mora sources",
                ("No rendered source contribution is available yet."
                 if mora is None else
                 "Mora: %s\nPhones: %s\n\nGenerate or Re-render to inspect "
                 "its UTAU source contributions." %
                 (mora.surface or mora.reading,
                  " ".join(phone.symbol for phone in mora.phones))))
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Japanese mora source contributions")
        dialog.resize(920, 360)
        layout = QtWidgets.QVBoxLayout(dialog)
        label = ((mora.surface or mora.reading or str(mora_index))
                 if mora is not None else str(mora_index))
        layout.addWidget(QtWidgets.QLabel(
            "Selected mora: %s  |  %d source contribution%s" %
            (label, len(rows), "" if len(rows) == 1 else "s")))
        table = QtWidgets.QTableWidget(len(rows), 6)
        table.setHorizontalHeaderLabels([
            "Role", "Alias", "Source WAV", "Source interval",
            "Target edge", "Fallback",
        ])
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for row_index, row in enumerate(rows):
            source_slice = dict(row.get("source_slice") or {})
            target = dict(row.get("target_interval") or {})
            values = (
                row.get("role") or row.get("source_kind") or "unknown",
                row.get("source_alias") or row.get("alias") or "",
                row.get("source_wav") or row.get("wav") or "",
                ("%.3f - %.3f s" %
                 (float(source_slice.get("start") or 0.0),
                  float(source_slice.get("end") or 0.0))
                 if source_slice else "n/a"),
                ("%s  %.3f s" %
                 (row.get("diphone") or row.get("component_of") or "",
                  float(target.get("phone_boundary") or 0.0))),
                row.get("fallback_reason") or "",
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                table.setItem(row_index, column, item)
        layout.addWidget(table)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def on_analyze_japanese_bank(self):
        source = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Japanese UTAU bank or subbank")
        if not source:
            return
        self.statusBar().showMessage(
            "Status: analyzing Japanese OTO aliases read-only...")
        QtWidgets.QApplication.processEvents()
        try:
            analysis = je.analyze_bank(source)
        except Exception as error:
            QtWidgets.QMessageBox.critical(
                self, "Japanese bank analysis", str(error))
            return
        suggested = str(Path(self._project_root) /
                        "japanese-bank-profile.json") \
            if self._project_root else "japanese-bank-profile.json"
        dialog = JapaneseBankAnalysisDialog(
            analysis, suggested_profile=suggested, parent=self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self.statusBar().showMessage(
                "Status: Japanese bank analysis closed without project changes")
            return
        before = copy.deepcopy(self._active_japanese_state())
        after = copy.deepcopy(before)
        after["bank_analysis"] = dialog.analysis.to_state_dict()
        after["profile_path"] = dialog.profile_path
        after["needs_voice_rebuild"] = bool(
            before.get("needs_voice_rebuild") or dialog.profile_changed)
        after = je.normalize_edit_state(after)
        invalidation = "rebuild" if dialog.profile_changed else "none"
        self._apply_japanese_state_snapshot(after, invalidation)
        if before != after:
            self._push_applied_undo(
                "Japanese bank analysis",
                lambda state=before:
                self._apply_japanese_state_snapshot(state, "none"),
                lambda state=after, mode=invalidation:
                self._apply_japanese_state_snapshot(state, mode))
        report = dialog.analysis.graph.coverage
        self.statusBar().showMessage(
            "Status: Japanese bank analyzed -- %d candidates, %d unresolved%s"
            % (report.candidate_count, report.unresolved_count,
               "; voice rebuild required" if dialog.profile_changed else ""))

    def _show_recording_details(self, row):
        if self.current is None:
            return
        index = int(row.get("index", -1))
        segments = self.waveform.segments
        if not (0 <= index < len(segments) - 1):
            return
        phones = [str(segment.phone) for segment in segments]
        outer_left = phones[index - 1] if index else "*"
        outer_right = phones[index + 2] if index + 2 < len(phones) else "*"
        left, right = phones[index], phones[index + 1]
        l_class = "*"
        if left.rstrip("_") == "l":
            l_class = ("light" if fc.is_vowel_phone(right) or
                       right.rstrip("_") == "y" else "dark")
        elif right.rstrip("_") == "l":
            l_class = ("light" if fc.is_vowel_phone(outer_right) or
                       outer_right.rstrip("_") == "y"
                       else "dark")
        choices = list(row.get("choices") or [])
        safe = fc.contextual_unit_choice(
            choices, outer_left, outer_right, l_class,
            right_phone=right) or {}
        selected_name = str(row.get("manual") or row.get("actual") or "")
        selected = next((choice for choice in choices
                         if str(choice.get("left_name") or "") == selected_name),
                        row.get("choice") or {})
        safe_name = (safe.get("left_name") or left) if choices else \
            "metadata unavailable"
        safe_score = safe.get("context_score", 0) if choices else "n/a"

        def evidence_text(info, missing_direction):
            kind = str(info.get("kind") or "unclassified")
            token = str(info.get("context") or "*")
            edge_phone = str(info.get("phone") or token)
            context_class = str(info.get("class") or "other").replace("_", " ")
            source = str(info.get("source") or "")
            provenance = (
                " (recovered from the immediately adjacent ordered OTO edge)"
                if source == "adjacent_oto_edge" else
                " (from a chained adjacent OTO transition)"
                if source == "adjacent_transition" else "")
            if kind == "wildcard_unknown":
                return "unknown (no %s OTO transition)" % missing_direction
            if kind == "compound_cv":
                return "%s: compound CV, adjacent %s (%s)%s" % \
                    (token, edge_phone, context_class, provenance)
            if kind == "atomic":
                return "%s: known %s%s" % (
                    token, context_class, provenance)
            return "%s: literal alias, phonetic class unresolved%s" % (
                token, provenance)

        left_evidence = fc.choice_context_info(selected, "left")
        right_evidence = fc.choice_context_info(selected, "right")
        lines = [
            "Transition: %s-%s" % (left, right),
            "Utterance context: %s > %s-%s > %s" %
            (outer_left, left, right, outer_right),
            "Selection: %s (%s)" %
            (selected_name or "base", "manual" if row.get("manual") else "automatic"),
            "Safe automatic choice: %s (score %s)" %
            (safe_name, safe_score),
            "Automatic reason: %s" %
            (safe.get("selection_reason") or "metadata unavailable"),
            "",
            "Alias: %s" % (selected.get("alias") or "unknown"),
            "Source WAV: %s" % (selected.get("wav") or "unknown"),
            "OTO line: %s" % (selected.get("oto_line") or "unknown"),
            "Recorded context: %s > %s > %s" %
            (fc.choice_recorded_context(selected, "left"),
             row.get("pair") or "",
             fc.choice_recorded_context(selected, "right")),
            "Left OTO evidence: %s" %
            evidence_text(left_evidence, "preceding"),
            "Right OTO evidence: %s" %
            evidence_text(right_evidence, "following"),
            "Context classification uses OTO aliases only; WAV filenames "
            "are ignored.",
        ]
        if all(key in selected for key in ("start", "mid", "end")):
            lines.append("Slice: %.3f / %.3f / %.3f s" %
                         (float(selected["start"]), float(selected["mid"]),
                          float(selected["end"])))
        if selected.get("tail_clamped"):
            lines.append(
                "Context-tail guard: %.3f -> %.3f s (next phone boundary)" %
                (float(selected.get("raw_end", selected["end"])),
                 float(selected["end"])))
        if len(choices) == 1 and (
                fc.choice_recorded_context(selected, "left") != outer_left or
                fc.choice_recorded_context(selected, "right") != outer_right):
            lines.extend((
                "",
                "This is the only recording for the transition, and its "
                "outer context does not match this occurrence."))
        if row.get("pending_manual") and not row.get("manual"):
            lines.extend(("", "The saved manual take belongs to a transition "
                          "that was edited. It is retained for remapping and "
                          "will be dropped only if that transition no longer exists."))
        QtWidgets.QMessageBox.information(
            self, "Recording source", "\n".join(lines))

    def _show_unit_pitchmarks(self, row):
        choice = dict(row.get("choice") or {})
        if not choice:
            QtWidgets.QMessageBox.information(
                self, "PSOLA source pitchmarks",
                "No generated unit metadata is available for this edge.")
            return None
        backend = self._ab()
        inspect = getattr(backend, "unit_pitchmark_diagnostic", None)
        if inspect is None:
            QtWidgets.QMessageBox.information(
                self, "PSOLA source pitchmarks",
                "This renderer does not use generated UniSyn pitchmarks.")
            return None
        try:
            diagnostic = inspect(
                self._current_voicebank() or "",
                str(row.get("pair") or ""), choice)
        except (fc.BackendError, OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(
                self, "PSOLA source pitchmarks", str(error))
            return None
        dialog = SourcePitchmarkDialog(diagnostic, self)
        dialog.exec_()
        return diagnostic

    def _show_join_loudness_for_row(self, row):
        return self._show_join_loudness_diagnostic(
            focus_edge=int(row.get("index", -1)))

    def on_join_loudness_diagnostic(self):
        return self._show_join_loudness_diagnostic()

    def on_rendered_formant_diagnostic(self):
        """Analyze and display formants measured from the final waveform."""
        if self.current is None or not len(self.waveform.audio):
            QtWidgets.QMessageBox.information(
                self, "Rendered formants", "Nothing rendered yet."
            )
            return None
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        join_diagnostic = None
        join_warning = ""
        try:
            try:
                join_diagnostic = diphone_loudness.analyze_rendered_joins(
                    self.waveform.audio,
                    int(self.waveform.sr),
                    self.waveform.segments,
                    target_pitchmarks=getattr(
                        self.current, "target_pitchmarks", ()),
                    splice_records=getattr(
                        self.current, "splice_records", ()),
                    selected_units=self.current.selected_units,
                    alternatives=self._unit_alternatives(),
                )
                join_diagnostic["frame_trajectory_records"] = [
                    dict(row) for row in getattr(
                        self.current, "frame_trajectory_records", ())
                ]
            except (TypeError, ValueError) as error:
                # Whole-render tracking remains useful without exact splice
                # evidence. Preserve the reason instead of hiding the gap.
                join_warning = str(error)
            diagnostic = rendered_formant_diagnostic.analyze_rendered_formants(
                self.waveform.audio,
                int(self.waveform.sr),
                self.waveform.segments,
                join_diagnostic=join_diagnostic,
            )
            if join_warning:
                diagnostic["join_analysis_warning"] = join_warning
        except (TypeError, ValueError) as error:
            QtWidgets.QMessageBox.warning(
                self, "Rendered formants", str(error)
            )
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        dialog = RenderedFormantDialog(
            diagnostic, self.waveform.audio, parent=self
        )
        dialog.exec_()
        return diagnostic

    def _show_join_loudness_diagnostic(self, focus_edge=None):
        diagnostic = self._analyze_rendered_join_discontinuities()
        if diagnostic is None:
            return None
        state = (
            self.sentences[self._active_sentence_index]
            if 0 <= self._active_sentence_index < len(self.sentences) else
            {})
        legacy_active = bool(
            dict(state.get("fault_mode") or self._fault_mode()).get(
                "legacy_joins"))
        dialog = JoinDiscontinuityDialog(
            diagnostic, self.waveform.audio,
            focus_edge=focus_edge,
            requested_join_settings=state.get("join_settings"),
            effective_join_settings=getattr(
                self.current, "join_settings", None),
            editable=(self._engine() == "festival_wsl"),
            legacy_active=legacy_active,
            parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self._set_join_settings(dialog.requested_join_settings())
        return diagnostic

    @staticmethod
    def _project_join_settings(settings):
        normalized = fc.FestivalWSLBackend.normalize_join_settings(settings)
        return (
            {} if normalized ==
            fc.FestivalWSLBackend.normalize_join_settings(None)
            else normalized)

    @classmethod
    def _faults_with_join_settings(cls, faults, settings):
        """Attach private render controls without polluting Fault Mode data."""
        render_faults = copy.deepcopy(dict(faults or {}))
        render_faults["_join_settings"] = (
            fc.FestivalWSLBackend.normalize_join_settings(settings))
        return render_faults

    def _apply_join_settings(self, settings):
        if not (0 <= self._active_sentence_index < len(self.sentences)):
            return
        state = self.sentences[self._active_sentence_index]
        state["join_settings"] = self._project_join_settings(settings)
        self.waveform.set_requested_join_settings(
            state["join_settings"])
        action = self._mark_active_pending(
            "rerender", "UniSyn join settings changed")
        self._capture_active_sentence()
        self.statusBar().showMessage(
            "Status: UniSyn crossover/window settings changed -- %s "
            "applies them; "
            "phone timing, F0, and recording choices are preserved" %
            ("Generate Audio" if action == "generate" else
             "Re-render Phonemes"))

    def _join_edit_snapshot(self, state):
        return {
            "join_settings": self._project_join_settings(
                state.get("join_settings")),
            "needs_generate": bool(state.get("needs_generate")),
            "needs_rerender": bool(state.get("needs_rerender")),
            "rendered": bool(state.get("rendered")),
            "pending_reason": str(state.get("pending_reason") or ""),
        }

    def _restore_join_edit_snapshot(self, state, snapshot):
        state["join_settings"] = self._project_join_settings(
            snapshot.get("join_settings"))
        for key in ("needs_generate", "needs_rerender", "rendered"):
            state[key] = bool(snapshot.get(key))
        state["pending_reason"] = str(
            snapshot.get("pending_reason") or "")
        if (0 <= self._active_sentence_index < len(self.sentences) and
                self.sentences[self._active_sentence_index] is state):
            self.waveform.set_requested_join_settings(
                state["join_settings"])
            self._refresh_pending_ui()
            self._capture_active_sentence()

    def _set_join_crossover_override(
            self, unit_index, left_ms, right_ms):
        """Store one occurrence's editable crossover without touching units."""
        if not (0 <= self._active_sentence_index < len(self.sentences)):
            return
        state = self.sentences[self._active_sentence_index]
        before = self._join_edit_snapshot(state)
        before_settings = before["join_settings"]
        after = fc.FestivalWSLBackend.normalize_join_settings(
            before_settings)
        unit_index = int(unit_index)
        left_ms = max(0.0, min(100.0, float(left_ms)))
        right_ms = max(0.0, min(100.0 - left_ms, float(right_ms)))
        overrides = dict(after.get("crossover_overrides") or {})
        default_side = float(after.get("crossover_ms") or 0.0) * 0.5
        if (abs(left_ms - default_side) <= 0.001 and
                abs(right_ms - default_side) <= 0.001):
            overrides.pop(str(unit_index), None)
        else:
            overrides[str(unit_index)] = {
                "left_ms": round(left_ms, 3),
                "right_ms": round(right_ms, 3),
            }
        after["crossover_overrides"] = overrides
        after = self._project_join_settings(after)
        if before_settings == after:
            return
        self._apply_join_settings(after)
        after_snapshot = self._join_edit_snapshot(state)
        self._push_applied_undo(
            "join crossover",
            lambda target=state, value=before:
                self._restore_join_edit_snapshot(target, value),
            lambda target=state, value=after_snapshot:
                self._restore_join_edit_snapshot(target, value))
        self.statusBar().showMessage(
            "Status: join %d requested crossover %.1f ms left / %.1f ms "
            "right -- Re-render applies the renderer's pitch-synchronous "
            "bounds" % (unit_index, left_ms, right_ms))

    def _set_join_settings(self, settings):
        if not (0 <= self._active_sentence_index < len(self.sentences)):
            return
        state = self.sentences[self._active_sentence_index]
        before = self._join_edit_snapshot(state)
        before_settings = before["join_settings"]
        after = self._project_join_settings(settings)
        if before_settings == after:
            return
        self._apply_join_settings(after)
        after_snapshot = self._join_edit_snapshot(state)
        self._push_applied_undo(
            "UniSyn join settings",
            lambda target=state, value=before:
                self._restore_join_edit_snapshot(target, value),
            lambda target=state, value=after_snapshot:
                self._restore_join_edit_snapshot(target, value))

    def _analyze_rendered_join_discontinuities(self):
        if self.current is None or not len(self.waveform.audio):
            QtWidgets.QMessageBox.information(
                self, "Rendered join discontinuities",
                "Nothing rendered yet.")
            return None
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            diagnostic = diphone_loudness.analyze_rendered_joins(
                self.waveform.audio,
                int(self.waveform.sr),
                self.waveform.segments,
                target_pitchmarks=getattr(
                    self.current, "target_pitchmarks", ()),
                splice_records=getattr(self.current, "splice_records", ()),
                selected_units=self.current.selected_units,
                alternatives=self._unit_alternatives(),
            )
            diagnostic["frame_trajectory_records"] = [
                dict(row) for row in getattr(
                    self.current, "frame_trajectory_records", ())
            ]
            return diagnostic
        except (TypeError, ValueError) as error:
            QtWidgets.QMessageBox.warning(
                self, "Rendered join discontinuities", str(error))
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_export_broadband_impulse_join_audit(self):
        """Write the named broadband-click diagnostic for the current render."""
        diagnostic = self._analyze_rendered_join_discontinuities()
        if diagnostic is None:
            return None
        sentence_number = max(1, int(self._active_sentence_index) + 1)
        folder = (FESTVOX_TOOL_DIR / "diagnostic_images" /
                  "broadband_impulse_join_audit")
        destination = folder / (
            "sentence_%03d_broadband_impulse_join_audit.png" %
            sentence_number)
        try:
            join_spectrogram.render_join_spectrogram(
                self.waveform.audio,
                int(self.waveform.sr),
                destination,
                diagnostic=diagnostic,
                title="Broadband Impulse Join Audit - sentence %d" %
                sentence_number,
            )
        except (OSError, TypeError, ValueError) as error:
            QtWidgets.QMessageBox.warning(
                self, "Broadband Impulse Join Audit", str(error))
            return None
        self.statusBar().showMessage(
            "Status: Broadband Impulse Join Audit saved to " +
            str(destination))
        QtWidgets.QMessageBox.information(
            self, "Broadband Impulse Join Audit",
            "Diagnostic image saved to:\n" + str(destination))
        return destination

    def _unit_alternatives(self):
        vb = self._current_voicebank()
        if not vb:
            return {}
        backend = self._ab()
        refresh = getattr(backend, "refresh_voice_metadata", None)
        token = refresh(vb) if refresh is not None else None
        key = (self._engine(), vb, repr(token))
        for stale in [item for item in self._variant_cache
                      if item[:2] == key[:2] and item != key]:
            self._variant_cache.pop(stale, None)
        if key not in self._variant_cache:
            try:
                self._variant_cache[key] = backend.unit_alternatives(vb)
            except (fc.BackendError, AttributeError):
                self._variant_cache[key] = {}
            try:
                limit = max(1, int(self.cfg.get(
                    "voice_variant_cache_limit", 16)))
            except (TypeError, ValueError):
                limit = 16
            while len(self._variant_cache) > limit:
                self._variant_cache.popitem(last=False)
        else:
            self._variant_cache.move_to_end(key)
        return self._variant_cache[key]

    def _resolve_source_selection_phones(
        self, display, *, backend=None, voice=None,
        allow_unverified_inventory=False,
    ):
        """Resolve one canonical sequence against a selected voice."""
        display = [str(phone) for phone in (display or [])]
        if not display:
            return []
        backend = backend or self._ab()
        voice = str(voice or self._current_voicebank() or "")
        metadata = {}
        try:
            reader = getattr(backend, "voice_metadata", None)
            if reader is not None and voice:
                metadata = dict(reader(voice) or {})
            else:
                db_reader = getattr(backend, "db", None)
                if db_reader is not None and voice:
                    metadata = dict(
                        getattr(db_reader(voice), "metadata", {}) or {})
        except (fc.BackendError, OSError, ValueError, TypeError):
            metadata = {}
        inventory = {}
        if isinstance(metadata.get("index"), dict):
            inventory = dict(metadata.get("index") or {})
        if not inventory:
            try:
                reader = getattr(backend, "unit_alternatives", None)
                if reader is not None and voice:
                    inventory = dict(reader(voice) or {})
            except (fc.BackendError, OSError, ValueError, TypeError):
                inventory = {}
        resolution = fc.resolve_voice_special_phones(
            display,
            metadata,
            voicebank=voice,
            available_diphones=(
                inventory.keys() if inventory else None
            ),
            allow_unverified_inventory=allow_unverified_inventory,
        )
        return list(resolution.render_phones)

    def _source_selection_phones(self):
        """Return source phones aligned one-to-one with visible segments."""
        display = [str(segment.phone) for segment in self.waveform.segments]
        if not display:
            return []
        if self.current is not None:
            canonical = [
                str(segment.phone) for segment in self.current.segments
            ]
            rendered = [
                str(phone)
                for phone in (getattr(self.current, "render_phones", ()) or ())
            ]
            if (
                len(canonical) == len(display) == len(rendered)
                and canonical == display
            ):
                return rendered
        try:
            return self._resolve_source_selection_phones(display)
        except fc.BackendError:
            # Keep the source identity structural in the editor even when an
            # old generated bank lacks its required C-C hold. Re-render will
            # issue the actionable rebuild error; the UI must never fall back
            # to offering the coincidentally named literal cl OTO.
            try:
                return self._resolve_source_selection_phones(
                    display,
                    allow_unverified_inventory=True,
                )
            except fc.BackendError:
                return display

    def _refresh_voice_metadata(self, backend=None, engine=None, voice=None):
        """Refresh only metadata whose source fingerprint has changed."""
        backend = backend if backend is not None else self._ab()
        engine = str(engine if engine is not None else self._engine())
        voice = str(voice if voice is not None else
                    (self._current_voicebank() or ""))
        refresh = getattr(backend, "refresh_voice_metadata", None)
        token = refresh(voice) if refresh is not None and voice else None
        current = (engine, voice, repr(token))
        for stale in [item for item in self._variant_cache
                      if item[:2] == current[:2] and item != current]:
            self._variant_cache.pop(stale, None)

    def _add_unit_variant_menu(self, menu, selected_idx):
        if self.current is None:
            return
        inventory = self._unit_alternatives()
        segments = self.waveform.segments
        source_phones = self._source_selection_phones()
        added = False
        for label, pair_idx in (("Incoming recording", selected_idx - 1),
                                ("Outgoing recording", selected_idx)):
            if pair_idx < 0 or pair_idx + 1 >= len(segments):
                continue
            p1, p2 = source_phones[pair_idx:pair_idx + 2]
            pair = "%s-%s" % (p1, p2)
            choices = list(inventory.get(pair) or [])
            if len(choices) < 2:
                continue
            if not added:
                menu.addSeparator()
                added = True
            submenu = menu.addMenu(label)
            current = self.current.unit_overrides.get(pair_idx)
            auto = submenu.addAction("Auto (context match)")
            auto.setCheckable(True)
            auto.setChecked(not current)
            auto.triggered.connect(
                lambda _on=False, i=pair_idx: self._set_unit_override(i, None))
            submenu.addSeparator()
            for number, choice in enumerate(choices):
                key = str(choice.get("left_name") or p1)
                left = fc.choice_recorded_context(choice, "left")
                right = fc.choice_recorded_context(choice, "right")
                take = str(choice.get("id") or (number + 1))
                action = submenu.addAction(
                    "%s: %s > %s > %s" % (take, left, pair, right))
                action.setCheckable(True)
                action.setChecked(current == key)
                action.setToolTip(str(choice.get("wav") or ""))
                action.triggered.connect(
                    lambda _on=False, i=pair_idx, k=key:
                    self._set_unit_override(i, k))

    def _set_unit_override(self, pair_idx, unit_name):
        if self.current is None:
            return
        pair_idx = int(pair_idx)
        old = self.current.unit_overrides.get(pair_idx)
        self._apply_unit_override(pair_idx, unit_name)
        if old != unit_name:
            self._push_applied_undo(
                "recording choice",
                lambda index=pair_idx, value=old:
                self._apply_unit_override(index, value),
                lambda index=pair_idx, value=unit_name:
                self._apply_unit_override(index, value))

    def _apply_unit_override(self, pair_idx, unit_name):
        if self.current is None:
            return
        if unit_name:
            self.current.unit_overrides[int(pair_idx)] = str(unit_name)
        else:
            self.current.unit_overrides.pop(int(pair_idx), None)
        self.recordings.set_data(
            self.waveform.segments, self._unit_alternatives(),
            self.current.selected_units, self.current.unit_overrides,
            self._source_selection_phones())
        self._mark_active_pending("rerender", "Recording choice changed")
        self.statusBar().showMessage(
            "Status: recorded unit %s -- Re-render applies this occurrence only"
            % ("selected" if unit_name else "set to automatic context matching"))

    def _show_synthesis(self, syn: fc.Synthesis, *, display=None,
                        timing_factors=(), focus_timeline=True,
                        preserve_view=False):
        saved_view = None
        saved_playhead = None
        if preserve_view:
            saved_view = tuple(
                float(value) for value in
                self.waveform.plot.getViewBox().viewRange()[0])
            saved_playhead = self.waveform.playhead_time()
        self.current = syn
        self.waveform.set_synthesis(display if display is not None else syn)
        if 0 <= self._active_sentence_index < len(self.sentences):
            self.waveform.set_requested_join_settings(
                self.sentences[self._active_sentence_index].get(
                    "join_settings"))
        self._refresh_join_overlay_controls()
        for position, factor in enumerate(timing_factors or ()):
            if (position < len(self.waveform.base_durs) and
                    float(factor) > 1e-4):
                self.waveform.base_durs[position] = max(
                    1e-4, self.waveform.segments[position].dur /
                    float(factor))
        dur = self.waveform.duration() or 1.0
        self._sync_timing_track(reset=True)
        if saved_playhead is not None:
            self.waveform.set_playhead(saved_playhead)
        if saved_view is not None:
            self.waveform.plot.getViewBox().setXRange(
                saved_view[0], saved_view[1], padding=0)
        if 0 <= self._active_sentence_index < len(self.sentences):
            self._editor_sentence_state = self.sentences[
                self._active_sentence_index]
        tag = "festival/WSL" if self._engine() == "festival_wsl" else "diphone"
        msg = "Ready [%s] -- %d phones" % (tag, len(syn.phones))
        if syn.diphones:
            msg += ", %d diphones" % len(syn.diphones)
        alternate_count = sum("__u" in str(v)
                              for v in syn.selected_units.values())
        if alternate_count:
            msg += ", %d alternate take%s" % (
                alternate_count, "" if alternate_count == 1 else "s")
        if syn.output_bit_depth:
            msg += ", %d-bit" % syn.output_bit_depth
        japanese_prosody = dict(
            getattr(syn, "japanese_prosody", None) or {})
        if japanese_prosody:
            msg += ", JA %s [%s | %s]" % (
                japanese_prosody.get("duration_model") or "unknown timing",
                japanese_prosody.get("duration_model_id") or
                "unknown timing model",
                japanese_prosody.get("pitch_model_id") or
                "unknown pitch model",
            )
        msg += ", %.2fs [%s @ x%.2f]" % (dur, syn.voicebank,
                                         self._speed_factor())
        if syn.skipped:
            msg += "  |  %d MISSING: %s" % (len(syn.skipped),
                                            ", ".join(syn.skipped[:6]))
            if len(syn.skipped) > 6:
                msg += ", ..."
        if syn.warning and not syn.skipped:
            msg += "  |  " + syn.warning[:120]
        self.statusBar().showMessage("Status: " + msg)
        self._mark_gain_applied(syn)
        self._refresh_pending_ui()
        if 0 <= self._active_sentence_index < len(self.sentences):
            self.japanese_editor.set_state(
                self.sentences[self._active_sentence_index].get(
                    "japanese_state"))
            self.asaxi_editor.set_state(
                self.sentences[self._active_sentence_index].get(
                    "asaxi_state"))
        if focus_timeline:
            self.waveform.timeline.setFocus(Qt.OtherFocusReason)

    def _display_background_render_if_active(self, state, *,
                                             preserve_view=False):
        if (state is None or
                not (0 <= self._active_sentence_index < len(self.sentences)) or
                self.sentences[self._active_sentence_index] is not state or
                (hasattr(self, "mode_tabs") and
                 self.mode_tabs.currentIndex() != 0)):
            return
        syn = state.get("synthesis")
        if syn is None:
            return
        focus = QtWidgets.QApplication.focusWidget()
        cursor = self.text.cursorPosition() if focus is self.text else None
        display = copy.copy(syn)
        display.samples = np.asarray(
            state.get("preview_audio") if
            state.get("preview_audio") is not None else syn.samples,
            np.float32)
        display.sr = int(state.get("preview_sr") or syn.sr)
        if state.get("editor_segments"):
            display.segments = state["editor_segments"]
        self._show_synthesis(
            syn, display=display,
            timing_factors=state.get("timing_factors") or [],
            focus_timeline=False, preserve_view=preserve_view)
        self._editor_sentence_state = state
        if focus is not None:
            try:
                focus.setFocus(Qt.OtherFocusReason)
                if cursor is not None:
                    self.text.setCursorPosition(min(
                        int(cursor), len(self.text.text())))
            except RuntimeError:
                pass

    # -- actions ------------------------------------------------------------
    def on_generate(self):
        return self._generate_for_sentence_mode(confirm_replace=True)

    def _generate_for_sentence_mode(self, confirm_replace=False,
                                    show_error=True, target_state=None,
                                    display_result=True):
        self._begin_generation_progress("Generating current sentence...")
        try:
            return self._generate_current(
                confirm_replace=confirm_replace, show_error=show_error,
                target_state=target_state, display_result=display_result)
        finally:
            self._end_generation_progress()

    def _phrase_dictionary(self, phrase, fallback, backend=None):
        key = str(phrase.get("dictionary") or "")
        path = (self.cfg.get("voice_dictionaries") or {}).get(key)
        if not path:
            return fallback
        try:
            backend = backend if backend is not None else self._ab()
            return backend.read_installed_dictionary(path)
        except (fc.BackendError, OSError, ValueError, AttributeError):
            return fallback

    def _japanese_runtime(self, voicebank):
        runtime = self.fest.japanese_runtime_metadata(voicebank)
        if runtime:
            return runtime
        raise fc.BackendError(
            "The selected Festival voice is not an isolated generated "
            "Japanese voice. Build or register a Phase 3 Japanese voice "
            "whose runtime metadata has language='ja' and a *_ja entry point.")

    def _refresh_japanese_runtime_controls(self, runtime=None):
        if not hasattr(self, "japanese_editor"):
            return
        if runtime is None:
            runtime = {}
            if (self._engine() == "festival_wsl" and
                    self._current_lang_code() in ("ja", "jp")):
                try:
                    runtime = self.fest.japanese_runtime_metadata(
                        self._current_voicebank() or "")
                except (fc.BackendError, OSError, ValueError, AttributeError):
                    runtime = {}
        self.japanese_editor.set_runtime_metadata(runtime)

    @staticmethod
    def _asaxi_edit_overlays(state, text):
        asaxi_state = asaxi_editing.normalize_edit_state(
            (state or {}).get("asaxi_state"))
        same_text = (
            str(asaxi_state.get("source_text") or "").strip().lower()
            == str(text or "").strip().lower()
        )
        if not same_text:
            return {"tone": {}, "pitch": {}, "voicing": {}}
        return {
            "tone": dict(asaxi_state.get("mora_tone_overrides") or {}),
            "pitch": dict(
                asaxi_state.get("mora_pitch_offsets_cents") or {}),
            "voicing": dict(
                asaxi_state.get("mora_voicing_overrides") or {}),
        }

    def _reconcile_asaxi_result(self, state, syn, text,
                                refresh_controls=True):
        if state is None or str(getattr(syn, "lang", "")) != "asaxi":
            return
        metadata = dict(getattr(syn, "asaxi_prosody", None) or {})
        state["asaxi_state"] = asaxi_editing.reconcile_plan(
            state.get("asaxi_state"), text, metadata)
        if refresh_controls and hasattr(self, "asaxi_editor"):
            self.asaxi_editor.set_state(state["asaxi_state"])

    def _prepare_asaxi_rerender(self, state, text, segments, pitch, fall,
                                user_dict=None):
        path = self.fest.cfg.get("asaxi_synthesis_dictionary") \
            or asaxi_prosody.DEFAULT_DICTIONARY_PATH
        try:
            dictionary = asaxi_prosody.load_dictionary(path)
            plans = tuple(
                asaxi_prosody.analyze_utterance(
                    chunk,
                    dictionary,
                    phone_overrides=dict(user_dict or {}),
                )
                for chunk, _mark in fc.text_phrase_chunks(text)
            )
            overlays = self._asaxi_edit_overlays(state, text)
            realization, diagnostics = \
                asaxi_prosody.realize_pitch_for_plans(
                plans,
                segments,
                base_pitch_hz=float(pitch or 160.0),
                fall_percent=float(fall or 0.0),
                mora_tone_overrides=overlays["tone"],
                mora_pitch_offsets_cents=overlays["pitch"],
            )
            metadata = self.fest._asaxi_metadata(
                plans,
                diagnostics,
                segments,
                pitch_realization=realization,
            )
        except (OSError, TypeError, ValueError) as error:
            raise fc.BackendError(
                "Asaxi re-render planning failed:\n%s" % error
            ) from error
        return list(realization.targets), metadata

    def _prepare_japanese_plan(self, state, text, voicebank, speed, pitch,
                               analyze=True, refresh_controls=True):
        japanese_state = je.normalize_edit_state(
            state.get("japanese_state"))
        raw = japanese_state.get("utterance")
        try:
            utterance = je.utterance_from_dict(raw) if raw else None
        except (TypeError, ValueError, KeyError):
            utterance = None
        if (analyze or utterance is None or
                utterance.source_text != str(text)):
            try:
                utterance = japanese_frontend.analyze_japanese(
                    str(text), mode=japanese_state.get(
                        "frontend_mode") or "auto")
            except Exception as error:
                diagnostic = getattr(error, "diagnostic", None)
                message = getattr(diagnostic, "message", None) or str(error)
                raise fc.BackendError(
                    "Japanese text analysis failed:\n" + message) from error
            japanese_state = je.reconcile_analyzed_utterance(
                japanese_state, utterance)
        runtime = self._japanese_runtime(voicebank)
        if refresh_controls:
            self._refresh_japanese_runtime_controls(runtime)
        configured_pauses = fc.normalize_phrase_pauses_ms(
            self.cfg.get("phrase_pauses_ms"))
        # The shared config retains the legacy English defaults.  For Japanese,
        # an untouched default means "use the fitted voice profile"; any
        # changed value remains an explicit project-wide user override.
        japanese_pauses = (
            None if configured_pauses == fc.DEFAULT_PHRASE_PAUSES_MS
            else configured_pauses
        )
        try:
            plan = je.create_edited_plan(
                utterance, japanese_state,
                runtime_metadata=runtime,
                base_pitch_hz=float(pitch), speed=float(speed),
                phrase_pauses_ms=japanese_pauses,
                duration_model=self.cfg.get(
                    "japanese_duration_model", "contextual"))
        except (TypeError, ValueError) as error:
            raise fc.BackendError(
                "Japanese synthesis planning failed:\n%s" % error) from error
        plan_state = plan.to_dict()
        plan_state["mora_voicing_predictions"] = [
            item.to_dict() for item in
            japanese_devoicing.predict_mora_voicing(
                plan, japanese_state.get("mora_voicing_overrides") or {})
        ]
        try:
            plan_state["source_contributions"] = \
                japanese_assembly.create_source_contribution_plan(
                    plan, runtime).to_dict()
        except (TypeError, ValueError, KeyError) as error:
            plan_state["source_contributions"] = {
                "kind": "japanese_source_contribution_plan",
                "contributions": [],
                "diagnostics": [{
                    "code": "source_contribution_plan_unavailable",
                    "severity": "warning",
                    "message": str(error),
                }],
            }
        japanese_state["last_plan"] = plan_state
        state["japanese_state"] = je.normalize_edit_state(japanese_state)
        self.japanese_editor.set_state(state["japanese_state"])
        self._sync_japanese_timeline()
        return plan

    @staticmethod
    def _japanese_fault_plan(plan, faults):
        entries = list(plan.segment_durations)
        original = list(entries)
        if dict(faults or {}).get("disable_phone_timing"):
            entries = fc.equalize_phone_durations(
                entries, phone_dur=0.10 / max(0.25, float(plan.speed)))
        if dict(faults or {}).get("single_pause"):
            entries = fc.collapse_pause_runs(entries)
        targets = list(plan.pitch_targets)
        overrides = dict(plan.unit_overrides)
        if entries != original:
            old_segments = fc.segments_from_durations(original)
            targets = fc.remap_targets(
                targets, old_segments,
                [duration for _phone, duration in entries])
            overrides = fc.remap_unit_overrides(
                [phone for phone, _duration in original],
                [phone for phone, _duration in entries], overrides)
        return entries, targets, overrides

    @staticmethod
    def _japanese_prosody_metadata(plan):
        frequencies = [float(target.hz) for target in plan.f0_targets
                       if float(target.hz) > 0.0]
        phrase_indices = {int(target.phrase_index)
                          for target in plan.f0_targets}
        later_shape_targets = sum(
            any(str(name).startswith("later_phrase_")
                and abs(float(value)) > 1.0e-10
                for name, value in target.components_semitones.items())
            for target in plan.f0_targets
        )
        return {
            "duration_model": plan.duration_model,
            "duration_model_id": plan.duration_model_id,
            "pitch_model_id": plan.pitch_model_id,
            "contour_model": plan.contour_model,
            "frontend_name": plan.frontend_name,
            "plan_schema_version": plan.schema_version,
            "segment_count": len(plan.segments),
            "mora_count": len(plan.mora_timings),
            "phrase_count": len(phrase_indices),
            "total_duration_seconds": round(sum(
                float(segment.duration) for segment in plan.segments), 6),
            "f0_target_count": len(frequencies),
            "f0_min_hz": min(frequencies) if frequencies else None,
            "f0_max_hz": max(frequencies) if frequencies else None,
            "f0_span_semitones": (
                12.0 * math.log2(max(frequencies) / min(frequencies))
                if frequencies else None
            ),
            "later_phrase_shape_target_count": later_shape_targets,
            "phrase_position_model": "mean_centered_shape",
            "cumulative_register_drift_enabled": False,
        }

    def _set_language_render_features(
            self, syn, japanese_plan=None, voicing_override=None,
            mora_voicing_overrides=None, asaxi_metadata=None,
            asaxi_voicing_overrides=None):
        """Attach linguistic directives without changing rendered samples.

        Language frontends may produce different timing, F0, and voicing
        directives, but their audio all enters the same post-render stages.
        Keeping the directives on the render also lets a later timing/pitch
        pass transfer them before any waveform transform is applied.
        """
        features = dict(getattr(syn, "_language_render_features", {}) or {})
        if japanese_plan is not None:
            features["japanese_plan"] = japanese_plan
            features["mora_voicing_overrides"] = dict(
                mora_voicing_overrides or {})
            if isinstance(syn, fc.Synthesis):
                syn.japanese_prosody = self._japanese_prosody_metadata(
                    japanese_plan)
        if asaxi_metadata is not None:
            features["asaxi_metadata"] = copy.deepcopy(
                dict(asaxi_metadata or {}))
            features["asaxi_voicing_overrides"] = dict(
                asaxi_voicing_overrides or {})
            if isinstance(syn, fc.Synthesis):
                syn.asaxi_prosody = copy.deepcopy(
                    dict(asaxi_metadata or {}))
        if voicing_override is not None:
            features["voicing_override"] = [
                (float(time), float(value))
                for time, value in voicing_override
            ]
        syn._language_render_features = features
        return syn

    @staticmethod
    def _transfer_language_render_features(source, target):
        features = dict(getattr(
            source, "_language_render_features", {}) or {})
        if features:
            target._language_render_features = features
        target.japanese_prosody = dict(getattr(
            source, "japanese_prosody", None) or {})
        target.asaxi_prosody = copy.deepcopy(dict(getattr(
            source, "asaxi_prosody", None) or {}))
        return target

    def _apply_shared_voicing_stage(self, syn):
        """Apply the common voicing capability after the final WSL render."""
        if getattr(syn, "_shared_voicing_stage_applied", False):
            return syn
        features = dict(getattr(
            syn, "_language_render_features", {}) or {})
        plan = features.get("japanese_plan")
        override = features.get("voicing_override")
        if plan is not None:
            japanese_devoicing.apply_vowel_realizations(
                syn,
                plan,
                mode=self.cfg.get(
                    "japanese_vowel_devoicing", "contextual"),
                renderer=self.cfg.get(
                    "japanese_devoicing_renderer", "auto"),
                voicing_override=override,
                mora_voicing_overrides=features.get(
                    "mora_voicing_overrides") or {},
            )
        elif (
                features.get("asaxi_metadata") is not None or
                (
                    str(getattr(syn, "lang", "")) == "asaxi" and
                    bool(getattr(syn, "asaxi_prosody", None))
                )
        ):
            metadata = (
                features.get("asaxi_metadata")
                if features.get("asaxi_metadata") is not None
                else getattr(syn, "asaxi_prosody", {}) or {}
            )
            asaxi_phonation.apply_phonation(
                syn,
                metadata,
                voicing_overrides=features.get(
                    "asaxi_voicing_overrides") or {},
                continuous_voicing_override=override,
            )
        elif override:
            japanese_devoicing.apply_voicing_override(syn, override)
        elif not getattr(syn, "generated_voicing_targets", None):
            japanese_devoicing.initialize_voicing_metadata(syn)
        syn._shared_voicing_stage_applied = True
        stages = list(getattr(syn, "_render_pipeline_stages", ()) or ())
        if "voicing" not in stages:
            stages.append("voicing")
        syn._render_pipeline_stages = stages
        return syn

    def _render_japanese_plan(self, plan, voicebank, text, pitch, fall,
                              monotone, faults,
                              mora_voicing_overrides=None):
        entries, baseline, overrides = self._japanese_fault_plan(plan, faults)
        blocks = fc.phrase_blocks(
            fc.segments_from_durations(entries), text)
        use_intonation = fc.intonation_overlay_required(blocks, float(fall))
        targets = (fc.overlay_intonation_targets(
            baseline, blocks, float(pitch), float(fall))
            if use_intonation else list(baseline))
        syn = self._call_synthesis_backend(
            self.fest.synth_phones,
            [phone for phone, _duration in entries], voicebank, 1.0,
            text=text, lang="ja", seg_durs=entries,
            pitch=pitch, fall=fall, monotone=monotone,
            fault_mode=faults, pitch_targets=targets,
            ground_truth_targets=baseline,
            intonation_blocks=blocks if use_intonation else None,
            pitch_mode="intonation" if use_intonation else "",
            unit_overrides=overrides)
        # Real backends carry the linguistic plan into the common post-render
        # pipeline. Metadata-only host proxies have no waveform to transform.
        if isinstance(syn, fc.Synthesis):
            self._set_language_render_features(
                syn, japanese_plan=plan,
                mora_voicing_overrides=mora_voicing_overrides)
        # Structural Japanese F0 remains the dashed ground truth. Punctuation
        # is the ordinary Intonation-block layer, and a later continuous Pitch
        # edit still replaces both through the existing re-render path.
        if plan.diagnostics:
            note = "; ".join(item.message for item in plan.diagnostics[:2])
            syn.warning = (syn.warning + "; " + note) if syn.warning else note
        return syn

    def _synthesize_phrase_request(self, text, input_mode, code, voicebank,
                                   speed, pitch, fall, monotone, faults,
                                   user_dict, *, backend=None,
                                   festival=None, refresh_controls=True,
                                   asaxi_tone_overrides=None,
                                   asaxi_pitch_offsets_cents=None):
        fest = (self._engine() == "festival_wsl" if festival is None else
                bool(festival))
        backend = backend if backend is not None else self._ab()
        if input_mode == "phones":
            phones = [phone for phone in str(text).split() if phone]
            if fest:
                return self._call_synthesis_backend(
                    self.fest.synth_phones,
                    phones, voicebank, speed, text=text, lang=code,
                    seg_durs=fc.class_seg_durs(
                        phones, speed,
                        equal=faults.get("disable_phone_timing")),
                    pitch=pitch, fall=fall, monotone=monotone,
                    fault_mode=faults)
            return self._call_synthesis_backend(
                backend.synth_phones,
                phones, voicebank, speed, text=text, lang=code)
        if fest and code in ("ja", "jp"):
            temporary = {"japanese_state": je.new_edit_state()}
            plan = self._prepare_japanese_plan(
                temporary, text, voicebank, speed, pitch, analyze=True,
                refresh_controls=refresh_controls)
            return self._render_japanese_plan(
                plan, voicebank, text, pitch, fall, monotone, faults)
        if fest:
            expanded, extra, _dropped = self._prepare_inline_phones(
                text, code, voicebank, backend=backend)
            combined = dict(user_dict or {})
            combined.update(extra)
            return self._call_synthesis_backend(
                self.fest.synth,
                expanded, code, voicebank, speed, pitch=pitch, fall=fall,
                monotone=monotone, user_dict=combined or None,
                fault_mode=faults,
                asaxi_tone_overrides=(
                    asaxi_tone_overrides if code == "asaxi" else None
                ),
                asaxi_pitch_offsets_cents=(
                    asaxi_pitch_offsets_cents if code == "asaxi" else None
                ))
        return self._call_synthesis_backend(
            backend.synth,
            text, code, voicebank, speed, user_dict=user_dict,
            fault_mode=faults)

    def _apply_phrase_profile(self, syn, phrase, voicebank, speed, pitch,
                              fall, monotone, faults, *, backend=None,
                              lang_code=None):
        backend = backend if backend is not None else self._ab()
        lang_code = (self._current_lang_code() if lang_code is None else
                     str(lang_code))
        factors = [float(value) for value in
                   (phrase.get("timing_factors") or [])]
        local_pitch = [(float(time), float(value)) for time, value in
                       (phrase.get("pitch_override") or [])]
        local_voicing = [(float(time), float(value)) for time, value in
                         (phrase.get("voicing_override") or [])]
        voiced_start = next((segment.start for segment in syn.segments
                             if segment.phone != "pau"), 0.0)
        if not factors and not local_pitch:
            if local_voicing:
                self._set_language_render_features(
                    syn, voicing_override=[
                        (voiced_start + time, value)
                        for time, value in local_voicing])
            return syn
        entries = []
        cursor = 0
        for segment in syn.segments:
            factor = 1.0
            if segment.phone != "pau" and cursor < len(factors):
                factor = max(.125, factors[cursor])
                cursor += 1
            entries.append((segment.phone, max(.01, segment.dur * factor)))
        pitch_targets = [(voiced_start + time, value)
                         for time, value in local_pitch]
        rendered = self._call_synthesis_backend(
            backend.synth_phones,
            [phone for phone, _duration in entries], voicebank, speed=1.0,
            text=phrase.get("text") or "", lang=lang_code,
            seg_durs=entries, old_segments=syn.segments,
            prev_targets=syn.targets, pitch=pitch, fall=fall,
            monotone=monotone, fault_mode=faults,
            pitch_targets=pitch_targets or None,
            ground_truth_targets=syn.generated_targets or syn.targets,
            pitch_mode="curve" if pitch_targets else "")
        self._transfer_language_render_features(syn, rendered)
        if local_voicing:
            voicing_targets = [(voiced_start + time, value)
                               for time, value in local_voicing]
            self._set_language_render_features(
                rendered, voicing_override=voicing_targets)
        return rendered

    def _generate_phrase_sequence(self, state, code, default_voice, speed,
                                  pitch, fall, monotone, faults, user_dict,
                                  *, backend=None, festival=None,
                                  refresh_controls=True):
        backend = backend if backend is not None else self._ab()
        festival = (self._engine() == "festival_wsl"
                    if festival is None else bool(festival))
        phrases = self._ensure_phrase_states(state)
        rendered = []
        applied_bits = []
        asaxi_overlays = (
            self._asaxi_edit_overlays(state, state.get("text") or "")
            if code == "asaxi" else
            {"tone": {}, "pitch": {}, "voicing": {}}
        )
        asaxi_mora_cursor = 0
        for phrase in phrases:
            voice = str(phrase.get("speaker") or default_voice)
            phrase_faults = dict(faults)
            local_faults = dict(phrase.get("fault_mode") or {})
            for key, value in local_faults.items():
                if key == "bit_depth":
                    if int(value or 0) > 0:
                        phrase_faults[key] = int(value)
                elif bool(value):
                    phrase_faults[key] = True
            dictionary = self._phrase_dictionary(
                phrase, user_dict, backend=backend)
            local_asaxi_offsets = None
            local_asaxi_tones = None
            if code == "asaxi":
                local_asaxi_offsets = {
                    int(index) - asaxi_mora_cursor: value
                    for index, value in asaxi_overlays["pitch"].items()
                    if int(index) >= asaxi_mora_cursor
                }
                local_asaxi_tones = {
                    int(index) - asaxi_mora_cursor: value
                    for index, value in asaxi_overlays["tone"].items()
                    if int(index) >= asaxi_mora_cursor
                }
            syn = self._synthesize_phrase_request(
                phrase.get("text") or "", state.get("input_mode") or "text",
                code, voice, speed, pitch, fall, monotone,
                phrase_faults, dictionary, backend=backend,
                festival=festival, refresh_controls=refresh_controls,
                asaxi_tone_overrides=local_asaxi_tones,
                asaxi_pitch_offsets_cents=local_asaxi_offsets)
            if code == "asaxi":
                asaxi_mora_cursor += int(
                    (getattr(syn, "asaxi_prosody", {}) or {}).get(
                        "mora_count") or 0
                )
            syn = self._apply_phrase_profile(
                syn, phrase, voice, speed, pitch, fall, monotone,
                phrase_faults, backend=backend, lang_code=code)
            # Asaxi's mora phonation is sentence-contextual and is applied
            # once after all phrases share their final combined timeline.
            if code != "asaxi":
                syn = self._run_synthesis_task(
                    lambda rendered=syn: self._apply_shared_voicing_stage(
                        rendered))
            self._apply_voice_output_calibration(
                syn, voice, backend=backend)
            bits = int(phrase_faults.get("bit_depth") or 0)
            applied_bits.append(bits)
            if bits:
                syn.samples = fc.apply_bit_depth(syn.samples, bits)
                syn.output_bit_depth = bits
            rendered.append(syn)
        combined = fc.combine_syntheses(
            rendered, text=state.get("text") or "", lang=code,
            single_pause=bool(faults.get("single_pause")))
        combined._per_phrase_bit_depth = True
        combined._per_phrase_bit_depth_value = (
            applied_bits[0] if applied_bits and
            all(bits == applied_bits[0] for bits in applied_bits) else 0)
        return combined

    def _apply_voice_output_calibration(
            self, syn, voicebank="", backend=None):
        """Apply a generated voice's one-scalar phrase calibration policy."""
        if getattr(syn, "output_calibration", None):
            return syn
        voice = str(voicebank or getattr(syn, "voicebank", "") or "")
        if not voice or " + " in voice:
            return syn
        backend = backend if backend is not None else self._ab()
        metadata_reader = getattr(backend, "voice_metadata", None)
        if metadata_reader is None:
            return syn
        try:
            metadata = metadata_reader(voice)
        except (fc.BackendError, OSError, TypeError, ValueError):
            return syn
        return fc.apply_active_speech_calibration(
            syn, fc.generated_voice_output_calibration(metadata)
        )

    def _apply_vocal_tract_transform(
            self, syn, ratio=1.0, targets=None, chipmunk_range=False):
        result = vocal_tract.transform_vocal_tract(
            syn.samples,
            syn.sr,
            float(ratio),
            chipmunk_range=bool(chipmunk_range),
            ratio_targets=targets,
            segments=syn.segments,
        )
        syn.samples = result.samples
        syn.vocal_tract_requested_ratio = result.requested_ratio
        syn.vocal_tract_length_ratio = result.applied_ratio
        syn.chipmunk_range = result.chipmunk_range
        duration = syn.duration
        syn.generated_vocal_tract_targets = [
            (0.0, 1.0), (duration, 1.0)
        ]
        non_identity = any(abs(value - 1.0) > 1e-10
                           for _time, value in result.requested_targets)
        syn.vocal_tract_override = (
            list(result.requested_targets) if non_identity else [])
        syn.applied_vocal_tract_targets = list(result.applied_targets)
        syn.vocal_tract_mode = "curve" if non_identity else ""
        syn.vocal_tract_diagnostics = [result.diagnostic_dict()]
        return syn

    def _apply_output_faults(
            self, syn, faults=None, vocal_tract_ratio=1.0,
            vocal_tract_targets=None, chipmunk_range=False, *, backend=None,
            voicebank="", output_gain_db=None, allow_clipping=None,
            update_gain_control=True):
        def signal_transforms():
            self._apply_shared_voicing_stage(syn)
            self._apply_vocal_tract_transform(
                syn, vocal_tract_ratio, vocal_tract_targets, chipmunk_range)
            return syn

        # Source-filter voicing and tract-envelope processing can be more
        # expensive than Festival itself on long sentences. Keep those DSP
        # stages off Qt's event thread as well.
        syn = self._run_synthesis_task(signal_transforms)
        self._apply_voice_output_calibration(
            syn, voicebank or getattr(syn, "voicebank", "") or
            self._current_voicebank(), backend=backend
        )
        bits = int((faults or self._fault_mode()).get("bit_depth") or 0)
        if bits and not getattr(syn, "_per_phrase_bit_depth", False):
            syn.samples = fc.apply_bit_depth(syn.samples, bits)
        pre_gain_peak = (float(np.max(np.abs(syn.samples)))
                         if np.asarray(syn.samples).size else 0.0)
        gain = float(self.output_gain.value() if output_gain_db is None else
                     output_gain_db)
        allow_clipping = (self._allow_output_clipping
                          if allow_clipping is None else bool(allow_clipping))
        if not allow_clipping:
            safe = safe_gain_ceiling_db(pre_gain_peak, 0.0)
            if gain > safe:
                gain = safe
                if update_gain_control:
                    self.speech_gain.set_value(gain, emit=False)
        syn.pre_gain_peak = pre_gain_peak
        syn.applied_gain_db = gain
        syn.samples = fc.apply_gain_db(syn.samples, gain)
        syn.output_bit_depth = (int(getattr(
            syn, "_per_phrase_bit_depth_value", 0)) if getattr(
                syn, "_per_phrase_bit_depth", False) else bits)
        return syn

    def _confirm_generate_reset(self, states=None):
        candidates = ([self.sentences[self._active_sentence_index]]
                      if states is None and
                      0 <= self._active_sentence_index < len(self.sentences)
                      else list(states or []))
        has_existing_audio = bool(
            self.current is not None if states is None else
            any(state.get("synthesis") is not None or state.get("rendered")
                for state in candidates))
        if not has_existing_audio:
            return True
        return QtWidgets.QMessageBox.question(
            self, "Generate audio",
            "Generate may reset manual timing, pitch, segment, or recording "
            "edits. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes

    def _state_render_context(self, state):
        engine = str(state.get("engine") or "diphone")
        festival = engine == "festival_wsl"
        faults = self._faults_with_join_settings(
            state.get("fault_mode"), state.get("join_settings"))
        synthesis = state.get("synthesis")
        tract_targets = (
            list(getattr(synthesis, "vocal_tract_override", []) or [])
            if synthesis is not None and
            getattr(synthesis, "vocal_tract_mode", "") == "curve" else None
        )
        return {
            "engine": engine,
            "festival": festival,
            "backend": self._backend_for_engine(engine),
            "code": str(state.get("lang_code") or "asaxi"),
            "voicebank": str(state.get("voicebank") or ""),
            "speed": max(.25, min(4.0, float(state.get("speed") or 1.0))),
            "pitch": (float(state.get("pitch_hz") or 185.0)
                      if festival else None),
            "fall": (float(state.get("fall_pct") or 0.0)
                     if festival else None),
            "monotone": bool(festival and faults.get("monotone")),
            "faults": faults,
            "input_mode": str(state.get("input_mode") or "text"),
            "user_dict": self.user_dicts.get(
                str(state.get("lang_code") or "asaxi")) or None,
            "tract_targets": tract_targets,
            "tract_ratio": (
                vocal_tract.ratio_curve_summary(tract_targets)
                if tract_targets else
                float(state.get("vocal_tract_length_ratio", 1.0))),
            "chipmunk_range": bool(state.get("chipmunk_range", False)),
            "output_gain_db": float(state.get("output_gain_db", 0.0)),
            "allow_clipping": bool(self._allow_output_clipping),
        }

    def _generate_current(self, confirm_replace=True, show_error=True,
                          target_state=None, display_result=True):
        if self._synthesis_busy:
            self.statusBar().showMessage(
                "Status: synthesis is already in progress...")
            return None
        self._last_generation_error = ""
        background = target_state is not None
        state = target_state
        if state is None:
            if not self._need_backend():
                return None
            state = (self.sentences[self._active_sentence_index]
                     if 0 <= self._active_sentence_index < len(self.sentences)
                     else self._new_sentence_state(self.text.text()))
            text = self.text.text().strip()
            engine = self._engine()
            festival = engine == "festival_wsl"
            backend = self._ab()
            code = self._current_lang_code()
            vb = self._current_voicebank() or ""
            speed = self._speed_factor()
            pitch, fall = self._pitch()
            mono = self._monotone()
            faults = self._faults_with_join_settings(
                self._fault_mode(), state.get("join_settings"))
            input_mode = self.input_mode.currentData() or "text"
            udict = self._active_dict()
            tract_targets = (
                self.vocal_tract_track.targets()
                if self.current is not None and
                self.current.vocal_tract_mode == "curve" else None)
            tract_ratio = (
                vocal_tract.ratio_curve_summary(tract_targets)
                if tract_targets else
                float(state.get("vocal_tract_length_ratio", 1.0)))
            chipmunk_range = bool(state.get("chipmunk_range", False))
            output_gain_db = float(self.output_gain.value())
            allow_clipping = bool(self._allow_output_clipping)
        else:
            context = self._state_render_context(state)
            text = str(state.get("text") or "").strip()
            engine = context["engine"]
            festival = context["festival"]
            backend = context["backend"]
            code = context["code"]
            vb = context["voicebank"]
            speed = context["speed"]
            pitch = context["pitch"]
            fall = context["fall"]
            mono = context["monotone"]
            faults = context["faults"]
            input_mode = context["input_mode"]
            udict = context["user_dict"]
            tract_targets = context["tract_targets"]
            tract_ratio = context["tract_ratio"]
            chipmunk_range = context["chipmunk_range"]
            output_gain_db = context["output_gain_db"]
            allow_clipping = context["allow_clipping"]
            if backend is None:
                self._last_generation_error = (
                    "The selected sentence's synthesis engine is unavailable.")
                return None
        if confirm_replace and not self._confirm_generate_reset(
                [state] if background else None):
            return None
        if not text:
            self.statusBar().showMessage("Status: enter some text to synthesize.")
            return None
        if not vb:
            self._last_generation_error = "No voicebank is selected."
            if show_error:
                QtWidgets.QMessageBox.information(
                    self, "Voicebank", "No voicebank selected.\n" +
                    ("Voicebank > Scan Festival voices (WSL) or Add Festival "
                     "voice folder..." if festival else
                     "Voicebank > Add diphone DB folder... to register one."))
            return None
        if background:
            self._refresh_voice_metadata(backend, engine, vb)
        else:
            self._refresh_voice_metadata()
        fest = festival
        self.statusBar().showMessage(
            "Status: synthesizing via Festival in WSL (first call may take a "
            "while -- WSL boots and the voice loads)..." if fest
            else "Status: synthesizing...")
        QtWidgets.QApplication.processEvents()
        revision = int(state.get("_edit_revision") or 0)
        if not background:
            state["text"] = text
            state["input_mode"] = input_mode
        phrases = self._ensure_phrase_states(state)
        explicit_line_break = bool(re.search(r"[\r\n]", text))
        phrase_mode = explicit_line_break or any(
            phrase.get("speaker") or phrase.get("dictionary") or
            any(dict(phrase.get("fault_mode") or {}).values()) or
            phrase.get("pitch_override") or
            any(abs(float(value) - 1.0) > 1e-4 for value in
                (phrase.get("timing_factors") or []))
            for phrase in phrases)
        if fest and code in ("ja", "jp") and not explicit_line_break:
            phrase_mode = False
        try:
            if phrase_mode:
                syn = self._generate_phrase_sequence(
                    state, code, vb, speed, pitch, fall,
                    mono, faults, udict, backend=backend, festival=fest,
                    refresh_controls=not background)
            elif input_mode == "phones":
                # direct phoneme prompt: space-separated phone names.
                # On Festival this goes through a Segments utterance with
                # class-based durations (so Speed works) and the Pitch/Fall
                # contour (bare Phones utterances are hardwired to
                # FP_F0=120Hz / 100ms -- the old wrong-pitch monotone).
                phones = text.split()
                if fest:
                    syn = self._call_synthesis_backend(
                        self.fest.synth_phones,
                        phones, vb, speed, text=text,
                        lang=code,
                        seg_durs=fc.class_seg_durs(phones,
                                                   speed,
                                                   equal=faults.get(
                                                       "disable_phone_timing")),
                        pitch=pitch, fall=fall, monotone=mono,
                        fault_mode=faults)
                else:
                    syn = self._call_synthesis_backend(
                        backend.synth_phones, phones, vb,
                        speed, text=text, lang=code)
            elif fest and code in ("ja", "jp"):
                plan = self._prepare_japanese_plan(
                    state, text, vb, speed, pitch, analyze=True,
                    refresh_controls=not background)
                syn = self._render_japanese_plan(
                    plan, vb, text, pitch, fall, mono, faults,
                    mora_voicing_overrides=(state.get("japanese_state") or
                                            {}).get(
                                                "mora_voicing_overrides"))
            else:
                # dictionary overrides + inline [phones] ride the normal text
                # pipeline as Festival lexicon addenda (full prosody + timing);
                # reduction is disabled in _dict_addenda so vowels stay verbatim
                exp, extra, dropped = self._prepare_inline_phones(
                    text, code, vb, backend=backend)
                if dropped:
                    self.statusBar().showMessage(
                        "Status: %s does not provide %s; omitted. "
                        "Synthesizing..." %
                        (vb, ", ".join(sorted(dropped))))
                    QtWidgets.QApplication.processEvents()
                combined = dict(udict or {})
                combined.update(extra)
                synth_kwargs = {
                    "pitch": pitch,
                    "fall": fall,
                    "monotone": mono,
                    "user_dict": combined or None,
                    "fault_mode": faults,
                }
                if fest and code == "asaxi":
                    overlays = self._asaxi_edit_overlays(state, text)
                    synth_kwargs[
                        "asaxi_tone_overrides"] = overlays["tone"]
                    synth_kwargs[
                        "asaxi_pitch_offsets_cents"] = overlays["pitch"]
                syn = self._call_synthesis_backend(
                    backend.synth, exp, code, vb, speed,
                    **synth_kwargs)
        except fc.BackendError as e:
            self._last_generation_error = str(e)
            if show_error:
                QtWidgets.QMessageBox.critical(
                    self, "Synthesis error", str(e))
            self.statusBar().showMessage("Status: Ready")
            return None
        if fest and code == "asaxi" and isinstance(syn, fc.Synthesis):
            overlays = self._asaxi_edit_overlays(state, text)
            self._set_language_render_features(
                syn,
                asaxi_metadata=getattr(syn, "asaxi_prosody", {}) or {},
                asaxi_voicing_overrides=overlays["voicing"],
            )
        syn = self._apply_output_faults(
            syn, faults,
            vocal_tract_ratio=tract_ratio,
            vocal_tract_targets=tract_targets,
            chipmunk_range=chipmunk_range, backend=backend,
            voicebank=vb, output_gain_db=output_gain_db,
            allow_clipping=allow_clipping,
            update_gain_control=not background,
        )
        if int(state.get("_edit_revision") or 0) != revision:
            self._last_generation_error = (
                "The sentence changed while it was generating; its older "
                "result was discarded and the edit remains pending.")
            return None
        self._reconcile_asaxi_result(
            state, syn, text, refresh_controls=not background)
        if background or not display_result:
            self._commit_synthesis_to_state(state, syn, text)
            self._display_background_render_if_active(state)
        else:
            self._clear_state_pending(state)
            state["rendered"] = True
            self._show_synthesis(syn)
            self._commit_rendered_state(syn)
            self._capture_active_sentence()
            self._refresh_sentences_after_render()
        return syn

    def _rerender_seg_durs(self):
        """(phone, dur) pairs for the CURRENT editor state, expanding
        multi-phone fields (their duration is split evenly)."""
        return self._rerender_seg_durs_from_segments(
            self.waveform.segments, self._fault_mode())

    @staticmethod
    def _rerender_seg_durs_from_segments(segments, faults=None):
        out = []
        for seg in segments:
            ph = [p for p in seg.phone.replace(",", " ").split() if p]
            if not ph:
                continue
            d = max(0.01, seg.dur) / len(ph)
            out.extend((p, d) for p in ph)
        if dict(faults or {}).get("single_pause"):
            return fc.collapse_pause_runs(out)
        return fc.normalize_internal_pause_runs(out)

    def _retimed_ground_truth(
            self, seg_durs, targets, pitch, faults, synthesis=None,
            editor_segments=None):
        """Move the displayed generated F0 onto the edited phone timeline."""
        synthesis = synthesis if synthesis is not None else self.current
        if synthesis is None:
            return []
        entries = [(str(phone), max(.01, float(duration)))
                   for phone, duration in (seg_durs or [])]
        source_segments = list(synthesis.segments or [])
        remapped = list(targets or [])
        edited = list(editor_segments or [])
        if remapped and source_segments and edited:
            remapped = fc.remap_targets_aligned(
                remapped, source_segments, edited)
            source_segments = edited
        final_segments = fc.segments_from_durations(entries)
        if remapped and source_segments:
            remapped = fc.remap_targets_aligned(
                remapped, source_segments, final_segments)
        if remapped and not dict(faults or {}).get("disable_f0_correction"):
            remapped = fc.anchor_phrase_targets(
                entries, remapped, float(pitch or 160.0))
        return remapped

    def on_rerender(self, _checked=False, confirm_generate=True,
                    show_error=True, target_state=None,
                    display_result=True):
        self._begin_generation_progress("Re-rendering current sentence...")
        try:
            return self._rerender_current(
                _checked=_checked, confirm_generate=confirm_generate,
                show_error=show_error, target_state=target_state,
                display_result=display_result)
        finally:
            self._end_generation_progress()

    def _rerender_current(self, _checked=False, confirm_generate=True,
                          show_error=True, target_state=None,
                          display_result=True):
        """Feed the (edited) phoneme fields back through the engine. On the
        Festival engine the editor's timings and the previous render's F0
        contour are passed along, so prosody survives the edit."""
        if self._synthesis_busy:
            self.statusBar().showMessage(
                "Status: synthesis is already in progress...")
            return None
        self._last_generation_error = ""
        background = target_state is not None
        state = target_state
        if state is None:
            if not self._need_backend():
                return None
            state = (self.sentences[self._active_sentence_index]
                     if 0 <= self._active_sentence_index < len(self.sentences)
                     else None)
            context = None
        else:
            context = self._state_render_context(state)
            if context["backend"] is None:
                self._last_generation_error = (
                    "The selected sentence's synthesis engine is unavailable.")
                return None
        if state is not None and self._pending_action(state) == "generate":
            return self._generate_for_sentence_mode(
                confirm_replace=confirm_generate, show_error=show_error,
                target_state=state if background else None,
                display_result=display_result)
        current_render = (state.get("synthesis") if background and state
                          is not None else self.current)
        editor_segments = (copy.deepcopy(
            state.get("editor_segments") or current_render.segments)
            if background and current_render is not None else
            copy.deepcopy(self.waveform.segments))
        if current_render is None or not editor_segments:
            return self._generate_for_sentence_mode(
                confirm_replace=confirm_generate, show_error=show_error,
                target_state=state if background else None,
                display_result=display_result)
        old_editor_segments = copy.deepcopy(editor_segments)
        revision = int((state or {}).get("_edit_revision") or 0)
        if background:
            engine = context["engine"]
            fest = context["festival"]
            backend = context["backend"]
            code = context["code"]
            vb = context["voicebank"]
            speed = context["speed"]
            pitch = context["pitch"]
            fall = context["fall"]
            mono = context["monotone"]
            faults = context["faults"]
            input_mode = context["input_mode"]
            text = str(state.get("text") or "").strip()
            output_gain_db = context["output_gain_db"]
            allow_clipping = context["allow_clipping"]
            chipmunk_range = context["chipmunk_range"]
        else:
            engine = self._engine()
            fest = engine == "festival_wsl"
            backend = self._ab()
            code = self._current_lang_code()
            vb = self._current_voicebank()
            speed = self._speed_factor()
            pitch, fall = self._pitch()
            mono = self._monotone()
            faults = self._faults_with_join_settings(
                self._fault_mode(), (state or {}).get("join_settings"))
            input_mode = self.input_mode.currentData() or "text"
            text = self.text.text().strip()
            output_gain_db = float(self.output_gain.value())
            allow_clipping = bool(self._allow_output_clipping)
            chipmunk_range = self.vocal_tract_chipmunk.isChecked()
        phones = [phone for segment in editor_segments for phone in
                  segment.phone.replace(",", " ").split() if phone]
        seg_durs = self._rerender_seg_durs_from_segments(
            editor_segments, faults)
        active_state = state or {}
        if active_state.get("pending_reason") == \
                "Phrase pause duration changed":
            seg_durs = fc.retime_internal_phrase_pauses(
                seg_durs, text, speed,
                self.cfg.get("phrase_pauses_ms"))
        if background:
            self._refresh_voice_metadata(backend, engine, vb)
        else:
            self._refresh_voice_metadata()
        old_display_phones = [
            segment.phone for segment in current_render.segments
        ]
        new_display_phones = [
            phone for phone, _duration in seg_durs
        ]
        old_source_phones = list(
            getattr(current_render, "render_phones", ()) or ()
        )
        try:
            if len(old_source_phones) != len(old_display_phones):
                old_source_phones = self._resolve_source_selection_phones(
                    old_display_phones, backend=backend, voice=vb)
            new_source_phones = self._resolve_source_selection_phones(
                new_display_phones, backend=backend, voice=vb)
        except fc.BackendError as error:
            self._last_generation_error = str(error)
            if show_error:
                QtWidgets.QMessageBox.critical(
                    self, "Re-render error", str(error))
            return None
        remapped_overrides = fc.remap_unit_overrides(
            old_display_phones,
            new_display_phones,
            current_render.unit_overrides,
            old_source_phones=old_source_phones,
            new_source_phones=new_source_phones,
        )
        self.statusBar().showMessage("Status: re-rendering edited phonemes...")
        QtWidgets.QApplication.processEvents()
        mode = current_render.pitch_mode or ""
        # The dashed generated contour is the reset/reference source.  Using
        # current.targets here can resurrect a curve or intonation override
        # that was rendered before the user pressed Reset.
        reference_targets = self._synthesis_pitch_source(
            current_render, text=text, base=float(pitch or 160.0),
            fall=float(fall or 0.0), allow_baseline=fest)
        target_segments = current_render.segments
        effective_overrides = dict(remapped_overrides)
        is_japanese = (
            fest and code in ("ja", "jp") and input_mode == "text")
        is_asaxi = (
            fest and code == "asaxi" and input_mode == "text")
        japanese_plan = None
        asaxi_metadata = None
        if is_japanese and state is not None:
            try:
                plan = self._prepare_japanese_plan(
                    state, text, vb, speed, pitch, analyze=False,
                    refresh_controls=not background)
            except fc.BackendError as error:
                self._last_generation_error = str(error)
                if show_error:
                    QtWidgets.QMessageBox.critical(
                        self, "Japanese re-render", str(error))
                return None
            japanese_plan = plan
            plan_entries = list(plan.segment_durations)
            # Re-render is an acoustic refresh of the editor state, not a new
            # linguistic timing pass.  The fresh plan supplies structural F0,
            # voicing, and unit metadata, while ``seg_durs`` remains the exact
            # current timeline returned by _rerender_seg_durs().  Generate
            # Audio is the only action allowed to adopt a new duration model.
            target_segments = fc.segments_from_durations(plan_entries)
            reference_targets = list(plan.pitch_targets)
            plan_display_phones = [
                phone for phone, _duration in plan_entries
            ]
            try:
                plan_source_phones = self._resolve_source_selection_phones(
                    plan_display_phones, backend=backend, voice=vb)
            except fc.BackendError as error:
                self._last_generation_error = str(error)
                if show_error:
                    QtWidgets.QMessageBox.critical(
                        self, "Japanese re-render", str(error))
                return None
            planned_overrides = fc.remap_unit_overrides(
                plan_display_phones,
                new_display_phones,
                plan.unit_overrides,
                old_source_phones=plan_source_phones,
                new_source_phones=new_source_phones,
            )
            planned_overrides.update(remapped_overrides)
            effective_overrides = planned_overrides
            ground_truth = fc.remap_targets_aligned(
                reference_targets, target_segments,
                fc.segments_from_durations(seg_durs))
            if ground_truth and not faults.get("disable_f0_correction"):
                ground_truth = fc.anchor_phrase_targets(
                    seg_durs, ground_truth, float(pitch))
        elif is_asaxi and state is not None:
            try:
                reference_targets, asaxi_metadata = \
                    self._prepare_asaxi_rerender(
                        state,
                        text,
                        fc.segments_from_durations(seg_durs),
                        pitch,
                        fall,
                        user_dict=(
                            context.get("user_dict") if background
                            else self._active_dict()
                        ),
                    )
            except fc.BackendError as error:
                self._last_generation_error = str(error)
                if show_error:
                    QtWidgets.QMessageBox.critical(
                        self, "Asaxi re-render", str(error))
                return None
            target_segments = fc.segments_from_durations(seg_durs)
            ground_truth = list(reference_targets)
        else:
            ground_truth = self._retimed_ground_truth(
                seg_durs, reference_targets, pitch, faults,
                synthesis=current_render,
                editor_segments=editor_segments)
        rendered_pitch = active_state.get("rendered_pitch_hz")
        pitch_changed = (
            rendered_pitch is not None and pitch is not None and
            abs(float(rendered_pitch) - float(pitch)) > 1.0e-7
        )
        if (fest and ground_truth and pitch_changed and
                not is_japanese and not is_asaxi):
            ground_truth = fc.pitch_domain.recenter_targets_log(
                ground_truth, float(pitch),
                fc.PITCH_MIN_HZ, fc.PITCH_MAX_HZ)
        if background:
            pitch_targets = (list(current_render.pitch_override or [])
                             if mode == "curve" else None)
            voicing_targets = (
                list(current_render.voicing_override or [])
                if current_render.voicing_mode == "curve" else None)
            vocal_tract_targets = (
                list(current_render.vocal_tract_override or [])
                if current_render.vocal_tract_mode == "curve" else None)
            blocks = (copy.deepcopy(current_render.intonation_blocks)
                      if mode == "intonation" else None)
        else:
            pitch_targets = (self.pitch_track.render_targets()
                             if mode == "curve" else None)
            voicing_targets = (
                self.voicing_track.targets()
                if current_render.voicing_mode == "curve" else None)
            vocal_tract_targets = (
                self.vocal_tract_track.targets()
                if current_render.vocal_tract_mode == "curve" else None)
            blocks = (self.intonation.blocks()
                      if mode == "intonation" else None)
        backend_target_segments = target_segments
        backend_reference_targets = reference_targets
        if fest and ground_truth:
            # ``ground_truth`` is already aligned to the exact editor phone
            # sequence. Feeding the earlier contour and segment list into the
            # backend would remap and recenter it again, accumulating a
            # whole-sentence pitch shift after each edit.
            backend_target_segments = fc.segments_from_durations(seg_durs)
            backend_reference_targets = list(ground_truth)
        synth_kwargs = {
            "text": text,
            "lang": code,
            "seg_durs": seg_durs,
            "old_segments": backend_target_segments,
            "prev_targets": backend_reference_targets,
            "pitch": pitch,
            "fall": fall,
            "monotone": mono,
            "fault_mode": faults,
            "pitch_targets": pitch_targets,
            "ground_truth_targets": ground_truth,
            "intonation_blocks": blocks,
            "pitch_mode": mode,
            "unit_overrides": effective_overrides,
        }
        if fest:
            synth_kwargs["preserve_pitch_register"] = True
        try:
            syn = self._call_synthesis_backend(
                backend.synth_phones,
                phones, vb, speed,
                **synth_kwargs)
        except fc.BackendError as e:
            self._last_generation_error = str(e)
            if show_error:
                QtWidgets.QMessageBox.critical(
                    self, "Re-render error", str(e))
            return None
        if japanese_plan is not None:
            self._set_language_render_features(
                syn,
                japanese_plan=japanese_plan,
                voicing_override=voicing_targets,
                mora_voicing_overrides=(
                    dict((state or {}).get("japanese_state") or {}).get(
                        "mora_voicing_overrides") or {}),
            )
        elif asaxi_metadata is not None:
            overlays = self._asaxi_edit_overlays(state, text)
            self._set_language_render_features(
                syn,
                asaxi_metadata=asaxi_metadata,
                voicing_override=voicing_targets,
                asaxi_voicing_overrides=overlays["voicing"],
            )
        elif voicing_targets is not None:
            self._set_language_render_features(
                syn, voicing_override=voicing_targets)
        syn = self._apply_output_faults(
            syn, faults,
            vocal_tract_ratio=(
                vocal_tract.ratio_curve_summary(vocal_tract_targets)
                if vocal_tract_targets else 1.0),
            vocal_tract_targets=vocal_tract_targets,
            chipmunk_range=chipmunk_range, backend=backend,
            voicebank=vb, output_gain_db=output_gain_db,
            allow_clipping=allow_clipping,
            update_gain_control=not background,
        )
        fc.transfer_segment_uids(old_editor_segments, syn.segments)
        if int((state or {}).get("_edit_revision") or 0) != revision:
            self._last_generation_error = (
                "The sentence changed while it was re-rendering; its older "
                "result was discarded and the edit remains pending.")
            return None
        self._reconcile_asaxi_result(
            state, syn, text, refresh_controls=not background)
        if background or not display_result:
            self._commit_synthesis_to_state(
                state, syn, text,
                timing_factors=(state.get("timing_factors") or []))
            self._display_background_render_if_active(
                state, preserve_view=True)
        else:
            self._clear_state_pending(state)
            state["rendered"] = True
            self._show_synthesis(syn, preserve_view=True)
            self._commit_rendered_state(syn)
            self._capture_active_sentence()
            self._refresh_sentences_after_render()
        return syn

    def on_render_details(self):
        if self.current is None:
            QtWidgets.QMessageBox.information(self, "Render details",
                                              "Nothing rendered yet.")
            return
        s = self.current
        join_settings = dict(getattr(s, "join_settings", None) or {})
        if join_settings:
            join_method = (
                "symmetric" if join_settings.get(
                    "window_symmetric", True) else "asymmetric")
            join_details = (
                "\n\nUniSyn joins:\n"
                "method: %s\nwindow radius: %.3f periods\n"
                "requested crossover: %.1f ms\n"
                "effective runtime crossover: %.1f ms\n"
                "per-join overrides: %d\nruntime: %s\n"
                "scope: %s\nsource: %s\nLegacy override: %s"
                % (
                    join_method,
                    float(join_settings.get("window_factor") or 1.0),
                    float(join_settings.get(
                        "requested_crossover_ms") or 0.0),
                    float(join_settings.get("crossover_ms") or 0.0),
                    len(join_settings.get(
                        "crossover_overrides") or {}),
                    str(join_settings.get("runtime") or "unknown"),
                    str(join_settings.get("scope") or "utterance"),
                    str(join_settings.get("source") or "unknown"),
                    ("active" if join_settings.get("legacy_active") else
                     "inactive"),
                ))
        else:
            join_details = (
                "\n\nUniSyn joins:\n"
                "No render metadata (older project or non-UniSyn backend).")
        timing_details = ""
        if (self._current_lang_code() in ("ja", "jp") and
                0 <= self._active_sentence_index < len(self.sentences)):
            japanese_state = dict(self.sentences[
                self._active_sentence_index].get("japanese_state") or {})
            plan = dict(japanese_state.get("last_plan") or {})
            utterance = dict(japanese_state.get("utterance") or {})
            mora_labels = {}
            for phrase in utterance.get("phrases") or []:
                for accent in dict(phrase).get("accent_phrases") or []:
                    for mora in dict(accent).get("moras") or []:
                        row = dict(mora)
                        mora_labels[int(row.get("index", -1))] = str(
                            row.get("reading") or row.get("surface") or "?")
            lines = []
            for mora in plan.get("mora_timings") or []:
                row = dict(mora)
                index = int(row.get("mora_index", -1))
                phones = []
                for phone in row.get("phone_allocation") or []:
                    item = dict(phone)
                    reference = item.get("source_reference_duration")
                    stretch = item.get("requested_stretch")
                    phones.append(
                        "%s %.1fms [safe %.1f-%.1fms%s%s]" % (
                            item.get("phone") or "?",
                            1000.0 * float(item.get("final_duration") or 0.0),
                            1000.0 * float(item.get("source_safe_min") or 0.0),
                            1000.0 * float(item.get("source_safe_max") or 0.0),
                            (", source %.1fms" % (1000.0 * float(reference))
                             if reference is not None else ""),
                            (", %.2fx" % float(stretch)
                             if stretch is not None else ""),
                        ))
                lines.append(
                    "%d %s: predicted %.1fms, final %.1fms; %s" % (
                        index,
                        mora_labels.get(index, "?"),
                        1000.0 * float(
                            row.get("predicted_mora_duration") or 0.0),
                        1000.0 * float(row.get("final_duration") or 0.0),
                        "; ".join(phones) or "no spoken phones",
                    ))
            realization_lines = []
            for decision in getattr(s, "vowel_realizations", ()):
                row = dict(decision)
                before = row.get("periodicity_before")
                after = row.get("periodicity_after")
                periodicity = ""
                if before is not None or after is not None:
                    periodicity = ", periodicity %s -> %s" % (
                        "?" if before is None else "%.3f" % float(before),
                        "?" if after is None else "%.3f" % float(after),
                    )
                realization_lines.append(
                    "%s mora %s: %s%s (%s)" % (
                        row.get("phone") or "?",
                        row.get("mora_index", "?"),
                        row.get("strategy") or "unknown",
                        periodicity,
                        row.get("reason") or "no reason reported",
                    )
                )
            prosody = dict(getattr(s, "japanese_prosody", None) or {})
            if not prosody and plan:
                plan_segments = list(plan.get("segments") or ())
                plan_moras = list(plan.get("mora_timings") or ())
                plan_f0 = list(plan.get("f0_targets") or ())
                frequencies = [
                    float(row["hz"]) for row in plan_f0
                    if row.get("hz") is not None and float(row["hz"]) > 0.0
                ]
                prosody = {
                    "duration_model": plan.get("duration_model"),
                    "duration_model_id": plan.get("duration_model_id"),
                    "pitch_model_id": plan.get("pitch_model_id"),
                    "contour_model": plan.get("contour_model"),
                    "frontend_name": plan.get("frontend_name"),
                    "plan_schema_version": plan.get("schema_version"),
                    "segment_count": len(plan_segments),
                    "mora_count": len(plan_moras),
                    "phrase_count": len({
                        int(row.get("phrase_index") or 0)
                        for row in plan_f0
                    }),
                    "total_duration_seconds": sum(
                        float(row.get("duration") or 0.0)
                        for row in plan_segments),
                    "f0_target_count": len(frequencies),
                    "f0_min_hz": min(frequencies) if frequencies else None,
                    "f0_max_hz": max(frequencies) if frequencies else None,
                    "f0_span_semitones": (
                        12.0 * math.log2(
                            max(frequencies) / min(frequencies))
                        if frequencies else None),
                    "phrase_position_model": "mean_centered_shape",
                    "cumulative_register_drift_enabled": False,
                }
            if lines or realization_lines or prosody:
                timing_details = (
                    "\n\nJapanese active prosody models:\n"
                    "duration mode: %s\n"
                    "duration model: %s\n"
                    "pitch model: %s\n"
                    "contour: %s\n"
                    "frontend: %s\n"
                    "plan schema: %s\n"
                    "timeline: %s segments, %s morae, %.3fs\n"
                    "F0: %s targets, %s to %s Hz, %s semitone span\n"
                    "phrase position: %s; cumulative register drift: %s\n"
                    "\nJapanese mora timing / source safety:\n"
                    % (
                        prosody.get("duration_model") or "unknown",
                        prosody.get("duration_model_id") or "unknown",
                        prosody.get("pitch_model_id") or "unknown",
                        prosody.get("contour_model") or "unknown",
                        prosody.get("frontend_name") or "unknown",
                        prosody.get("plan_schema_version") or "unknown",
                        prosody.get("segment_count") or len(s.phones),
                        prosody.get("mora_count") or "unknown",
                        float(prosody.get("total_duration_seconds") or
                              s.duration),
                        prosody.get("f0_target_count") or "unknown",
                        ("%.2f" % float(prosody["f0_min_hz"])
                         if prosody.get("f0_min_hz") is not None else
                         "unknown"),
                        ("%.2f" % float(prosody["f0_max_hz"])
                         if prosody.get("f0_max_hz") is not None else
                         "unknown"),
                        ("%.3f" % float(prosody["f0_span_semitones"])
                         if prosody.get("f0_span_semitones") is not None else
                         "unknown"),
                        prosody.get("phrase_position_model") or "unknown",
                        ("enabled" if prosody.get(
                            "cumulative_register_drift_enabled") else
                         "disabled"),
                    )
                    + ("\n".join(lines) or "(no mora timing rows)")
                )
                if realization_lines:
                    timing_details += (
                        "\n\nJapanese vowel realization:\n"
                        + "\n".join(realization_lines)
                    )
        message = (
            "text: %s\nlanguage: %s   voicebank: %s\n\nphones (%d):\n%s\n\n"
            "diphones used (%d):\n%s\n\nselected outgoing units:\n%s\n\n"
            "missing/skipped (%d):\n%s%s%s"
            % (s.text, s.lang, s.voicebank, len(s.phones), " ".join(s.phones),
               len(s.diphones), " ".join(s.diphones),
               " ".join("%d:%s" % item for item in
                        sorted(s.selected_units.items())) or "(not reported)",
               len(s.skipped),
               " ".join(s.skipped) or "(none)", join_details,
               timing_details)
        )
        QtWidgets.QMessageBox.information(
            self, "Render details", message)

    def _on_audio_changed(self):
        dur = self.waveform.duration() or 1.0
        self._sync_timing_track()
        if 0 <= self._active_sentence_index < len(self.sentences):
            state = self.sentences[self._active_sentence_index]
            if self._editor_sentence_state is not state:
                return
            state["editor_segments"] = copy.deepcopy(self.waveform.segments)
            state["timing_factors"] = list(self.waveform.factors())
            state["preview_audio"] = np.asarray(
                self.waveform.audio, np.float32)
            state["preview_sr"] = int(self.waveform.sr)
            applied = float(state.get("applied_gain_db") or 0.0)
            peak = (float(np.max(np.abs(self.waveform.audio)))
                    if self.waveform.audio.size else 0.0)
            state["pre_gain_peak"] = (
                peak / (10.0 ** (applied / 20.0)))
            self._refresh_gain_controls()
        self._mark_active_pending("rerender", "Phoneme timing changed")
        self.statusBar().showMessage(
            "Status: re-timed -- %.2fs total (Re-render Phonemes = optimal "
            "quality at these timings)" % dur)

    def _begin_sentence_batch(self, total, verb):
        if self._batch_active:
            return False
        self._batch_active = True
        self._batch_cancel_requested = False
        self.batch_progress.setRange(0, max(1, int(total)))
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat("%s 0 / %d" % (verb, int(total)))
        self.batch_progress.show()
        self._refresh_synthesis_progress_visibility()
        self.batch_cancel.setEnabled(True)
        self.batch_cancel.show()
        self.btn_stop.setEnabled(True)
        return True

    def _update_sentence_batch(self, completed, total, verb):
        self.batch_progress.setRange(0, max(1, int(total)))
        self.batch_progress.setValue(max(0, min(int(total), int(completed))))
        self.batch_progress.setFormat(
            "%s %d / %d" % (verb, int(completed), int(total)))
        self.batch_progress.setToolTip(
            "%s %d / %d" % (verb, int(completed), int(total)))

    def _request_batch_cancel(self):
        if not self._batch_active:
            return
        self._batch_cancel_requested = True
        self.batch_cancel.setEnabled(False)
        self.statusBar().showMessage(
            "Status: stopping batch after the current sentence...")

    def _end_sentence_batch(self):
        self._batch_active = False
        self.batch_cancel.hide()
        self.batch_progress.hide()
        self._refresh_synthesis_progress_visibility()
        self.btn_stop.setEnabled(self._playback_active)

    @staticmethod
    def _batch_failure_details(failures):
        grouped = {}
        for index, reason in failures:
            reason = str(reason or "Unknown synthesis error").strip()
            grouped.setdefault(reason, []).append(int(index) + 1)
        rows = []
        for reason, indices in grouped.items():
            label = ("Sentence " if len(indices) == 1 else "Sentences ") + \
                ", ".join(str(index) for index in indices)
            rows.append("%s:\n%s" % (label, reason[:1200]))
        return "\n\n".join(rows)

    def on_generate_all(self, _checked=False, only_indices=None):
        if only_indices is None and isinstance(_checked, (list, tuple, set)):
            only_indices = list(_checked)
        if (self._synthesis_busy or self._batch_active or
                not self._need_backend() or
                not self.sentences):
            return
        self._capture_active_sentence()
        wanted = (list(range(len(self.sentences))) if only_indices is None
                  else list(dict.fromkeys(
                      int(index) for index in only_indices
                      if 0 <= int(index) < len(self.sentences))))
        if not wanted:
            return
        replacing = [self.sentences[index] for index in wanted]
        if not self._confirm_generate_reset(replacing):
            return
        total = len(wanted)
        rendered, attempted, failures = 0, 0, []
        self._begin_sentence_batch(total, "Generate")
        try:
            for position, index in enumerate(wanted):
                if self._batch_cancel_requested:
                    break
                state = self.sentences[index]
                attempted += 1
                self._update_sentence_batch(position, total, "Generate")
                self.statusBar().showMessage(
                    "Status: generating sentence %d of %d..." %
                    (position + 1, total))
                QtWidgets.QApplication.processEvents()
                if self._batch_cancel_requested:
                    break
                if not str(state.get("text") or "").strip():
                    failures.append((index, "The sentence is empty."))
                elif self._generate_for_sentence_mode(
                        confirm_replace=False, show_error=False,
                        target_state=state,
                        display_result=False) is None:
                    failures.append((
                        index, self._last_generation_error or
                        "The sentence could not be generated."))
                else:
                    rendered += 1
                self._update_sentence_batch(
                    position + 1, total, "Generate")
                QtWidgets.QApplication.processEvents()
        finally:
            cancelled = self._batch_cancel_requested
            self._end_sentence_batch()
            self._refresh_sentences_view_preserving_focus()
        message = "Generated %d of %d sentences" % (rendered, total)
        if cancelled:
            message += " (stopped after %d)" % attempted
        self.statusBar().showMessage("Status: " + message)
        if failures:
            QtWidgets.QMessageBox.warning(
                self, "Generate All", message + ".\n\n" +
                self._batch_failure_details(failures))

    def on_rerender_all(self, _checked=False, only_indices=None):
        if only_indices is None and isinstance(_checked, (list, tuple, set)):
            only_indices = list(_checked)
        if (self._synthesis_busy or self._batch_active or
                not self._need_backend() or
                not self.sentences):
            return
        self._capture_active_sentence()
        wanted = (list(range(len(self.sentences))) if only_indices is None
                  else [int(index) for index in only_indices])
        replacing = [self.sentences[index] for index in wanted
                     if 0 <= index < len(self.sentences) and
                     self._pending_action(self.sentences[index]) == "generate"
                     and (self.sentences[index].get("synthesis") is not None or
                          self.sentences[index].get("rendered"))]
        if replacing and not self._confirm_generate_reset(replacing):
            return
        rendered, attempted, failures = 0, 0, []
        total = len(wanted)
        self._begin_sentence_batch(total, "Re-render")
        try:
            for position, index in enumerate(wanted):
                if self._batch_cancel_requested:
                    break
                if not (0 <= index < len(self.sentences)):
                    continue
                attempted += 1
                self._update_sentence_batch(position, total, "Re-render")
                self.statusBar().showMessage(
                    "Status: re-rendering sentence %d of %d..." %
                    (position + 1, total))
                QtWidgets.QApplication.processEvents()
                if self._batch_cancel_requested:
                    break
                state = self.sentences[index]
                if (self._pending_action(state) != "generate" and
                        state.get("synthesis") is not None and
                        state.get("editor_segments")):
                    result = self.on_rerender(
                        confirm_generate=False, show_error=False,
                        target_state=state, display_result=False)
                else:
                    result = self._generate_for_sentence_mode(
                        confirm_replace=False, show_error=False,
                        target_state=state, display_result=False)
                if result is None:
                    failures.append((
                        index, self._last_generation_error or
                        "The sentence could not be re-rendered."))
                else:
                    rendered += 1
                self._update_sentence_batch(position + 1, total, "Re-render")
                QtWidgets.QApplication.processEvents()
        finally:
            cancelled = self._batch_cancel_requested
            self._end_sentence_batch()
            self._refresh_sentences_view_preserving_focus()
        self.statusBar().showMessage(
            "Status: re-rendered %d of %d requested sentences%s" %
            (rendered, total,
             " (stopped after %d)" % attempted if cancelled else ""))
        if failures:
            QtWidgets.QMessageBox.warning(
                self, "Re-render All",
                "Re-rendered %d of %d requested sentences.\n\n%s" %
                (rendered, len(wanted),
                 self._batch_failure_details(failures)))

    @staticmethod
    def _batch_slug(text):
        slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")[:48]
        return slug or "sentence"

    def on_export_batch(self):
        self._capture_active_sentence()
        choice = QtWidgets.QMessageBox(self)
        choice.setWindowTitle("Batch export")
        choice.setText("Export sentences separately or merge them in order?")
        separate_button = choice.addButton(
            "Separate WAV files", QtWidgets.QMessageBox.AcceptRole)
        merged_button = choice.addButton(
            "One merged WAV", QtWidgets.QMessageBox.ActionRole)
        choice.addButton(QtWidgets.QMessageBox.Cancel)
        choice.setDefaultButton(separate_button)
        choice.exec_()
        if choice.clickedButton() not in (separate_button, merged_button):
            return
        ready, skipped = [], []
        for index, state in enumerate(self.sentences):
            raw_audio = state.get("preview_audio")
            samples = np.asarray(
                raw_audio if raw_audio is not None else [], np.float32)
            sr = int(state.get("preview_sr") or 16000)
            if not state.get("rendered") or samples.size <= 1:
                skipped.append(index + 1)
            else:
                ready.append((index, state, samples, sr))
        if choice.clickedButton() is merged_button:
            default_export = os.path.join(
                self._project_root, "exports", "sentences_merged.wav") \
                if self._project_root else "sentences_merged.wav"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export merged sentence audio", default_export,
                "WAV (*.wav)")
            if not path:
                return
            if not ready:
                QtWidgets.QMessageBox.information(
                    self, "Batch export", "Generate sentences first.")
                return
            samples, sr = fc.concat_audio(
                [(samples, rate) for _index, _state, samples, rate in ready])
            try:
                fc.write_wav(path, samples, sr)
            except Exception as error:
                QtWidgets.QMessageBox.critical(
                    self, "Batch export", "Could not write:\n%s\n\n%s" %
                    (path, error))
                return
            self.statusBar().showMessage(
                "Status: exported merged audio for %d sentences" % len(ready))
            if skipped:
                QtWidgets.QMessageBox.information(
                    self, "Batch export",
                    "Merged the generated sentences. Generate these first: " +
                    ", ".join("sentence %d" % index for index in skipped))
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export one WAV per generated sentence",
            os.path.join(self._project_root, "exports")
            if self._project_root else "")
        if not folder:
            return
        exported = 0
        for index, state, samples, sr in ready:
            base = "%03d_%s" % (index + 1, self._batch_slug(state.get("text")))
            path = os.path.join(folder, base + ".wav")
            suffix = 2
            while os.path.exists(path):
                path = os.path.join(folder, "%s_%d.wav" % (base, suffix))
                suffix += 1
            try:
                fc.write_wav(path, samples, sr)
                exported += 1
            except Exception as error:
                QtWidgets.QMessageBox.critical(
                    self, "Batch export", "Could not write:\n%s\n\n%s" %
                    (path, error))
                return
        message = "exported %d WAV file%s" % (
            exported, "" if exported == 1 else "s")
        self.statusBar().showMessage("Status: " + message)
        if skipped:
            QtWidgets.QMessageBox.information(
                self, "Batch export", message.capitalize() +
                ".\n\nGenerate these first: " +
                ", ".join("sentence %d" % i for i in skipped))

    def _set_playback_active(self, active):
        active = bool(active)
        self._playback_active = active
        self.btn_play.setEnabled(not active)
        self.btn_stop.setEnabled(active)
        if hasattr(self, "sentences_view"):
            self.sentences_view.set_playing(active)
        self._update_shortcut_hints()

    def _finish_playback(self, token):
        if int(token) != self._playback_token:
            return
        reset_timeline = self._playback_timeline_start is not None
        self._playback_timer.stop()
        self._playback_finish_timer.stop()
        self._playback_timeline_start = None
        self._playback_highlights = []
        self.sentences_view.set_playing_item()
        if reset_timeline:
            self.waveform.set_playhead(0.0)
        self._set_playback_active(False)
        self.statusBar().showMessage("Status: playback finished")

    def _finish_scheduled_playback(self):
        self._finish_playback(self._playback_finish_token)

    def _advance_playhead(self):
        if (self._playback_timeline_start is None and
                not self._playback_highlights):
            return
        elapsed = self._playback_elapsed.elapsed() / 1000.0
        if self._playback_timeline_start is not None:
            when = self._playback_timeline_start + elapsed
            self.waveform.set_playhead(when)
            self._follow_playhead_if_needed(when)
        active = next((item for item in self._playback_highlights
                       if item[0] <= elapsed < item[1]), None)
        if active is None:
            self.sentences_view.set_playing_item()
        else:
            self.sentences_view.set_playing_item(active[2], active[3])

    def _on_follow_playhead_toggled(self, enabled):
        self.cfg["follow_playhead"] = bool(enabled)
        if enabled and self._playback_active:
            self._follow_playhead_if_needed(self.waveform.playhead_time())
        self._persist_config()

    def _set_join_overlay_visible(self, enabled):
        self.cfg["show_rendered_joins"] = bool(enabled)
        if hasattr(self, "waveform"):
            self.waveform.set_join_overlay_visible(enabled)
            self._refresh_join_overlay_controls()
        self._persist_config()

    def _refresh_join_overlay_controls(self):
        """Keep requested join handles scoped to native non-Legacy renders."""
        if not hasattr(self, "waveform"):
            return
        active = (
            self.sentences[self._active_sentence_index]
            if 0 <= self._active_sentence_index < len(self.sentences)
            else None)
        if active is not None:
            self.waveform.set_requested_join_settings(
                active.get("join_settings"))
        legacy = bool(self._fault_mode().get("legacy_joins"))
        self.waveform.set_join_overlay_editable(bool(
            self.current is not None and
            self._engine() == "festival_wsl" and
            not legacy))

    def _on_follow_spoken_sentence_toggled(self, enabled):
        self.cfg["follow_spoken_sentence"] = bool(enabled)
        if enabled and self._playback_highlights:
            self._advance_playhead()
        self._persist_config()

    def _follow_playhead_if_needed(self, when):
        if not self.follow_playhead.isChecked():
            return False
        viewbox = self.waveform.plot.getViewBox()
        start, end = viewbox.viewRange()[0]
        width = max(.01, end - start)
        tolerance = max(1e-6, width * 1e-6)
        when = float(when)
        if start - tolerance <= when <= end + tolerance:
            return False
        new_start = max(WAVEFORM_LEFT_LIMIT, when - width * 0.08)
        viewbox.setXRange(new_start, new_start + width, padding=0)
        return True

    def _start_playback(self, samples, sr, timeline_start=None,
                        highlights=None):
        samples = np.asarray(samples, np.float32)
        sr = int(sr)
        if self._playback_active:
            try:
                self.player.stop()
            except Exception:
                pass
        self.player.play(samples, sr)
        self._playback_token += 1
        token = self._playback_token
        self._playback_timeline_start = (
            None if timeline_start is None else float(timeline_start))
        self._playback_highlights = list(highlights or [])
        if (self._playback_timeline_start is not None or
                self._playback_highlights):
            self._playback_elapsed.restart()
            self._playback_timer.start()
            self._advance_playhead()
        self._set_playback_active(True)
        duration_ms = max(1, int(round(len(samples) * 1000.0 / max(1, sr))))
        self._playback_finish_token = token
        self._playback_finish_timer.start(duration_ms)
        self.statusBar().showMessage(
            "Status: playing... (%s)" % self.player.mode)

    def on_play(self):
        if self._playback_active:
            self.on_stop()
            return
        if (0 <= self._active_sentence_index < len(self.sentences) and
                not self.sentences[self._active_sentence_index].get("rendered")):
            self.statusBar().showMessage("Status: generate this sentence first.")
            return
        try:
            samples, sr = self._output_audio()
            when = self.waveform.playhead_time()
            start = max(0, min(len(samples), int(round(when * sr))))
            if len(samples) - start <= 1:
                when, start = 0.0, 0
                self.waveform.set_playhead(0.0)
            self._start_playback(samples[start:], sr, timeline_start=when)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Playback", str(e))

    def on_stop(self):
        if self._batch_active:
            self._request_batch_cancel()
            if not self._playback_active:
                return
        tracking_timeline = self._playback_timeline_start is not None
        if tracking_timeline:
            self._advance_playhead()
        try:
            self.player.stop()
        except Exception:
            pass
        self._playback_token += 1
        self._playback_timer.stop()
        self._playback_finish_timer.stop()
        self._playback_timeline_start = None
        self._playback_highlights = []
        self.sentences_view.set_playing_item()
        self._set_playback_active(False)
        message = ("Status: stopped at %.2fs" % self.waveform.playhead_time()
                   if tracking_timeline else "Status: stopped")
        self.statusBar().showMessage(message)

    def on_export(self):
        samples, sr = self._output_audio()
        state_ready = (0 <= self._active_sentence_index < len(self.sentences)
                       and self.sentences[self._active_sentence_index].get(
                           "rendered"))
        if samples.size <= 1 or not state_ready:
            QtWidgets.QMessageBox.information(self, "Export", "Generate audio first.")
            return
        start_dir = (os.path.join(self._project_root, "exports")
                     if self._project_root else
                     self.backend.synth_output_dir() if self.backend else "")
        name = "output"
        if self.current and self.current.text:
            try:
                slug = (self.backend.sd.safe_name(self.current.text)
                        if self.backend else
                        re.sub(r"[^\w]+", "_", self.current.text,
                               flags=re.UNICODE).strip("_")[:48] or "output")
            except Exception:
                slug = "output"
            slug = str(slug or "output")[:48].rstrip("_. ") or "output"
            name = "%s_%s" % (self.current.lang or self._engine(), slug)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export WAV", os.path.join(start_dir, name + ".wav"),
            "WAV (*.wav)")
        if not path:
            return
        try:
            fc.write_wav(path, samples, sr)
            self.statusBar().showMessage("Status: exported " + os.path.basename(path))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", str(e))

    def _sentence_project_row(self, state):
        syn = state.get("synthesis")
        phrases = copy.deepcopy(state.get("phrases") or [])
        for phrase in phrases:
            phrase.pop("_speaker_name", None)
            phrase.pop("_speaker_icon", None)
        editor_segments = copy.deepcopy(state.get("editor_segments") or [])
        rendered_segments = copy.deepcopy(syn.segments if syn else [])
        phones = []
        for seg in editor_segments:
            phones.extend(p for p in seg.phone.replace(",", " ").split()
                          if p and p != "pau")
        if not phones and syn is not None:
            phones = list(syn.phones)
        if not phones:
            phones = [phone for phrase in (state.get("phrases") or [])
                      for phone in (phrase.get("phones") or [])
                      if phone != "pau"]
        return {
            "text": state.get("text") or "",
            "rendered_text": state.get("rendered_text",
                                        state.get("text") or ""),
            "input_mode": state.get("input_mode") or "text",
            "language": state.get("language") or "",
            "lang_code": state.get("lang_code") or "",
            "engine": state.get("engine") or "diphone",
            "voicebank": state.get("voicebank") or "",
            "speed": float(state.get("speed") or 1.0),
            "pitch_hz": float(state.get("pitch_hz") or 185.0),
            "rendered_pitch_hz": (
                None if state.get("rendered_pitch_hz") is None else
                float(state.get("rendered_pitch_hz"))
            ),
            "pitch_manual": bool(state.get("pitch_manual", False)),
            "fall_pct": float(state.get("fall_pct") or 0.0),
            "rendered_fall_pct": (
                None if state.get("rendered_fall_pct") is None else
                float(state.get("rendered_fall_pct"))
            ),
            "output_gain_db": float(state.get("output_gain_db") or 0.0),
            "applied_gain_db": state.get("applied_gain_db"),
            "pre_gain_peak": float(state.get("pre_gain_peak") or 0.0),
            "vocal_tract_length_ratio": float(
                state.get("vocal_tract_length_ratio", 1.0)),
            "chipmunk_range": bool(state.get("chipmunk_range", False)),
            "applied_vocal_tract_length_ratio": state.get(
                "applied_vocal_tract_length_ratio"),
            "fault_mode": dict(state.get("fault_mode") or {}),
            "join_settings": self._project_join_settings(
                state.get("join_settings")),
            "effective_join_settings": dict(getattr(
                syn, "join_settings", {}) if syn else {}),
            "pitch_fault_target": state.get("pitch_fault_target"),
            "parameter_mode": state.get("parameter_mode") or "timing",
            "view_mode": state.get("view_mode") or "speech",
            "needs_rerender": bool(state.get("needs_rerender")),
            "needs_generate": bool(state.get("needs_generate")),
            "pending_reason": str(state.get("pending_reason") or ""),
            "phrases": phrases,
            "phones": phones,
            "render_phones": list(
                getattr(syn, "render_phones", ()) if syn else ()),
            "special_phone_realizations": [
                dict(item) for item in
                (getattr(syn, "special_phone_realizations", ()) if syn else ())
            ],
            "segments": rendered_segments,
            "editor_segments": editor_segments,
            "timing_factors": list(state.get("timing_factors") or []),
            "generated_targets": list(syn.generated_targets if syn else []),
            "pitch_override": list(syn.pitch_override if syn else []),
            "intonation_blocks": [dict(b) for b in
                                  (syn.intonation_blocks if syn else [])],
            "pitch_mode": str(syn.pitch_mode if syn else ""),
            "unit_overrides": dict(syn.unit_overrides if syn else {}),
            "selected_units": dict(syn.selected_units if syn else {}),
            "target_pitchmarks": list(
                syn.target_pitchmarks if syn else []),
            "splice_records": [dict(item) for item in
                               (syn.splice_records if syn else [])],
            "frame_trajectory_records": [dict(item) for item in
                                         (getattr(
                                             syn,
                                             "frame_trajectory_records",
                                             ()) if syn else [])],
            "vowel_realizations": [dict(item) for item in
                                    (syn.vowel_realizations if syn else [])],
            "source_voicing_targets": list(
                syn.source_voicing_targets if syn else []),
            "generated_voicing_targets": list(
                syn.generated_voicing_targets if syn else []),
            "voicing_override": list(
                syn.voicing_override if syn else []),
            "voicing_mode": str(syn.voicing_mode if syn else ""),
            "voicing_diagnostics": [dict(item) for item in
                                     (syn.voicing_diagnostics if syn else [])],
            "vocal_tract_requested_ratio": float(getattr(
                syn, "vocal_tract_requested_ratio", 1.0) if syn else 1.0),
            "generated_vocal_tract_targets": list(getattr(
                syn, "generated_vocal_tract_targets", []) if syn else []),
            "vocal_tract_override": list(getattr(
                syn, "vocal_tract_override", []) if syn else []),
            "applied_vocal_tract_targets": list(getattr(
                syn, "applied_vocal_tract_targets", []) if syn else []),
            "vocal_tract_mode": str(getattr(
                syn, "vocal_tract_mode", "") if syn else ""),
            "vocal_tract_diagnostics": copy.deepcopy(getattr(
                syn, "vocal_tract_diagnostics", []) if syn else []),
            "asaxi_prosody": copy.deepcopy(getattr(
                syn, "asaxi_prosody", {}) if syn else {}),
            "english_syllabification": copy.deepcopy(getattr(
                syn, "english_syllabification", {}) if syn else {}),
            "japanese_state": je.normalize_edit_state(
                state.get("japanese_state")),
            "asaxi_state": asaxi_editing.normalize_edit_state(
                state.get("asaxi_state")),
        }

    def _state_from_project_row(self, row, project_dir=None):
        state = self._new_sentence_state(row.get("text") or "")
        state.update({
            "rendered_text": row.get("rendered_text",
                                      row.get("text") or ""),
            "input_mode": row.get("input_mode") or "text",
            "engine": row.get("engine") or "diphone",
            "language": row.get("language") or "",
            "lang_code": row.get("lang_code") or "",
            "voicebank": row.get("voicebank") or "",
            "speed": float(row.get("speed") or 1.0),
            "pitch_hz": float(row.get("pitch_hz") or self.pitch.value()),
            "rendered_pitch_hz": float(
                row.get("rendered_pitch_hz")
                if row.get("rendered_pitch_hz") is not None else
                row.get("pitch_hz") or self.pitch.value()),
            "pitch_manual": bool(row.get("pitch_manual", False)),
            "fall_pct": float(row.get("fall_pct") or 0.0),
            "rendered_fall_pct": float(
                row.get("rendered_fall_pct")
                if row.get("rendered_fall_pct") is not None else
                row.get("fall_pct") or 0.0),
            "output_gain_db": float(row.get("output_gain_db") or 0.0),
            "applied_gain_db": row.get("applied_gain_db"),
            "pre_gain_peak": float(row.get("pre_gain_peak") or 0.0),
            "vocal_tract_length_ratio": float(
                row.get("vocal_tract_length_ratio", 1.0)),
            "chipmunk_range": bool(row.get("chipmunk_range", False)),
            "applied_vocal_tract_length_ratio": row.get(
                "applied_vocal_tract_length_ratio"),
            "applied_vocal_tract_targets": list(
                row.get("applied_vocal_tract_targets") or []),
            "fault_mode": dict(row.get("fault_mode") or {}),
            "join_settings": self._project_join_settings(
                row.get("join_settings")),
            "pitch_fault_target": row.get("pitch_fault_target"),
            "parameter_mode": row.get("parameter_mode") or "timing",
            "view_mode": row.get("view_mode") or "speech",
            "phrases": copy.deepcopy(row.get("phrases") or []),
            "timing_factors": list(row.get("timing_factors") or []),
            "needs_rerender": bool(row.get("needs_rerender", True)),
            "needs_generate": bool(row.get("needs_generate", False)),
            "pending_reason": str(row.get("pending_reason") or ""),
            "rendered": False,
            "cache_loaded": False,
            "japanese_state": je.normalize_edit_state(
                row.get("japanese_state")),
            "asaxi_state": asaxi_editing.normalize_edit_state(
                row.get("asaxi_state")),
        })
        segments = copy.deepcopy(row.get("segments") or [])
        if (not segments and row.get("phones") and
                not row.get("needs_rerender")):
            segments = fc.segments_from_durations(fc.class_seg_durs(
                row.get("phones") or [], state["speed"]))
        editor_segments = copy.deepcopy(
            row.get("editor_segments") or segments)
        state["editor_segments"] = editor_segments
        cached_samples = None
        cached_sr = 16000
        cache_rel = str(row.get("cache_wav") or "")
        if cache_rel and project_dir and not os.path.isabs(cache_rel):
            parent = os.path.abspath(project_dir)
            cache_path = os.path.abspath(os.path.join(parent, cache_rel))
            try:
                contained = (os.path.commonpath((parent, cache_path)) ==
                             parent)
            except ValueError:
                contained = False
            if contained and os.path.isfile(cache_path):
                try:
                    cached_samples, cached_sr = fc.read_wav(cache_path)
                except (OSError, ValueError, wave.Error):
                    cached_samples = None
        if segments or row.get("phones") or cached_samples is not None:
            sr = int(cached_sr)
            duration = segments[-1].end if segments else 0.01
            generated = list(row.get("generated_targets") or [])
            samples = (np.asarray(cached_samples, np.float32)
                       if cached_samples is not None else
                       np.zeros(max(1, int(round(duration * sr))),
                                np.float32))
            legacy_ratio = float(row.get(
                "vocal_tract_requested_ratio",
                row.get("vocal_tract_length_ratio", 1.0)))
            generated_tract = list(
                row.get("generated_vocal_tract_targets") or
                [(0.0, 1.0), (float(duration), 1.0)])
            tract_override = list(row.get("vocal_tract_override") or [])
            tract_mode = str(row.get("vocal_tract_mode") or "")
            if not tract_override and abs(legacy_ratio - 1.0) > 1e-10:
                tract_override = [
                    (0.0, legacy_ratio), (float(duration), legacy_ratio)
                ]
                tract_mode = "curve"
            applied_tract = list(
                row.get("applied_vocal_tract_targets") or
                tract_override or generated_tract)
            syn = fc.Synthesis(
                samples, sr,
                segments, text=state["text"], lang=state["lang_code"],
                voicebank=state["voicebank"],
                phones=list(row.get("phones") or []),
                render_phones=list(row.get("render_phones") or []),
                special_phone_realizations=[
                    dict(item) for item in
                    (row.get("special_phone_realizations") or [])
                ],
                targets=list(generated), generated_targets=list(generated),
                pitch_override=list(row.get("pitch_override") or []),
                intonation_blocks=[dict(b) for b in
                                   (row.get("intonation_blocks") or [])],
                pitch_mode=str(row.get("pitch_mode") or ""),
                unit_overrides=dict(row.get("unit_overrides") or {}),
                selected_units=dict(row.get("selected_units") or {}),
                target_pitchmarks=[float(value) for value in
                                   (row.get("target_pitchmarks") or [])],
                splice_records=[dict(item) for item in
                                (row.get("splice_records") or [])],
                frame_trajectory_records=[dict(item) for item in
                                          (row.get(
                                              "frame_trajectory_records") or
                                           [])],
                join_settings=dict(
                    row.get("effective_join_settings") or
                    row.get("join_settings") or {}),
                vowel_realizations=[dict(item) for item in
                                     (row.get("vowel_realizations") or [])],
                source_voicing_targets=list(
                    row.get("source_voicing_targets") or []),
                generated_voicing_targets=list(
                    row.get("generated_voicing_targets") or []),
                voicing_override=list(row.get("voicing_override") or []),
                voicing_mode=str(row.get("voicing_mode") or ""),
                voicing_diagnostics=[dict(item) for item in
                                     (row.get("voicing_diagnostics") or [])],
                vocal_tract_requested_ratio=float(row.get(
                    "vocal_tract_requested_ratio",
                    row.get("vocal_tract_length_ratio", 1.0))),
                vocal_tract_length_ratio=float(row.get(
                    "applied_vocal_tract_length_ratio",
                    row.get("vocal_tract_length_ratio", 1.0)) or 1.0),
                chipmunk_range=bool(row.get("chipmunk_range", False)),
                generated_vocal_tract_targets=generated_tract,
                vocal_tract_override=tract_override,
                applied_vocal_tract_targets=applied_tract,
                vocal_tract_mode=tract_mode,
                vocal_tract_diagnostics=copy.deepcopy(
                    row.get("vocal_tract_diagnostics") or []),
                asaxi_prosody=copy.deepcopy(
                    row.get("asaxi_prosody") or {}),
                english_syllabification=copy.deepcopy(
                    row.get("english_syllabification") or {}),
                output_bit_depth=int(
                    state["fault_mode"].get("bit_depth") or 0))
            state["synthesis"] = syn
            if "rendered_text" not in row and syn.text:
                state["rendered_text"] = str(syn.text)
            applied = state.get("applied_gain_db")
            if applied is None:
                applied = float(state.get("output_gain_db") or 0.0)
                state["applied_gain_db"] = applied
            pre_gain_peak = float(state.get("pre_gain_peak") or 0.0)
            if pre_gain_peak <= 0.0 and samples.size:
                pre_gain_peak = (float(np.max(np.abs(samples))) /
                                 (10.0 ** (float(applied) / 20.0)))
                state["pre_gain_peak"] = pre_gain_peak
            syn.applied_gain_db = float(applied)
            syn.pre_gain_peak = pre_gain_peak
            state["preview_audio"] = syn.samples
            state["preview_sr"] = sr
            if cached_samples is not None:
                state["rendered"] = True
                state["needs_rerender"] = bool(
                    row.get("needs_rerender", False))
                state["cache_loaded"] = True
        return state

    def on_save_project(self, path=None):
        if isinstance(path, bool):
            path = None
        if path is None and self._project_root:
            path = self._project_root
        if path is None:
            suggested = str(Path.cwd() / "FestVox Project")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Project Folder As", suggested,
                "Project folder (*)")
        if not path:
            return False
        try:
            self._capture_active_sentence()
            project_root = Path(path).expanduser()
            if project_root.name.lower() == fc.PROJECT_MANIFEST_NAME:
                project_root = project_root.parent
            elif project_root.suffix.lower() == ".json":
                project_root = project_root.with_suffix("")
            project_root = fc.prepare_project_folder(project_root).resolve()
            cache_dir = project_root / "cache"
            rows = []
            for index, state in enumerate(self.sentences):
                row = self._sentence_project_row(state)
                cached_audio, cached_sr = self._state_audio(state)
                if state.get("rendered") and cached_audio.size > 1:
                    filename = "sentence_%04d.wav" % (index + 1)
                    cache_path = cache_dir / filename
                    fc.write_wav(cache_path, cached_audio, cached_sr)
                    row["cache_wav"] = "cache/" + filename
                rows.append(row)
            fc.save_project_folder(
                project_root, rows, self._active_sentence_index,
                settings={
                    "phrase_pauses_ms": fc.normalize_phrase_pauses_ms(
                        self.cfg.get("phrase_pauses_ms")),
                })
            expected_cache = {
                "sentence_%04d.wav" % (index + 1)
                for index, state in enumerate(self.sentences)
                if state.get("rendered") and self._state_audio(state)[0].size > 1
            }
            for stale in cache_dir.glob("sentence_*.wav"):
                if stale.name not in expected_cache:
                    stale.unlink()
            self._project_root = str(project_root)
            self.statusBar().showMessage(
                "Status: saved project " + project_root.name)
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))
            return False

    def on_open_project(self, path=None):
        if isinstance(path, bool):
            path = None
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Project JSON", "",
                "FestVox project (project.json);;JSON (*.json)")
        if not path:
            return False
        if Path(path).is_dir():
            QtWidgets.QMessageBox.warning(
                self, "Open project",
                "Select the project's project.json file, not its folder.")
            return False
        return self._open_project_path(path)

    def _open_project_path(self, path):
        try:
            d = fc.load_project(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open error", str(e))
            return False
        if (int(d.get("version") or 0) < 4 or
                str(d.get("layout") or "") != "folder" or
                not d.get("_project_root")):
            QtWidgets.QMessageBox.warning(
                self, "Open project",
                "Only version-4 FestVox project.json files are supported.")
            return False
        project_settings = dict(d.get("settings") or {})
        if "phrase_pauses_ms" in project_settings:
            self.cfg["phrase_pauses_ms"] = fc.normalize_phrase_pauses_ms(
                project_settings.get("phrase_pauses_ms"))
        rows = d.get("sentences") if isinstance(d.get("sentences"), list) \
            else [d]
        if not rows:
            QtWidgets.QMessageBox.warning(
                self, "Open project", "This project contains no sentences.")
            return False
        self._switching_sentence = True
        project_dir = (str(d.get("_project_root") or "") or
                       os.path.dirname(str(d.get("_project_manifest") or
                                           os.path.abspath(path))))
        self.sentences = [self._state_from_project_row(row, project_dir)
                          for row in rows]
        self._active_sentence_index = max(0, min(
            len(self.sentences) - 1, int(d.get("active_sentence") or 0)))
        self._refresh_sentence_selector(self._active_sentence_index)
        self._switching_sentence = False
        self._project_root = str(d.get("_project_root") or "")
        self._restore_sentence(self._active_sentence_index)
        self._refresh_sentences_view()
        missing = [index for index, state in enumerate(self.sentences)
                   if not state.get("cache_loaded") and
                   (state.get("synthesis") is not None or
                    str(state.get("text") or "").strip())]
        cached = sum(bool(state.get("cache_loaded"))
                     for state in self.sentences)
        self._refresh_sentences_view()
        self._refresh_gain_controls()
        self.statusBar().showMessage(
            "Status: opened %s (%d cached%s)" %
            (Path(project_dir).name, cached,
             ", %d need rendering" % len(missing) if missing else ""))
        return True

    def on_import_text_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import text as sentence entries", "",
            "Text files (*.txt *.md);;All files (*)")
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            QtWidgets.QMessageBox.critical(self, "Text import", str(error))
            return
        pieces = fc.split_document_sentences(text)
        if not pieces:
            QtWidgets.QMessageBox.information(
                self, "Text import", "The file contains no sentence text.")
            return
        self._capture_active_sentence()
        has_content = any(str(state.get("text") or "").strip()
                          for state in self.sentences)
        replace = False
        if has_content:
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Import text")
            box.setText("Add %d sentence entries to this project?" %
                        len(pieces))
            append_button = box.addButton("Append",
                                          QtWidgets.QMessageBox.AcceptRole)
            replace_button = box.addButton("Replace project",
                                           QtWidgets.QMessageBox.DestructiveRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.setDefaultButton(append_button)
            box.exec_()
            if box.clickedButton() is replace_button:
                replace = True
            elif box.clickedButton() is not append_button:
                return
        imported = [self._new_sentence_state(piece) for piece in pieces]
        if replace or not has_content:
            self.sentences = imported
            index = 0
        else:
            index = len(self.sentences)
            self.sentences.extend(imported)
        self._active_sentence_index = index
        self._refresh_sentence_selector(index)
        self._restore_sentence(index)
        self.statusBar().showMessage(
            "Status: imported %d sentences from %s" %
            (len(pieces), os.path.basename(path)))

    # -- voicebank / engine management ----------------------------------------
    def on_add_voice_folder(self):
        if not self._need_backend():
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select a diphone DB folder (contains dic/diphone_index.json)")
        if not d:
            return
        try:
            name = self.backend.add_voicebank_dir(d)
        except fc.BackendError as e:
            QtWidgets.QMessageBox.critical(self, "Voicebank", str(e))
            return
        self._variant_cache.clear()
        self._persist_config()
        self._refresh_voicebanks(keep=name)
        self.statusBar().showMessage(
            "Status: added '%s' (%d diphones) -- saved to config.json"
            % (name, self.backend.db_size(name)))

    def on_set_festvox_config(self):
        if not self._need_backend():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Locate festvox.json", self.backend.fcfg_path or "",
            "JSON (*.json)")
        if not path:
            return
        self.cfg["festvox_config"] = path
        try:
            self.backend.reload_festvox_config()
        except fc.BackendError as e:
            QtWidgets.QMessageBox.critical(self, "festvox.json", str(e))
            return
        self._variant_cache.clear()
        self._persist_config()
        self._refresh_voicebanks()
        self.statusBar().showMessage("Status: festvox.json = " + path)

    def on_reload_voicebanks(self):
        if not self._need_backend():
            return
        try:
            self.backend.reload_festvox_config()
        except fc.BackendError as e:
            QtWidgets.QMessageBox.critical(self, "festvox.json", str(e))
            return
        report = None
        if self._engine() == "festival_wsl":
            report = self._refresh_configured_voice_roots(show_errors=True)
        self._variant_cache.clear()
        self._refresh_voicebanks()
        changes = 0 if not report else sum(len(report.get(key) or []) for key in
                                           ("added", "updated", "removed"))
        self.statusBar().showMessage(
            "Status: voicebanks reloaded (%d discovered changes)" % changes)

    def _select_voicebank_name(self, name):
        for row in range(self.voicebank.count()):
            item = self.voicebank.item(row)
            if item.data(Qt.UserRole) == name:
                self.voicebank.setCurrentItem(item)
                return True
        return False

    def on_voicebank_manager(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Voicebank Manager")
        dlg.resize(820, 430)
        layout = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTreeWidget()
        table.setHeaderLabels(["Voicebank", "Location", "Source", "Status"])
        table.setRootIsDecorated(False)
        table.setAlternatingRowColors(True)
        table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        table.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        table.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        table.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        table.header().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        layout.addWidget(table, 1)

        buttons = QtWidgets.QHBoxLayout()
        add_windows = QtWidgets.QPushButton("Add Windows folder...")
        add_wsl = QtWidgets.QPushButton("Add WSL folder...")
        refresh = QtWidgets.QPushButton("Refresh")
        delete = QtWidgets.QPushButton("Delete...")
        delete.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_TrashIcon))
        close = QtWidgets.QPushButton("Close")
        buttons.addWidget(add_windows)
        buttons.addWidget(add_wsl)
        buttons.addWidget(refresh)
        buttons.addStretch(1)
        buttons.addWidget(delete)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        def populate(keep=""):
            table.clear()
            try:
                voices = self._ab().voicebanks()
            except Exception:
                voices = []
            if self._engine() == "festival_wsl":
                configured = set(self.fest.fcfg().get("voices") or {})
                configured.update(str(name) for name in
                                  (self.fest.fcfg().get(
                                      "installed_voices") or []))
            else:
                configured = set(self.cfg.get("extra_voicebanks") or {})
            configured.difference_update(fc.BUILTIN_FESTIVAL_VOICES)
            selected = None
            for voice in voices:
                item = QtWidgets.QTreeWidgetItem([
                    str(voice.get("name") or ""),
                    str(voice.get("dir") or "(Festival built-in)"),
                    str(voice.get("source") or ""),
                    "Ready" if voice.get("ok") else "Missing",
                ])
                name = str(voice.get("name") or "")
                item.setData(0, Qt.UserRole, name)
                item.setData(0, Qt.UserRole + 1, name in configured)
                table.addTopLevelItem(item)
                if name == keep:
                    selected = item
            if selected is None and table.topLevelItemCount():
                selected = table.topLevelItem(0)
            if selected is not None:
                table.setCurrentItem(selected)

        def selected_names():
            selected_items = list(table.selectedItems())
            if not selected_items and table.currentItem() is not None:
                selected_items.append(table.currentItem())
            return [
                str(table.topLevelItem(row).data(0, Qt.UserRole) or "")
                for row in range(table.topLevelItemCount())
                if any(table.topLevelItem(row) is item
                       for item in selected_items)
            ]

        def update_delete():
            items = table.selectedItems()
            delete.setEnabled(bool(items) and all(
                bool(item.data(0, Qt.UserRole + 1)) for item in items))
            delete.setText(
                "Delete %d..." % len(items) if len(items) > 1 else "Delete...")

        def do_refresh():
            names = selected_names()
            keep = names[0] if names else ""
            if self._engine() == "festival_wsl":
                self._refresh_configured_voice_roots(show_errors=True)
            self._refresh_voicebanks(keep=keep)
            populate(keep)

        def do_add_windows():
            before = self._current_voicebank()
            if self._engine() == "festival_wsl":
                self.on_add_fest_voice_folder()
            else:
                self.on_add_voice_folder()
            populate(self._current_voicebank() or before)

        def do_add_wsl():
            before = self._current_voicebank()
            self.on_add_fest_voice_wsl()
            populate(self._current_voicebank() or before)

        def do_delete():
            names = selected_names()
            if not names:
                return
            self.on_uninstall_voicebanks(names)
            populate()

        table.currentItemChanged.connect(lambda *_args: update_delete())
        table.itemSelectionChanged.connect(update_delete)
        add_windows.clicked.connect(do_add_windows)
        add_wsl.clicked.connect(do_add_wsl)
        refresh.clicked.connect(do_refresh)
        delete.clicked.connect(do_delete)
        close.clicked.connect(dlg.accept)
        add_wsl.setEnabled(self._engine() == "festival_wsl")
        populate(self._current_voicebank() or "")
        update_delete()
        dlg.exec_()

    def on_uninstall_voicebank(self):
        name = self._current_voicebank()
        if not name:
            QtWidgets.QMessageBox.information(
                self, "Uninstall voicebank", "Select a voicebank first.")
            return
        self.on_uninstall_voicebanks([name])

    def on_uninstall_voicebanks(self, names):
        names = list(dict.fromkeys(
            str(name) for name in (names or []) if str(name)))
        if not names:
            return []
        backend = self._ab()
        infos = []
        try:
            infos = [backend.voicebank_removal_info(name) for name in names]
        except (fc.BackendError, AttributeError) as e:
            QtWidgets.QMessageBox.warning(self, "Uninstall voicebank", str(e))
            return []

        existing = [info for info in infos if info.get("exists", True)]
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Critical if existing else
                    QtWidgets.QMessageBox.Warning)
        box.setWindowTitle(
            "Permanent file deletion" if existing else
            "Remove missing voicebanks")
        box.setText(
            "Permanently uninstall %d voicebank%s?" %
            (len(infos), "" if len(infos) == 1 else "s")
            if existing else
            "Remove %d missing voicebank entr%s?" %
            (len(infos), "y" if len(infos) == 1 else "ies"))
        lines = [
            "%s\n  %s%s" % (
                info["name"], info["path"],
                "  (entry only; folder is missing)"
                if not info.get("exists", True) else "")
            for info in infos
        ]
        warning = (
            "The generated folders listed below will be permanently deleted, "
            "including audio, pitchmarks, indexes, and Scheme files.\n\n"
            if existing else
            "Only the saved registrations below will be removed. No files "
            "exist at these locations.\n\n")
        box.setInformativeText(
            warning + "\n\n".join(lines) +
            "\n\nThis cannot be undone. Source UTAU banks are protected and "
            "cannot be targeted by this action.")
        confirm = box.addButton(
            ("Delete %d voicebank%s" %
             (len(infos), "" if len(infos) == 1 else "s"))
            if existing else
            ("Remove %d entr%s" %
             (len(infos), "y" if len(infos) == 1 else "ies")),
            QtWidgets.QMessageBox.DestructiveRole if existing else
            QtWidgets.QMessageBox.AcceptRole)
        cancel = box.addButton(QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(cancel)
        box.exec_()
        if box.clickedButton() is not confirm:
            return []

        removed = []
        errors = []
        for info in infos:
            try:
                backend.uninstall_voicebank(
                    info["name"], delete_files=bool(info.get("exists", True)))
                removed.append(info["name"])
            except (fc.BackendError, AttributeError) as error:
                errors.append("%s: %s" % (info["name"], error))
        if removed:
            self._variant_cache.clear()
            self._persist_config()
        self._refresh_voicebanks()
        if errors:
            QtWidgets.QMessageBox.critical(
                self, "Uninstall incomplete",
                "Some selected voicebanks could not be removed:\n\n" +
                "\n".join(errors))
        self.statusBar().showMessage(
            "Status: removed %d voicebank%s%s" %
            (len(removed), "" if len(removed) == 1 else "s",
             "; %d failed" % len(errors) if errors else ""))
        return removed

    def _on_input_mode(self):
        phones = self.input_mode.currentData() == "phones"
        self.text.setPlaceholderText(
            "space-separated phones, e.g.  hh eh l ow pau l eh m"
            if phones else "text; inline exact phones in [brackets], [pau]=pause")
        self.statusBar().showMessage(
            "Status: input = " + ("direct phonemes" if phones else "text"))
        self._update_fault_availability()
        if not self._switching_sentence and getattr(self, "sentences", None):
            state = self.sentences[self._active_sentence_index]
            state["input_mode"] = self.input_mode.currentData() or "text"
            self._mark_active_pending("generate", "Input mode changed")

    def _on_engine_changed(self):
        eng = self._engine()
        self.cfg["engine"] = eng
        fest = eng == "festival_wsl"
        self.pitch.setEnabled(fest)
        self._update_fault_availability()
        self._update_parameter_availability()
        self.lang.setToolTip(
            "Festival engine: English uses the voice's English front end "
            "(voice_*_en,\nbuilt by build_festival_voice.py; kal_diphone is "
            "natively English);\nAsaxi uses the voice's own letter rules; "
            "Japanese is g2p'd on this side." if fest else "")
        if fest:
            self._refresh_configured_voice_roots()
        self._refresh_voicebanks()
        if not self._switching_sentence and getattr(self, "sentences", None):
            state = self.sentences[self._active_sentence_index]
            state["engine"] = eng
            self._mark_active_pending("generate", "Synthesis engine changed")
        self._persist_config()
        self.statusBar().showMessage(
            "Status: engine = " + ("Festival via WSL (multisyn-capable)"
                                   if fest else "diphone (pure Python)"))

    def on_add_fest_voice_folder(self):
        root = self.fest.generated_voice_root()
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select a festvox / Multisyn voice folder "
                  "(contains festvox/*.scm)", root)
        if not d:
            return
        try:
            d = self.fest.validate_generated_voice_location(d)
            info = self.fest.scan_voice_dir(d)
        except fc.BackendError as e:
            QtWidgets.QMessageBox.critical(self, "Festival voice", str(e))
            return
        name = self.fest.add_voice(info)
        self._variant_cache.clear()
        self._persist_config()
        if self._engine() == "festival_wsl":
            self._refresh_voicebanks(keep=name)
        note = "" if info.get("scm") else \
            "  (no festvox/*.scm auto-detected -- edit config.json if it fails)"
        self.statusBar().showMessage(
            "Status: added Festival voice '%s' -> %s%s"
            % (name, info.get("voice"), note))

    def on_add_fest_voice_wsl(self):
        path, ok = QtWidgets.QInputDialog.getText(
            self, "Festival voice (WSL path)",
            "Voice folder INSIDE the WSL filesystem\n"
            "(e.g. /home/you/voices/asaxi_multisyn -- recommended for "
            "Multisyn,\nloading from /mnt/<drive> is slow):")
        if not ok or not path.strip():
            return
        self.statusBar().showMessage("Status: scanning the folder inside WSL...")
        QtWidgets.QApplication.processEvents()
        try:
            info = self.fest.scan_voice_dir_wsl(path.strip())
        except fc.BackendError as e:
            QtWidgets.QMessageBox.critical(self, "Festival voice (WSL)", str(e))
            self.statusBar().showMessage("Status: Ready")
            return
        name = self.fest.add_voice(info)
        self._variant_cache.clear()
        self._persist_config()
        if self._engine() == "festival_wsl":
            self._refresh_voicebanks(keep=name)
        self.statusBar().showMessage(
            "Status: added Festival voice '%s' -> %s"
            % (name, info.get("voice")))

    def on_scan_fest_voices(self):
        self.statusBar().showMessage(
            "Status: asking Festival (WSL) for its voice list...")
        QtWidgets.QApplication.processEvents()
        try:
            voices = self.fest.list_installed_voices()
        except fc.BackendError as e:
            QtWidgets.QMessageBox.critical(self, "Scan Festival voices", str(e))
            self.statusBar().showMessage("Status: Ready")
            return
        if not voices:
            QtWidgets.QMessageBox.information(
                self, "Scan Festival voices",
                "Festival reported no voices.\nInside WSL, install some, "
                "e.g.:  sudo apt install festvox-kallpc16k")
            self.statusBar().showMessage("Status: Ready")
            return
        have = set(self.fest.fcfg().get("installed_voices") or [])
        self.fest.fcfg()["installed_voices"] = sorted(have | set(voices))
        self._persist_config()
        if self._engine() == "festival_wsl":
            self._refresh_voicebanks()
        self.statusBar().showMessage(
            "Status: Festival knows %d voices (%d new) -- saved"
            % (len(voices), len(set(voices) - have)))

    def on_wsl_settings(self):
        f = self.fest.fcfg()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("WSL / Festival settings")
        dlg.setMinimumWidth(560)
        dlg.setMaximumWidth(760)
        form = QtWidgets.QFormLayout(dlg)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow
        )
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)

        wsl_exe = QtWidgets.QLineEdit(f.get("wsl_exe") or "")
        wsl_exe.setPlaceholderText("auto (wsl.exe on PATH)")
        distro = QtWidgets.QLineEdit(f.get("distro") or "")
        distro.setPlaceholderText("default distro")
        fbin = QtWidgets.QLineEdit(f.get("festival_bin") or "festival")
        voice_root = QtWidgets.QLineEdit(self.fest.generated_voice_root())
        voice_root.setObjectName("generatedVoiceRoot")
        voice_root.setAccessibleName("Generated voice root")
        voice_root.setMinimumWidth(340)
        voice_root.setMaximumWidth(520)
        btn_root = QtWidgets.QPushButton("...")
        btn_root.setFixedWidth(28)

        def browse_root():
            path = QtWidgets.QFileDialog.getExistingDirectory(
                dlg, "Generated voice root", voice_root.text().strip()
            )
            if path:
                voice_root.setText(path)
        btn_root.clicked.connect(browse_root)
        root_row = QtWidgets.QHBoxLayout()
        root_row.addWidget(voice_root, 1); root_row.addWidget(btn_root)
        wsl_voice_root = QtWidgets.QLineEdit(
            f.get("generated_voice_wsl_root") or "")
        wsl_voice_root.setObjectName("generatedVoiceWSLRoot")
        wsl_voice_root.setPlaceholderText(
            "optional, e.g. /home/you/voices")
        wsl_voice_root.setAccessibleName("WSL voice scan root")
        runtime_root = QtWidgets.QLabel("")
        runtime_root.setObjectName("generatedVoiceRuntimePath")
        runtime_root.setAccessibleName("Derived Festival runtime path")
        runtime_root.setWordWrap(True)
        runtime_root.setMinimumWidth(340)
        runtime_root.setMaximumWidth(520)
        runtime_root.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )

        def refresh_runtime_root():
            runtime_root.setText(
                fc.win_to_wsl_path(voice_root.text().strip())
            )
        voice_root.textChanged.connect(refresh_runtime_root)
        refresh_runtime_root()
        extra = QtWidgets.QLineEdit(f.get("extra_scheme") or "")
        extra.setPlaceholderText("optional .scm loaded before every synth")
        btn_extra = QtWidgets.QPushButton("...")
        btn_extra.setFixedWidth(28)

        def browse_extra():
            p, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg, "Extra scheme file", "", "Scheme (*.scm);;All files (*)")
            if p:
                extra.setText(p)
        btn_extra.clicked.connect(browse_extra)
        extra_row = QtWidgets.QHBoxLayout()
        extra_row.addWidget(extra, 1); extra_row.addWidget(btn_extra)

        timeout = QtWidgets.QSpinBox()
        timeout.setRange(30, 1800); timeout.setSuffix(" s")
        timeout.setValue(int(f.get("timeout_s") or 180))

        form.addRow("WSL executable:", wsl_exe)
        form.addRow("Distro (wsl -d):", distro)
        form.addRow("festival binary (in WSL):", fbin)
        form.addRow("Windows voice scan root:", root_row)
        form.addRow("Runtime path:", runtime_root)
        form.addRow("WSL voice scan root:", wsl_voice_root)
        form.addRow("Extra scheme:", extra_row)
        form.addRow("Timeout:", timeout)

        result = QtWidgets.QLabel("")
        result.setWordWrap(True)
        test = QtWidgets.QPushButton("Test connection")

        def do_test():
            f["wsl_exe"] = wsl_exe.text().strip()
            f["distro"] = distro.text().strip()
            f["festival_bin"] = fbin.text().strip() or "festival"
            f["generated_voice_root"] = voice_root.text().strip()
            f["generated_voice_wsl_root"] = wsl_voice_root.text().strip()
            f["timeout_s"] = timeout.value()
            result.setText("testing... (a cold WSL start can take a moment)")
            QtWidgets.QApplication.processEvents()
            ok, msg = self.fest.available()
            result.setText(("OK: " if ok else "FAILED: ") + msg)
            result.setStyleSheet("color:%s" % ("#006000" if ok else "#A00000"))
        test.clicked.connect(do_test)
        form.addRow(test, result)

        note = QtWidgets.QLabel(
            "Immediate child folders under both scan roots are refreshed "
            "automatically. Missing auto-discovered folders leave the list; "
            "manual registrations are preserved. Kal is mirrored into the "
            "Windows root and loaded through its derived WSL path.")
        note.setWordWrap(True)
        note.setMaximumWidth(680)
        note.setStyleSheet("color:#555")
        form.addRow(note)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        f["wsl_exe"] = wsl_exe.text().strip()
        f["distro"] = distro.text().strip()
        f["festival_bin"] = fbin.text().strip() or "festival"
        f["generated_voice_root"] = voice_root.text().strip()
        self.fest.generated_voice_root()
        f["generated_voice_wsl_root"] = wsl_voice_root.text().strip()
        f["extra_scheme"] = extra.text().strip()
        f["timeout_s"] = timeout.value()
        self._persist_config()
        self._refresh_configured_voice_roots(show_errors=True)
        self._refresh_voicebanks()
        self.statusBar().showMessage(
            "Status: WSL / Festival settings saved; voice roots refreshed")

    def on_locate_engine(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Locate synth_diphone.py", "", "Python (*.py)")
        if not path:
            return
        self.cfg["synth_diphone_dir"] = os.path.dirname(path)
        self._init_backend()
        if self.backend:
            self._persist_config()
            self._populate_from_backend()
            self.statusBar().showMessage("Status: engine loaded from " + path)
        else:
            QtWidgets.QMessageBox.critical(self, "Engine", str(self.backend_err))

    def _set_japanese_duration_model(self, mode):
        self._set_japanese_synthesis_option(
            "japanese_duration_model", mode
        )

    def _set_japanese_synthesis_option(self, key, value):
        allowed = {
            "japanese_duration_model": {"contextual", "legacy"},
            "japanese_vowel_devoicing": {"contextual", "legacy"},
            "japanese_devoicing_renderer": {
                "auto", "source_filter", "residual", "natural_source",
                "shortened_voiced"
            },
        }
        value = str(value).casefold()
        if key not in allowed or value not in allowed[key]:
            return
        if self.cfg.get(key) == value:
            return
        self.cfg[key] = value
        self._persist_config()
        labels = {
            "japanese_duration_model": "duration model",
            "japanese_vowel_devoicing": "vowel devoicing",
            "japanese_devoicing_renderer": "devoicing renderer",
        }
        affected = 0
        for state in self.sentences:
            if str(state.get("language") or "").casefold() != "japanese":
                continue
            if state.get("rendered") or state.get("synthesis") is not None:
                self._set_state_pending(
                    state, "rerender", "Japanese synthesis option changed"
                )
                affected += 1
        self._refresh_pending_ui()
        self._refresh_sentences_view()
        self.statusBar().showMessage(
            "Status: Japanese %s set to %s%s" % (
                labels[key], value,
                " -- Re-render applies it to %d sentence%s" % (
                    affected, "" if affected == 1 else "s"
                ) if affected else "",
            )
        )

    def on_phrase_pauses(self):
        before = fc.normalize_phrase_pauses_ms(
            self.cfg.get("phrase_pauses_ms"))
        dialog = PhrasePauseDialog(before, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return False
        after = fc.normalize_phrase_pauses_ms(dialog.values())
        if before == after:
            return False
        self.cfg["phrase_pauses_ms"] = after
        self._persist_config()
        affected = 0
        for state in self.sentences:
            if (state.get("engine") == "festival_wsl"
                    and state.get("input_mode", "text") == "text"
                    and (state.get("rendered") or
                         state.get("synthesis") is not None)):
                self._set_state_pending(
                    state, "rerender", "Phrase pause duration changed")
                affected += 1
        self._refresh_pending_ui()
        self._refresh_sentences_view()
        self.statusBar().showMessage(
            "Status: phrase pauses saved%s" %
            (" -- Re-render applies them to %d sentence%s" %
             (affected, "" if affected == 1 else "s")
             if affected else ""))
        return True

    def on_advanced(self):
        adv = dict(self.cfg.get("advanced") or {})
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Advanced synthesis settings")
        form = QtWidgets.QFormLayout(dlg)

        def dspin(val, lo, hi, step=1.0):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(1)
            s.setValue(float(val))
            return s

        cross = dspin(adv.get("crossfade_ms", 15.0), 0, 100)
        edge = dspin(adv.get("edge_fade_ms", 8.0), 0, 50)
        half = dspin(adv.get("half_ms", 150.0), 30, 500, 10)
        form.addRow("Diphone crossfade (ms):", cross)
        form.addRow("Utterance edge fade (ms):", edge)
        form.addRow("Phone window (ms/side):", half)
        note = QtWidgets.QLabel(
            "Phone window caps audio kept per side of each diphone boundary;\n"
            "the speed slider divides it. These map to the engine's\n"
            "CROSSFADE_MS / EDGE_FADE_MS / HALF_MS constants.")
        note.setStyleSheet("color:#555")
        form.addRow(note)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        self.cfg["advanced"] = {"crossfade_ms": cross.value(),
                                "edge_fade_ms": edge.value(),
                                "half_ms": half.value()}
        self._persist_config()
        self.statusBar().showMessage(
            "Status: synthesis settings saved -- regenerate to hear them")


def parse_launch_args(argv=None):
    """Separate FestVox launch options from arguments intended for Qt."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project", default="")
    return parser.parse_known_args(
        list(sys.argv[1:] if argv is None else argv)
    )


def main():
    launch, qt_arguments = parse_launch_args()
    try:
        cfg = fc.load_config(CONFIG_PATH)
    except Exception as e:
        cfg = fc.load_config("nonexistent")  # defaults
        sys.stderr.write("config.json problem, using defaults: %s\n" % e)
    configure_qt_high_dpi()
    app = QtWidgets.QApplication([sys.argv[0], *qt_arguments])
    app.setQuitOnLastWindowClosed(True)
    app.setFont(select_ui_font())
    app.setStyle(ArrowProxyStyle(QtWidgets.QStyleFactory.create("Fusion")))
    app.setStyleSheet(XP_QSS)
    win = MainWindow(cfg)
    app.aboutToQuit.connect(win._shutdown_resources)
    app.lastWindowClosed.connect(app.quit)
    win.destroyed.connect(lambda *_args: app.quit())
    win.show()
    if launch.project:
        project = str(Path(launch.project).expanduser())
        QtCore.QTimer.singleShot(
            0, lambda path=project: win._open_project_path(path)
        )
    exit_code = app.exec_()
    win._shutdown_resources()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
# end of festvox_gui

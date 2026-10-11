"""Κοινά widgets του GUI: νήματα εργασίας, γραμμή ρύθμισης με επεξήγηση."""
from __future__ import annotations
import traceback
from PySide6.QtCore import Qt, QObject, QRunnable, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox, QSpinBox, QCheckBox
from .config import Params


# ------------------------------------------------------------------------------------------------
class Signals(QObject):
    progress = Signal(float, str)
    done = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.setAutoDelete(False)          # το κρατάμε ζωντανό εμείς (αλλιώς χάνονται τα σήματα)
        self.fn = fn
        self.s = Signals()

    def run(self):
        try:
            res = self.fn(lambda f, m="": self.s.progress.emit(float(f), str(m)))
            self.s.done.emit(res)
        except Exception as e:  # noqa
            self.s.error.emit(f"{e}\n\x00\n{traceback.format_exc()[-1800:]}")


# ------------------------------------------------------------------------------------------------
class ParamRow(QWidget):
    """Μία ρύθμιση: ετικέτα + slider/πεδίο + (προαιρετικά ορατή) επεξήγηση από κάτω."""
    help_labels: list = []
    help_visible = True

    @classmethod
    def set_help_visible(cls, on: bool):
        cls.help_visible = bool(on)
        for lb in list(cls.help_labels):
            try:
                lb.setVisible(cls.help_visible)
            except RuntimeError:                      # το widget έχει καταστραφεί
                cls.help_labels.remove(lb)

    def __init__(self, params: Params, spec, on_change=None):
        super().__init__()
        key, label, kind, lo, hi, step, tip = spec
        self.params, self.key, self.kind, self.lo, self.hi, self.on_change = params, key, kind, lo, hi, on_change
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(1)
        top = QWidget()
        lay = QHBoxLayout(top)
        lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(top)
        if tip:
            hl = QLabel(tip)
            hl.setWordWrap(True)
            hl.setStyleSheet("color:#8a90a6; font-size:11px; margin-left:4px;")
            hl.setVisible(ParamRow.help_visible)
            ParamRow.help_labels.append(hl)
            outer.addWidget(hl)
        lb = QLabel(label)
        lb.setMinimumWidth(170)
        lb.setToolTip(tip)
        lay.addWidget(lb)
        val = getattr(params, key)
        if kind == "b":
            self.w = QCheckBox()
            self.w.setChecked(bool(val))
            self.w.toggled.connect(self._set)
            lay.addWidget(self.w)
            lay.addStretch(1)
        elif kind == "i":
            self.w = QSpinBox()
            self.w.setRange(int(lo), int(hi))
            self.w.setSingleStep(int(step) or 1)
            self.w.setValue(int(val))
            self.w.valueChanged.connect(self._set)
            lay.addWidget(self.w)
        else:
            self.sl = QSlider(Qt.Horizontal)
            self.sl.setRange(0, 1000)
            self.w = QDoubleSpinBox()
            self.w.setRange(lo, hi)
            self.w.setDecimals(3 if step < 0.01 else 2)
            self.w.setSingleStep(step)
            self.w.setValue(float(val))
            self.sl.setValue(self._to_sl(float(val)))
            self.sl.valueChanged.connect(self._from_sl)
            self.w.valueChanged.connect(self._from_spin)
            lay.addWidget(self.sl, 1)
            lay.addWidget(self.w)
        self.setToolTip(tip)

    def _to_sl(self, v):
        return int(round((v - self.lo) / (self.hi - self.lo) * 1000))

    def _from_sl(self, i):
        v = self.lo + (self.hi - self.lo) * i / 1000.0
        self.w.blockSignals(True)
        self.w.setValue(v)
        self.w.blockSignals(False)
        self._set(self.w.value())

    def _from_spin(self, v):
        self.sl.blockSignals(True)
        self.sl.setValue(self._to_sl(v))
        self.sl.blockSignals(False)
        self._set(v)

    def _set(self, v):
        cur = getattr(self.params, self.key)
        setattr(self.params, self.key, type(cur)(v))
        if self.on_change:
            self.on_change(self.key)

    def refresh(self):
        v = getattr(self.params, self.key)
        self.w.blockSignals(True)
        if self.kind == "b":
            self.w.setChecked(bool(v))
        else:
            self.w.setValue(v)
            if self.kind == "f":
                self.sl.blockSignals(True)
                self.sl.setValue(self._to_sl(float(v)))
                self.sl.blockSignals(False)
        self.w.blockSignals(False)



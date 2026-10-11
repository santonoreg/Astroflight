"""Καρτέλα «Αυτόματο»: ένα κουμπί που τρέχει ASTAP → StarNet → Gaia → 3D σκηνή → πτήση, με πρόοδο ανά βήμα
και επεξήγηση σε κάθε ρύθμιση."""
from __future__ import annotations
import os, time, traceback
from PySide6.QtCore import Qt, QObject, QRunnable, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QGroupBox,
                               QDoubleSpinBox, QComboBox, QProgressBar, QFileDialog, QScrollArea, QFormLayout)

from .widgets import ParamRow
from .pipeline import AutoOptions, Reporter, run_auto, STEPS
from .runner import CancelToken, Cancelled
from . import wcs_solve as ws
from . import starnet as sn

HELP_CSS = "color:#8a90a6; font-size:11px;"
ICONS = {"pending": ("○", "#6b7080"), "running": ("◔", "#6aa0ff"), "ok": ("✔", "#4cc27a"), "warn": ("⚠", "#e0b040"),
         "skip": ("–", "#8a90a6"), "fail": ("✖", "#e06060")}


class AutoSignals(QObject):
    step = Signal(str, str, str)
    frac = Signal(float, str)
    log = Signal(str)
    done = Signal(object)
    error = Signal(str)
    cancelled = Signal()


class AutoWorker(QRunnable):
    def __init__(self, proj, opts, sig: AutoSignals, token: CancelToken):
        super().__init__()
        self.setAutoDelete(False)
        self.proj, self.opts, self.sig, self.token = proj, opts, sig, token

    def run(self):
        rep = Reporter(on_step=self.sig.step.emit, on_frac=lambda f, m="": self.sig.frac.emit(float(f), str(m)),
                       on_log=self.sig.log.emit)
        try:
            res = run_auto(self.proj, self.opts, rep, self.token)
            self.sig.done.emit(res)
        except Cancelled:
            self.sig.cancelled.emit()
        except Exception as e:  # noqa
            self.sig.error.emit(f"{e}\n\x00\n{traceback.format_exc()[-1800:]}")


def _help(text: str) -> QLabel:
    l = QLabel(text)
    l.setWordWrap(True)
    l.setStyleSheet(HELP_CSS)
    l.setOpenExternalLinks(True)
    ParamRow.help_labels.append(l)
    l.setVisible(ParamRow.help_visible)
    return l


class AutoPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.proj = win.proj
        self.token: CancelToken | None = None
        self.sig: AutoSignals | None = None
        self.worker: AutoWorker | None = None
        self.t0 = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._build()

    # ------------------------------------------------------------------ UI
    def _group(self, title):
        g = QGroupBox(title)
        lay = QVBoxLayout(g)
        return g, lay

    def _tool_row(self, lay, title, what, attr, finder, placeholder, install_html):
        """Γραμμή εργαλείου: κατάσταση ✔/✖ + διαδρομή + αναζήτηση."""
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 6)
        top = QHBoxLayout()
        chk = QCheckBox(title)
        chk.setChecked(True)
        f = QFont()
        f.setBold(True)
        chk.setFont(f)
        top.addWidget(chk)
        status = QLabel()
        top.addWidget(status, 1)
        v.addLayout(top)
        v.addWidget(_help(what))
        row = QHBoxLayout()
        edit = QLineEdit(getattr(self.proj, attr) or "")
        edit.setMinimumWidth(120)
        edit.setPlaceholderText(placeholder)
        row.addWidget(edit, 1)
        b1 = QPushButton("…")
        b1.setToolTip("Επιλογή αρχείου")
        b2 = QPushButton("Αυτόματη αναζήτηση")
        row.addWidget(b1)
        row.addWidget(b2)
        v.addLayout(row)
        inst = QLabel(install_html)
        inst.setWordWrap(True)
        inst.setOpenExternalLinks(True)
        inst.setStyleSheet(HELP_CSS)
        v.addWidget(inst)
        lay.addWidget(box)

        def upd():
            path = edit.text().strip()
            ok = bool(path) and os.path.isfile(path)
            status.setText(("✔ βρέθηκε" if ok else "✖ δεν βρέθηκε") + (f"  ({os.path.basename(path)})" if ok else ""))
            status.setStyleSheet("color:#4cc27a;" if ok else "color:#e06060;")
            inst.setVisible(not ok)
            if ok:
                setattr(self.proj, attr, path)
                self.win.settings.setValue(attr.replace("_path", ""), path)
        edit.textChanged.connect(lambda _t: upd())

        def browse():
            p, _ = QFileDialog.getOpenFileName(self, f"{title} – εκτελέσιμο", os.path.dirname(edit.text()) if edit.text() else "")
            if p:
                edit.setText(p)
        b1.clicked.connect(browse)

        def find():
            p = finder()
            if p:
                edit.setText(p)
            else:
                self.win._log(f"{title}: δεν βρέθηκε σε γνωστές θέσεις· πάτα «…» και διάλεξέ το χειροκίνητα.")
        b2.clicked.connect(find)
        upd()
        return chk, edit, upd

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        w = QWidget()
        lay = QVBoxLayout(w)
        intro = QLabel("Άνοιξε τη φωτογραφία σου και πάτα «Έναρξη». Η εφαρμογή κάνει μόνη της όλα τα βήματα: "
                       "βρίσκει πού κοιτάζει η φωτογραφία (ASTAP), χωρίζει τα αστέρια από το νεφέλωμα (StarNet), "
                       "βρίσκει πραγματικές αποστάσεις αστεριών (Gaia), χτίζει το 3D και σχεδιάζει την πτήση. "
                       "Αν λείπει κάποιο εργαλείο ή αποτύχει ένα βήμα, συνεχίζει με εφεδρική λύση και σε ενημερώνει.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#8d93a8;")
        lay.addWidget(intro)

        # --- εικόνα
        b = QPushButton("📂  Άνοιγμα αστροφωτογραφίας (TIFF / PNG / JPG)…")
        b.setMinimumHeight(40)
        b.clicked.connect(self.open_image)
        lay.addWidget(b)
        self.img_label = QLabel("Δεν έχει φορτωθεί εικόνα.")
        self.img_label.setWordWrap(True)
        lay.addWidget(self.img_label)

        # --- εκτέλεση
        row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Έναρξη αυτόματης επεξεργασίας")
        self.btn_run.setMinimumHeight(46)
        f = QFont()
        f.setBold(True)
        f.setPointSize(11)
        self.btn_run.setFont(f)
        self.btn_run.clicked.connect(self.start)
        self.btn_cancel = QPushButton("■ Ακύρωση")
        self.btn_cancel.setMinimumHeight(46)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        row.addWidget(self.btn_run, 3)
        row.addWidget(self.btn_cancel, 1)
        lay.addLayout(row)

        # --- πρόοδος
        g, gl = self._group("Πρόοδος")
        self.step_lbls: dict[str, QLabel] = {}
        self.step_state: dict[str, tuple[str, str]] = {}
        from .pipeline import STEP_TITLES
        for sid, title, _w in STEPS:
            l = QLabel()
            l.setWordWrap(True)
            l.setTextFormat(Qt.RichText)
            self.step_lbls[sid] = l
            gl.addWidget(l)
            self._set_step(sid, "pending", "")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        gl.addWidget(self.bar)
        self.time_lbl = QLabel("")
        gl.addWidget(self.time_lbl)
        lay.addWidget(g)
        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        self.result_lbl.setTextFormat(Qt.RichText)
        lay.addWidget(self.result_lbl)
        row = QHBoxLayout()
        self.btn_view = QPushButton("🎬  Δες την πτήση")
        self.btn_view.setMinimumHeight(40)
        self.btn_view.setVisible(False)
        self.btn_view.clicked.connect(self.win._show_flight)
        self.btn_export = QPushButton("💾  Εξαγωγή βίντεο MP4…")
        self.btn_export.setMinimumHeight(40)
        self.btn_export.setVisible(False)
        self.btn_export.clicked.connect(self.win.export_mp4)
        row.addWidget(self.btn_view)
        row.addWidget(self.btn_export)
        lay.addLayout(row)


        sep = QLabel("Ρυθμίσεις")
        sep.setStyleSheet("font-weight:bold; margin-top:8px;")
        lay.addWidget(sep)
        # --- εργαλεία
        g, gl = self._group("1 · Εργαλεία")
        self.chk_astap, self.ed_astap, self.upd_astap = self._tool_row(
            gl, "ASTAP – αστρομετρία (plate solve)",
            "Αναγνωρίζει τα αστέρια της φωτογραφίας και βρίσκει ΠΟΥ στον ουρανό κοιτάζει και με ποια κλίμακα. "
            "Χωρίς αυτό δεν μπορούμε να ξέρουμε ποια αστέρια είναι κοντά και ποια μακριά.",
            "astap_path", ws.find_astap, r"π.χ. C:\Program Files\astap\astap.exe",
            'Κατέβασέ το από <a href="https://hnsky.org/astap.htm">hnsky.org/astap.htm</a> και εγκατάστησε ΚΑΙ μία βάση αστεριών (D50 ή D20).')
        self.chk_starnet, self.ed_starnet, self.upd_starnet = self._tool_row(
            gl, "StarNet – αφαίρεση αστεριών",
            "Αφαιρεί τα αστέρια από την εικόνα ώστε να μείνει μόνο το νεφέλωμα. Τα αστέρια μπαίνουν ξανά ως 3D σημεία στις σωστές αποστάσεις. "
            "Χωρίς StarNet χρησιμοποιείται απλούστερη μέθοδος (αφήνει σκούρα σημάδια γύρω από φωτεινά αστέρια).",
            "starnet_path", sn.find_starnet, r"π.χ. C:\Program Files\StarNet2\bin\starnet2.exe",
            'Κατέβασέ το (starnet2, command line) από <a href="https://starnetastro.com/cli-tools/starnet/">starnetastro.com</a>.')
        self.chk_gaia = QCheckBox("Gaia – πραγματικές αποστάσεις αστεριών (χρειάζεται internet)")
        self.chk_gaia.setChecked(True)
        f = QFont()
        f.setBold(True)
        self.chk_gaia.setFont(f)
        gl.addWidget(self.chk_gaia)
        gl.addWidget(_help("Κατεβάζει από τον κατάλογο Gaia DR3 τις μετρημένες αποστάσεις των αστεριών του πεδίου σου. "
                           "Απαιτεί επιτυχημένο plate solve. Αν δεν υπάρχει internet χρησιμοποιούνται εύλογες συνθετικές αποστάσεις."))
        lay.addWidget(g)

        # --- πληροφορίες εικόνας
        g, gl = self._group("2 · Τι ξέρεις για την εικόνα (όλα προαιρετικά)")
        form = QFormLayout()
        self.sp_scale = QDoubleSpinBox()
        self.sp_scale.setRange(0.0, 600.0)
        self.sp_scale.setDecimals(3)
        self.sp_scale.setSpecialValueText("άγνωστη")
        self.sp_scale.setSuffix(' "/px')
        form.addRow("Κλίμακα εικόνας:", self.sp_scale)
        gl.addLayout(form)
        gl.addWidget(_help("Πόσα δευτερόλεπτα τόξου (\") καλύπτει κάθε pixel. Αν την ξέρεις, ο ASTAP λύνει πολύ πιο γρήγορα· αν είναι «άγνωστη» "
                           "θα την ανακαλύψει μόνος του. Είναι απαραίτητη μόνο όταν ΔΕΝ υπάρχει ASTAP (τότε καθορίζει το μέγεθος του αντικειμένου)."))
        opt = QHBoxLayout()
        self.sp_focal = QDoubleSpinBox()
        self.sp_focal.setRange(1, 20000)
        self.sp_focal.setValue(500)
        self.sp_focal.setSuffix(" mm")
        self.sp_pix = QDoubleSpinBox()
        self.sp_pix.setRange(0.5, 50)
        self.sp_pix.setDecimals(2)
        self.sp_pix.setValue(3.76)
        self.sp_pix.setSuffix(" µm")
        btn = QPushButton("Υπολογισμός από τα οπτικά")
        btn.clicked.connect(lambda: self.sp_scale.setValue(ws.scale_from_optics(self.sp_focal.value(), self.sp_pix.value())))
        opt.addWidget(QLabel("Εστιακή:"))
        opt.addWidget(self.sp_focal)
        opt.addWidget(QLabel("Pixel:"))
        opt.addWidget(self.sp_pix)
        opt.addWidget(btn)
        gl.addLayout(opt)
        gl.addWidget(_help("Κλίμακα = 206.265 × μέγεθος pixel (µm) ÷ εστιακή απόσταση (mm). Αν έκανες binning ή μίκρυνες την εικόνα, πολλαπλασίασε ανάλογα."))
        form = QFormLayout()
        self.ed_object = QLineEdit()
        self.ed_object.setPlaceholderText("π.χ. Rosette Nebula, M42")
        form.addRow("Όνομα αντικειμένου:", self.ed_object)
        gl.addLayout(form)
        gl.addWidget(_help("Βοηθά τον ASTAP να ξέρει πού να ψάξει στον ουρανό (ταχύτερη και πιο αξιόπιστη λύση) και την εφαρμογή να βρει την απόσταση του "
                           "αντικειμένου. Γνωστά ονόματα βρίσκονται και χωρίς internet."))
        form = QFormLayout()
        self.cb_stretch = QComboBox()
        self.cb_stretch.addItems(["Αυτόματα", "Ναι – έχει ήδη τεντωθεί (τελική εικόνα)", "Όχι – γραμμικά δεδομένα (linear)"])
        form.addRow("Η εικόνα είναι τεντωμένη;", self.cb_stretch)
        gl.addLayout(form)
        gl.addWidget(_help("Οι τελικές, επεξεργασμένες εικόνες είναι «τεντωμένες» (non-linear). Τα ακατέργαστα stacked δεδομένα είναι σκοτεινά και γραμμικά. "
                           "Η εφαρμογή το εντοπίζει μόνη της από τη φωτεινότητα· άλλαξέ το μόνο αν το μαντέψει λάθος."))
        row = QHBoxLayout()
        self.chk_dist = QCheckBox("Απόσταση αντικειμένου αυτόματα")
        self.chk_dist.setChecked(True)
        self.sp_dist = QDoubleSpinBox()
        self.sp_dist.setRange(5, 30000)
        self.sp_dist.setDecimals(0)
        self.sp_dist.setValue(self.proj.p.object_distance_pc)
        self.sp_dist.setSuffix(" pc")
        self.sp_dist.setEnabled(False)
        self.chk_dist.toggled.connect(lambda on: self.sp_dist.setEnabled(not on))
        row.addWidget(self.chk_dist)
        row.addWidget(self.sp_dist)
        gl.addLayout(row)
        gl.addWidget(_help("Πόσο μακριά είναι το νεφέλωμα (1 pc = 3.26 έτη φωτός). Αυτόματα: από τα δεδομένα Gaia ή από κατάλογο γνωστών αντικειμένων. "
                           "Επηρεάζει την κλίμακα του 3D χώρου και πόσο γρήγορα «περνούν» τα αστέρια."))
        lay.addWidget(g)

        # --- πτήση
        g, gl = self._group("3 · Πτήση (αλλάζει ζωντανά μετά την επεξεργασία)")
        self.prows = []
        for spec in (
            ("duration", "Διάρκεια (s)", "f", 4, 120, 0.5, "Πόσο διαρκεί το βίντεο. Η ταχύτητα προσαρμόζεται αυτόματα."),
            ("dive_zoom", "Zoom προσέγγισης (×)", "f", 1.0, 8.0, 0.1,
             "Πόσες φορές μεγαλύτερο φαίνεται το νεφέλωμα στο κάδρο όταν η κάμερα φτάνει στην είσοδό του. 1 = μένει όσο στη φωτογραφία (έντονη παράλλαξη, "
             "όλο το αντικείμενο φαίνεται), μεγάλο = το νεφέλωμα «ξεχειλίζει» από το κάδρο."),
            ("dive_fov_max", "Μέγιστο οπτικό πεδίο (°)", "f", 20, 100, 1,
             "Πόσο ανοίγει το κάδρο όταν η κάμερα μπαίνει μέσα στα νέφη. Μεγαλύτερο = βλέπεις περισσότερα στρώματα γύρω σου, μικρότερο = πιο «τηλεσκοπικό» πλάνο."),
            ("dive_depth", "Βάθος τέλους (0 μπροστά … 1 πίσω)", "f", 0.0, 1.0, 0.01,
             "Πόσο μέσα στα νέφη τελειώνει η πτήση. Μικρή τιμή = μένεις στα εξωτερικά, φωτεινά στρώματα· μεγάλη = μπαίνεις βαθιά (πιο σκοτεινά)."),
            ("dive_auto_poi", "Αυτόματος στόχος", "b", 0, 1, 0,
             "Η κάμερα πετά προς τη φωτεινότερη περιοχή. Ξετσέκαρέ το για να διαλέξεις εσύ σημείο (βήμα «Πτήση & βίντεο» → στόχος X/Y)."),
            ("depth_extent", "Συνολικό βάθος νεφελώματος (× πλάτος)", "f", 0.2, 3, 0.05,
             "Πόσο «βαθύ» είναι το νεφέλωμα. Μεγαλύτερο = πιο έντονο 3D και παράλλαξη (εφαρμόζεται με νέα «Έναρξη» – γρήγορη, δεν ξανατρέχει StarNet)."),
        ):
            r = ParamRow(self.proj.p, spec, self.win._param_changed)
            self.win.rows.append(r)
            gl.addWidget(r)
        lay.addWidget(g)

        lay.addStretch(1)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(w)
        outer.addWidget(sa)
        self._inputs = [self.chk_astap, self.ed_astap, self.chk_starnet, self.ed_starnet, self.chk_gaia, self.sp_scale,
                        self.sp_focal, self.sp_pix, self.ed_object, self.cb_stretch, self.chk_dist, btn]

    # ------------------------------------------------------------------ βοηθητικά
    def _set_step(self, sid, state, msg):
        ic, col = ICONS[state]
        title = [t for k, t, _ in STEPS if k == sid][0]
        txt = f'<span style="color:{col}; font-weight:bold;">{ic}</span>&nbsp; <b>{title}</b>'
        if msg:
            txt += f'<br><span style="color:#8a90a6; font-size:11px;">&nbsp;&nbsp;&nbsp;&nbsp;{msg}</span>'
        self.step_lbls[sid].setText(txt)
        self.step_state[sid] = (state, msg)

    def refresh_image(self):
        p = self.proj
        if p.image_path:
            self.img_label.setText(f"✔ {os.path.basename(p.image_path)}  –  {p.W}×{p.H}px"
                                   + (f"  (μικρύνθηκε ×{p.down:.2f})" if p.down < 1 else ""))
        else:
            self.img_label.setText("Δεν έχει φορτωθεί εικόνα.")
        for sid, _t, _w in STEPS:
            self._set_step(sid, "pending", "")
        self.bar.setValue(0)
        self.result_lbl.setText("")
        self.btn_view.setVisible(False)
        self.btn_export.setVisible(False)

    def set_busy(self, busy: bool):
        running = self.token is not None
        for w in self._inputs:
            w.setEnabled(not busy)
        self.btn_run.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy and running)

    def open_image(self):
        d = self.win.settings.value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "Άνοιγμα εικόνας", d, "Εικόνες (*.tif *.tiff *.png *.jpg *.jpeg)")
        if not path:
            return
        self.win.settings.setValue("last_dir", os.path.dirname(path))
        self.win.run_task(lambda prog: self.proj.load_image(path), lambda r: (self.win._after_load(False), self.refresh_image()),
                          "Φόρτωση εικόνας…")

    # ------------------------------------------------------------------ εκτέλεση
    def options(self) -> AutoOptions:
        scale = self.sp_scale.value() or None
        stretch = self.cb_stretch.currentIndex()
        p = self.proj.p
        if stretch == 1:
            p.stretched_input = True
        elif stretch == 2:
            p.stretched_input = False
        if not self.chk_dist.isChecked():
            p.object_distance_pc = float(self.sp_dist.value())
        return AutoOptions(use_astap=self.chk_astap.isChecked(), use_starnet=self.chk_starnet.isChecked(),
                           use_gaia=self.chk_gaia.isChecked(), scale_arcsec=scale, object_name=self.ed_object.text(),
                           auto_stretch=stretch == 0, auto_distance=self.chk_dist.isChecked())

    def start(self):
        if self.proj.img is None:
            self.win.statusBar().showMessage("Άνοιξε πρώτα μια εικόνα.")
            self.img_label.setText("⚠ Άνοιξε πρώτα μια αστροφωτογραφία (κουμπί από πάνω).")
            return
        for e in (self.ed_astap, self.ed_starnet):
            e.textChanged.emit(e.text())
        opts = self.options()
        for sid, _t, _w in STEPS:
            self._set_step(sid, "pending", "")
        self.result_lbl.setText("")
        self.btn_view.setVisible(False)
        self.btn_export.setVisible(False)
        self.bar.setValue(0)
        self.token = CancelToken()
        self.sig = AutoSignals()
        self.sig.step.connect(self._on_step)
        self.sig.frac.connect(self._on_frac)
        self.sig.log.connect(self.win._log)
        self.sig.done.connect(self._on_done)
        self.sig.error.connect(self._on_error)
        self.sig.cancelled.connect(self._on_cancelled)
        self.worker = AutoWorker(self.proj, opts, self.sig, self.token)
        self.t0 = time.time()
        self._timer.start(500)
        self.win._set_busy(True)
        self.btn_cancel.setEnabled(True)
        self.win.pool.start(self.worker)

    def cancel(self):
        if self.token:
            self.token.cancel()
            self.btn_cancel.setEnabled(False)
            self.time_lbl.setText("Ακύρωση… (περιμένω να σταματήσει το τρέχον βήμα)")

    def _tick(self):
        el = time.time() - self.t0
        self.time_lbl.setText(f"Χρόνος: {int(el) // 60}:{int(el) % 60:02d}")

    def _on_step(self, sid, state, msg):
        self._set_step(sid, state, msg)

    def _on_frac(self, f, msg):
        self.bar.setValue(int(f * 1000))
        if msg:
            self.win.statusBar().showMessage(msg)

    def _finish_run(self):
        self._timer.stop()
        self.token = None
        self.win._set_busy(False)

    def _on_done(self, summary):
        self._finish_run()
        self.bar.setValue(1000)
        el = summary.get("seconds", 0.0)
        parts = [f"<b>✔ Έτοιμο σε {int(el) // 60}:{int(el) % 60:02d}.</b>"]
        if summary.get("object"):
            parts.append(f"Αντικείμενο: {summary['object']}.")
        if summary["warnings"]:
            parts.append('<span style="color:#e0b040;">Προσοχή:</span><br>' + "<br>".join("• " + w for w in summary["warnings"]))
        self.result_lbl.setText("<br>".join(parts))
        self.btn_view.setVisible(True)
        self.btn_export.setVisible(True)
        self.win._auto_finished(summary)

    def _on_error(self, msg):
        self._finish_run()
        self.result_lbl.setText('<span style="color:#e06060;"><b>✖ Η επεξεργασία απέτυχε:</b><br>' + msg.split("\x00")[0].strip().replace("\n", "<br>") + "</span>")
        self.win._log("ΣΦΑΛΜΑ: " + msg.replace("\x00", ""))
        for sid, (st, m) in list(self.step_state.items()):
            if st == "running":
                self._set_step(sid, "fail", m)

    def _on_cancelled(self):
        self._finish_run()
        self.result_lbl.setText("Η επεξεργασία ακυρώθηκε.")
        for sid, (st, m) in list(self.step_state.items()):
            if st == "running":
                self._set_step(sid, "skip", "Ακυρώθηκε")
        self.time_lbl.setText("")

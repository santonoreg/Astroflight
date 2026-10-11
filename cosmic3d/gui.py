"""Γραφικό περιβάλλον του launcher (PyQt6): επιλογή εικόνας, ρυθμίσεις με επεξήγηση, πρόοδος, άνοιγμα των scripts."""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QSettings, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPalette, QPixmap
from PyQt6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QComboBox,
                             QVBoxLayout, QWidget)

from . import astap
from .optics import CAMERAS
from .prepare import Prepared, PrepareOptions, prepare, read_image
from .runner import CancelToken, Cancelled

HELP = "color:#8d93a8; font-size:11px;"


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    base, alt, text, hl = QColor(30, 31, 38), QColor(40, 42, 52), QColor(220, 222, 232), QColor(59, 95, 192)
    for role, col in ((QPalette.ColorRole.Window, QColor(37, 38, 46)), (QPalette.ColorRole.WindowText, text),
                      (QPalette.ColorRole.Base, base), (QPalette.ColorRole.AlternateBase, alt),
                      (QPalette.ColorRole.ToolTipBase, alt), (QPalette.ColorRole.ToolTipText, text), (QPalette.ColorRole.Text, text),
                      (QPalette.ColorRole.Button, QColor(50, 52, 64)), (QPalette.ColorRole.ButtonText, text),
                      (QPalette.ColorRole.Highlight, hl), (QPalette.ColorRole.HighlightedText, QColor(255, 255, 255)),
                      (QPalette.ColorRole.PlaceholderText, QColor(120, 124, 140)), (QPalette.ColorRole.Link, QColor(120, 170, 255))):
        pal.setColor(role, col)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(110, 113, 128))
    app.setPalette(pal)


def _help(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setOpenExternalLinks(True)
    lb.setStyleSheet(HELP)
    return lb


class _Sig(QObject):
    progress = pyqtSignal(float, str)
    log = pyqtSignal(str)
    done = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()


class _Job(QRunnable):
    def __init__(self, path, opts, token, sig):
        super().__init__()
        self.setAutoDelete(False)
        self.path, self.opts, self.token, self.sig = path, opts, token, sig

    def run(self):
        try:
            r = prepare(self.path, self.opts, progress=lambda f, m="": self.sig.progress.emit(float(f), str(m)),
                        log=self.sig.log.emit, cancel=self.token)
            self.sig.done.emit(r)
        except Cancelled:
            self.sig.cancelled.emit()
        except Exception as e:  # noqa
            self.sig.error.emit(f"{e}\n\x00\n{traceback.format_exc()[-1500:]}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmic 3D – CosmicView 3D / CosmicDepth 3D για οποιαδήποτε αστροφωτογραφία")
        self.resize(760, 980)
        self.settings = QSettings("Cosmic3D", "Launcher")
        self.pool = QThreadPool.globalInstance()
        self.image_path: str | None = None
        self.prepared: Prepared | None = None
        self.token: CancelToken | None = None
        self._job = None
        self._sig = None
        self._windows: list = []
        self._t0 = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._build()

    # ------------------------------------------------------------------ UI
    def _group(self, title: str):
        g = QGroupBox(title)
        return g, QVBoxLayout(g)

    def _build(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        w = QWidget()
        lay = QVBoxLayout(w)

        intro = QLabel("Διάλεξε μια αστροφωτογραφία. Η εφαρμογή κάνει ό,τι χρειάζονται τα δύο scripts ως είσοδο (αστρομετρική λύση μέσω ASTAP) "
                       "και τα ανοίγει. Δεν χρειάζεται Siril.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#8d93a8;")
        lay.addWidget(intro)

        # --- 1. εικόνα
        g, gl = self._group("1 · Εικόνα")
        b = QPushButton("📂  Επιλογή αστροφωτογραφίας (TIFF / PNG / JPG / FITS)…")
        b.setMinimumHeight(42)
        b.clicked.connect(self.pick_image)
        gl.addWidget(b)
        self.img_label = QLabel("Δεν έχει επιλεγεί εικόνα.")
        self.img_label.setWordWrap(True)
        gl.addWidget(self.img_label)
        self.thumb = QLabel()
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setMinimumHeight(10)
        gl.addWidget(self.thumb)
        gl.addWidget(_help("Μπορεί να είναι τελική (τεντωμένη) εικόνα ή linear. Αν είναι FITS που έχει ήδη λυθεί (π.χ. από το Siril) "
                           "δεν θα γίνει ξανά plate solve."))
        lay.addWidget(g)

        # --- 2. ASTAP
        g, gl = self._group("2 · ASTAP – αστρομετρία (plate solve)")
        self.astap_status = QLabel()
        gl.addWidget(self.astap_status)
        row = QHBoxLayout()
        self.astap_edit = QLineEdit(self.settings.value("astap", astap.find_astap() or "", type=str))
        self.astap_edit.setPlaceholderText(r"π.χ. C:\Program Files\astap\astap.exe")
        self.astap_edit.textChanged.connect(self._astap_changed)
        bb = QPushButton("…")
        bb.clicked.connect(self._browse_astap)
        bf = QPushButton("Αναζήτηση")
        bf.setToolTip("Ψάχνει τον ASTAP στις συνηθισμένες θέσεις εγκατάστασης")
        bf.clicked.connect(lambda: self.astap_edit.setText(astap.find_astap() or self.astap_edit.text()))
        row.addWidget(self.astap_edit, 1)
        row.addWidget(bb)
        row.addWidget(bf)
        gl.addLayout(row)
        gl.addWidget(_help("Τα scripts πρέπει να ξέρουν ΠΟΥ στον ουρανό κοιτάζει η φωτογραφία. Ο ASTAP αναγνωρίζει τα αστέρια και το βρίσκει. "
                           'Κατέβασέ τον από <a href="https://hnsky.org/astap.htm">hnsky.org/astap.htm</a> και εγκατάστησε ΚΑΙ μία βάση αστεριών (D50 ή D20).'))
        lay.addWidget(g)

        # --- 3. πληροφορίες
        g, gl = self._group("3 · Τι ξέρεις για την εικόνα (όλα προαιρετικά)")
        gl.addWidget(_help("<b>Δεν χρειάζεται να ξέρεις την κλίμακα.</b> Η εφαρμογή δοκιμάζει μόνη της πολλές κλίμακες (από 0.7 έως 80 δευτερόλεπτα τόξου ανά pixel). "
                           "Αν όμως της δώσεις τα οπτικά σου, ξεκινά από την ονομαστική κλίμακα και λύνει ταχύτερα. "
                           "Σε mosaic ή εικόνα που έχει μικρύνει/μεγαλώσει, η πραγματική κλίμακα διαφέρει – δεν πειράζει."))
        form = QFormLayout()
        self.cb_cam = QComboBox()
        self.cb_cam.addItem("— δεν ξέρω / άλλη —", 0.0)
        for name, um in CAMERAS.items():
            self.cb_cam.addItem(f"{name}  ({um} µm)", um)
        self.cb_cam.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cb_cam.setMinimumContentsLength(14)
        form.addRow("Κάμερα:", self.cb_cam)
        self.sp_pix = QDoubleSpinBox()
        self.sp_pix.setRange(0.0, 50.0)
        self.sp_pix.setDecimals(2)
        self.sp_pix.setSpecialValueText("άγνωστο")
        self.sp_pix.setSuffix(" µm")
        form.addRow("Μέγεθος pixel:", self.sp_pix)
        self.sp_focal = QDoubleSpinBox()
        self.sp_focal.setRange(0.0, 20000.0)
        self.sp_focal.setSpecialValueText("άγνωστη")
        self.sp_focal.setSuffix(" mm")
        form.addRow("Εστιακή απόσταση:", self.sp_focal)
        self.sp_bin = QSpinBox()
        self.sp_bin.setRange(1, 6)
        form.addRow("Binning:", self.sp_bin)
        self.lbl_nominal = QLabel("")
        form.addRow("Ονομαστική κλίμακα:", self.lbl_nominal)
        self.sp_scale = QDoubleSpinBox()
        self.sp_scale.setRange(0.0, 600.0)
        self.sp_scale.setDecimals(3)
        self.sp_scale.setSpecialValueText("άγνωστη")
        self.sp_scale.setSuffix(' "/px')
        form.addRow("Κλίμακα αν την ξέρεις:", self.sp_scale)
        gl.addLayout(form)
        gl.addWidget(_help("Ονομαστική κλίμακα = 206.265 × μέγεθος pixel (µm) × binning ÷ εστιακή (mm), π.χ. Atik 383L+ με 300 mm: 3.71″/px. "
                           "Ισχύει αν η εικόνα είναι στο αρχικό μέγεθος. Το «Κλίμακα αν την ξέρεις» υπερισχύει της ονομαστικής."))
        self.cb_cam.currentIndexChanged.connect(lambda _i: (self.sp_pix.setValue(self.cb_cam.currentData() or self.sp_pix.value()), self._nominal()))
        for sp in (self.sp_pix, self.sp_focal, self.sp_bin):
            sp.valueChanged.connect(lambda _v: self._nominal())
        # θυμάται τα οπτικά από τη φορά που τα έδωσες
        self.sp_pix.setValue(float(self.settings.value("pixel_um", 0.0)))
        self.sp_focal.setValue(float(self.settings.value("focal_mm", 0.0)))
        self.sp_bin.setValue(int(self.settings.value("binning", 1)))
        self._nominal()
        form = QFormLayout()
        self.ed_object = QLineEdit()
        self.ed_object.setPlaceholderText("π.χ. Rosette Nebula, M42, NGC 7000")
        form.addRow("Όνομα αντικειμένου:", self.ed_object)
        gl.addLayout(form)
        gl.addWidget(_help("Βοηθά τον ASTAP να ξέρει πού να ψάξει (ταχύτερη λύση) και δίνει το όνομα του στόχου στο CosmicView 3D, "
                           "ώστε να βρει σωστά την απόσταση. Η αναζήτηση ονόματος χρειάζεται internet (SIMBAD)."))
        self.chk_approx = QCheckBox("Αν αποτύχει ο ASTAP, χρησιμοποίησε προσεγγιστική λύση (όνομα αντικειμένου + κλίμακα)")
        self.chk_approx.setChecked(True)
        gl.addWidget(self.chk_approx)
        gl.addWidget(_help("Εφεδρική λύση: κέντρο από το όνομα που έγραψες, κλίμακα από τα οπτικά σου, βορράς πάνω. Αρκεί για το CosmicView 3D (πού κοιτάζει η φωτογραφία)· "
                           "οι θέσεις αντικειμένων στο CosmicDepth 3D μπορεί να μην ταιριάζουν ακριβώς. Σημαίνεται ΠΑΝΤΑ ως προσεγγιστική στο αποτέλεσμα."))
        self.chk_force = QCheckBox("Νέο plate solve (αγνόησε την αποθηκευμένη λύση)")
        gl.addWidget(self.chk_force)
        gl.addWidget(_help("Για την ίδια εικόνα η λύση αποθηκεύεται και δεν ξαναγίνεται. Τσέκαρέ το αν άλλαξες κλίμακα/όνομα ή η λύση βγήκε λάθος."))
        lay.addWidget(g)

        # --- 4. άνοιγμα
        g, gl = self._group("4 · Άνοιγμα")
        self.btn_view = QPushButton("🌌  CosmicView 3D – πού κοιτάζει η φωτογραφία στον Γαλαξία (πτήση «Journey»)")
        self.btn_depth = QPushButton("📊  CosmicDepth 3D – τρισδιάστατος χάρτης των αντικειμένων του πεδίου με τις αποστάσεις τους")
        for kind, btn in (("view", self.btn_view), ("depth", self.btn_depth)):
            btn.setMinimumHeight(46)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _=False, k=kind: self.open_script(k))
            gl.addWidget(btn)
        gl.addWidget(_help("Και τα δύο χρειάζονται internet (SIMBAD / Gaia) για τις αποστάσεις· οι απαντήσεις αποθηκεύονται και μετά δουλεύουν και offline. "
                           "Την πρώτη φορά μπορεί να εγκατασταθούν αυτόματα βιβλιοθήκες που λείπουν (plotly, astroquery κ.ά.)."))
        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        self.result_lbl.setTextFormat(Qt.TextFormat.RichText)
        gl.addWidget(self.result_lbl)
        lay.addWidget(g)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(110)
        lay.addWidget(self.log_box)
        lay.addStretch(1)

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.Shape.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setWidget(w)
        outer.addWidget(sa, 1)

        # --- σταθερό υποσέλιδο
        foot = QWidget()
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(8, 6, 8, 8)
        row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Προετοιμασία εικόνας (plate solve)")
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
        fl.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        fl.addWidget(self.bar)
        self.status = QLabel("")
        fl.addWidget(self.status)
        outer.addWidget(foot)
        self.setCentralWidget(central)
        self._astap_changed()

    # ------------------------------------------------------------------ βοηθητικά
    def log(self, s: str):
        self.log_box.appendPlainText(s)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _astap_changed(self, *_):
        p = self.astap_edit.text().strip()
        ok = bool(p) and os.path.isfile(p)
        dbs = astap.find_database(p) if ok else []
        if ok:
            self.settings.setValue("astap", p)
            self.astap_status.setText(f"✔ βρέθηκε ({os.path.basename(p)})  –  βάσεις αστεριών: " + (", ".join(d.upper() for d in dbs) if dbs else "ΚΑΜΙΑ (εγκατάστησε π.χ. D50)"))
            self.astap_status.setStyleSheet("color:#4cc27a;" if dbs else "color:#e0b040;")
        else:
            self.astap_status.setText("✖ δεν βρέθηκε – χωρίς ASTAP δουλεύουν μόνο FITS που έχουν ήδη αστρομετρική λύση")
            self.astap_status.setStyleSheet("color:#e06060;")

    def _nominal(self):
        from .optics import nominal_scale
        px, fl = self.sp_pix.value(), self.sp_focal.value()
        if px > 0 and fl > 0:
            n = nominal_scale(px, fl, self.sp_bin.value())
            self.lbl_nominal.setText(f"{n:.2f} \"/px")
        else:
            self.lbl_nominal.setText("— (δώσε κάμερα και εστιακή)")

    def _browse_astap(self):
        p, _ = QFileDialog.getOpenFileName(self, "ASTAP εκτελέσιμο", os.path.dirname(self.astap_edit.text()) if self.astap_edit.text() else "")
        if p:
            self.astap_edit.setText(p)

    def set_image(self, path: str):
        self.image_path = path
        self.prepared = None
        self.btn_view.setEnabled(False)
        self.btn_depth.setEnabled(False)
        self.result_lbl.setText("")
        try:
            rgb, hdr = read_image(path)
            _, H, W = rgb.shape
            has = hdr is not None and "CRVAL1" in hdr
            self.img_label.setText(f"{os.path.basename(path)}  –  {W}×{H}px" + ("  (έχει ήδη αστρομετρία)" if has else ""))
            step = max(1, max(H, W) // 380)
            small = np.clip(rgb[:, ::-1][:, ::step, ::step], 0, 1)             # πρώτη γραμμή = ΠΑΝΩ για εμφάνιση
            if float(np.median(rgb[:, ::8, ::8])) < 0.04:                    # linear εικόνα: απλό τέντωμα για προεπισκόπηση
                small = np.arcsinh(small * 60.0) / np.arcsinh(60.0)
            top = np.ascontiguousarray((small * 255).astype(np.uint8).transpose(1, 2, 0))
            qi = QImage(top.data, top.shape[1], top.shape[0], top.shape[1] * 3, QImage.Format.Format_RGB888).copy()
            self.thumb.setPixmap(QPixmap.fromImage(qi))
        except Exception as e:  # noqa
            self.img_label.setText(f"{os.path.basename(path)}  –  δεν μπόρεσα να τη διαβάσω: {e}")
            self.thumb.clear()

    def pick_image(self):
        d = self.settings.value("last_dir", "", type=str)
        p, _ = QFileDialog.getOpenFileName(self, "Άνοιγμα αστροφωτογραφίας", d,
                                           "Εικόνες (*.tif *.tiff *.png *.jpg *.jpeg *.fit *.fits *.fts);;Όλα (*)")
        if p:
            self.settings.setValue("last_dir", os.path.dirname(p))
            self.set_image(p)

    # ------------------------------------------------------------------ προετοιμασία
    def _set_busy(self, busy: bool):
        self.btn_run.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)

    def start(self):
        if not self.image_path:
            self.status.setText("⚠ Διάλεξε πρώτα μια εικόνα.")
            return
        self.settings.setValue("pixel_um", self.sp_pix.value())
        self.settings.setValue("focal_mm", self.sp_focal.value())
        self.settings.setValue("binning", self.sp_bin.value())
        opts = PrepareOptions(astap_path=self.astap_edit.text().strip() or None, scale_arcsec=self.sp_scale.value() or None,
                              pixel_um=self.sp_pix.value() or None, focal_mm=self.sp_focal.value() or None, binning=float(self.sp_bin.value()),
                              allow_approx=self.chk_approx.isChecked(), object_name=self.ed_object.text(), force=self.chk_force.isChecked())
        self.token = CancelToken()
        sig = self._sig = _Sig()
        sig.progress.connect(lambda f, m: (self.bar.setValue(int(f * 1000)), self.status.setText(m)))
        sig.log.connect(self.log)
        sig.done.connect(self._done)
        sig.error.connect(self._error)
        sig.cancelled.connect(self._cancelled)
        self._job = _Job(self.image_path, opts, self.token, sig)
        self.bar.setValue(0)
        self.result_lbl.setText("")
        self.btn_view.setEnabled(False)
        self.btn_depth.setEnabled(False)
        self._set_busy(True)
        self._t0 = time.time()
        self._timer.start(500)
        self.pool.start(self._job)

    def cancel(self):
        if self.token:
            self.token.cancel()
            self.btn_cancel.setEnabled(False)
            self.status.setText("Ακύρωση…")

    def _tick(self):
        pass

    def _finish(self):
        self._timer.stop()
        self.token = None
        self._set_busy(False)

    def _done(self, r: Prepared):
        self._finish()
        self.prepared = r
        self.bar.setValue(1000)
        txt = f"<b>✔ Έτοιμο ({time.time() - self._t0:.0f} s).</b><br>{r.summary()}"
        if r.warnings:
            txt += '<br><span style="color:#e0b040;">Προσοχή:</span><br>' + "<br>".join("• " + x for x in r.warnings)
        self.result_lbl.setText(txt)
        self.status.setText("Έτοιμο – διάλεξε ποιο script θα ανοίξεις.")
        self.btn_view.setEnabled(True)
        self.btn_depth.setEnabled(True)
        if getattr(self, "_auto_open", None):
            kinds = ["view", "depth"] if self._auto_open == "both" else [self._auto_open]
            self._auto_open = None
            for k in kinds:
                self.open_script(k)

    def _error(self, msg: str):
        self._finish()
        self.log("ΣΦΑΛΜΑ: " + msg.replace("\x00", ""))
        self.status.setText("✖ Η προετοιμασία απέτυχε")
        self.result_lbl.setText('<span style="color:#e06060;"><b>✖ Η προετοιμασία απέτυχε:</b><br>' + msg.split("\x00")[0].strip().replace("\n", "<br>") + "</span>")

    def _cancelled(self):
        self._finish()
        self.status.setText("Ακυρώθηκε.")

    # ------------------------------------------------------------------ άνοιγμα script
    def open_script(self, kind: str):
        if self.prepared is None:
            return
        import sirilpy
        from launcher import SCRIPTS, load_script
        _fname, cls, title = SCRIPTS[kind]
        sirilpy.configure(self.prepared, log=self.log)
        self.status.setText(f"Φόρτωση {title}… (την πρώτη φορά μπορεί να εγκατασταθούν βιβλιοθήκες)")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            mod = load_script(kind)
            win = getattr(mod, cls)(sirilpy.SirilInterface())
            win.showMaximized()
            self._windows.append(win)
            self.status.setText(f"{title} άνοιξε – πάτα εκεί «Render».")
        except Exception as e:  # noqa
            self.log(traceback.format_exc())
            QMessageBox.critical(self, title, f"Δεν μπόρεσα να ανοίξω το {title}:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()


def run_gui(args) -> int:
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    if args.astap:
        win.astap_edit.setText(args.astap)
    if args.scale:
        win.sp_scale.setValue(args.scale)
    if args.object:
        win.ed_object.setText(args.object)
    if args.force:
        win.chk_force.setChecked(True)
    if args.image:
        win.set_image(args.image)
        if args.open:
            win._auto_open = args.open
            QTimer.singleShot(200, win.start)
    return app.exec()

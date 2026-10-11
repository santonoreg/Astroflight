"""Καρτέλα «Αυτόματο»: ένα κουμπί που τρέχει ASTAP → StarNet → Gaia → 3D σκηνή → πτήση, με πρόοδο ανά βήμα
και επεξήγηση σε κάθε ρύθμιση."""
from __future__ import annotations
import os, time, traceback
from PySide6.QtCore import Qt, QObject, QRunnable, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QGroupBox,
                               QDoubleSpinBox, QComboBox, QProgressBar, QFileDialog, QScrollArea, QFormLayout, QSpinBox, QMessageBox)

from .widgets import ParamRow
from .pipeline import AutoOptions, Reporter, run_auto, STEPS
from .runner import CancelToken, Cancelled
from .config import Params
from . import wcs_solve as ws
from . import starnet as sn
from . import gaia as gm

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


class ExportSignals(QObject):
    progress = Signal(float, str)
    done = Signal(str)
    error = Signal(str)
    cancelled = Signal()


class ExportWorker(QRunnable):
    """Εξαγωγή βίντεο σε δικό της νήμα και δικό της OpenGL context (η εφαρμογή δεν «παγώνει», υπάρχει ETA και ακύρωση)."""

    def __init__(self, scene, tone, keyframes, p: Params, out: str, png: bool, sig: ExportSignals, token: CancelToken, log):
        super().__init__()
        self.setAutoDelete(False)
        self.scene, self.tone, self.keyframes, self.p, self.out, self.png = scene, tone, keyframes, p, out, png
        self.sig, self.token, self.log = sig, token, log

    def run(self):
        ctx = r = None
        try:
            import moderngl
            from .renderer import Renderer
            from .camera import CameraPath
            from . import export as ex
            backend = os.environ.get("ASTROFLIGHT_BACKEND") or None
            ctx = moderngl.create_standalone_context(backend=backend) if backend else moderngl.create_standalone_context()
            r = Renderer(ctx)
            r.set_scene(self.scene, self.tone)
            path = CameraPath(self.keyframes, self.p.duration, self.p.ease)
            n = max(2, int(round(self.p.duration * self.p.fps)))
            t0 = time.time()

            def prog(f):
                el = time.time() - t0
                done = f * n
                eta = el / f * (1 - f) if f > 0.01 else 0.0
                msg = f"Καρέ {int(done)}/{n}" + (f" – απομένουν περίπου {int(eta) // 60}:{int(eta) % 60:02d}" if f > 0.01 else "")
                self.sig.progress.emit(float(f), msg)
            ex.render_video(r, path, self.p, self.out, prog, lambda: self.token.cancelled, self.log, self.png)
            if self.token.cancelled:
                if not self.png and os.path.exists(self.out):
                    try:
                        os.remove(self.out)
                    except OSError:
                        pass
                self.sig.cancelled.emit()
            else:
                self.sig.done.emit(self.out)
        except Exception as e:  # noqa
            self.sig.error.emit(f"{e}\n\x00\n{traceback.format_exc()[-1800:]}")
        finally:
            try:
                if r is not None:
                    r.release()
                if ctx is not None:
                    ctx.release()
            except Exception:
                pass


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
        b2 = QPushButton("Αναζήτηση")
        b2.setToolTip("Ψάχνει το πρόγραμμα στις συνηθισμένες θέσεις εγκατάστασης")
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
        intro = QLabel("Άνοιξε τη φωτογραφία σου, διάλεξε (αν θες) το κέντρο ενδιαφέροντος και πάτα «Έναρξη» (κάτω, είναι πάντα ορατό). Η εφαρμογή κάνει μόνη της όλα τα βήματα: "
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

        # --- κέντρο ενδιαφέροντος (πρώτη επιλογή)
        g, gl = self._group("🎯 1 · Κέντρο ενδιαφέροντος")
        self.target_lbl = QLabel("")
        self.target_lbl.setWordWrap(True)
        gl.addWidget(self.target_lbl)
        trow = QHBoxLayout()
        b_pick = QPushButton("🎯 Διάλεξε κέντρο ενδιαφέροντος")
        b_pick.setToolTip("Δείχνει τη φωτογραφία στο δεξί παράθυρο· κάνεις κλικ στο σημείο που θες να πετάξει η κάμερα")
        b_pick.clicked.connect(self.win.begin_pick_target)
        b_auto = QPushButton("↺ Αυτόματος")
        b_auto.setToolTip("Η κάμερα πετά προς τη φωτεινότερη περιοχή του νεφελώματος")
        b_auto.clicked.connect(self.win.use_auto_target)
        trow.addWidget(b_pick, 3)
        trow.addWidget(b_auto, 1)
        gl.addLayout(trow)
        gl.addWidget(_help("Η κάμερα πετά σε ευθεία γραμμή από το σημείο λήψης προς το σημείο που θα μαρκάρεις πάνω στη φωτογραφία και συνεχίζει μέσα από το νεφέλωμα. "
                           "Διάλεξε π.χ. την καρδιά του αντικειμένου, ένα σμήνος ή μια κοιλότητα. Μπορείς να το αλλάξεις και μετά την επεξεργασία: η πτήση ξαναϋπολογίζεται αμέσως."))
        self._inputs_extra = [b_pick, b_auto]
        lay.addWidget(g)

        sep = QLabel("Ρυθμίσεις")
        sep.setStyleSheet("font-weight:bold; margin-top:8px;")
        lay.addWidget(sep)
        # --- εργαλεία
        g, gl = self._group("2 · Εργαλεία")
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
        self.chk_gaia = QCheckBox("Gaia – πραγματικές αποστάσεις αστεριών")
        self.chk_gaia.setChecked(True)
        f = QFont()
        f.setBold(True)
        self.chk_gaia.setFont(f)
        gl.addWidget(self.chk_gaia)
        gl.addWidget(_help("Κατεβάζει από τον κατάλογο Gaia DR3 τις μετρημένες αποστάσεις των αστεριών του πεδίου σου. "
                           "Απαιτεί επιτυχημένο plate solve. Αν δεν υπάρχει internet χρησιμοποιούνται εύλογες συνθετικές αποστάσεις."))
        grow = QHBoxLayout()
        b_diag = QPushButton("🔍 Έλεγχος σύνδεσης")
        b_diag.setToolTip("Δοκιμάζει αν φτάνει η εφαρμογή στους servers του Gaia (ESA, ARI Heidelberg, VizieR) και εξηγεί τι δεν δουλεύει")
        b_diag.clicked.connect(self.check_gaia_connection)
        b_file = QPushButton("📄 Τοπικό αρχείο Gaia…")
        b_file.setToolTip("Φόρτωσε CSV/FITS/VOTable που κατέβασες εσύ από το Gaia archive (δεν χρειάζεται internet στην εφαρμογή)")
        b_file.clicked.connect(self.pick_gaia_file)
        b_clr = QPushButton("✖")
        b_clr.setFixedWidth(30)
        b_clr.setToolTip("Αφαίρεση τοπικού αρχείου (θα χρησιμοποιηθεί internet)")
        b_clr.clicked.connect(lambda: self.set_gaia_file(None))
        b_q = QPushButton("📋 Ερώτημα Gaia archive")
        b_q.setToolTip("Αντιγράφει το ερώτημα ADQL για τη δική σου περιοχή του ουρανού (χρειάζεται πρώτα επιτυχημένο plate solve)")
        b_q.clicked.connect(self.copy_gaia_query)
        grow.addWidget(b_diag)
        grow.addWidget(b_q)
        gl.addLayout(grow)
        grow2 = QHBoxLayout()
        grow2.addWidget(b_file, 1)
        grow2.addWidget(b_clr)
        gl.addLayout(grow2)
        self.gaia_file_lbl = QLabel("")
        self.gaia_file_lbl.setWordWrap(True)
        gl.addWidget(self.gaia_file_lbl)
        gl.addWidget(_help("Αν η εφαρμογή δεν μπορεί να κατεβάσει Gaia (firewall/proxy): (1) πάτα «Έλεγχος σύνδεσης» για να δεις γιατί, ή (2) μετά από ένα επιτυχημένο "
                           "plate solve πάτα «Ερώτημα για το Gaia archive», επικόλλησέ το στο <a href=\"https://gea.esac.esa.int/archive/\">gea.esac.esa.int/archive</a> "
                           "(Search → Advanced ADQL), κατέβασε το αποτέλεσμα ως CSV και φόρτωσέ το εδώ. Ολόκληρο το Gaia (~1.5 TB) δεν χρειάζεται: μόνο η μικρή περιοχή της φωτογραφίας σου."))
        lay.addWidget(g)

        # --- πληροφορίες εικόνας
        g, gl = self._group("3 · Τι ξέρεις για την εικόνα (όλα προαιρετικά)")
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
        self.cb_stretch.addItems(["Αυτόματα", "Ναι – τελική εικόνα", "Όχι – γραμμικά (linear)"])
        self.cb_stretch.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_stretch.setMinimumContentsLength(12)
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
        g, gl = self._group("4 · Πτήση (αλλάζει ζωντανά μετά την επεξεργασία)")
        self.prows = []
        for spec in (
            ("duration", "Διάρκεια (s)", "f", 6, 180, 1,
             "Πόσο διαρκεί το βίντεο. Μεγαλύτερη διάρκεια = πιο αργή κίνηση και πιο αργό ζουμ (π.χ. 30 s ήρεμο, 60 s πολύ αργό)."),
            ("dive_zoom_rate", "Μέγιστη ταχύτητα ζουμ", "f", 0.05, 1.0, 0.01,
             "Πόσο γρήγορα επιτρέπεται να μεγαλώνει το νεφέλωμα στο κάδρο (ln ανά δευτερόλεπτο: 0.1 = πολύ αργά, 0.25 = ήρεμα, 0.5 = γρήγορα). "
             "Αν η διάρκεια είναι πολύ μικρή για να τηρηθεί, το όριο αυξάνεται αυτόματα."),
            ("dive_zoom", "Zoom προσέγγισης (×)", "f", 1.0, 8.0, 0.1,
             "Πόσες φορές μεγαλύτερο φαίνεται το νεφέλωμα στο κάδρο όταν η κάμερα φτάνει στην είσοδό του. 1 = μένει όσο στη φωτογραφία (έντονη παράλλαξη, "
             "όλο το αντικείμενο φαίνεται), μεγάλο = το νεφέλωμα «ξεχειλίζει» από το κάδρο."),
            ("dive_fov_max", "Μέγιστο οπτικό πεδίο (°)", "f", 20, 100, 1,
             "Πόσο ανοίγει το κάδρο όταν η κάμερα μπαίνει μέσα στα νέφη. Μεγαλύτερο = βλέπεις περισσότερα στρώματα γύρω σου, μικρότερο = πιο «τηλεσκοπικό» πλάνο."),
            ("dive_exit", "Έξοδος από το νεφέλωμα (× πλάτος)", "f", 0.0, 1.5, 0.05,
             "Η κάμερα συνεχίζει πέρα από την πίσω πλευρά του νεφελώματος και σταματά αυτή την απόσταση έξω, σε πεδίο μόνο με αστέρια. 0 = μένει μέσα στα νέφη (τότε ισχύει το «Βάθος τέλους»)."),
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
        lay.addWidget(g)
        self.progress_group = g
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
        self.btn_export.clicked.connect(self.export_video)
        row.addWidget(self.btn_view)
        row.addWidget(self.btn_export)
        lay.addLayout(row)

        # --- εξαγωγή βίντεο
        g, gl = self._group("🎬 Εξαγωγή βίντεο")
        form = QFormLayout()
        self.cb_res = QComboBox()
        for name, wh in (("1280×720 (HD)", (1280, 720)), ("1920×1080 (Full HD)", (1920, 1080)), ("2560×1440 (2K)", (2560, 1440)),
                         ("3840×2160 (4K)", (3840, 2160)), ("1080×1080 (τετράγωνο)", (1080, 1080)), ("1080×1920 (κάθετο, Reels/Shorts)", (1080, 1920))):
            self.cb_res.addItem(name, wh)
        self.cb_res.setCurrentIndex(1)
        form.addRow("Ανάλυση:", self.cb_res)
        self.sp_fps = QSpinBox()
        self.sp_fps.setRange(12, 120)
        self.sp_fps.setValue(self.proj.p.fps)
        form.addRow("Καρέ ανά δευτερόλεπτο:", self.sp_fps)
        self.cb_ssaa = QComboBox()
        self.cb_ssaa.addItems(["1× – γρήγορο", "2× – καθαρότερο (≈4× αργό)", "3× – μέγιστο (≈9× αργό)"])
        form.addRow("Εξομάλυνση (antialiasing):", self.cb_ssaa)
        self.cb_q = QComboBox()
        for name, crf in (("Υψηλή (μεγάλο αρχείο)", 16), ("Μέτρια", 21), ("Μικρό αρχείο", 26)):
            self.cb_q.addItem(name, crf)
        form.addRow("Ποιότητα συμπίεσης:", self.cb_q)
        self.cb_fmt = QComboBox()
        self.cb_fmt.addItem("MP4 (H.264)", False)
        self.cb_fmt.addItem("Εικόνες PNG (για άλλο πρόγραμμα)", True)
        form.addRow("Μορφή:", self.cb_fmt)
        gl.addLayout(form)
        self.exp_info = _help("")
        gl.addWidget(self.exp_info)
        gl.addWidget(_help("Η διάρκεια και η ταχύτητα του ζουμ ορίζονται στο «3 · Πτήση» παρακάτω (μεγαλύτερη διάρκεια = πιο αργό ζουμ). "
                           "Η εξαγωγή τρέχει στο παρασκήνιο: μπορείς να κάνεις άλλα πράγματα και να την ακυρώσεις. "
                           "Αν η κάρτα γραφικών σου έχει λίγη μνήμη, δοκίμασε μικρότερη ανάλυση ή «Ανάλυση όγκου» στο Βήμα-βήμα."))
        row = QHBoxLayout()
        self.btn_do_export = QPushButton("💾  Εξαγωγή βίντεο…")
        self.btn_do_export.setMinimumHeight(42)
        f2 = QFont()
        f2.setBold(True)
        self.btn_do_export.setFont(f2)
        self.btn_do_export.clicked.connect(self.export_video)
        self.btn_exp_cancel = QPushButton("■ Ακύρωση εξαγωγής")
        self.btn_exp_cancel.setMinimumHeight(42)
        self.btn_exp_cancel.setEnabled(False)
        self.btn_exp_cancel.clicked.connect(self.cancel_export)
        row.addWidget(self.btn_do_export, 3)
        row.addWidget(self.btn_exp_cancel, 2)
        gl.addLayout(row)
        self.exp_bar = QProgressBar()
        self.exp_bar.setRange(0, 1000)
        gl.addWidget(self.exp_bar)
        self.exp_lbl = QLabel("")
        self.exp_lbl.setWordWrap(True)
        self.exp_lbl.setTextFormat(Qt.RichText)
        self.exp_lbl.setOpenExternalLinks(True)
        gl.addWidget(self.exp_lbl)
        for cb in (self.cb_res, self.cb_ssaa, self.cb_q, self.cb_fmt):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)     # να μην φουσκώνει το πάνελ
            cb.setMinimumContentsLength(12)
        self._export_widgets = [self.cb_res, self.sp_fps, self.cb_ssaa, self.cb_q, self.cb_fmt, self.btn_do_export]
        for w_ in (self.cb_res, self.sp_fps, self.cb_ssaa):
            (w_.currentIndexChanged if hasattr(w_, "currentIndexChanged") else w_.valueChanged).connect(lambda *_: self._update_export_info())
        self.token_exp: CancelToken | None = None
        lay.addWidget(g)
        self.exp_group = g


        lay.addStretch(1)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(w)
        outer.addWidget(sa, 1)
        self.sa = sa

        # --- σταθερό υποσέλιδο: κουμπί έναρξης + πρόοδος (φαίνονται πάντα, χωρίς scroll)
        foot = QWidget()
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(8, 6, 8, 8)
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
        fl.addLayout(row)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        fl.addWidget(self.bar)
        self.time_lbl = QLabel("")
        fl.addWidget(self.time_lbl)
        outer.addWidget(foot)
        self.refresh_target_label()
        self._inputs = [self.chk_astap, self.ed_astap, self.chk_starnet, self.ed_starnet, self.chk_gaia, self.sp_scale,
                        self.sp_focal, self.sp_pix, self.ed_object, self.cb_stretch, self.chk_dist, btn]

    # ------------------------------------------------------------------ εξαγωγή βίντεο
    def _update_export_info(self):
        p = self.proj.p
        n = int(round(p.duration * self.sp_fps.value()))
        w, h = self.cb_res.currentData()
        ss = self.cb_ssaa.currentIndex() + 1
        self.exp_info.setText(f"{p.duration:.0f} s × {self.sp_fps.value()} fps = {n} καρέ στα {w}×{h}" + (f" με εξομάλυνση {ss}×" if ss > 1 else "")
                              + ". Ο χρόνος εξαγωγής εξαρτάται από την κάρτα γραφικών· θα δεις ETA μόλις ξεκινήσει.")

    def refresh_export_state(self):
        ok = self.proj.scene is not None and self.token is None and self.token_exp is None and not self.win.busy
        for w in self._export_widgets:
            w.setEnabled(ok)
        self.btn_export.setEnabled(ok)
        self._update_export_info()

    def export_video(self):
        p = self.proj
        if p.scene is None or not p.keyframes:
            QMessageBox.information(self, "AstroFlight", "Πρώτα τρέξε την αυτόματη επεξεργασία ώστε να υπάρχει πτήση.")
            return
        png = bool(self.cb_fmt.currentData())
        d = self.win.settings.value("last_export_dir", os.path.expanduser("~"))
        if png:
            out = QFileDialog.getExistingDirectory(self, "Φάκελος για τα καρέ PNG", d)
        else:
            out, _ = QFileDialog.getSaveFileName(self, "Αποθήκευση βίντεο", os.path.join(d, "astroflight.mp4"), "MP4 (*.mp4)")
            if out and not out.lower().endswith(".mp4"):
                out += ".mp4"
        if not out:
            return
        self.win.settings.setValue("last_export_dir", out if png else os.path.dirname(out))
        pp = Params(**p.p.to_dict())
        pp.out_width, pp.out_height = self.cb_res.currentData()
        pp.fps = int(self.sp_fps.value())
        pp.ssaa = self.cb_ssaa.currentIndex() + 1
        pp.video_crf = int(self.cb_q.currentData())
        pp.steps = max(int(pp.steps), 256)                       # λιγότερος θόρυβος (κόκκος) στο τελικό βίντεο
        self.token_exp = CancelToken()
        sig = ExportSignals()
        self._exp_sig = sig
        sig.progress.connect(lambda f, m: (self.exp_bar.setValue(int(f * 1000)), self.exp_lbl.setText(m)))
        sig.done.connect(lambda path: self._export_finished(f"✔ Έτοιμο: <b>{path}</b>"))
        sig.cancelled.connect(lambda: self._export_finished("Η εξαγωγή ακυρώθηκε."))
        sig.error.connect(self._export_error)
        self.exp_bar.setValue(0)
        self.exp_lbl.setText("Εκκίνηση…")
        self.btn_exp_cancel.setEnabled(True)
        self.win._set_busy(True)
        self.btn_exp_cancel.setEnabled(True)
        self._exp_worker = ExportWorker(p.scene, p.tone, list(p.keyframes), pp, out, png, sig, self.token_exp, self.win.log_sig.emit)
        self.win.pool.start(self._exp_worker)

    def cancel_export(self):
        if self.token_exp:
            self.token_exp.cancel()
            self.btn_exp_cancel.setEnabled(False)
            self.exp_lbl.setText("Ακύρωση…")

    def _export_finished(self, msg):
        self.token_exp = None
        self.btn_exp_cancel.setEnabled(False)
        self.exp_lbl.setText(msg)
        if msg.startswith("✔"):
            self.exp_bar.setValue(1000)
        self.win._set_busy(False)

    def _export_error(self, msg):
        self._export_finished('<span style="color:#e06060;"><b>✖ Η εξαγωγή απέτυχε:</b> ' + msg.split("\x00")[0].strip().replace("\n", "<br>") + "</span>")
        self.win._log("ΣΦΑΛΜΑ εξαγωγής: " + msg.replace("\x00", ""))

    def refresh_target_label(self):
        p = self.proj.p
        if p.dive_auto_poi:
            self.target_lbl.setText("Στόχος: <b>αυτόματος</b> (η φωτεινότερη περιοχή του νεφελώματος)")
        else:
            self.target_lbl.setText(f"Στόχος: <b>δικός σου</b> – x = {p.dive_fx:+.2f}, y = {p.dive_fy:+.2f} (από το κέντρο της φωτογραφίας)")

    # ------------------------------------------------------------------ Gaia
    def set_gaia_file(self, path):
        self.proj.gaia_file = path
        self.gaia_file_lbl.setText(f"Χρήση τοπικού αρχείου: {os.path.basename(path)}" if path else "")

    def pick_gaia_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Αρχείο Gaia", self.win.settings.value("last_dir", ""),
                                           "Gaia (*.csv *.fits *.fit *.vot *.xml *.ecsv);;Όλα (*)")
        if p:
            self.set_gaia_file(p)

    def check_gaia_connection(self):
        self.win.run_task(lambda prog: gm.diagnose_connection(log=self.win.log_sig.emit), lambda r: None, "Έλεγχος σύνδεσης Gaia…")

    def copy_gaia_query(self):
        from PySide6.QtWidgets import QApplication
        p = self.proj
        if p.sky is None:
            self.win._log("Το ερώτημα Gaia χρειάζεται συντεταγμένες: κάνε πρώτα επιτυχημένο plate solve (Έναρξη με ASTAP).")
            return
        ra, dec = p.sky.center(p.W)
        q = gm.gaia_query_text(ra, dec, p._field_radius_deg(), min(p.p.gaia_glim, 15.5))
        QApplication.clipboard().setText(q)
        self.win._log("Αντιγράφηκε στο πρόχειρο το ερώτημα για το https://gea.esac.esa.int/archive/ :\n" + q)

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
        for w in self._inputs + getattr(self, "_inputs_extra", []):
            w.setEnabled(not busy)
        self.btn_run.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy and running)
        if hasattr(self, "_export_widgets"):
            self.refresh_export_state()
            self.btn_exp_cancel.setEnabled(self.token_exp is not None)

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
        self.sa.ensureWidgetVisible(self.progress_group, 0, 20)
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
        self.refresh_export_state()
        self.sa.ensureWidgetVisible(self.result_lbl, 0, 60)

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

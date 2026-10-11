"""Γραφικό περιβάλλον (PySide6)."""
from __future__ import annotations
import os, sys, math, time, traceback
import numpy as np
import moderngl
from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, Signal, QTimer, QSettings
from PySide6.QtGui import QSurfaceFormat, QImage, QPixmap, QFont
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
                               QLabel, QSlider, QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox, QFileDialog, QLineEdit,
                               QPlainTextEdit, QProgressBar, QSplitter, QScrollArea, QListWidget, QComboBox, QMessageBox,
                               QFormLayout, QProgressDialog, QSizePolicy, QStackedWidget, QButtonGroup, QRadioButton, QTabWidget)

from .config import Params, UI_GROUPS
from .widgets import Signals, Worker, ParamRow
from .gui_auto import AutoPanel
from .project import Project
from .renderer import Renderer
from .camera import CamState, Keyframe, CameraPath, pose_from_state, make_flythrough, make_dive
from . import wcs_solve as ws
from . import imageio_ as io
from . import export as ex
from . import depth_ai


# ------------------------------------------------------------------------------------------------
class GLView(QOpenGLWidget):
    cam_changed = Signal()

    def __init__(self, proj: Project):
        super().__init__()
        self.proj = proj
        self.ctx = None
        self.renderer: Renderer | None = None
        self.pending = None
        self.cam = CamState()
        self.setMinimumSize(520, 380)
        self.setFocusPolicy(Qt.StrongFocus)
        self._last = None
        self._fbos = {}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.playing = False
        self._t0 = 0.0
        self._play_t = 0.0
        self.path: CameraPath | None = None
        self.time_changed = None

    # --- GL
    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.renderer = Renderer(self.ctx)
        if self.pending is not None:
            sc, tone = self.pending
            self.renderer.set_scene(sc, tone)
            self.pending = None

    def set_scene(self, sc, tone):
        if self.renderer is None:
            self.pending = (sc, tone)
        else:
            self.makeCurrent()
            self.renderer.set_scene(sc, tone)
            self.doneCurrent()
        self.cam = CamState()
        self.update()

    def paintGL(self):
        if self.ctx is None:
            return
        glo = self.defaultFramebufferObject()
        fbo = self._fbos.get(glo)
        if fbo is None:
            fbo = self._fbos[glo] = self.ctx.detect_framebuffer(glo)
        r = self.devicePixelRatio()
        size = (max(8, int(self.width() * r)), max(8, int(self.height() * r)))
        if self.renderer.scene is None:
            fbo.use()
            self.ctx.viewport = (0, 0, *size)
            self.ctx.clear(0.06, 0.06, 0.08)
            return
        m = self.renderer.scene.meta
        pose = pose_from_state(self.cam, m["D0"], m["W0"])
        self.renderer.render(fbo, size, pose, self.proj.p, hdr_scale=self.proj.p.preview_scale,
                             fov_mul=math.exp(self.cam.logfov))

    # --- αλληλεπίδραση
    def mousePressEvent(self, e):
        self._last = e.position()
        self.setFocus()

    def mouseMoveEvent(self, e):
        if self._last is None or self.renderer is None or self.renderer.scene is None:
            return
        d = e.position() - self._last
        self._last = e.position()
        c = self.cam
        m = self.renderer.scene.meta
        if e.buttons() & Qt.LeftButton and not (e.modifiers() & Qt.ShiftModifier):
            c.az -= d.x() * 0.25
            c.el = max(-85.0, min(85.0, c.el + d.y() * 0.25))
        elif e.buttons() & Qt.LeftButton:
            c.roll += d.x() * 0.25
        elif e.buttons() & (Qt.RightButton | Qt.MiddleButton):
            R = m["D0"] * math.exp(c.logr)
            k = 0.0018 * R / m["W0"]
            c.tx -= d.x() * k
            c.ty += d.y() * k
        self.cam_changed.emit()
        self.update()

    def wheelEvent(self, e):
        self.cam.logr = max(-9.0, min(0.6, self.cam.logr - e.angleDelta().y() * 0.0012))
        self.cam_changed.emit()
        self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space:
            self.toggle_play()
        elif e.key() == Qt.Key_Home:
            self.cam = CamState()
            self.cam_changed.emit()
            self.update()

    # --- αναπαραγωγή
    def toggle_play(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
            return
        if self.path is None:
            return
        self.playing = True
        self._t0 = time.time() - self._play_t
        self.timer.start(15)

    def set_time(self, t):
        self._play_t = t
        if self.path:
            self.cam = self.path.eval(t)
            self.cam_changed.emit()
            self.update()

    def _tick(self):
        t = time.time() - self._t0
        dur = self.proj.p.duration
        if t >= dur:
            t = dur
            self.timer.stop()
            self.playing = False
        self.set_time(t)
        if self.time_changed:
            self.time_changed(t)



# ------------------------------------------------------------------------------------------------
class ImagePreview(QLabel):
    """Προεπισκόπηση 2D εικόνας (πριν χτιστεί η 3D σκηνή)."""

    def __init__(self):
        super().__init__("Φόρτωσε μια αστροφωτογραφία για να ξεκινήσεις")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(520, 380)
        self.setStyleSheet("background:#0e0e12; color:#8a8a99; font-size:15px;")
        self._pm = None

    def set_array(self, rgb01: np.ndarray):
        a = (np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        a = np.ascontiguousarray(a)
        qi = QImage(a.data, a.shape[1], a.shape[0], a.shape[1] * 3, QImage.Format_RGB888).copy()
        self._pm = QPixmap.fromImage(qi)
        self._fit()

    def clear_image(self, text="Φόρτωσε μια αστροφωτογραφία για να ξεκινήσεις"):
        self._pm = None
        self.clear()
        self.setText(text)

    def _fit(self):
        if self._pm is not None:
            self.setPixmap(self._pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()


STEP_NAMES = ["Φόρτωση εικόνας", "Κλίμακα & θέση", "Αστέρια", "Βάθος & 3D σκηνή", "Εμφάνιση", "Πτήση & βίντεο"]
STEP_HELP = [
    "Διάλεξε την αστροφωτογραφία σου (TIFF 16-bit προτιμάται). Αν είναι ήδη επεξεργασμένη/τεντωμένη (όπως οι περισσότερες τελικές εικόνες) άφησε τσεκαρισμένο το «τεντωμένη».",
    "Η εφαρμογή χρειάζεται να ξέρει πόσο μεγάλο είναι το πεδίο (arcsec/pixel). Προαιρετικά, με plate solve (ASTAP) μαθαίνει και πού στον ουρανό κοιτάς, ώστε να πάρει ΠΡΑΓΜΑΤΙΚΕΣ αποστάσεις αστεριών από το Gaia.",
    "Εντοπίζονται τα αστέρια, αφαιρούνται από την εικόνα (έτσι μένει μόνο το νεφέλωμα) και παίρνουν αποστάσεις. Τα αστέρια γίνονται πραγματικά 3D σημεία.",
    "Ορίζεται πόσο «βαθύ» είναι το νεφέλωμα και χτίζεται ο τρισδιάστατος όγκος. Από την αρχική οπτική γωνία το αποτέλεσμα ταυτίζεται με τη φωτογραφία σου.",
    "Ρύθμισε την εμφάνιση ζωντανά στο παράθυρο δεξιά (σύρε με το ποντίκι για να περιστρέψεις, ροδέλα για πλησίασμα).",
    "Φτιάξε τη διαδρομή της κάμερας, δες την προεπισκόπηση και εξήγαγε το βίντεο.",
]
BUILD_KEYS = ({s[0] for g in ("Βάθος νεφελώματος", "Όγκος (ποιότητα)") for s in UI_GROUPS[g]}
              | {"object_distance_pc"})
STAR_KEYS = {"stretched_input", "gamma", "star_sigma", "max_stars", "gaia_glim", "match_tol_px",
             "extra_enable", "extra_radius_deg", "extra_glim"}
LAST = len(STEP_NAMES) - 1
DIVE_KEYS = {"dive_zoom_rate", "dive_zoom", "dive_fov_max", "dive_depth", "dive_auto_poi", "dive_fx", "dive_fy", "dive_drift", "duration"}


class MainWindow(QMainWindow):
    log_sig = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AstroFlight – 3D πτήση μέσα σε αστροφωτογραφίες")
        self.resize(1500, 920)
        self.settings = QSettings("AstroFlight", "AstroFlight")
        self.proj = Project(Params(), log=lambda s: self.log_sig.emit(str(s)))
        if self.settings.value("astap"):
            self.proj.astap_path = self.settings.value("astap")
        if self.settings.value("starnet"):
            self.proj.starnet_path = self.settings.value("starnet")
        self.pool = QThreadPool.globalInstance()
        self.rows: list[ParamRow] = []
        self._tasks = set()
        self.step = 0
        self.busy = False
        self.stale_stars = False       # τα αστέρια πρέπει να ξαναϋπολογιστούν
        self.stale_scene = False       # η σκηνή πρέπει να ξαναχτιστεί
        self.view = GLView(self.proj)
        self.view.time_changed = self._on_play_time
        self.view.cam_changed.connect(self._update_cam_label)
        self.log_sig.connect(self._log)
        self._prev_timer = QTimer(self)
        self._prev_timer.setSingleShot(True)
        self._prev_timer.timeout.connect(self._show_preview)
        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.timeout.connect(self._regen_path)
        self._build_ui()
        self._build_menu()
        self._goto(0)
        self.statusBar().showMessage("Βήμα 1: άνοιξε μια αστροφωτογραφία.")

    # ================================================================== UI βοηθητικά
    def _group(self, title):
        g = QGroupBox(title)
        lay = QVBoxLayout(g)
        return g, lay

    def _btn(self, text, fn, tip=""):
        b = QPushButton(text)
        b.clicked.connect(fn)
        b.setToolTip(tip)
        return b

    def _big(self, text, fn, tip=""):
        b = self._btn(text, fn, tip)
        b.setMinimumHeight(42)
        f = QFont()
        f.setBold(True)
        b.setFont(f)
        return b

    def _intro(self, text):
        l = QLabel(text)
        l.setWordWrap(True)
        l.setStyleSheet("color:#8d93a8;")
        return l

    def _note(self, text=""):
        l = QLabel(text)
        l.setWordWrap(True)
        l.setOpenExternalLinks(True)
        return l

    def _prow(self, lay, key, label, kind, lo, hi, step, tip=""):
        r = ParamRow(self.proj.p, (key, label, kind, lo, hi, step, tip), self._param_changed)
        self.rows.append(r)
        lay.addWidget(r)
        return r

    def _spec_rows(self, lay, group):
        for spec in UI_GROUPS[group]:
            r = ParamRow(self.proj.p, spec, self._param_changed)
            self.rows.append(r)
            lay.addWidget(r)

    def _scroll(self, widget):
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(widget)
        return sa

    # ================================================================== δομή wizard
    def _build_ui(self):
        # --- αριστερά: wizard
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        self.title_lbl = QLabel()
        tf = QFont()
        tf.setPointSize(15)
        tf.setBold(True)
        self.title_lbl.setFont(tf)
        ll.addWidget(self.title_lbl)
        chips = QHBoxLayout()
        self.chips: list[QPushButton] = []
        for i, n in enumerate(STEP_NAMES):
            b = QPushButton(str(i + 1))
            b.setFixedSize(40, 30)
            b.setToolTip(n)
            b.clicked.connect(lambda _=False, k=i: self._goto(k))
            self.chips.append(b)
            chips.addWidget(b)
        chips.addStretch(1)
        ll.addLayout(chips)
        self.help_lbl = self._intro("")
        ll.addWidget(self.help_lbl)

        self.stack = QStackedWidget()
        for fn in (self._page_load, self._page_scale, self._page_stars, self._page_depth, self._page_look, self._page_video):
            self.stack.addWidget(self._scroll(fn()))
        ll.addWidget(self.stack, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        ll.addWidget(self.progress)
        nav = QHBoxLayout()
        self.back_btn = QPushButton("◀  Πίσω")
        self.back_btn.setMinimumHeight(36)
        self.back_btn.clicked.connect(self._back)
        self.next_btn = QPushButton("Επόμενο  ▶")
        self.next_btn.setMinimumHeight(36)
        self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.back_btn)
        nav.addWidget(self.next_btn)
        ll.addLayout(nav)

        # --- δεξιά: προεπισκόπηση / 3D
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.preview = ImagePreview()
        self.rstack = QStackedWidget()
        self.rstack.addWidget(self.preview)
        self.rstack.addWidget(self.view)
        rl.addWidget(self.rstack, 1)
        bar = QHBoxLayout()
        self.play_btn = QPushButton("▶ Αναπαραγωγή")
        self.play_btn.clicked.connect(self._toggle_play)
        self.t_slider = QSlider(Qt.Horizontal)
        self.t_slider.setRange(0, 1000)
        self.t_slider.valueChanged.connect(self._scrub)
        self.cam_label = QLabel("")
        bar.addWidget(self.play_btn)
        bar.addWidget(self.t_slider, 1)
        bar.addWidget(self.cam_label)
        self.playbar = QWidget()
        self.playbar.setLayout(bar)
        rl.addWidget(self.playbar)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumHeight(120)
        rl.addWidget(self.logbox)
        self.mouse_hint = QLabel("Ποντίκι: αριστερό = περιστροφή • Shift+αριστερό = roll • δεξί/μεσαίο = μετατόπιση • ροδέλα = πλησίασμα • Space = αναπαραγωγή • Home = αρχική θέση")
        self.mouse_hint.setWordWrap(True)
        rl.addWidget(self.mouse_hint)

        self.auto = AutoPanel(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.auto, "⚡ Αυτόματο (προτεινόμενο)")
        self.tabs.addTab(left, "🛠 Βήμα-βήμα (προχωρημένο)")
        self.tabs.currentChanged.connect(self._tab_changed)
        sp = QSplitter()
        sp.addWidget(self.tabs)
        sp.addWidget(right)
        sp.setSizes([560, 940])
        self.setCentralWidget(sp)

    def _build_menu(self):
        m = self.menuBar().addMenu("Αρχείο")
        a = m.addAction("Άνοιγμα project…")
        a.triggered.connect(self.load_project)
        a = m.addAction("Αποθήκευση project…")
        a.triggered.connect(self.save_project)
        m.addSeparator()
        a = m.addAction("Έξοδος")
        a.triggered.connect(self.close)
        v = self.menuBar().addMenu("Προβολή")
        a = v.addAction("Εμφάνιση επεξηγήσεων κάτω από τις ρυθμίσεις")
        a.setCheckable(True)
        a.setChecked(True)
        a.toggled.connect(ParamRow.set_help_visible)

    # ================================================================== πλοήγηση
    def _done(self, i: int) -> bool:
        p = self.proj
        d0 = p.lin is not None
        d2 = d0 and not self.stale_stars and p.stars is not None and p.dist is not None and p.starless_lin is not None
        d3 = d2 and not self.stale_scene and p.scene is not None
        return [d0, d0, d2, d3, d3, d3][i]

    def _unlocked(self, i: int) -> bool:
        return all(self._done(j) for j in range(i))

    def _goto(self, i: int):
        if self.busy or not (0 <= i <= LAST) or not self._unlocked(i):
            return
        self.step = i
        self.stack.setCurrentIndex(i)
        self._on_enter(i)
        self._update_nav()

    def _back(self):
        self._goto(self.step - 1)

    def _next(self):
        if self.busy:
            return
        if self.step == 1:
            self._ensure_scale()
        if self._done(self.step) and self.step < LAST:
            self._goto(self.step + 1)

    def _on_enter(self, i):
        if i == 2:
            self._refresh_stars_page()
        if i == 3:
            self._refresh_depth_page()
        if i == 5:
            self._refresh_kf()
        if self._done(3) and i >= 3:
            self.rstack.setCurrentWidget(self.view)
        else:
            self.rstack.setCurrentWidget(self.preview)
            self._show_preview()
        gl = self.rstack.currentWidget() is self.view
        self.playbar.setVisible(gl)
        self.mouse_hint.setVisible(gl)
        if gl:
            self.view.setFocus()
            self.view.update()

    def _update_nav(self):
        self.title_lbl.setText(f"Βήμα {self.step + 1} από {len(STEP_NAMES)}:  {STEP_NAMES[self.step]}")
        self.help_lbl.setText(STEP_HELP[self.step])
        for i, b in enumerate(self.chips):
            done = self._done(i) if i < LAST else self._done(3)
            cur = i == self.step
            col = "#2f7d4f" if done and not cur else ("#3b5fc0" if cur else "#3a3a46")
            b.setStyleSheet(f"QPushButton{{background:{col};color:white;border-radius:6px;font-weight:{'bold' if cur else 'normal'};}}"
                            "QPushButton:disabled{background:#26262e;color:#666;}")
            b.setEnabled(self._unlocked(i) and not self.busy)
        self.back_btn.setEnabled(self.step > 0 and not self.busy)
        self.next_btn.setEnabled(self.step < LAST and self._done(self.step) and not self.busy)
        self.next_btn.setVisible(self.step < LAST)
        self.stack.setEnabled(not self.busy)

    def _set_busy(self, b: bool):
        self.busy = b
        self._update_nav()
        if hasattr(self, "auto"):
            self.auto.set_busy(b)

    def _invalidate(self, stars=False, scene=True):
        if stars:
            self.stale_stars = True
        if scene:
            self.stale_scene = True
        self._update_nav()

    # ================================================================== σελίδες
    # --- 1
    def _page_load(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(self._big("📂  Άνοιγμα αστροφωτογραφίας (TIFF/PNG/JPG)…", self.open_image))
        self.img_label = QLabel("Δεν έχει φορτωθεί εικόνα.")
        self.img_label.setWordWrap(True)
        lay.addWidget(self.img_label)
        g, gl = self._group("Τύπος δεδομένων")
        self._prow(gl, "stretched_input", "Η εικόνα είναι ήδη τεντωμένη (non-linear)", "b", 0, 1, 0,
                   "Τσέκαρέ το για τελικές, επεξεργασμένες εικόνες (αυτές που βλέπεις κανονικά φωτεινές). Ξετσέκαρέ το μόνο για ακατέργαστα, σκοτεινά γραμμικά (linear) δεδομένα.")
        self._prow(gl, "gamma", "Gamma τεντώματος", "f", 1.0, 3.0, 0.05,
                   "Πώς μετατρέπεται η τεντωμένη εικόνα σε γραμμικά δεδομένα ώστε να προσομοιωθεί σωστά το φως. Σχεδόν ποτέ δεν χρειάζεται αλλαγή (2.2 = τυπική οθόνη).")
        gl.addWidget(self._intro("Σημείωση: το StarNet++ δουλεύει καλύτερα σε τεντωμένες εικόνες. Για γραμμικά δεδομένα η εφαρμογή τα τεντώνει προσωρινά."))
        lay.addWidget(g)
        lay.addWidget(self._btn("Άνοιγμα αποθηκευμένου project…", self.load_project))
        lay.addStretch(1)
        return w

    # --- 2
    def _page_scale(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        g, gl = self._group("Κλίμακα εικόνας")
        form = QFormLayout()
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 600)
        self.scale_spin.setDecimals(4)
        self.scale_spin.setValue(2.0)
        self.scale_spin.setSuffix(' "/px')
        self.scale_spin.valueChanged.connect(lambda _v: self._invalidate(scene=True))
        form.addRow("Κλίμακα:", self.scale_spin)
        self.focal = QDoubleSpinBox()
        self.focal.setRange(1, 20000)
        self.focal.setValue(500)
        self.focal.setSuffix(" mm")
        self.pixsz = QDoubleSpinBox()
        self.pixsz.setRange(0.5, 50)
        self.pixsz.setDecimals(2)
        self.pixsz.setValue(3.76)
        self.pixsz.setSuffix(" µm")
        self.binning = QSpinBox()
        self.binning.setRange(1, 4)
        form.addRow("Εστιακή απόσταση:", self.focal)
        form.addRow("Μέγεθος pixel:", self.pixsz)
        form.addRow("Binning / resample:", self.binning)
        gl.addLayout(form)
        gl.addWidget(self._btn("Υπολογισμός κλίμακας από τα οπτικά", self._calc_scale, "206.265 × pixel(µm) / εστιακή(mm)"))
        lay.addWidget(g)

        g, gl = self._group("Αστρομετρία – plate solve (προαιρετικό, για πραγματικές αποστάσεις Gaia)")
        self.astap_edit = QLineEdit(self.proj.astap_path or "")
        self.astap_edit.setPlaceholderText("διαδρομή astap.exe")
        row = QHBoxLayout()
        row.addWidget(self.astap_edit, 1)
        row.addWidget(self._btn("…", self._browse_astap))
        gl.addLayout(row)
        gl.addWidget(self._note('Δεν έχεις ASTAP; Κατέβασέ το από <a href="https://hnsky.org/astap.htm">hnsky.org/astap.htm</a> '
                                '(πρόγραμμα + μία βάση αστεριών, π.χ. D50 ή D20).'))
        gl.addWidget(self._intro("Για γρήγορη λύση η κλίμακα πάνω πρέπει να είναι σωστή (±10%). Αν αποτύχει, η εφαρμογή δοκιμάζει αυτόματα και με άγνωστο πεδίο."))
        gl.addWidget(self._big("🔭  Plate solve με ASTAP", self.solve_astap, "Χρειάζεται ASTAP + βάση αστεριών (D50/D20/D80)"))
        form = QFormLayout()
        self.ra_edit = QLineEdit()
        self.ra_edit.setPlaceholderText("05 35 17  ή  83.82")
        self.dec_edit = QLineEdit()
        self.dec_edit.setPlaceholderText("-05 23 28  ή  -5.39")
        self.rot_spin = QDoubleSpinBox()
        self.rot_spin.setRange(-360, 360)
        self.rot_spin.setSuffix("°")
        form.addRow("RA κέντρου:", self.ra_edit)
        form.addRow("Dec κέντρου:", self.dec_edit)
        form.addRow("Περιστροφή:", self.rot_spin)
        gl.addLayout(form)
        gl.addWidget(self._intro("Αν γράψεις RA/Dec, βοηθά το ASTAP (ταχύτερη λύση) ή μπορείς να ορίσεις χειροκίνητα WCS:"))
        gl.addWidget(self._btn("Χειροκίνητο WCS (RA/Dec + κλίμακα)", self.manual_wcs))
        gl.addWidget(self._btn("Φόρτωση .wcs / .fits…", self.load_wcs))
        lay.addWidget(g)
        self.wcs_label = QLabel("")
        self.wcs_label.setWordWrap(True)
        lay.addWidget(self.wcs_label)
        self._set_wcs_text()
        lay.addWidget(self._intro("Το «Επόμενο» χρησιμοποιεί την κλίμακα που βλέπεις (ή τη λύση του plate solve αν υπάρχει)."))
        lay.addStretch(1)
        return w

    # --- 3
    def _page_stars(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        g, gl = self._group("Εντοπισμός αστεριών")
        self._prow(gl, "star_sigma", "Κατώφλι (σ)", "f", 3, 30, 0.5,
                   "Πόσο έντονο πρέπει να είναι ένα σημείο για να θεωρηθεί αστέρι (σε φορές τον θόρυβο). Μικρότερο = περισσότερα, πιο αμυδρά αστέρια (και κάποιες ψευδείς ανιχνεύσεις).")
        self._prow(gl, "max_stars", "Μέγιστος αριθμός", "i", 100, 30000, 500,
                   "Όριο στον αριθμό αστεριών που γίνονται 3D. Περισσότερα = πιο πυκνό πεδίο αστεριών, πιο αργή απόδοση.")
        lay.addWidget(g)

        g, gl = self._group("Αφαίρεση αστεριών (starless)")
        self.sl_group = QButtonGroup(self)
        self.rb_sn = QRadioButton("StarNet (starnet2 / StarNet++) – αυτόματα, μέσα από την εφαρμογή")
        self.rb_file = QRadioButton("Έχω έτοιμο αρχείο starless (StarXTerminator, StarNet, κλπ.)")
        self.rb_auto = QRadioButton("Ενσωματωμένη μέθοδος (απλή, χωρίς εξωτερικό πρόγραμμα)")
        for i, rb in enumerate((self.rb_sn, self.rb_file, self.rb_auto)):
            self.sl_group.addButton(rb, i)
            gl.addWidget(rb)
        (self.rb_sn if self.proj.starnet_path else self.rb_auto).setChecked(True)
        self.sl_group.idToggled.connect(lambda *_: (self._invalidate(stars=True), self._sl_visibility()))
        self.sn_box = QWidget()
        sl = QVBoxLayout(self.sn_box)
        sl.setContentsMargins(18, 0, 0, 0)
        self.sn_edit = QLineEdit(self.proj.starnet_path or "")
        self.sn_edit.setPlaceholderText(r"διαδρομή starnet2.exe  (π.χ. C:\Program Files\StarNet2\bin\starnet2.exe)")
        row = QHBoxLayout()
        row.addWidget(self.sn_edit, 1)
        row.addWidget(self._btn("…", self._browse_starnet))
        sl.addLayout(row)
        self.sn_adv = QCheckBox("Προχωρημένα: πρότυπο ορισμάτων / stride")
        sl.addWidget(self.sn_adv)
        self.sn_adv_box = QWidget()
        al = QFormLayout(self.sn_adv_box)
        self.sn_args = QLineEdit(self.proj.starnet_args)
        self.sn_args.setPlaceholderText("κενό = αυτόματο")
        self.sn_args.setToolTip("{in} = αρχείο εισόδου, {out} = αρχείο εξόδου, {stride} = τιμή stride. Κενό = αυτόματο: starnet2 → -i {in} -o {out} -s {stride} · StarNet++ v1 → {in} {out} {stride}")
        self.sn_stride = QSpinBox()
        self.sn_stride.setRange(0, 1024)
        self.sn_stride.setSpecialValueText("προεπιλογή")
        al.addRow("Ορίσματα:", self.sn_args)
        al.addRow("Stride:", self.sn_stride)
        self.sn_adv_box.setVisible(False)
        self.sn_adv.toggled.connect(self.sn_adv_box.setVisible)
        sl.addWidget(self.sn_adv_box)
        gl.addWidget(self.sn_box)
        self.file_box = QWidget()
        fl = QVBoxLayout(self.file_box)
        fl.setContentsMargins(18, 0, 0, 0)
        fl.addWidget(self._btn("Επιλογή αρχείου starless…", self.open_starless))
        self.starless_label = QLabel("—")
        self.starless_label.setWordWrap(True)
        fl.addWidget(self.starless_label)
        gl.addWidget(self.file_box)
        lay.addWidget(g)

        g, gl = self._group("Αποστάσεις αστεριών")
        self.dist_group = QButtonGroup(self)
        self.rb_gaia = QRadioButton("Πραγματικές (Gaia DR3) – χρειάζεται internet και plate solve")
        self.rb_syn = QRadioButton("Συνθετικές (εύλογες, όχι μετρήσεις)")
        for i, rb in enumerate((self.rb_gaia, self.rb_syn)):
            self.dist_group.addButton(rb, i)
            gl.addWidget(rb)
        self.rb_syn.setChecked(True)
        self.dist_group.idToggled.connect(lambda *_: self._invalidate(stars=True))
        self._prow(gl, "gaia_glim", "Gaia: όριο G", "f", 12, 21, 0.5,
                   "Πόσο αμυδρά αστέρια του καταλόγου Gaia κατεβαίνουν για αντιστοίχιση (μεγαλύτερο = πιο αμυδρά, πιο αργό κατέβασμα). 15.5 αρκεί συνήθως.")
        self._prow(gl, "match_tol_px", "Ανοχή αντιστοίχισης (px)", "f", 1, 10, 0.5,
                   "Πόσο μακριά (σε pixel) μπορεί να πέφτει ένα αστέρι της φωτογραφίας από το αντίστοιχο του Gaia ώστε να θεωρηθούν το ίδιο αστέρι.")
        lay.addWidget(g)

        g, gl = self._group("Αστέρια εκτός κάδρου (για περιστροφή της κάμερας)")
        self._prow(gl, "extra_enable", "Ενεργό", "b", 0, 1, 0,
                   "Προσθέτει αστέρια του Gaia ΕΚΤΟΣ του κάδρου της φωτογραφίας. Χωρίς αυτά, όταν το οπτικό πεδίο ανοίγει στην πτήση, γύρω από το νεφέλωμα θα φαινόταν κενό.")
        self._prow(gl, "extra_radius_deg", "Ακτίνα κώνου (°, 0=αυτόματο)", "f", 0, 20, 0.5,
                   "Πόσο μακριά από το κέντρο της φωτογραφίας (σε μοίρες ουρανού) προστίθενται αστέρια. 0 = αυτόματα με βάση το πεδίο.")
        self._prow(gl, "extra_glim", "Gaia όριο G (εκτός κάδρου)", "f", 8, 17, 0.5,
                   "Πόσο αμυδρά είναι τα επιπλέον αστέρια (μικρότερος αριθμός = λιγότερα, μόνο τα φωτεινότερα).")
        lay.addWidget(g)

        lay.addWidget(self._big("▶  Επεξεργασία αστεριών", self.process_stars))
        self.star_label = QLabel("")
        self.star_label.setWordWrap(True)
        lay.addWidget(self.star_label)
        self.show_sl = QCheckBox("Δείξε την starless στην προεπισκόπηση")
        self.show_sl.toggled.connect(lambda _b: self._show_preview())
        lay.addWidget(self.show_sl)
        lay.addStretch(1)
        self._sl_visibility()
        return w

    def _sl_visibility(self):
        self.sn_box.setVisible(self.rb_sn.isChecked())
        self.file_box.setVisible(self.rb_file.isChecked())

    # --- 4
    def _page_depth(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        g, gl = self._group("Αντικείμενο")
        self._prow(gl, "object_distance_pc", "Απόσταση αντικειμένου (pc)", "f", 5, 30000, 1,
                   "Πόσο μακριά είναι το νεφέλωμα. Π.χ. M42 ≈ 412 pc, Πλειάδες ≈ 136 pc, Rosette ≈ 1500 pc (1 pc = 3.26 έτη φωτός). Καθορίζει την κλίμακα του 3D χώρου.")
        self.suggest_label = self._note("")
        gl.addWidget(self.suggest_label)
        self.suggest_btn = self._btn("Χρήση προτεινόμενης απόστασης", self._apply_suggest)
        gl.addWidget(self.suggest_btn)
        lay.addWidget(g)

        g, gl = self._group("Χάρτης βάθους νεφελώματος")
        self._spec_rows(gl, "Βάθος νεφελώματος")
        b = QHBoxLayout()
        b.addWidget(self._btn("Φόρτωση δικού μου χάρτη…", self.load_depth_map))
        b.addWidget(self._btn("Εξαγωγή τρέχοντος χάρτη…", self.export_depth_map))
        gl.addLayout(b)
        self.ai_btn = self._btn("Υπολογισμός AI χάρτη (Depth Anything)", self.ai_depth)
        self.ai_btn.setEnabled(depth_ai.available())
        if not depth_ai.available():
            self.ai_btn.setToolTip("pip install torch transformers pillow")
        gl.addWidget(self.ai_btn)
        self.depth_img = QLabel()
        self.depth_img.setMinimumHeight(140)
        self.depth_img.setAlignment(Qt.AlignCenter)
        gl.addWidget(self.depth_img)
        lay.addWidget(g)
        g, gl = self._group("Όγκος")
        self._spec_rows(gl, "Όγκος (ποιότητα)")
        lay.addWidget(g)
        lay.addWidget(self._big("▶  Χτίσιμο 3D σκηνής", self.build_scene))
        lay.addStretch(1)
        return w

    # --- 5
    def _page_look(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        g, gl = self._group("Απόδοση (ζωντανά)")
        self._spec_rows(gl, "Απόδοση")
        lay.addWidget(g)
        lay.addStretch(1)
        return w

    # --- 6
    def _page_video(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        g, gl = self._group("Πτήση μέσα στο νεφέλωμα (προτεινόμενη)")
        gl.addWidget(self._intro("Ευθεία πτήση από το σημείο λήψης προς το εσωτερικό του αντικειμένου: περνάς ανάμεσα από τα αστέρια "
                                 "και βυθίζεσαι μέσα στα νέφη. Χωρίς περιστροφή γύρω από το αντικείμενο."))
        self._prow(gl, "dive_auto_poi", "Αυτόματος στόχος", "b", 0, 1, 0, "Η κάμερα πετά προς τη φωτεινότερη περιοχή του νεφελώματος. Ξετσέκαρέ το για να ορίσεις εσύ στόχο με τα X/Y παρακάτω.")
        self._prow(gl, "dive_fx", "Στόχος X (−1 αριστερά … 1 δεξιά)", "f", -0.9, 0.9, 0.01,
                   "Προς ποιο σημείο της φωτογραφίας (οριζόντια) πετά η κάμερα. 0 = κέντρο.")
        self._prow(gl, "dive_fy", "Στόχος Y (−1 κάτω … 1 πάνω)", "f", -0.9, 0.9, 0.01,
                   "Προς ποιο σημείο της φωτογραφίας (κατακόρυφα) πετά η κάμερα. 0 = κέντρο.")
        self._prow(gl, "dive_depth", "Βάθος τέλους (0 μπροστά … 1 πίσω)", "f", 0.0, 1.0, 0.01,
                   "Πόσο μέσα στα νέφη τελειώνει η πτήση. Μικρή τιμή = μένεις στα εξωτερικά, φωτεινά στρώματα· μεγάλη = βαθιά μέσα (πιο σκοτεινά).")
        self._prow(gl, "dive_zoom", "Zoom προσέγγισης (×)", "f", 1.0, 8.0, 0.1,
                   "Πόσες φορές μεγαλύτερο φαίνεται το νεφέλωμα όταν η κάμερα φτάνει στην είσοδό του. 1 = όσο στη φωτογραφία, μεγάλο = «ξεχειλίζει» από το κάδρο.")
        self._prow(gl, "dive_zoom_rate", "Μέγιστη ταχύτητα ζουμ", "f", 0.05, 1.0, 0.01,
                   "Πόσο γρήγορα επιτρέπεται να μεγαλώνει το νεφέλωμα στο κάδρο (0.1 = πολύ αργά, 0.25 = ήρεμα, 0.5 = γρήγορα). Μικρότερο = πιο αργό ζουμ, αλλά και μεγαλύτερη ελάχιστη διάρκεια.")
        self._prow(gl, "dive_fov_max", "Μέγιστο οπτικό πεδίο (°)", "f", 20, 100, 1,
                   "Πόσο ανοίγει το κάδρο όταν η κάμερα μπαίνει μέσα στα νέφη (κατακόρυφη γωνία). Μεγαλύτερο = βλέπεις περισσότερα στρώματα γύρω σου.")
        self._prow(gl, "dive_drift", "Ήπια καμπύλη (× πλάτος)", "f", -0.3, 0.3, 0.01, "Πόσο ξεφεύγει πλάγια η διαδρομή στη μέση (σε πλάτη νεφελώματος). 0 = απολύτως ευθεία γραμμή.")
        gl.addWidget(self._big("✈  Δημιουργία πτήσης μέσα στα νέφη", self.make_dive_path))
        lay.addWidget(g)
        g, gl = self._group("Διαδρομή κάμερας (κυκλική, παλιά)")
        form = QFormLayout()
        self.end_dist = QDoubleSpinBox()
        self.end_dist.setRange(0.02, 5)
        self.end_dist.setSingleStep(0.05)
        self.end_dist.setValue(0.6)
        self.end_dist.setToolTip("Τελική απόσταση από το κέντρο, σε πλάτη νεφελώματος")
        self.end_az = QDoubleSpinBox()
        self.end_az.setRange(-180, 180)
        self.end_az.setValue(20)
        self.end_el = QDoubleSpinBox()
        self.end_el.setRange(-85, 85)
        self.end_el.setValue(8)
        form.addRow("Τελική απόσταση (× πλάτος):", self.end_dist)
        form.addRow("Τελική γωνία az (°):", self.end_az)
        form.addRow("Τελική γωνία el (°):", self.end_el)
        gl.addLayout(form)
        gl.addWidget(self._btn("Δημιουργία κυκλικής πτήσης", self.make_flythrough))
        self.kf_list = QListWidget()
        self.kf_list.setMaximumHeight(130)
        self.kf_list.currentRowChanged.connect(self._kf_selected)
        gl.addWidget(self.kf_list)
        row = QHBoxLayout()
        self.kf_time = QDoubleSpinBox()
        self.kf_time.setRange(0, 600)
        self.kf_time.setSuffix(" s")
        row.addWidget(self.kf_time)
        row.addWidget(self._btn("Προσθήκη keyframe", self.add_kf, "Αποθηκεύει την τρέχουσα θέση της κάμερας"))
        row.addWidget(self._btn("Αντικατάσταση", self.replace_kf))
        row.addWidget(self._btn("Διαγραφή", self.del_kf))
        gl.addLayout(row)
        self._prow(gl, "duration", "Διάρκεια (s)", "f", 1, 300, 0.5, "Πόσο διαρκεί το βίντεο, σε δευτερόλεπτα.")
        self._prow(gl, "ease", "Ομαλή εκκίνηση/τέλος", "b", 0, 1, 0,
                   "Εφαρμόζεται μόνο σε κυκλικές/χειροκίνητες διαδρομές (η ευθεία πτήση έχει δική της ομαλή εκκίνηση και προσγείωση).")
        lay.addWidget(g)
        g, gl = self._group("Εξαγωγή")
        form = QFormLayout()
        self.res_combo = QComboBox()
        for name, wh in (("1280×720", (1280, 720)), ("1920×1080", (1920, 1080)), ("2560×1440", (2560, 1440)),
                         ("3840×2160 (4K)", (3840, 2160)), ("1080×1080", (1080, 1080)), ("1080×1920 (κάθετο)", (1080, 1920))):
            self.res_combo.addItem(name, wh)
        self.res_combo.setCurrentIndex(1)
        self.res_combo.currentIndexChanged.connect(self._res_changed)
        form.addRow("Ανάλυση:", self.res_combo)
        gl.addLayout(form)
        self._prow(gl, "fps", "FPS", "i", 12, 120, 1, "Καρέ ανά δευτερόλεπτο του βίντεο. 30 είναι το τυπικό, 60 πιο ρευστό (διπλάσιος χρόνος εξαγωγής).")
        self._prow(gl, "ssaa", "Antialiasing (SSAA ×)", "i", 1, 3, 1, "2 = πολύ καλύτερη ποιότητα, 4× πιο αργό")
        gl.addWidget(self._big("🎬  Εξαγωγή MP4…", self.export_mp4))
        gl.addWidget(self._btn("Εξαγωγή ακολουθίας PNG…", self.export_png))
        gl.addWidget(self._btn("Αποθήκευση τρέχοντος καρέ (PNG)…", self.save_frame))
        lay.addWidget(g)
        lay.addStretch(1)
        return w

    # ================================================================== βοηθητικά
    def _log(self, s):
        self.logbox.appendPlainText(s)
        self.logbox.verticalScrollBar().setValue(self.logbox.verticalScrollBar().maximum())

    def _param_changed(self, key):
        if key in DIVE_KEYS and self.proj.scene is not None and self.proj.path_kind == "dive":
            self._regen_timer.start(250)
        if key in STAR_KEYS:
            self._invalidate(stars=True)
            if key in ("stretched_input", "gamma"):
                self._prev_timer.start(400)
        elif key in BUILD_KEYS:
            self._invalidate()
        self.view.update()

    def _refresh_rows(self):
        for r in self.rows:
            r.refresh()

    def _err(self, msg):
        self.progress.setValue(0)
        self._set_busy(False)
        self._log("ΣΦΑΛΜΑ: " + msg.replace("\x00", ""))
        QMessageBox.warning(self, "Σφάλμα", msg.split("\x00")[0].strip()[:1500])

    def _release(self, wk):
        QTimer.singleShot(3000, lambda: self._tasks.discard(wk))   # μικρή καθυστέρηση: το νήμα μπορεί να τελειώνει ακόμη

    def run_task(self, fn, done=None, label=""):
        wk = Worker(fn)
        self._tasks.add(wk)
        wk.s.progress.connect(lambda f, m: (self.progress.setValue(int(f * 1000)), self.statusBar().showMessage(m or label)))
        wk.s.error.connect(lambda m, w=wk: (self._release(w), self._err(m)))

        def _done(res):
            self._release(wk)
            self.progress.setValue(1000)
            self._set_busy(False)
            if done:
                done(res)
        wk.s.done.connect(_done)
        self.progress.setValue(0)
        self.statusBar().showMessage(label)
        self._set_busy(True)
        self.pool.start(wk)

    def need_image(self):
        if self.proj.lin is None:
            QMessageBox.information(self, "AstroFlight", "Άνοιξε πρώτα μια εικόνα (βήμα 1).")
            return False
        return True

    def _show_preview(self):
        p = self.proj
        if p.lin is None:
            self.preview.clear_image()
            return
        p.relinearize()
        src = p.lin
        if self.step == 2 and getattr(self, "show_sl", None) is not None and self.show_sl.isChecked() and p.starless_lin is not None:
            src = p.starless_lin
        try:
            tone = io.tone_params(p.lin, bool(p.p.stretched_input), float(p.p.gamma))
            a = io.apply_tone(src, tone)
            if max(a.shape[:2]) > 1800:
                import cv2
                k = 1800.0 / max(a.shape[:2])
                a = cv2.resize(a, (int(a.shape[1] * k), int(a.shape[0] * k)), interpolation=cv2.INTER_AREA)
            self.preview.set_array(a)
        except Exception as e:  # noqa
            self._log(f"Προεπισκόπηση: {e}")

    # ================================================================== βήμα 1: αρχεία
    def open_image(self):
        d = self.settings.value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "Άνοιγμα εικόνας", d, "Εικόνες (*.tif *.tiff *.png *.jpg *.jpeg)")
        if not path:
            return
        self.settings.setValue("last_dir", os.path.dirname(path))
        self.run_task(lambda prog: self.proj.load_image(path), lambda r: self._after_load(True), "Φόρτωση εικόνας…")

    def _after_load(self, advance=False):
        p = self.proj
        self.stale_stars = self.stale_scene = False
        self.img_label.setText(f"{os.path.basename(p.image_path)}  –  {p.W}×{p.H}px" + (f"  (μικρύνθηκε από το πρωτότυπο ×{p.down:.2f})" if p.down < 1 else ""))
        self.star_label.setText("")
        self.suggest_label.setText("")
        self.starless_label.setText("—")
        self._set_wcs_text()
        self.show_sl.setChecked(False)
        self._show_preview()
        self.rstack.setCurrentWidget(self.preview)
        self.playbar.setVisible(False)
        self.mouse_hint.setVisible(False)
        self.auto.refresh_image()
        self._update_nav()
        if advance:
            self._goto(1)
            self.statusBar().showMessage("Βήμα 2: όρισε την κλίμακα της εικόνας.")

    def open_starless(self):
        if not self.need_image():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Starless εικόνα", self.settings.value("last_dir", ""), "Εικόνες (*.tif *.tiff *.png *.jpg)")
        if path:
            def done(_):
                self.starless_label.setText(os.path.basename(path))
                self._invalidate(stars=True)
            self.run_task(lambda prog: self.proj.load_starless(path), done, "Φόρτωση starless…")

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Αποθήκευση project", "", "AstroFlight (*.afproj.json)")
        if path:
            self.proj.save(path)

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Άνοιγμα project", "", "AstroFlight (*.json)")
        if not path:
            return

        def job(prog):
            self.proj.load(path)

        def done(_):
            self._refresh_rows()
            if self.proj.lin is not None:
                self._after_load(True)
                if self.proj.scale_arcsec:
                    self.scale_spin.setValue(self.proj.scale_arcsec * self.proj.down)
                    self.stale_scene = False
                self.astap_edit.setText(self.proj.astap_path or "")
                self.sn_edit.setText(self.proj.starnet_path or "")
                self.sn_args.setText(self.proj.starnet_args)
            self._refresh_kf()
        self.run_task(job, done, "Φόρτωση project…")

    # ================================================================== βήμα 2: κλίμακα & θέση
    def _set_wcs_text(self):
        p = self.proj
        if p.sky is not None:
            c = p.sky.center(p.W)
            self.wcs_label.setText(f"✔ Αστρομετρική λύση: RA {c[0]:.3f}°  Dec {c[1]:.3f}°  –  {p.scale_arcsec:.3f}\"/px. Θα χρησιμοποιηθούν πραγματικές αποστάσεις Gaia.")
        elif p.scale_arcsec:
            self.wcs_label.setText(f"Κλίμακα {p.scale_arcsec * p.down:.3f}\"/px, χωρίς συντεταγμένες – οι αποστάσεις αστεριών θα είναι συνθετικές.")
        else:
            self.wcs_label.setText("Χωρίς αστρομετρία: οι αποστάσεις αστεριών θα είναι συνθετικές.")

    def _calc_scale(self):
        s = ws.scale_from_optics(self.focal.value(), self.pixsz.value(), self.binning.value())
        self.scale_spin.setValue(s)

    def _ensure_scale(self):
        """Εφαρμόζει την κλίμακα του spinbox αν δεν υπάρχει αστρομετρική λύση."""
        if not self.need_image():
            return
        if self.proj.sky is None:
            old = self.proj.scale_arcsec
            self.proj.set_scale(self.scale_spin.value())
            if old is None or abs(old - self.proj.scale_arcsec) > 1e-9:
                self._invalidate()
            self._set_wcs_text()

    def _browse_astap(self):
        path, _ = QFileDialog.getOpenFileName(self, "ASTAP εκτελέσιμο")
        if path:
            self.astap_edit.setText(path)

    def solve_astap(self):
        if not self.need_image():
            return
        self.proj.astap_path = self.astap_edit.text().strip() or self.proj.astap_path
        self.settings.setValue("astap", self.proj.astap_path or "")
        fov = None
        if self.scale_spin.value() > 0:
            fov = self.proj.H * self.scale_spin.value() * self.proj.down / 3600.0
        ra = dec = None
        if self.ra_edit.text().strip() and self.dec_edit.text().strip():
            try:
                ra, dec = ws.parse_ra_dec(self.ra_edit.text(), self.dec_edit.text())
            except Exception:
                pass
        self.run_task(lambda prog: self.proj.solve_astap(fov, ra, dec), lambda r: self._after_solve(), "Plate solving (ASTAP)…")

    def _after_solve(self):
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(self.proj.scale_arcsec * self.proj.down)
        self.scale_spin.blockSignals(False)
        self.dist_group.button(0).setChecked(True)          # πραγματικές αποστάσεις Gaia
        self._invalidate(stars=True)
        self._set_wcs_text()

    def manual_wcs(self):
        if not self.need_image():
            return
        try:
            ra, dec = ws.parse_ra_dec(self.ra_edit.text(), self.dec_edit.text())
        except Exception:
            QMessageBox.warning(self, "AstroFlight", "Μη έγκυρα RA/Dec.")
            return
        self.proj.set_manual_wcs(ra, dec, self.scale_spin.value(), self.rot_spin.value())
        self.dist_group.button(0).setChecked(True)
        self._invalidate(stars=True)
        self._set_wcs_text()

    def load_wcs(self):
        if not self.need_image():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Αρχείο WCS", "", "WCS/FITS (*.wcs *.fits *.fit *.fts)")
        if path:
            try:
                self.proj.load_wcs(path)
                self._after_solve()
            except Exception as e:
                self._err(str(e))

    # ================================================================== βήμα 3: αστέρια
    def _browse_starnet(self):
        path, _ = QFileDialog.getOpenFileName(self, "StarNet εκτελέσιμο (starnet2.exe ή starnet++.exe)")
        if path:
            self.sn_edit.setText(path)

    def _refresh_stars_page(self):
        p = self.proj
        self.rb_gaia.setEnabled(p.sky is not None)
        if p.sky is None and self.rb_gaia.isChecked():
            self.rb_syn.setChecked(True)
        self.rb_gaia.setToolTip("" if p.sky is not None else "Χρειάζεται πρώτα plate solve (βήμα 2)")

    def process_stars(self):
        if not self.need_image():
            return
        p = self.proj
        method = ("starnet", "file", "auto")[self.sl_group.checkedId()]
        dist = "gaia" if self.rb_gaia.isChecked() else "synthetic"
        if method == "starnet":
            p.starnet_path = self.sn_edit.text().strip() or p.starnet_path
            if not p.starnet_path or not os.path.isfile(p.starnet_path):
                QMessageBox.information(self, "AstroFlight", "Δώσε τη διαδρομή του StarNet (το εκτελέσιμο, π.χ. starnet2.exe) ή διάλεξε άλλη μέθοδο.")
                return
            p.starnet_args = self.sn_args.text().strip()
            p.starnet_stride = int(self.sn_stride.value())
            self.settings.setValue("starnet", p.starnet_path)
        elif method == "file" and p.starless_img is None:
            QMessageBox.information(self, "AstroFlight", "Διάλεξε πρώτα το αρχείο starless.")
            return
        if dist == "gaia" and p.sky is None:
            dist = "synthetic"
        if p.scale_arcsec is None:
            self._ensure_scale()

        def job(prog):
            p.process_stars(method, dist, prog)

        def done(_):
            self.stale_stars = False
            self.stale_scene = True
            st = p.stars
            msg = f"✔ {len(st)} αστέρια (FWHM ≈ {np.median(st.fwhm):.1f}px) • starless: {('StarNet', 'αρχείο', 'ενσωματωμένη')[('starnet', 'file', 'auto').index(method)]} • "
            if p.dist and p.dist.get("zp") is not None:
                msg += f"αποστάσεις Gaia για {p.dist['n_matched']}/{len(st)}"
            else:
                msg += "αποστάσεις συνθετικές" + (" (το Gaia απέτυχε)" if getattr(p, "dist_fallback", False) else "")
            self.star_label.setText(msg)
            if p.suggest:
                self.suggest_label.setText(f"Πιθανή απόσταση σμήνους/νεφελώματος από τα δεδομένα Gaia: ~{p.suggest[0]:.0f} pc")
            self._show_preview()
            self._update_nav()
            self._goto(3)
        self.run_task(job, done, "Επεξεργασία αστεριών…")

    def _apply_suggest(self):
        self.proj.apply_suggested_distance()
        self._refresh_rows()
        self._invalidate()

    # ================================================================== βήμα 4: βάθος & σκηνή
    def _refresh_depth_page(self):
        s = self.proj.suggest
        self.suggest_btn.setVisible(bool(s))
        if s:
            self.suggest_label.setText(f"Πιθανή απόσταση σμήνους/νεφελώματος από τα δεδομένα Gaia: ~{s[0]:.0f} pc")

    def load_depth_map(self):
        path, _ = QFileDialog.getOpenFileName(self, "Χάρτης βάθους (ασπρόμαυρο)", "", "Εικόνες (*.tif *.tiff *.png *.jpg)")
        if path:
            self.proj.imported_depth = io.load_image(path).mean(axis=2)
            self.proj.p.w_import = max(self.proj.p.w_import, 1.0)
            self._refresh_rows()
            self._invalidate()
            self._log("Φορτώθηκε χάρτης βάθους (λευκό = κοντά).")

    def export_depth_map(self):
        sc = self.proj.scene
        if sc is None or sc.depth_preview is None:
            QMessageBox.information(self, "AstroFlight", "Χτίσε πρώτα τη σκηνή.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Αποθήκευση χάρτη βάθους", "depth.png", "PNG (*.png)")
        if path:
            d = 1.0 - sc.depth_preview
            io.save_png(path, np.repeat(d[..., None], 3, axis=2))

    def ai_depth(self):
        if not self.need_image():
            return

        def job(prog):
            self.proj.ai_depth = depth_ai.estimate_depth(self.proj.img, log=self.proj.log)

        def done(_):
            self.proj.p.w_ai = max(self.proj.p.w_ai, 0.5)
            self._refresh_rows()
            self._invalidate()
        self.run_task(job, done, "AI βάθος…")

    def build_scene(self):
        if not self.need_image():
            return
        if self.proj.scale_arcsec is None:
            self._ensure_scale()
        self.run_task(lambda prog: self.proj.build(prog), self._scene_ready, "Χτίσιμο σκηνής…")

    def _scene_ready(self, sc):
        self.stale_scene = False
        self.rstack.setCurrentWidget(self.view)
        self.view.set_scene(sc, self.proj.tone)
        self._refresh_kf()
        self._show_depth(sc)
        self._update_nav()
        self.auto.refresh_export_state()
        self._goto(4)
        self.statusBar().showMessage("Έτοιμο! Πειραματίσου με το ποντίκι· στο επόμενο βήμα ρύθμισε την εμφάνιση.")

    def _show_depth(self, sc):
        d = sc.depth_preview
        if d is None:
            return
        g = (np.clip(1.0 - d, 0, 1) * 255).astype(np.uint8)
        g = np.ascontiguousarray(g)
        qi = QImage(g.data, g.shape[1], g.shape[0], g.shape[1], QImage.Format_Grayscale8).copy()
        self.depth_img.setPixmap(QPixmap.fromImage(qi).scaled(300, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ================================================================== αυτόματη επεξεργασία: ολοκλήρωση
    def _auto_finished(self, summary):
        p = self.proj
        self.stale_stars = self.stale_scene = False
        self._refresh_rows()
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(p.scale_arcsec * p.down)
        self.scale_spin.blockSignals(False)
        self.dist_group.button(0 if summary.get("gaia") else 1).setChecked(True)
        (self.rb_sn if summary.get("starnet") else self.rb_auto).setChecked(True)
        self.stale_stars = self.stale_scene = False
        self._set_wcs_text()
        self.suggest_label.setText("")
        self.rstack.setCurrentWidget(self.view)
        self.view.set_scene(p.scene, p.tone)
        self._refresh_kf()
        self._show_depth(p.scene)
        self.view.set_time(0.0)
        self.t_slider.blockSignals(True)
        self.t_slider.setValue(0)
        self.t_slider.blockSignals(False)
        self.playbar.setVisible(True)
        self.mouse_hint.setVisible(True)
        self._update_nav()
        self.statusBar().showMessage("Έτοιμο! Πάτα «Αναπαραγωγή» (Space) για να δεις την πτήση.")

    def _show_flight(self):
        if self.proj.scene is None:
            return
        self.tabs.setCurrentIndex(1)
        self._goto(5)
        self.view.set_time(0.0)
        if not self.view.playing:
            self._toggle_play()

    def _tab_changed(self, i):
        if i == 0:
            gl = self.proj.scene is not None and not self.stale_scene
            self.rstack.setCurrentWidget(self.view if gl else self.preview)
            self.playbar.setVisible(gl)
            self.mouse_hint.setVisible(gl)
        else:
            self._on_enter(self.step)

    def _regen_path(self):
        if self.proj.scene is None or self.proj.path_kind != "dive":
            return
        self.proj.make_dive_path()
        for r in self.rows:
            if r.key in ("dive_fx", "dive_fy"):
                r.refresh()
        self._refresh_kf()
        self.view.update()

    # ================================================================== βήμα 6: κάμερα / keyframes
    def _update_path(self):
        self.view.path = CameraPath(self.proj.keyframes, self.proj.p.duration, self.proj.p.ease) if self.proj.keyframes else None

    def _refresh_kf(self):
        self.kf_list.blockSignals(True)
        self.kf_list.clear()
        for k in self.proj.keyframes:
            s = k.state
            self.kf_list.addItem(f"t={k.t:5.2f}s  az={s.az:6.1f}° el={s.el:5.1f}° dist×{math.exp(s.logr):.3f}"
                                 + (f"  yaw={s.yaw:.1f}° pitch={s.pitch:.1f}°" if abs(s.yaw) + abs(s.pitch) > 0.05 else ""))
        self.kf_list.blockSignals(False)
        self._update_path()

    def _kf_selected(self, row):
        if 0 <= row < len(self.proj.keyframes):
            k = self.proj.keyframes[row]
            self.view.cam = CamState(**k.state.to_dict())
            self.kf_time.setValue(k.t)
            self.view.update()
            self._update_cam_label()

    def make_dive_path(self):
        if self.proj.scene is None:
            QMessageBox.information(self, "AstroFlight", "Χτίσε πρώτα τη σκηνή.")
            return
        self.proj.make_dive_path()
        for r in self.rows:
            if r.key in ("dive_fx", "dive_fy"):
                r.refresh()
        self._refresh_kf()
        self.view.cam = CamState(**self.proj.keyframes[0].state.to_dict())
        self.view.update()

    def make_flythrough(self):
        sc = self.proj.scene
        if sc is None:
            QMessageBox.information(self, "AstroFlight", "Χτίσε πρώτα τη σκηνή.")
            return
        m = sc.meta
        self.proj.keyframes = make_flythrough(m["D0"], m["W0"], self.end_dist.value(), self.end_az.value(),
                                              self.end_el.value(), self.proj.p.duration)
        self.proj.path_kind = "orbit"
        self._refresh_kf()

    def add_kf(self):
        self.proj.path_kind = "custom"
        t = self.kf_time.value()
        ks = [k for k in self.proj.keyframes if abs(k.t - t) > 1e-3]
        ks.append(Keyframe(t, CamState(**self.view.cam.to_dict())))
        self.proj.keyframes = sorted(ks, key=lambda k: k.t)
        self._refresh_kf()

    def replace_kf(self):
        self.proj.path_kind = "custom"
        r = self.kf_list.currentRow()
        if 0 <= r < len(self.proj.keyframes):
            self.proj.keyframes[r] = Keyframe(self.kf_time.value(), CamState(**self.view.cam.to_dict()))
            self.proj.keyframes.sort(key=lambda k: k.t)
            self._refresh_kf()

    def del_kf(self):
        self.proj.path_kind = "custom"
        r = self.kf_list.currentRow()
        if 0 <= r < len(self.proj.keyframes):
            del self.proj.keyframes[r]
            self._refresh_kf()

    def _toggle_play(self):
        self._update_path()
        if self.view.path is None:
            QMessageBox.information(self, "AstroFlight", "Δεν υπάρχει διαδρομή – πάτα «Δημιουργία πτήσης» στο βήμα 6.")
            return
        if not self.view.playing and self.view._play_t >= self.proj.p.duration - 1e-3:
            self.view._play_t = 0.0
        self.view.toggle_play()
        self.play_btn.setText("⏸ Παύση" if self.view.playing else "▶ Αναπαραγωγή")

    def _on_play_time(self, t):
        self.t_slider.blockSignals(True)
        self.t_slider.setValue(int(1000 * t / max(self.proj.p.duration, 1e-6)))
        self.t_slider.blockSignals(False)
        if not self.view.playing:
            self.play_btn.setText("▶ Αναπαραγωγή")

    def _scrub(self, v):
        self._update_path()
        self.view.set_time(self.proj.p.duration * v / 1000.0)

    def _update_cam_label(self):
        c = self.view.cam
        self.cam_label.setText(f"az {c.az:.0f}°  el {c.el:.0f}°  απόσταση ×{math.exp(c.logr):.3f}")

    def _res_changed(self):
        w, h = self.res_combo.currentData()
        self.proj.p.out_width, self.proj.p.out_height = w, h

    # ================================================================== εξαγωγή
    def _export(self, png_sequence: bool):
        if self.proj.scene is None or self.view.renderer is None:
            QMessageBox.information(self, "AstroFlight", "Χτίσε πρώτα τη σκηνή.")
            return
        self._update_path()
        if self.view.path is None:
            QMessageBox.information(self, "AstroFlight", "Δεν υπάρχει διαδρομή κάμερας.")
            return
        if png_sequence:
            out = QFileDialog.getExistingDirectory(self, "Φάκελος για τα καρέ PNG")
        else:
            out, _ = QFileDialog.getSaveFileName(self, "Αποθήκευση βίντεο", "flight.mp4", "MP4 (*.mp4)")
        if not out:
            return
        p = self.proj.p
        n = int(round(p.duration * p.fps))
        dlg = QProgressDialog("Απόδοση καρέ…", "Ακύρωση", 0, 1000, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        cancelled = {"v": False}
        dlg.canceled.connect(lambda: cancelled.__setitem__("v", True))

        def prog(f):
            dlg.setValue(int(f * 1000))
            dlg.setLabelText(f"Απόδοση καρέ {int(f * n)}/{n}")
            QApplication.processEvents()
        t0 = time.time()
        self.view.makeCurrent()
        try:
            ex.render_video(self.view.renderer, self.view.path, p, out, prog, lambda: cancelled["v"], self.proj.log, png_sequence)
        except Exception as e:  # noqa
            self._err(str(e))
        finally:
            self.view.doneCurrent()
            dlg.close()
        self._log(f"Χρόνος εξαγωγής: {time.time() - t0:.0f} s")

    def export_mp4(self):
        self._export(False)

    def export_png(self):
        self._export(True)

    def save_frame(self):
        if self.proj.scene is None or self.view.renderer is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Αποθήκευση καρέ", "frame.png", "PNG (*.png)")
        if not path:
            return
        m = self.proj.scene.meta
        p = self.proj.p
        self.view.makeCurrent()
        try:
            a = self.view.renderer.render_to_array(pose_from_state(self.view.cam, m["D0"], m["W0"]), p,
                                                   (p.out_width, p.out_height), max(1, p.ssaa),
                                                   fov_mul=math.exp(self.view.cam.logfov))
        finally:
            self.view.doneCurrent()
        io.save_png(path, a / 255.0)


def apply_dark_theme(app):
    """Σκούρο θέμα (Fusion) – ταιριάζει με το σκούρο παράθυρο προεπισκόπησης και είναι πιο άνετο για αστροφωτογραφίες."""
    from PySide6.QtGui import QPalette, QColor
    app.setStyle("Fusion")
    pal = QPalette()
    base, alt, text, hl = QColor(30, 31, 38), QColor(40, 42, 52), QColor(220, 222, 232), QColor(59, 95, 192)
    for role, col in ((QPalette.Window, QColor(37, 38, 46)), (QPalette.WindowText, text), (QPalette.Base, base),
                      (QPalette.AlternateBase, alt), (QPalette.ToolTipBase, alt), (QPalette.ToolTipText, text),
                      (QPalette.Text, text), (QPalette.Button, QColor(50, 52, 64)), (QPalette.ButtonText, text),
                      (QPalette.BrightText, QColor(255, 255, 255)), (QPalette.Highlight, hl), (QPalette.HighlightedText, QColor(255, 255, 255)),
                      (QPalette.PlaceholderText, QColor(120, 124, 140))):
        pal.setColor(role, col)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(110, 113, 128))
    app.setPalette(pal)


def main():
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

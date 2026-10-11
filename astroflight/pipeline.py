"""Αυτόματη επεξεργασία: φωτογραφία → ASTAP → αστέρια → StarNet → Gaia → 3D σκηνή → πτήση.

Κάθε βήμα έχει εφεδρική λύση (fallback) ώστε το αποτέλεσμα να βγαίνει ακόμη κι αν λείπει ένα εργαλείο:
 • χωρίς ASTAP / αποτυχία plate solve  → συνθετικές αποστάσεις αστεριών (απαιτείται κλίμακα arcsec/pixel)
 • χωρίς StarNet / αποτυχία           → ενσωματωμένη αφαίρεση αστεριών (λιγότερο καθαρή)
 • χωρίς internet / αποτυχία Gaia     → συνθετικές αποστάσεις
Η πρόοδος αναφέρεται μέσω αντικειμένου Reporter (το GUI και το CLI δίνουν τα δικά τους callbacks)."""
from __future__ import annotations
import math, os, time
from dataclasses import dataclass, field

import numpy as np

from .runner import CancelToken, Cancelled
from . import wcs_solve as ws
from . import starnet as sn
from . import objects as obj

# (id, τίτλος, βάρος στη συνολική πρόοδο)
STEPS = [
    ("image", "Ανάλυση εικόνας", 0.03),
    ("solve", "Αστρομετρία (ASTAP plate solve)", 0.17),
    ("stars", "Εντοπισμός αστεριών", 0.05),
    ("starless", "Αφαίρεση αστεριών (StarNet)", 0.30),
    ("gaia", "Αποστάσεις αστεριών (Gaia)", 0.12),
    ("object", "Αναγνώριση αντικειμένου & απόσταση", 0.02),
    ("scene", "Χτίσιμο 3D σκηνής", 0.29),
    ("flight", "Σχεδιασμός πτήσης", 0.02),
]
STEP_TITLES = {k: t for k, t, _ in STEPS}
_W = {k: w for k, _, w in STEPS}
_START = {}
_acc = 0.0
for _k, _t, _w in STEPS:
    _START[_k] = _acc
    _acc += _w


@dataclass
class AutoOptions:
    use_astap: bool = True
    use_starnet: bool = True
    use_gaia: bool = True
    scale_arcsec: float | None = None      # arcsec/pixel της ΠΡΩΤΟΤΥΠΗΣ εικόνας (από οπτικά). None = άγνωστη
    object_name: str = ""                  # προαιρετικό: βοηθά στο plate solve και στην αναγνώριση
    ra_deg: float | None = None
    dec_deg: float | None = None
    auto_stretch: bool = True              # αυτόματος εντοπισμός «τεντωμένης» εικόνας
    auto_distance: bool = True             # απόσταση αντικειμένου από Gaia/κατάλογο
    solve_timeout: int = 300
    reuse: bool = True                     # αν δεν άλλαξε τίποτα στα βήματα 1–5, δεν τα ξανατρέχει (π.χ. StarNet)


class Reporter:
    """Δέχεται callbacks: on_step(id, state, msg), on_frac(f, msg), on_log(text).
    state ∈ running | ok | warn | skip | fail"""

    def __init__(self, on_step=None, on_frac=None, on_log=None):
        self.on_step = on_step or (lambda *a: None)
        self.on_frac = on_frac or (lambda *a: None)
        self.on_log = on_log or (lambda *a: None)
        self._cur = None

    def step(self, sid: str, state: str, msg: str = ""):
        self._cur = sid
        self.on_step(sid, state, msg)
        if state in ("ok", "warn", "skip", "fail"):
            self.on_frac(_START[sid] + _W[sid], msg)
        else:
            self.on_frac(_START[sid], msg)

    def sub(self, sid: str, f: float, msg: str = ""):
        f = min(max(f, 0.0), 1.0)
        self.on_frac(_START[sid] + _W[sid] * f, msg)

    def tick_fn(self, sid: str, label: str, est_s: float):
        """tick για εξωτερικά προγράμματα με άγνωστη διάρκεια: ασυμπτωτική πρόοδος + χρόνος που πέρασε."""
        def tick(el: float, last: str):
            f = 1.0 - math.exp(-el / max(est_s, 1.0))
            msg = f"{label} – {el:.0f} s"
            self.sub(sid, 0.95 * f, msg)
            self.on_step(sid, "running", msg)
        return tick

    def log(self, s: str):
        self.on_log(s)


def looks_stretched(lin_or_img: np.ndarray) -> bool:
    """Εικόνες με διάμεσο φωτεινότητας > ~4% θεωρούνται ήδη τεντωμένες (non-linear)."""
    a = lin_or_img[::4, ::4]
    return float(np.median(a)) > 0.04


def run_auto(proj, opts: AutoOptions, rep: Reporter, cancel: CancelToken | None = None) -> dict:
    """Εκτελεί όλα τα βήματα. Απαιτεί ήδη φορτωμένη εικόνα (proj.load_image). Επιστρέφει περίληψη (dict)."""
    cancel = cancel or CancelToken()
    p = proj.p
    old_log = proj.log
    proj.log = lambda s: (rep.log(str(s)), None)[1]
    summary: dict = dict(warnings=[], solved=False, starnet=False, gaia=False, object=None)
    t_all = time.time()

    def warn(msg: str):
        summary["warnings"].append(msg)
        rep.log("⚠ " + msg)

    try:
        if proj.img is None:
            raise RuntimeError("Δεν έχει φορτωθεί εικόνα.")

        sig = (proj.image_path, opts.use_astap, opts.use_starnet, opts.use_gaia, opts.scale_arcsec, opts.object_name.strip(),
               opts.auto_stretch, proj.astap_path, proj.starnet_path, p.star_sigma, p.max_stars, p.gaia_glim, p.match_tol_px,
               p.extra_enable, p.extra_radius_deg, p.extra_glim, None if opts.auto_stretch else p.stretched_input)
        if (opts.reuse and getattr(proj, "_auto_sig", None) == sig and proj.stars is not None and proj.dist is not None
                and proj.starless_lin is not None and proj.scale_arcsec is not None):
            for sid in ("image", "solve", "stars", "starless", "gaia"):
                rep.step(sid, "ok", "Από την προηγούμενη εκτέλεση (αμετάβλητο)")
            summary.update(solved=proj.sky is not None, starnet=bool(proj.starless_user), gaia=bool(proj.dist.get("zp") is not None))
            return _finish(proj, opts, rep, cancel, summary, warn, reuse_mode=True)

        # ---------------------------------------------------------------- 1. εικόνα
        rep.step("image", "running", "Ανάλυση εικόνας…")
        if opts.auto_stretch:
            p.stretched_input = bool(looks_stretched(proj.img.mean(axis=2)))
            rep.log(f"Τύπος δεδομένων: {'τεντωμένη (non-linear)' if p.stretched_input else 'γραμμική (linear)'} – "
                    f"αυτόματος εντοπισμός από τη φωτεινότητα.")
        proj.reset_derived()
        proj.relinearize()
        scale_orig = opts.scale_arcsec
        hint_ra, hint_dec = opts.ra_deg, opts.dec_deg
        named = None
        if opts.object_name.strip():
            named = obj.lookup_name(opts.object_name)
            if named:
                hint_ra, hint_dec = named[0], named[1]
                rep.log(f"Όνομα «{opts.object_name}» → RA {named[0]:.3f}° Dec {named[1]:.3f}° ({named[2]})")
            else:
                warn(f"Δεν βρέθηκε το όνομα «{opts.object_name}» (χρειάζεται internet για άγνωστα ονόματα). Αγνοείται.")
        rep.step("image", "ok", f"{proj.W}×{proj.H}px, {'τεντωμένη' if p.stretched_input else 'γραμμική'}")
        cancel.check()

        # ---------------------------------------------------------------- 2. plate solve
        proj.sky = None
        exe = proj.astap_path or ws.find_astap()
        if not opts.use_astap:
            rep.step("solve", "skip", "Απενεργοποιημένο από τις ρυθμίσεις")
        elif not exe or not os.path.isfile(exe):
            rep.step("solve", "skip", "Δεν βρέθηκε ο ASTAP – συνθετικές αποστάσεις")
            warn("Δεν βρέθηκε ο ASTAP. Ορίσε τη διαδρομή του για πραγματικές αποστάσεις αστεριών (Gaia).")
        else:
            rep.step("solve", "running", "ASTAP: αναζήτηση λύσης…")
            fov = None
            if scale_orig:
                fov = proj.H * scale_orig * proj.down / 3600.0
            try:
                proj.astap_path = exe
                proj.solve_astap(fov, hint_ra, hint_dec, cancel=cancel,
                                 tick=rep.tick_fn("solve", "ASTAP: αναζήτηση λύσης", 45.0))
                summary["solved"] = True
                c = proj.sky.center(proj.W)
                rep.step("solve", "ok", f"RA {c[0]:.3f}°  Dec {c[1]:.3f}°  –  {proj.scale_arcsec * proj.down:.2f}\"/px")
            except Cancelled:
                raise
            except Exception as e:  # noqa
                proj.sky = None
                lines = [l.strip() for l in str(e).splitlines() if l.strip()]
                first = " ".join(lines[:2])[:160]
                rep.step("solve", "warn", f"Απέτυχε: {first}")
                warn(f"Plate solve απέτυχε ({e}). Συνεχίζω με συνθετικές αποστάσεις.")
        # κλίμακα
        if proj.sky is None:
            if scale_orig:
                proj.set_scale(scale_orig)
            else:
                proj.set_scale(2.0)
                warn("Άγνωστη κλίμακα εικόνας: χρησιμοποιώ 2.0\"/pixel. Δώσε εστιακή απόσταση/μέγεθος pixel για σωστές αναλογίες.")
        cancel.check()

        # ---------------------------------------------------------------- 3. αστέρια
        rep.step("stars", "running", "Εντοπισμός αστεριών…")
        proj.detect()
        rep.step("stars", "ok", f"{len(proj.stars)} αστέρια")
        cancel.check()

        # ---------------------------------------------------------------- 4. starless
        sn_exe = proj.starnet_path or sn.find_starnet()
        used_sn = False
        sn_note = None                                   # (κατάσταση, μήνυμα) όταν χρησιμοποιείται fallback
        if not opts.use_starnet:
            sn_note = ("ok", "Ενσωματωμένη μέθοδος (το StarNet είναι απενεργοποιημένο)")
        elif not (sn_exe and os.path.isfile(sn_exe)):
            sn_note = ("warn", "Δεν βρέθηκε το StarNet – ενσωματωμένη μέθοδος")
            warn("Δεν βρέθηκε το StarNet. Ορίσε τη διαδρομή του για καθαρότερο αποτέλεσμα (ιδιαίτερα γύρω από φωτεινά αστέρια).")
        else:
            rep.step("starless", "running", "StarNet: αφαίρεση αστεριών…")
            mp = proj.W * proj.H / 1e6
            try:
                proj.starnet_path = sn_exe
                proj.starnet_starless(cancel=cancel, tick=rep.tick_fn("starless", "StarNet: αφαίρεση αστεριών", max(20.0, 12.0 * mp)))
                used_sn = True
                rep.step("starless", "ok", "Έτοιμο (StarNet)")
            except Cancelled:
                raise
            except Exception as e:  # noqa
                first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
                sn_note = ("warn", f"StarNet απέτυχε ({first[:90]}) – ενσωματωμένη μέθοδος")
                warn(f"StarNet απέτυχε ({first}). Χρησιμοποιώ την ενσωματωμένη αφαίρεση αστεριών.")
        if not used_sn:
            rep.step("starless", "running", "Ενσωματωμένη αφαίρεση αστεριών…")
            proj.auto_starless()
            rep.step("starless", *(sn_note or ("ok", "Ενσωματωμένη μέθοδος")))
        summary["starnet"] = used_sn
        cancel.check()

        # ---------------------------------------------------------------- 5. Gaia
        proj.dist_fallback = False
        if opts.use_gaia and proj.sky is not None:
            rep.step("gaia", "running", "Λήψη Gaia DR3 (internet)…")
            try:
                proj.fetch_gaia_and_match()
                if proj.dist["zp"] is None or proj.dist["n_matched"] < 8:
                    raise RuntimeError(f"αντιστοιχίστηκαν μόνο {proj.dist['n_matched']} αστέρια")
                summary["gaia"] = True
                rep.step("gaia", "ok", f"{proj.dist['n_matched']}/{len(proj.stars)} αστέρια με μετρημένη απόσταση")
            except Cancelled:
                raise
            except Exception as e:  # noqa
                proj.assign_synthetic()
                rep.step("gaia", "warn", "Συνθετικές αποστάσεις")
                warn(f"Gaia: {str(e).rstrip(".")}. Χρησιμοποιώ συνθετικές αποστάσεις αστεριών.")
        else:
            proj.assign_synthetic()
            rep.step("gaia", "skip", "Συνθετικές αποστάσεις" + ("" if opts.use_gaia else " (απενεργοποιημένο)"))
        cancel.check()

        proj._auto_sig = sig
        return _finish(proj, opts, rep, cancel, summary, warn)
    finally:
        proj.log = old_log
        summary["seconds"] = time.time() - t_all


def _finish(proj, opts: AutoOptions, rep: Reporter, cancel: CancelToken, summary: dict, warn, reuse_mode: bool = False) -> dict:
    """Βήματα 6–8: αναγνώριση αντικειμένου, 3D σκηνή, πτήση."""
    p = proj.p
    named = None
    if opts.object_name.strip():
        named = obj.lookup_name(opts.object_name)
    # ---------------------------------------------------------------- 6. αντικείμενο / απόσταση
    rep.step("object", "running", "Αναγνώριση αντικειμένου…")
    cat = None
    if proj.sky is not None:
        c = proj.sky.center(proj.W)
        fw, fh = proj.sky.fov_deg(proj.W)
        cat = obj.identify(c[0], c[1], max(fw, fh))
    elif named:
        cat = obj.identify(named[0], named[1], 0.0)
    gaia_d = proj.suggest[0] if (proj.suggest and summary["gaia"]) else None
    if cat:
        summary["object"] = cat[0]
        rep.log(f"Αναγνωρίστηκε: {cat[0]} (~{cat[1]:.0f} pc από τη βιβλιογραφία)")
    dist = None
    why = ""
    if opts.auto_distance:
        if gaia_d and cat:
            if abs(gaia_d / cat[1] - 1.0) < 0.4:
                dist, why = gaia_d, "Gaia (συμφωνεί με τον κατάλογο)"
            else:
                dist, why = cat[1], f"κατάλογος (το Gaia έδωσε {gaia_d:.0f} pc – πιθανώς άλλο σμήνος στο πεδίο)"
        elif gaia_d:
            dist, why = gaia_d, "Gaia"
        elif cat:
            dist, why = cat[1], "κατάλογος γνωστών αντικειμένων"
    if dist:
        p.object_distance_pc = float(dist)
        rep.step("object", "ok", f"{summary['object'] or 'αντικείμενο'}: {dist:.0f} pc [{why}]")
    else:
        rep.step("object", "warn" if opts.auto_distance else "skip",
                 f"Χρήση τιμής ρυθμίσεων: {p.object_distance_pc:.0f} pc")
        if opts.auto_distance:
            warn(f"Δεν βρέθηκε απόσταση αντικειμένου· χρησιμοποιείται {p.object_distance_pc:.0f} pc (άλλαξέ τη στις ρυθμίσεις).")
    cancel.check()

    # ---------------------------------------------------------------- 7. σκηνή
    rep.step("scene", "running", "Χτίσιμο όγκου νεφελώματος…")
    proj.keyframes = []

    def scene_prog(f, m=""):
        cancel.check()
        rep.sub("scene", f, m or "Χτίσιμο όγκου νεφελώματος…")
        rep.on_step("scene", "running", m or "Χτίσιμο όγκου νεφελώματος…")
    proj.build(scene_prog)
    rep.step("scene", "ok", f"{len(proj.scene.stars)} αστέρια σε 3D")

    # ---------------------------------------------------------------- 8. πτήση
    rep.step("flight", "running", "Σχεδιασμός πτήσης…")
    if not proj.keyframes:
        proj.make_dive_path()
    rep.step("flight", "ok", f"{len(proj.keyframes)} keyframes, {p.duration:.0f} s")
    summary["scene_ok"] = True
    return summary

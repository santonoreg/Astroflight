"""Κεντρική κλάση που κρατά την κατάσταση και εκτελεί τα βήματα του pipeline (χωρίς GUI)."""
from __future__ import annotations
import json, os, math
import numpy as np
import cv2
from .config import Params
from . import imageio_ as io
from .stars import Stars, detect_stars
from .starsplit import make_starless
from . import wcs_solve as ws
from . import gaia as gm
from . import starnet as sn
from .scene import build_scene, SceneData
from .camera import CamState, Keyframe, make_flythrough, make_dive

MAX_SIDE = 8000


class Project:
    def __init__(self, params: Params | None = None, log=print):
        self.p = params or Params()
        self.log = log
        self.reset()
        self.keyframes: list[Keyframe] = []
        self.path_kind = "dive"                      # dive | orbit | custom
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".astroflight", "cache")
        self.astap_path: str | None = ws.find_astap()
        self.starnet_path: str | None = sn.find_starnet()
        self.starnet_args: str = ""
        self.starnet_stride: int = 0
        self.gaia_file: str | None = None            # τοπικό αρχείο Gaia (CSV/FITS/VOT) αντί για internet

    def reset(self):
        self.image_path = self.starless_path = None
        self.img = None            # display-referred 0..1 (working size)
        self.lin = None
        self.starless_lin = None
        self.starless_img = None
        self.starless_user = False
        self._lin_key = None
        self.W = self.H = 0
        self.down = 1.0
        self.sky: ws.SkyWCS | None = None
        self.scale_arcsec: float | None = None      # arcsec ανά pixel της εικόνας εργασίας
        self.stars: Stars | None = None
        self.dist: dict | None = None
        self.gaia = None
        self.extra = None
        self.imported_depth = None
        self.ai_depth = None
        self.scene: SceneData | None = None
        self.tone = None
        self.suggest = None

    # ---------------------------------------------------------------- εικόνα
    def load_image(self, path: str):
        self.reset_derived()
        img = io.load_image(path)
        H, W = img.shape[:2]
        self.down = 1.0
        if max(W, H) > MAX_SIDE:
            self.down = MAX_SIDE / float(max(W, H))
            img = cv2.resize(img, (int(round(W * self.down)), int(round(H * self.down))), interpolation=cv2.INTER_AREA)
            self.log(f"Η εικόνα ({W}x{H}) μικρύνθηκε σε {img.shape[1]}x{img.shape[0]} για την επεξεργασία.")
        self.img = img
        self.H, self.W = img.shape[:2]
        self.image_path = path
        self.relinearize()
        self.log(f"Φορτώθηκε: {os.path.basename(path)}  {self.W}x{self.H}")

    def reset_derived(self):
        self.stars = self.dist = self.gaia = self.extra = self.scene = None
        self.lin = None
        self._lin_key = None
        self.starless_img = None
        self.starless_lin = None
        self.starless_user = False
        self.sky = None
        self.suggest = None

    def relinearize(self):
        key = (bool(self.p.stretched_input), float(self.p.gamma))
        if self.lin is not None and getattr(self, "_lin_key", None) == key:
            return
        changed = getattr(self, "_lin_key", None) is not None
        self._lin_key = key
        self.lin = io.to_linear(self.img, *key)
        self.tone = io.tone_params(self.lin, *key)
        if changed:
            self.stars = None
            self.dist = None
            self.scene = None
            if self.starless_user and self.starless_img is not None:
                self.starless_lin = io.to_linear(self.starless_img, *key)
            else:
                self.starless_lin = None

    def load_starless(self, path: str):
        sl = io.load_image(path)
        sl = io.resize_to(sl, self.W, self.H)
        self.starless_img = sl
        self.starless_lin = io.to_linear(sl, self.p.stretched_input, self.p.gamma)
        self.starless_user = True
        self.starless_path = path
        self.log(f"Φορτώθηκε starless: {os.path.basename(path)}")

    def starnet_starless(self, progress=None, cancel=None, tick=None):
        """Αφαίρεση αστεριών με StarNet++ (CLI). Το StarNet θέλει τεντωμένη εικόνα: αν τα δεδομένα είναι γραμμικά,
        τεντώνονται προσωρινά (MTF) και το αποτέλεσμα επιστρέφει στο γραμμικό."""
        if self.img is None:
            raise RuntimeError("Δεν έχει φορτωθεί εικόνα.")
        exe = self.starnet_path or sn.find_starnet()
        if not exe:
            raise RuntimeError("Δεν βρέθηκε το StarNet. Όρισε τη διαδρομή του εκτελέσιμου.")
        self.relinearize()
        if self.p.stretched_input:
            res = sn.run_starnet(exe, self.img, self.starnet_args, self.starnet_stride, log=self.log, cancel=cancel, tick=tick)
            res = np.minimum(res, self.img)             # η starless δεν μπορεί να είναι φωτεινότερη από το πρωτότυπο
            self.starless_img = res
            self.starless_lin = io.to_linear(res, True, self.p.gamma)
        else:
            tp = self.tone
            st = io.apply_tone(self.lin, tp).astype(np.float32)
            res = sn.run_starnet(exe, st, self.starnet_args, self.starnet_stride, log=self.log, cancel=cancel, tick=tick)
            res = np.minimum(res, st)
            x = io.mtf(1.0 - tp["m"], np.clip(res, 0, 1))          # αντίστροφο του MTF
            lin = tp["black"] + x * (tp["white"] - tp["black"])
            self.starless_lin = np.minimum(np.clip(lin, 0, None), self.lin).astype(np.float32)
            self.starless_img = None
        self.starless_user = True
        self.starless_path = None
        self.starnet_path = exe
        self.log("Η starless δημιουργήθηκε με StarNet.")

    # ---------------------------------------------------------------- αστέρια
    def detect(self, progress=None):
        self.stars = detect_stars(self.lin, self.p.star_sigma, int(self.p.max_stars), progress)
        self.dist = None
        self.log(f"Εντοπίστηκαν {len(self.stars)} αστέρια (FWHM διάμεσος {np.median(self.stars.fwhm) if len(self.stars) else 0:.1f} px)")

    def auto_starless(self):
        if self.stars is None:
            self.detect()
        self.starless_lin = make_starless(self.lin, self.stars)
        self.starless_user = False
        self.log("Δημιουργήθηκε starless (αυτόματα). Για καλύτερο αποτέλεσμα χρησιμοποίησε StarXTerminator.")

    def process_stars(self, starless_method: str = "auto", dist_mode: str = "synthetic", progress=None):
        """Βήμα «Αστέρια» ολόκληρο: εντοπισμός → starless (starnet|file|auto) → αποστάσεις (gaia|synthetic)."""
        if self.img is None:
            raise RuntimeError("Δεν έχει φορτωθεί εικόνα.")
        if self.scale_arcsec is None:
            raise RuntimeError("Δώσε πρώτα την κλίμακα της εικόνας (βήμα 2).")
        self.relinearize()
        if starless_method == "file" and self.starless_img is None:
            raise RuntimeError("Επίλεξε πρώτα το αρχείο starless.")
        self.detect(progress)
        if starless_method == "starnet":
            self.starnet_starless(progress)
        elif starless_method == "file":
            self.starless_lin = io.to_linear(self.starless_img, self.p.stretched_input, self.p.gamma)
            self.starless_user = True
        else:
            self.auto_starless()
        self.dist_fallback = False
        if dist_mode == "gaia":
            try:
                self.fetch_gaia_and_match(progress)
            except Exception as e:  # noqa
                self.log(f"Gaia: {e}. Χρήση συνθετικών αποστάσεων.")
                self.dist_fallback = True
                self.assign_synthetic()
        else:
            self.assign_synthetic()

    # ---------------------------------------------------------------- αστρομετρία
    def set_scale(self, arcsec_per_px_original: float):
        self.scale_arcsec = float(arcsec_per_px_original) / self.down

    def solve_astap(self, fov_hint_deg: float | None = None, ra: float | None = None, dec: float | None = None, progress=None,
                    cancel=None, tick=None):
        exe = self.astap_path or ws.find_astap()
        if not exe:
            raise RuntimeError("Δεν βρέθηκε ο ASTAP. Εγκατέστησέ τον (hnsky.org/astap.htm) με βάση αστεριών D50/D20 ή όρισε τη διαδρομή.")
        lum = self.lin.mean(axis=2)
        self.sky = ws.solve_with_astap(lum, exe, fov_hint_deg, ra, dec, log=self.log, cancel=cancel, tick=tick)
        self.scale_arcsec = self.sky.pixel_scale_arcsec
        c = self.sky.center(self.W)
        fw, fh = self.sky.fov_deg(self.W)
        self.log(f"Λύση ASTAP: κέντρο RA={c[0]:.4f}° Dec={c[1]:.4f}°, κλίμακα {self.scale_arcsec:.3f}\"/px, πεδίο {fw:.2f}°x{fh:.2f}°")

    def set_manual_wcs(self, ra_deg: float, dec_deg: float, scale_arcsec_original: float, rot_deg: float = 0.0):
        self.set_scale(scale_arcsec_original)
        self.sky = ws.make_manual_wcs(ra_deg, dec_deg, self.scale_arcsec, rot_deg, self.W, self.H)
        self.log(f"Χειροκίνητο WCS: RA={ra_deg:.4f}° Dec={dec_deg:.4f}°, {scale_arcsec_original:.3f}\"/px (πρωτότυπη εικόνα), περιστροφή {rot_deg:.1f}°")

    def load_wcs(self, path: str, flip: bool = True):
        self.sky = ws.read_wcs_file(path, self.H, flip)
        if abs(self.down - 1.0) > 1e-6:
            self.sky = self.sky.rescaled(self.down, self.H)
        self.scale_arcsec = self.sky.pixel_scale_arcsec
        self.log(f"Φορτώθηκε WCS: {self.scale_arcsec:.3f}\"/px (εικόνα εργασίας)")

    # ---------------------------------------------------------------- αποστάσεις
    def _field_radius_deg(self):
        return 0.5 * math.hypot(self.W, self.H) * self.scale_arcsec / 3600.0 * 1.05

    def assign_synthetic(self):
        if self.stars is None:
            self.detect()
        self.dist = gm.assign_distances(self.stars, None, None, self.W, self.H)
        self.log("Αποστάσεις αστεριών: ΣΥΝΘΕΤΙΚΕΣ (εύλογες, όχι μετρήσεις). Για πραγματικές χρειάζεται plate solve + Gaia.")
        self._make_extra(None)

    def fetch_gaia_and_match(self, progress=None):
        if self.sky is None:
            raise RuntimeError("Χρειάζεται πρώτα αστρομετρική λύση (ASTAP ή χειροκίνητο WCS).")
        if self.stars is None:
            self.detect()
        ra, dec = self.sky.center(self.W)
        rad = self._field_radius_deg()
        self.gaia = gm.fetch_gaia(ra, dec, rad, self.p.gaia_glim, cache_dir=self.cache_dir, log=self.log, local_file=self.gaia_file)
        self.dist = gm.assign_distances(self.stars, self.sky, self.gaia, self.W, self.H, self.p.match_tol_px)
        if self.dist["n_matched"] < 8 and len(self.stars) > 30:
            # ίσως η λύση αφορά κατοπτρισμένο προσανατολισμό: δοκιμή με αντίστροφη φορά y
            self.sky.flip = not self.sky.flip
            alt = gm.assign_distances(self.stars, self.sky, self.gaia, self.W, self.H, self.p.match_tol_px)
            if alt["n_matched"] > self.dist["n_matched"]:
                self.log("Η αντιστοίχιση βελτιώθηκε με αναστροφή του άξονα y του WCS.")
                self.dist = alt
            else:
                self.sky.flip = not self.sky.flip
        n, N = self.dist["n_matched"], len(self.stars)
        self.log(f"Αντιστοίχιση Gaia: {n}/{N} αστέρια με μετρημένη απόσταση (zp={self.dist['zp']}).")
        self.suggest = gm.suggest_cluster_distance(self.dist["dist_pc"], self.dist["matched"])
        if self.suggest:
            self.log(f"Κορυφή στην κατανομή αποστάσεων: ~{self.suggest[0]:.0f} pc (ένταση {self.suggest[1]:.1f}) – πιθανή απόσταση σμήνους/νεφελώματος.")
        self._make_extra(self.sky if self.dist["zp"] is not None else None)

    def _make_extra(self, sky):
        self.extra = None
        if not self.p.extra_enable or self.scale_arcsec is None:
            return
        field_r = self._field_radius_deg()
        R = self.p.extra_radius_deg if self.p.extra_radius_deg > 0 else max(4.0, field_r * 2.2)
        if sky is not None and self.dist and self.dist["zp"] is not None:
            try:
                ra, dec = sky.center(self.W)
                g_ext = gm.fetch_gaia(ra, dec, R, self.p.extra_glim, top=60000, cache_dir=self.cache_dir, log=self.log, local_file=self.gaia_file)
                self.extra = gm.extra_stars_gaia(g_ext, sky, self.W, self.H, self.dist["zp"], float(np.median(self.stars.fwhm)))
                self.log(f"Επιπλέον αστέρια Gaia εκτός κάδρου: {len(self.extra['x'])} (G<{self.p.extra_glim}, ακτίνα {R:.1f}°)")
                return
            except Exception as e:  # noqa
                self.log(f"Gaia (εκτός κάδρου) απέτυχε: {e}. Χρήση συνθετικών.")
        radius_px = R * 3600.0 / self.scale_arcsec
        n = int(min(60000, max(0, len(self.stars) * (R / max(field_r, 1e-6)) ** 2 * 0.6)))
        self.extra = gm.extra_stars_synthetic(self.stars, self.dist["dist_pc"], self.W, self.H, radius_px, n)
        self.log(f"Συνθετικά αστέρια εκτός κάδρου: {len(self.extra['x'])} (ακτίνα {R:.1f}°) – για αίσθηση βάθους στην περιστροφή.")

    def apply_suggested_distance(self):
        if self.suggest:
            self.p.object_distance_pc = float(self.suggest[0])

    # ---------------------------------------------------------------- σκηνή
    def build(self, progress=None) -> SceneData:
        if self.lin is None:
            raise RuntimeError("Δεν έχει φορτωθεί εικόνα.")
        if self.scale_arcsec is None:
            raise RuntimeError("Δώσε κλίμακα εικόνας (arcsec/pixel) ή κάνε plate solve.")
        self.relinearize()
        if self.stars is None:
            self.detect()
        if self.starless_lin is None:
            self.auto_starless()
        if self.dist is None:
            self.assign_synthetic()
        dist_pc = self.dist["dist_pc"]
        extra = self.extra
        if self.dist.get("syn_all"):
            # συνθετικές αποστάσεις: κλιμάκωση ώστε να ταιριάζουν με την απόσταση του αντικειμένου που δόθηκε
            f = float(self.p.object_distance_pc) / gm.SYN_REF
            dist_pc = np.where(self.dist["matched"], dist_pc, dist_pc * f)
            if extra is not None and len(extra["x"]):
                extra = dict(extra)
                extra["dist"] = np.asarray(extra["dist"]) * f
        self.scene = build_scene(self.lin, self.starless_lin, self.stars, dist_pc, self.scale_arcsec, self.p,
                                 self.imported_depth, self.ai_depth, progress, extra)
        self.log(f"Σκηνή: όγκος {self.scene.vol.shape[2]}x{self.scene.vol.shape[1]}x{self.scene.vol.shape[0]}, "
                 f"{len(self.scene.stars)} αστέρια, πλάτος νεφελώματος {self.scene.meta['W0']:.2f} pc, βάθος {self.scene.meta['E']:.2f} pc")
        if not self.keyframes:
            self.default_path()
        return self.scene

    def make_path(self):
        """CameraPath για την τρέχουσα πτήση. Η ευθεία πτήση (dive) έχει δική της εκκίνηση/πέδηση, άρα δεν παίρνει επιπλέον
        ομαλοποίηση του χρόνου (αλλιώς στρεβλώνεται ο χρονισμός, π.χ. τα δευτερόλεπτα στο τέλος)."""
        from .camera import CameraPath
        return CameraPath(self.keyframes, self.p.duration, bool(self.p.ease) and self.path_kind != "dive")

    def default_path(self):
        """Προεπιλεγμένη πτήση: ευθεία βουτιά ΜΕΣΑ στα νέφη του νεφελώματος."""
        if self.scene:
            self.make_dive_path()

    def make_dive_path(self):
        from .scene import column_depth_quantile, suggest_poi
        sc = self.scene
        if sc is None:
            return None
        m, p = sc.meta, self.p
        if p.dive_auto_poi and self.starless_lin is not None:
            fx, fy = suggest_poi(self.starless_lin)
            p.dive_fx, p.dive_fy = float(fx), float(fy)
        fx, fy = float(p.dive_fx), float(p.dive_fy)
        self.path_kind = "dive"
        z_end = column_depth_quantile(sc, fx, fy, float(p.dive_depth))
        self.keyframes = make_dive(m["D0"], m["W0"], m["E"], m["tx0"], m["ty0"], fx, fy, z_end,
                                   float(p.dive_drift), float(p.duration),
                                   zoom_end=float(p.dive_zoom), fov_max_deg=float(p.dive_fov_max),
                                   zoom_rate=float(p.dive_zoom_rate), hold=float(p.dive_hold), fade_s=float(p.dive_fade))
        self.log(f"Πτήση: στόχος ({fx:+.2f}, {fy:+.2f}), τέλος στα {z_end:.1f} pc (αρχή {m['D0']:.1f} pc).")
        return self.keyframes

    # ---------------------------------------------------------------- αποθήκευση
    def save(self, path: str):
        d = dict(params=self.p.to_dict(), image=self.image_path, starless=self.starless_path,
                 scale_arcsec=(self.scale_arcsec * self.down if self.scale_arcsec else None),
                 astap=self.astap_path, starnet=self.starnet_path, starnet_args=self.starnet_args,
                 keyframes=[dict(t=k.t, **k.state.to_dict()) for k in self.keyframes])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.p.update_from(d.get("params", {}))
        if d.get("astap"):
            self.astap_path = d["astap"]
        if d.get("starnet"):
            self.starnet_path = d["starnet"]
        self.starnet_args = d.get("starnet_args", self.starnet_args)
        if d.get("image") and os.path.exists(d["image"]):
            self.load_image(d["image"])
            if d.get("starless") and os.path.exists(d["starless"]):
                self.load_starless(d["starless"])
            if d.get("scale_arcsec"):
                self.set_scale(d["scale_arcsec"])
        self.keyframes = [Keyframe(k["t"], CamState(**{a: k.get(a, 1.0 if a == "fade" else 0.0) for a in ("az", "el", "roll", "logr", "tx", "ty", "tz", "yaw", "pitch", "logfov", "fade")}))
                          for k in d.get("keyframes", [])]

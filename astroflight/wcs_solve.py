"""Αστρομετρία: ASTAP plate solve, χειροκίνητο WCS, ανάγνωση .wcs/.fits."""
from __future__ import annotations
import os, sys, shutil, tempfile, math
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from .runner import run_cmd


class SkyWCS:
    """Περιτύλιγμα WCS με συντεταγμένες (στήλη, γραμμή) όπως στο TIFF (γραμμή 0 = πάνω).
    Το FITS έχει y προς τα πάνω, άρα γίνεται αναστροφή."""

    def __init__(self, wcs: WCS, height: int, flip: bool = True):
        self.wcs, self.H, self.flip = wcs, int(height), flip

    def world_to_pix(self, ra, dec):
        x, y = self.wcs.all_world2pix(np.asarray(ra, float), np.asarray(dec, float), 0)
        return x, (self.H - 1 - y) if self.flip else y

    def pix_to_world(self, x, row):
        y = (self.H - 1 - np.asarray(row, float)) if self.flip else np.asarray(row, float)
        return self.wcs.all_pix2world(np.asarray(x, float), y, 0)

    @property
    def pixel_scale_arcsec(self) -> float:
        return float(np.mean(proj_plane_pixel_scales(self.wcs))) * 3600.0

    def center(self, W):
        r = self.pix_to_world((W - 1) / 2.0, (self.H - 1) / 2.0)
        return float(r[0]), float(r[1])

    def fov_deg(self, W):
        s = self.pixel_scale_arcsec / 3600.0
        return W * s, self.H * s

    def rescaled(self, factor: float, newH: int):
        """WCS για εικόνα που έχει μικρύνει κατά factor (<1)."""
        w = self.wcs.deepcopy()
        w.wcs.crpix = (np.asarray(w.wcs.crpix) - 0.5) * factor + 0.5
        if w.wcs.has_cd():
            w.wcs.cd = w.wcs.cd / factor
        else:
            w.wcs.cdelt = w.wcs.cdelt / factor
        return SkyWCS(w, newH, self.flip)


def make_manual_wcs(ra_deg: float, dec_deg: float, scale_arcsec: float, rot_deg: float, W: int, H: int) -> SkyWCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra_deg, dec_deg]
    w.wcs.crpix = [(W + 1) / 2.0, (H + 1) / 2.0]
    s = scale_arcsec / 3600.0
    t = math.radians(rot_deg)
    w.wcs.cd = [[-s * math.cos(t), s * math.sin(t)], [s * math.sin(t), s * math.cos(t)]]
    return SkyWCS(w, H)


def scale_from_optics(focal_mm: float, pixel_um: float, binning: int = 1) -> float:
    return 206.265 * pixel_um * binning / focal_mm


def parse_ra_dec(ra: str, dec: str) -> tuple[float, float]:
    """RA: '05 35 17', '05h35m17s', '05:35:17' (ώρες) ή '83.82' (μοίρες). Dec: '-05 23 28' ή '-5.39'."""
    def nums(s):
        for ch in "hmsd°'\":":
            s = s.replace(ch, " ")
        return [float(t) for t in s.replace("+", " ").replace("-", " ").split()]
    r = nums(ra)
    if len(r) == 1:
        ra_deg = r[0] * 15.0 if "h" in ra.lower() else r[0]
    else:
        ra_deg = (r[0] + r[1] / 60.0 + (r[2] / 3600.0 if len(r) > 2 else 0.0)) * 15.0
    sign = -1.0 if dec.strip().startswith("-") else 1.0
    d = nums(dec)
    dec_deg = sign * (d[0] + (d[1] / 60.0 if len(d) > 1 else 0.0) + (d[2] / 3600.0 if len(d) > 2 else 0.0))
    return ra_deg, dec_deg


def find_astap() -> str | None:
    p = shutil.which("astap") or shutil.which("astap_cli")
    if p:
        return p
    for c in (r"C:\Program Files\astap\astap.exe", r"C:\Program Files (x86)\astap\astap.exe",
              "/Applications/ASTAP.app/Contents/MacOS/astap", "/opt/astap/astap", "/usr/local/bin/astap"):
        if os.path.exists(c):
            return c
    return None


def _parse_wcs_text(text: str) -> fits.Header:
    text = text.replace("\r", "")
    last = None
    for kw in ({"sep": "\n"}, {}):
        try:
            hdr = fits.Header.fromstring(text, **kw)
            if "CRVAL1" in hdr:
                return hdr
        except Exception as e:  # noqa
            last = e
    raise ValueError(f"Μη έγκυρο αρχείο WCS ({last})")


def read_wcs_file(path: str, H: int, flip: bool = True) -> SkyWCS:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fits", ".fit", ".fts"):
        hdr = fits.getheader(path)
    else:
        with open(path, "r", errors="ignore") as f:
            hdr = _parse_wcs_text(f.read())
    return SkyWCS(WCS(hdr), H, flip)


def find_database(astap_exe: str) -> list[str]:
    """Βάσεις αστεριών ASTAP στον φάκελο του προγράμματος (D05/D20/D50/D80, H17/H18, V50, G05, W08...)."""
    import re
    d = os.path.dirname(os.path.abspath(astap_exe))
    try:
        return sorted({re.match(r"([a-z]\d\d)_", f.lower()).group(1) for f in os.listdir(d) if re.match(r"[a-z]\d\d_", f.lower())})
    except Exception:
        return []


ASTAP_EXIT = {
    1: "δεν βρέθηκε λύση (το πεδίο ή η περιοχή του ουρανού δεν ταιριάζει με τη βάση αστεριών)",
    2: "ανιχνεύθηκαν πολύ λίγα αστέρια στην εικόνα",
    16: "ο ASTAP δεν μπόρεσε να διαβάσει το προσωρινό αρχείο εικόνας",
    32: "ο ASTAP δεν βρήκε βάση αστεριών (εγκατάστησε π.χ. D50 ή D20 στον φάκελο του ASTAP)",
    33: "σφάλμα ανάγνωσης της βάσης αστεριών",
    34: "ο ASTAP δεν μπόρεσε να γράψει το αποτέλεσμα (δικαιώματα φακέλου;)",
}


def _run_astap_once(astap_exe: str, fp: str, fov: float | None, ra_deg, dec_deg, timeout: int, log, cancel=None, tick=None):
    cmd = [astap_exe, "-f", fp, "-wcs", "-log", "-z", "0", "-s", "500", "-fov", f"{fov:.4f}" if fov else "0"]
    if ra_deg is not None and dec_deg is not None:
        cmd += ["-ra", f"{ra_deg / 15.0:.5f}", "-spd", f"{dec_deg + 90.0:.5f}", "-r", "20"]
    else:
        cmd += ["-r", "180"]
    if log:
        log("ASTAP: " + " ".join(cmd))
    rc, out = run_cmd(cmd, cwd=os.path.dirname(astap_exe) or None, timeout=timeout, tick=tick, cancel=cancel)
    d = os.path.dirname(fp)
    # ο ASTAP στα Windows είναι GUI εφαρμογή: το αποτέλεσμα γράφεται σε .log/.ini δίπλα στην εικόνα
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".log", ".ini")):
            try:
                with open(os.path.join(d, f), "r", errors="replace") as fh:
                    out += "\n" + fh.read()
            except Exception:
                pass
    ok = os.path.exists(os.path.join(d, "solve.wcs"))
    return ok, rc, out.strip()


def solve_with_astap(lum_for_solve: np.ndarray, astap_exe: str, fov_deg_h: float | None = None,
                     ra_deg: float | None = None, dec_deg: float | None = None, timeout: int = 300,
                     log=None, cancel=None, tick=None) -> SkyWCS:
    """lum_for_solve: 2D float (γραμμή 0 = πάνω). Γράφεται προσωρινό FITS (αναστραμμένο) και τρέχει ο ASTAP.
    Δοκιμάζει πρώτα με το FOV που δόθηκε και μετά με άγνωστο FOV (-fov 0)."""
    H, W = lum_for_solve.shape
    if not os.path.isfile(astap_exe):
        raise RuntimeError(f"Δεν υπάρχει το αρχείο: {astap_exe}")
    dbs = find_database(astap_exe)
    if log:
        log("Βάσεις αστεριών στον φάκελο του ASTAP: " + (", ".join(dbs).upper() if dbs else "ΚΑΜΙΑ ΔΕΝ ΒΡΕΘΗΚΕ"))
    tmp = tempfile.mkdtemp(prefix="astroflight_")
    try:
        a = np.sqrt(np.clip(lum_for_solve, 0, None)).astype(np.float32)
        a = a / max(float(a.max()), 1e-6) * 60000.0
        fp = os.path.join(tmp, "solve.fits")
        hdr = fits.Header()
        fits.PrimaryHDU(np.flipud(a).astype(np.float32), header=hdr).writeto(fp, overwrite=True)
        attempts = []
        if fov_deg_h:
            attempts.append(fov_deg_h)
        attempts.append(None)                                  # άγνωστο πεδίο
        tail = ""
        last_rc = None
        for fov in attempts:
            ok, rc, out = _run_astap_once(astap_exe, fp, fov, ra_deg, dec_deg, timeout, log, cancel, tick)
            last_rc = rc
            tail = out
            if log and out:
                log(out[-700:])
            if ok:
                return read_wcs_file(os.path.join(tmp, "solve.wcs"), H, flip=True)
            for f in os.listdir(tmp):
                if f != "solve.fits":
                    try:
                        os.remove(os.path.join(tmp, f))
                    except Exception:
                        pass
            if log and fov:
                log("Αποτυχία με το δοσμένο πεδίο· δοκιμή με άγνωστο πεδίο (-fov 0)…")
        hint = []
        if not dbs:
            hint.append("Δεν βρέθηκε βάση αστεριών στον φάκελο του ASTAP — τρέξε τον installer της βάσης (π.χ. d50_star_database.exe).")
        if fov_deg_h:
            hint.append(f"Το ύψος πεδίου που δόθηκε ήταν {fov_deg_h:.2f}° (από την κλίμακα του βήματος 2) — πρέπει να είναι σωστό εντός ~10%.")
        if last_rc in ASTAP_EXIT:
            hint.insert(0, f"Κωδικός ASTAP {last_rc}: {ASTAP_EXIT[last_rc]}.")
        raise RuntimeError("Ο ASTAP δεν βρήκε λύση.\n" + "\n".join(hint) + ("\n\nΈξοδος ASTAP:\n" + tail[-600:] if tail else ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

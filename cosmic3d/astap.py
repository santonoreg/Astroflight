"""Plate solve με ASTAP (γραμμή εντολών) – επιστρέφει κεφαλίδα FITS με WCS.

Δοκιμασμένο μοτίβο: το ASTAP τρέχει πάνω σε προσωρινό FITS και γράφει δίπλα του αρχείο .wcs με τη λύση.
Κωδικοί εξόδου ASTAP: 0 = λύση, 1 = δεν βρέθηκε λύση, 2 = λίγα αστέρια, 16 = σφάλμα ανάγνωσης εικόνας,
32 = δεν βρέθηκε βάση αστεριών, 33 = σφάλμα βάσης, 34 = σφάλμα εγγραφής."""
from __future__ import annotations
import os, re, shutil, tempfile
import numpy as np
from astropy.io import fits

from .runner import run_cmd

ASTAP_EXIT = {
    1: "δεν βρέθηκε λύση (το πεδίο ή η περιοχή του ουρανού δεν ταιριάζει με τη βάση αστεριών)",
    2: "ανιχνεύθηκαν πολύ λίγα αστέρια στην εικόνα",
    16: "ο ASTAP δεν μπόρεσε να διαβάσει το προσωρινό αρχείο εικόνας",
    32: "ο ASTAP δεν βρήκε βάση αστεριών (εγκατάστησε π.χ. D50 ή D20 στον φάκελο του ASTAP)",
    33: "σφάλμα ανάγνωσης της βάσης αστεριών",
    34: "ο ASTAP δεν μπόρεσε να γράψει το αποτέλεσμα (δικαιώματα φακέλου;)",
}


def find_astap() -> str | None:
    p = shutil.which("astap") or shutil.which("astap_cli")
    if p:
        return p
    for c in (r"C:\Program Files\astap\astap.exe", r"C:\Program Files (x86)\astap\astap.exe",
              "/Applications/ASTAP.app/Contents/MacOS/astap", "/opt/astap/astap", "/usr/local/bin/astap"):
        if os.path.exists(c):
            return c
    return None


def find_database(astap_exe: str) -> list[str]:
    """Βάσεις αστεριών ASTAP στον φάκελο του προγράμματος (D05/D20/D50/D80, H17/H18, V50, G05, W08...)."""
    d = os.path.dirname(os.path.abspath(astap_exe))
    try:
        return sorted({re.match(r"([a-z]\d\d)_", f.lower()).group(1) for f in os.listdir(d) if re.match(r"[a-z]\d\d_", f.lower())})
    except Exception:
        return []


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


def solve(lum_bottom_up: np.ndarray, astap_exe: str, fov_deg_h: float | None = None, ra_deg: float | None = None,
          dec_deg: float | None = None, timeout: int = 300, log=None, cancel=None, tick=None) -> fits.Header:
    """lum_bottom_up: 2D float με γραμμή 0 = ΚΑΤΩ (προσανατολισμός FITS). Δοκιμάζει πρώτα με το πεδίο που δόθηκε, μετά με
    άγνωστο πεδίο. Επιστρέφει την κεφαλίδα WCS (συντεταγμένες pixel FITS, αρχή κάτω-αριστερά)."""
    if not astap_exe or not os.path.isfile(astap_exe):
        raise RuntimeError(f"Δεν υπάρχει το αρχείο ASTAP: {astap_exe}")
    dbs = find_database(astap_exe)
    if log:
        log("Βάσεις αστεριών στον φάκελο του ASTAP: " + (", ".join(dbs).upper() if dbs else "ΚΑΜΙΑ ΔΕΝ ΒΡΕΘΗΚΕ"))
    tmp = tempfile.mkdtemp(prefix="cosmic3d_")
    try:
        a = np.sqrt(np.clip(lum_bottom_up, 0, None)).astype(np.float32)
        a = a / max(float(a.max()), 1e-6) * 60000.0
        fp = os.path.join(tmp, "solve.fits")
        fits.PrimaryHDU(a.astype(np.float32)).writeto(fp, overwrite=True)
        attempts = [fov_deg_h] if fov_deg_h else []
        attempts.append(None)
        tail, rc = "", None
        for fov in attempts:
            cmd = [astap_exe, "-f", fp, "-wcs", "-log", "-z", "0", "-s", "500", "-fov", f"{fov:.4f}" if fov else "0"]
            if ra_deg is not None and dec_deg is not None:
                cmd += ["-ra", f"{ra_deg / 15.0:.5f}", "-spd", f"{dec_deg + 90.0:.5f}", "-r", "20"]
            else:
                cmd += ["-r", "180"]
            if log:
                log("ASTAP: " + " ".join(cmd))
            rc, out = run_cmd(cmd, cwd=os.path.dirname(astap_exe) or None, timeout=timeout, tick=tick, cancel=cancel)
            # ο ASTAP στα Windows είναι GUI εφαρμογή: το αποτέλεσμα γράφεται και σε .log/.ini δίπλα στην εικόνα
            for f in sorted(os.listdir(tmp)):
                if f.lower().endswith((".log", ".ini")):
                    try:
                        with open(os.path.join(tmp, f), "r", errors="replace") as fh:
                            out += "\n" + fh.read()
                    except Exception:
                        pass
            tail = out.strip()
            if log and tail:
                log(tail[-600:])
            wcs_path = os.path.join(tmp, "solve.wcs")
            if os.path.exists(wcs_path):
                with open(wcs_path, "r", errors="replace") as fh:
                    return _parse_wcs_text(fh.read())
            for f in os.listdir(tmp):
                if f != "solve.fits":
                    try:
                        os.remove(os.path.join(tmp, f))
                    except Exception:
                        pass
            if log and fov:
                log("Αποτυχία με το δοσμένο πεδίο· δοκιμή με άγνωστο πεδίο…")
        hint = []
        if rc in ASTAP_EXIT:
            hint.append(f"Κωδικός ASTAP {rc}: {ASTAP_EXIT[rc]}.")
        if not dbs:
            hint.append("Δεν βρέθηκε βάση αστεριών στον φάκελο του ASTAP — τρέξε τον installer της βάσης (π.χ. d50_star_database.exe).")
        if fov_deg_h:
            hint.append(f"Το ύψος πεδίου που δόθηκε ήταν {fov_deg_h:.2f}° — πρέπει να είναι σωστό εντός ~10%.")
        raise RuntimeError("Ο ASTAP δεν βρήκε λύση.\n" + "\n".join(hint) + ("\n\nΈξοδος ASTAP:\n" + tail[-600:] if tail else ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

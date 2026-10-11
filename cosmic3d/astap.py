"""Plate solve με ASTAP (γραμμή εντολών) – επιστρέφει κεφαλίδα FITS με WCS.

Δοκιμασμένο μοτίβο: το ASTAP τρέχει πάνω σε προσωρινό FITS και γράφει δίπλα του αρχείο .wcs με τη λύση.
Κωδικοί εξόδου ASTAP: 0 = λύση, 1 = δεν βρέθηκε λύση, 2 = λίγα αστέρια, 16 = σφάλμα ανάγνωσης εικόνας,
32 = δεν βρέθηκε βάση αστεριών, 33 = σφάλμα βάσης, 34 = σφάλμα εγγραφής."""
from __future__ import annotations
import os, re, shutil, tempfile, time
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


def _run_once(exe: str, tmp: str, fov: float | None, ra_deg, dec_deg, timeout: float, log, cancel, tick, slow: bool = False):
    """Ένα τρέξιμο του ASTAP. Επιστρέφει (κεφαλίδα WCS ή None, κωδικός εξόδου, έξοδος)."""
    for f in os.listdir(tmp):
        if f != "solve.fits":
            try:
                os.remove(os.path.join(tmp, f))
            except Exception:
                pass
    fp = os.path.join(tmp, "solve.fits")
    cmd = [exe, "-f", fp, "-wcs", "-log", "-z", "0", "-s", "1000" if slow else "500", "-fov", f"{fov:.4f}" if fov else "0"]
    if slow:
        cmd += ["-speed", "slow"]
    if ra_deg is not None and dec_deg is not None:
        cmd += ["-ra", f"{ra_deg / 15.0:.5f}", "-spd", f"{dec_deg + 90.0:.5f}", "-r", "20"]
    else:
        cmd += ["-r", "180"]
    if log:
        log("ASTAP: " + " ".join(cmd))
    rc, out = run_cmd(cmd, cwd=os.path.dirname(exe) or None, timeout=timeout, tick=tick, cancel=cancel)
    # ο ASTAP στα Windows είναι GUI εφαρμογή: το αποτέλεσμα γράφεται και σε .log/.ini δίπλα στην εικόνα
    for f in sorted(os.listdir(tmp)):
        if f.lower().endswith((".log", ".ini")):
            try:
                with open(os.path.join(tmp, f), "r", errors="replace") as fh:
                    out += "\n" + fh.read()
            except Exception:
                pass
    wcs_path = os.path.join(tmp, "solve.wcs")
    hdr = None
    if os.path.exists(wcs_path):
        with open(wcs_path, "r", errors="replace") as fh:
            hdr = _parse_wcs_text(fh.read())
    return hdr, rc, out.strip()


def _upscale_wcs(h: fits.Header, f: float) -> fits.Header:
    """Η λύση βρέθηκε σε εικόνα που είναι f φορές (f<1) μικρότερη από την αρχική· επιστρέφει το WCS της αρχικής."""
    from astropy.wcs import WCS
    w = WCS(h, naxis=[1, 2])
    w.wcs.crpix = (np.asarray(w.wcs.crpix) - 0.5) / f + 0.5
    if w.wcs.has_cd():
        w.wcs.cd = w.wcs.cd * f
    else:
        w.wcs.cdelt = w.wcs.cdelt * f
    out = fits.Header()
    for k in h.keys():                              # κράτα ό,τι άλλο υπήρχε (π.χ. EQUINOX, RADESYS)
        if k and k not in ("COMMENT", "HISTORY") and not k.startswith(("CRPIX", "CD1_", "CD2_", "CDELT", "PC1_", "PC2_", "CROTA")):
            try:
                out[k] = (h[k], h.comments[k])
            except Exception:
                pass
    out.update(w.to_header())
    return out


def solve_auto(lum_bottom_up: np.ndarray, astap_exe: str, scale_hints: list[float] | None = None, ra_deg: float | None = None,
               dec_deg: float | None = None, budget_s: float = 600.0, per_try_s: float = 90.0, max_side: int = 2800,
               log=None, cancel=None, progress=None) -> tuple[fits.Header, dict]:
    """Plate solve ΧΩΡΙΣ να χρειάζεται να ξέρεις την κλίμακα.

    Δοκιμάζει με τη σειρά: (1) τις υποθέσεις κλίμακας που δόθηκαν (π.χ. ονομαστική από τα οπτικά), (2) τυφλή αναζήτηση
    (ο ASTAP βρίσκει μόνος του το πεδίο), (3) σκάλα κλιμάκων 0.7–80 arcsec/pixel (από την πιο πιθανή προς τις άλλες),
    (4) τυφλή αναζήτηση σε αργή λειτουργία. Η εικόνα μικραίνει πριν σταλεί (τεράστια mosaics) και η λύση επιστρέφεται
    στη διάσταση της αρχικής. Επιστρέφει (κεφαλίδα WCS, πληροφορίες προσπαθειών). progress(f, μήνυμα)."""
    from .optics import order_candidates, scale_ladder
    if not astap_exe or not os.path.isfile(astap_exe):
        raise RuntimeError(f"Δεν υπάρχει το αρχείο ASTAP: {astap_exe}")
    dbs = find_database(astap_exe)
    if log:
        log("Βάσεις αστεριών στον φάκελο του ASTAP: " + (", ".join(dbs).upper() if dbs else "ΚΑΜΙΑ ΔΕΝ ΒΡΕΘΗΚΕ"))
    H, W = lum_bottom_up.shape
    lum = np.nan_to_num(lum_bottom_up.astype(np.float32), nan=0.0)
    valid = lum > 0
    if valid.mean() < 0.999 and valid.any():          # mosaic: τα μηδενικά περιθώρια του stitching δεν είναι ουρανός
        lum = np.where(valid, lum, float(np.median(lum[valid])))
    f = min(1.0, max_side / float(max(H, W)))
    if f < 1.0:
        import cv2
        lum = cv2.resize(lum, (max(8, int(round(W * f))), max(8, int(round(H * f)))), interpolation=cv2.INTER_AREA)
        if log:
            log(f"Η εικόνα ({W}×{H}) μίκρυνε σε {lum.shape[1]}×{lum.shape[0]} για το plate solve.")
    tries: list[tuple[str, float | None, bool]] = []
    cand = order_candidates([x for x in (scale_hints or []) if x], scale_ladder())
    n_hint = len([x for x in (scale_hints or []) if x])
    for i, sc in enumerate(cand[:n_hint]):
        tries.append((f"κλίμακα ~{sc:.2f}\"/px", H * sc / 3600.0, False))
    tries.append(("άγνωστη κλίμακα (τυφλή αναζήτηση)", None, False))
    for sc in cand[n_hint:]:
        tries.append((f"κλίμακα ~{sc:.2f}\"/px", H * sc / 3600.0, False))
    tries.append(("τυφλή αναζήτηση, αργή λειτουργία", None, True))

    tmp = tempfile.mkdtemp(prefix="cosmic3d_")
    t_start = time.time()
    attempts: list[str] = []
    try:
        a = np.sqrt(np.clip(lum, 0, None))
        a = a / max(float(a.max()), 1e-6) * 60000.0
        fits.PrimaryHDU(a.astype(np.float32)).writeto(os.path.join(tmp, "solve.fits"), overwrite=True)
        last_rc, tail = None, ""
        for i, (label, fov, slow) in enumerate(tries):
            if time.time() - t_start > budget_s:
                attempts.append("Ο συνολικός χρόνος αναζήτησης εξαντλήθηκε.")
                break
            if cancel is not None:
                cancel.check()
            t_try = time.time()

            def tick(el, last, _i=i, _label=label):
                if progress:
                    progress(min(0.99, (_i + min(el / per_try_s, 1.0)) / len(tries)), f"ASTAP ({_i + 1}/{len(tries)}): {_label}… {time.time() - t_start:.0f} s")
            if progress:
                progress(i / len(tries), f"ASTAP ({i + 1}/{len(tries)}): {label}…")
            try:
                hdr, rc, out = _run_once(astap_exe, tmp, fov, ra_deg, dec_deg, per_try_s, log, cancel, tick, slow)
            except TimeoutError:
                attempts.append(f"{label}: δεν τελείωσε σε {per_try_s:.0f} s")
                if log:
                    log(f"ASTAP: {label} – λήξη χρόνου, επόμενη δοκιμή.")
                continue
            last_rc, tail = rc, out
            if log and out:
                log(out[-500:])
            if hdr is not None:
                if log:
                    log(f"Λύση βρέθηκε ({label}) σε {time.time() - t_try:.0f} s.")
                if f < 1.0:
                    hdr = _upscale_wcs(hdr, f)
                return hdr, dict(attempts=attempts, label=label, tries=i + 1, seconds=time.time() - t_start, downsample=f)
            attempts.append(f"{label}: {ASTAP_EXIT.get(rc, f'κωδικός {rc}')}")
        hint = []
        if last_rc in ASTAP_EXIT:
            hint.append(f"Τελευταίος κωδικός ASTAP {last_rc}: {ASTAP_EXIT[last_rc]}.")
        if not dbs:
            hint.append("Δεν βρέθηκε βάση αστεριών στον φάκελο του ASTAP — τρέξε τον installer της βάσης (π.χ. d50_star_database.exe).")
        hint.append("Δοκίμασε: (1) να γράψεις το όνομα του αντικειμένου (π.χ. «Rosette Nebula») – περιορίζει την αναζήτηση και την κάνει πολύ πιο σίγουρη· "
                    "(2) αν το πεδίο είναι πολύ ευρύ (>6° όπως σε mosaics), έλεγξε ότι η βάση αστεριών του ASTAP καλύπτει τόσο ευρύ πεδίο· "
                    "(3) ή τσέκαρε «προσεγγιστική λύση» (χρησιμοποιεί το όνομα και μια κλίμακα που εκτιμάς).")
        raise RuntimeError(f"Ο ASTAP δεν βρήκε λύση μετά από {len(attempts)} δοκιμές ({time.time() - t_start:.0f} s).\n" + "\n".join(hint)
                           + ("\n\nΔοκιμές:\n- " + "\n- ".join(attempts[:12]) if attempts else "")
                           + ("\n\nΈξοδος ASTAP (τελευταία δοκιμή):\n" + tail[-500:] if tail else ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

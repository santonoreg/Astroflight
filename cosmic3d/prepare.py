"""Προετοιμασία εικόνας: ό,τι χρειάζονται τα scripts CosmicView 3D / CosmicDepth 3D ως είσοδο.

Επιλέγεις ένα αρχείο (TIFF / PNG / JPG / FITS)· η προετοιμασία:
  1. διαβάζει την εικόνα (και ελέγχει αν έχει ήδη αστρομετρία/WCS),
  2. αν δεν έχει, κάνει plate solve με ASTAP,
  3. γράφει ένα FITS (προσανατολισμός Siril/FITS: η πρώτη γραμμή είναι η ΚΑΤΩ) με την κεφαλίδα WCS, που διαβάζουν τα scripts.
Το αποτέλεσμα αποθηκεύεται σε φάκελο εργασίας και ξαναχρησιμοποιείται αν δεν έχει αλλάξει η εικόνα."""
from __future__ import annotations
import hashlib, json, math, os
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from . import astap
from .runner import CancelToken


@dataclass
class PrepareOptions:
    astap_path: str | None = None
    scale_arcsec: float | None = None        # arcsec/pixel της εικόνας (προαιρετικό: επιταχύνει και σιγουρεύει τη λύση)
    object_name: str = ""                    # προαιρετικό: βοηθά το plate solve και δίνει το OBJECT στα scripts
    ra_deg: float | None = None
    dec_deg: float | None = None
    timeout: int = 300
    force: bool = False                      # αγνόησε τη λύση που έχει ήδη αποθηκευτεί


@dataclass
class Prepared:
    source: str
    fits_path: str
    workdir: str
    width: int
    height: int
    channels: int
    header: fits.Header
    solved_by: str                           # "file" (είχε ήδη WCS) | "astap" | "cache"
    object_name: str = ""
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    scale_arcsec: float = 0.0
    fov_deg: tuple[float, float] = (0.0, 0.0)
    rgb_bottom_up: np.ndarray | None = field(default=None, repr=False)   # (3,H,W) float32 [0,1], προσανατολισμός FITS
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        how = {"file": "η εικόνα είχε ήδη αστρομετρία", "astap": "plate solve με ASTAP", "cache": "από προηγούμενη προετοιμασία"}[self.solved_by]
        return (f"{self.width}×{self.height}px, RA {self.ra_deg:.4f}° Dec {self.dec_deg:+.4f}°, κλίμακα {self.scale_arcsec:.2f}\"/px, "
                f"πεδίο {self.fov_deg[0]:.2f}°×{self.fov_deg[1]:.2f}° ({how})")


def default_workdir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cosmic3d", "work")


# ---------------------------------------------------------------------------------------------------------------------
def _normalize(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind == "u":
        return a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    if a.dtype.kind == "i":
        return (a.astype(np.float32) - np.iinfo(a.dtype).min) / float(np.iinfo(a.dtype).max - np.iinfo(a.dtype).min)
    out = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(out.max()) if out.size else 1.0
    if mx > 1.0 + 1e-3:
        out = out / mx
    return np.clip(out, 0.0, None)


def read_image(path: str) -> tuple[np.ndarray, fits.Header | None]:
    """Επιστρέφει (rgb (3,H,W) float32 [0,1] σε προσανατολισμό FITS/Siril, κεφαλίδα FITS ή None)."""
    ext = os.path.splitext(path)[1].lower()
    header = None
    if ext in (".fit", ".fits", ".fts"):
        with fits.open(path) as hdul:
            hdu = next((h for h in hdul if h.data is not None and h.data.ndim >= 2), None)
            if hdu is None:
                raise ValueError("Το αρχείο FITS δεν περιέχει εικόνα.")
            header = hdu.header.copy()
            data = np.asarray(hdu.data)
            while data.ndim > 3:
                data = data[0]
            if data.ndim == 2:
                data = np.stack([data] * 3)
            elif data.shape[0] not in (1, 3, 4) and data.shape[-1] in (1, 3, 4):
                data = np.transpose(data, (2, 0, 1))
            if data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            data = _normalize(data[:3])
            return np.ascontiguousarray(data), header
    if ext in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(path)
    else:
        import cv2
        buf = np.fromfile(path, dtype=np.uint8)               # δουλεύει και με ελληνικά paths στα Windows
        a = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError(f"Δεν μπόρεσα να διαβάσω την εικόνα: {path}")
        if a.ndim == 3 and a.shape[2] >= 3:
            a = a[..., [2, 1, 0]]
    a = np.asarray(a)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[0] in (1, 3, 4) and a.shape[2] > 4:             # channel-first
        a = np.transpose(a, (1, 2, 0))
    if a.shape[2] == 2:
        a = a[..., :1]
    a = a[..., :3]
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    rgb = _normalize(a)                                        # (H,W,3), πρώτη γραμμή = ΠΑΝΩ
    return np.ascontiguousarray(np.transpose(rgb[::-1], (2, 0, 1))), None   # → (3,H,W), πρώτη γραμμή = ΚΑΤΩ


def _has_wcs(h: fits.Header | None) -> bool:
    if h is None or "CRVAL1" not in h:
        return False
    try:
        return bool(WCS(h, naxis=[1, 2]).has_celestial)
    except Exception:
        return False


def resolve_name(name: str, log=None) -> tuple[float, float] | None:
    """Συντεταγμένες από όνομα (SIMBAD μέσω astropy – χρειάζεται internet)."""
    try:
        from astropy.coordinates import SkyCoord
        c = SkyCoord.from_name(name.strip())
        return float(c.ra.deg), float(c.dec.deg)
    except Exception as e:  # noqa
        if log:
            log(f"Δεν βρέθηκε το όνομα «{name}» (χρειάζεται internet): {str(e).splitlines()[0][:100]}")
        return None


def _ascii(text: str) -> str:
    """Οι τιμές των καρτών FITS πρέπει να είναι ASCII."""
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in text)


def _wcs_cards(h: fits.Header) -> fits.Header:
    skip = ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "COMMENT", "HISTORY", "END")
    out = fits.Header()
    for k in h.keys():
        if k in skip or not k:
            continue
        try:
            out[k] = (h[k], h.comments[k])
        except Exception:
            pass
    return out


def _sig(path: str, opts: PrepareOptions) -> dict:
    st = os.stat(path)
    return dict(path=os.path.abspath(path), size=st.st_size, mtime=int(st.st_mtime), scale=opts.scale_arcsec,
                obj=opts.object_name.strip(), ra=opts.ra_deg, dec=opts.dec_deg)


def prepare(path: str, opts: PrepareOptions, progress=None, log=None, cancel: CancelToken | None = None,
            workdir: str | None = None) -> Prepared:
    """Κύρια συνάρτηση. progress(f, μήνυμα): f στο [0,1]. Σηκώνει RuntimeError με κατανοητό μήνυμα αν αποτύχει."""
    cancel = cancel or CancelToken()
    prog = progress or (lambda f, m="": None)
    log = log or (lambda m: None)
    warnings: list[str] = []

    prog(0.02, "Ανάγνωση εικόνας…")
    rgb, header = read_image(path)
    _, H, W = rgb.shape
    log(f"Εικόνα: {os.path.basename(path)}  {W}×{H}px")
    cancel.check()

    key = hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:8]
    stem = os.path.splitext(os.path.basename(path))[0]
    wd = os.path.join(workdir or default_workdir(), f"{stem}-{key}")
    os.makedirs(wd, exist_ok=True)
    out_fits = os.path.join(wd, "image_solved.fits")
    meta_path = os.path.join(wd, "prepared.json")
    sig = _sig(path, opts)

    def finish(hdr: fits.Header, by: str) -> Prepared:
        w = WCS(hdr, naxis=[1, 2])
        sc = proj_plane_pixel_scales(w)
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        ra, dec = (float(v) for v in w.all_pix2world([[cx, cy]], 0)[0])
        scale = float(np.mean(sc)) * 3600.0
        return Prepared(source=path, fits_path=out_fits, workdir=wd, width=W, height=H, channels=3, header=hdr, solved_by=by,
                        object_name=str(hdr.get("OBJECT", "") or "").strip(), ra_deg=ra, dec_deg=dec, scale_arcsec=scale,
                        fov_deg=(W * scale / 3600.0, H * scale / 3600.0), rgb_bottom_up=rgb, warnings=warnings)

    # --- 1. αποθηκευμένη προετοιμασία
    if not opts.force and os.path.isfile(out_fits) and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                if json.load(f) == sig:
                    hdr = fits.getheader(out_fits)
                    if _has_wcs(hdr):
                        log("Βρέθηκε αποθηκευμένη λύση για αυτή την εικόνα – δεν χρειάζεται νέο plate solve.")
                        prog(1.0, "Έτοιμο (από προηγούμενη προετοιμασία)")
                        return finish(hdr, "cache")
        except Exception as e:  # noqa
            log(f"Η αποθηκευμένη λύση δεν χρησιμοποιήθηκε ({e}).")

    # --- 2. αστρομετρία
    obj = opts.object_name.strip()
    ra_h, dec_h = opts.ra_deg, opts.dec_deg
    if obj and ra_h is None:
        r = resolve_name(obj, log)
        if r:
            ra_h, dec_h = r
            log(f"Όνομα «{obj}» → RA {ra_h:.3f}° Dec {dec_h:+.3f}°")
        else:
            warnings.append(f"Δεν βρέθηκε το όνομα «{obj}» (χρειάζεται internet)· αγνοείται ως υπόδειξη.")
    if _has_wcs(header) and not opts.force:
        log("Η εικόνα έχει ήδη αστρομετρική λύση (WCS) – δεν χρειάζεται ASTAP.")
        solved_hdr, by = _wcs_cards(header), "file"
        prog(0.8, "Η εικόνα έχει ήδη WCS")
    else:
        exe = opts.astap_path or astap.find_astap()
        if not exe or not os.path.isfile(exe):
            raise RuntimeError("Η εικόνα δεν έχει αστρομετρία (WCS) και δεν βρέθηκε ο ASTAP για plate solve.\n"
                               "Εγκατάστησε τον ASTAP + μία βάση αστεριών (D50 ή D20) από https://hnsky.org/astap.htm και δείξε τη διαδρομή του, "
                               "ή δώσε FITS που έχει ήδη λυθεί (π.χ. από το Siril).")
        lum = rgb.mean(axis=0)
        fov = H * opts.scale_arcsec / 3600.0 if opts.scale_arcsec else None

        def tick(el, last):
            f = 0.1 + 0.7 * (1.0 - math.exp(-el / 45.0))
            prog(f, f"ASTAP: αναζήτηση λύσης… {el:.0f} s")
        prog(0.1, "ASTAP: αναζήτηση λύσης…")
        solved_hdr = astap.solve(lum, exe, fov, ra_h, dec_h, timeout=opts.timeout, log=log, cancel=cancel, tick=tick)
        by = "astap"
    cancel.check()

    # --- 3. αποθήκευση FITS για τα scripts
    prog(0.88, "Αποθήκευση FITS…")
    hdr = fits.Header()
    hdr.update(solved_hdr)
    for k in ("NAXIS1", "NAXIS2"):
        hdr.remove(k, ignore_missing=True)
    if obj:
        hdr["OBJECT"] = (_ascii(obj), "Object name (from user)")
    elif ra_h is not None and not hdr.get("OBJECT"):
        pass
    hdr["PLTSOLVD"] = (True, "Plate solved")
    hdr["CREATOR"] = ("Cosmic3D launcher", "Prepared for CosmicView/CosmicDepth 3D")
    fits.PrimaryHDU(rgb.astype(np.float32), header=hdr).writeto(out_fits, overwrite=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sig, f)
    res = finish(fits.getheader(out_fits), by)
    log("Έτοιμο: " + res.summary())
    prog(1.0, "Έτοιμο")
    return res

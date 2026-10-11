"""Στρώμα συμβατότητας «sirilpy» για τα scripts Svenesis CosmicView 3D / CosmicDepth 3D.

Τα scripts γράφτηκαν ως plugins του Siril και παίρνουν την εικόνα και το WCS από ένα τρέχον Siril μέσω του πακέτου
``sirilpy``. Εδώ ο ρόλος του Siril παίζεται από ένα αρχείο εικόνας που έχει προετοιμαστεί (plate solve → FITS με WCS)
από το ``cosmic3d.prepare``. Υλοποιείται μόνο το υποσύνολο του API που χρησιμοποιούν τα δύο scripts, με τις ίδιες
συμβάσεις του Siril:
  • τα δεδομένα εικόνας είναι (κανάλια, ύψος, πλάτος) με την πρώτη γραμμή = ΚΑΤΩ (προσανατολισμός FITS),
  • τα pix2radec/radec2pix χρησιμοποιούν 0-based pixel του ίδιου προσανατολισμού.

(c) για το στρώμα συμβατότητας: SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import contextlib
import importlib.metadata as _md
import re
import subprocess
import sys
import types

import numpy as np

from .exceptions import NoImageError, SirilConnectionError, SirilError  # noqa: F401

__version__ = "0.0-standalone"
IS_STANDALONE_SHIM = True

_STATE: dict = {"prepared": None, "log": None}


def configure(prepared, log=None) -> None:
    """Ορίζει ποια προετοιμασμένη εικόνα «βλέπουν» τα scripts (καλείται από το cosmic3d)."""
    _STATE["prepared"] = prepared
    _STATE["log"] = log


def ensure_installed(*packages, **kwargs) -> None:
    """Όπως στο sirilpy: εγκαθιστά με pip ό,τι λείπει (δεν αγγίζει ό,τι είναι ήδη εγκατεστημένο)."""
    missing = []
    for spec in packages:
        name = re.split(r"[<>=!~\[ ]", str(spec), maxsplit=1)[0]
        try:
            _md.version(name)
        except _md.PackageNotFoundError:
            missing.append(str(spec))
    if not missing:
        return
    print("Εγκατάσταση βιβλιοθηκών που λείπουν:", ", ".join(missing), flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", *missing], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Αποτυχία εγκατάστασης ({', '.join(missing)}):\n{(r.stdout + r.stderr)[-600:]}")


class FFit:
    """Μίμηση του sirilpy.FFit: εικόνα (κανάλια, ύψος, πλάτος)."""

    def __init__(self, data: np.ndarray):
        self.data = np.ascontiguousarray(data)
        self.channels, self.height, self.width = self.data.shape
        self.bitpix = -32

    def ensure_data_type(self, dtype) -> None:
        self.data = self.data.astype(dtype, copy=False)


def _auto_stretch_uint8(rgb: np.ndarray) -> np.ndarray:
    """Προεπισκόπηση σαν το «linked autostretch» του Siril. Αν η εικόνα είναι ήδη τεντωμένη (non-linear) μένει ως έχει."""
    a = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    sub = a[:, ::4, ::4]
    if float(np.median(sub)) > 0.04:
        return (a * 255.0 + 0.5).astype(np.uint8)
    med = float(np.median(sub))
    mad = float(np.median(np.abs(sub - med))) * 1.4826
    black = max(0.0, med - 2.8 * mad)
    x = np.clip((a - black) / max(1.0 - black, 1e-6), 0.0, 1.0)
    m = float(np.median(np.clip((sub - black) / max(1.0 - black, 1e-6), 0.0, 1.0)))
    t = 0.25                                                    # στόχος διαμέσου
    m = min(max(m, 1e-6), 0.999)
    mid = (m * (t - 1.0)) / (2.0 * t * m - t - m)                # midtones transfer function
    y = ((mid - 1.0) * x) / ((2.0 * mid - 1.0) * x - mid)
    return (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


class SirilInterface:
    """Ελάχιστη υλοποίηση του sirilpy.SirilInterface πάνω σε ένα προετοιμασμένο αρχείο εικόνας."""

    def __init__(self, *args, **kwargs):
        self._wcs = None
        self._wcs_for = None

    # --- σύνδεση -------------------------------------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return True

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def log(self, msg, color=None) -> None:
        cb = _STATE.get("log")
        if cb:
            cb(str(msg))
        else:
            print(str(msg), flush=True)

    @contextlib.contextmanager
    def image_lock(self):
        yield

    # --- εικόνα --------------------------------------------------------------------------------------------------
    def _prep(self):
        p = _STATE.get("prepared")
        if p is None:
            raise NoImageError("Δεν έχει επιλεγεί εικόνα.")
        return p

    def get_siril_wd(self) -> str:
        return self._prep().workdir

    def get_image_filename(self) -> str:
        return self._prep().fits_path

    def get_image(self, *args, **kwargs) -> FFit:
        return FFit(self._prep().rgb_bottom_up.copy())

    def get_image_pixeldata(self, shape=None, preview: bool = False, linked: bool = False):
        rgb = self._prep().rgb_bottom_up
        if shape is not None:
            x, y, w, h = shape
            rgb = rgb[:, y:y + h, x:x + w]
        return _auto_stretch_uint8(rgb) if preview else rgb.copy()

    def get_image_fits_header(self, return_as: str = "str"):
        h = self._prep().header
        if return_as == "dict":
            return {k: h[k] for k in h.keys() if k and k not in ("COMMENT", "HISTORY")}
        return h.tostring(sep="\n")

    def get_image_keywords(self):
        p = self._prep()
        h = p.header
        return types.SimpleNamespace(
            pltsolvd=True, wcsdata=True, object=(p.object_name or ""), ra=p.ra_deg, dec=p.dec_deg,
            focal_length=float(h.get("FOCALLEN", 0.0) or 0.0), pixel_size_x=float(h.get("XPIXSZ", 0.0) or 0.0),
            pixel_size_y=float(h.get("YPIXSZ", 0.0) or 0.0))

    # --- αστρομετρία -------------------------------------------------------------------------------------------
    def _get_wcs(self):
        from astropy.wcs import WCS
        p = self._prep()
        if self._wcs is None or self._wcs_for is not p:
            self._wcs = WCS(p.header, naxis=[1, 2])
            self._wcs_for = p
        return self._wcs

    def pix2radec(self, x, y):
        ra, dec = self._get_wcs().all_pix2world(float(x), float(y), 0)
        return float(ra), float(dec)

    def radec2pix(self, ra, dec):
        x, y = self._get_wcs().all_world2pix(float(ra), float(dec), 0)
        return float(x), float(y)

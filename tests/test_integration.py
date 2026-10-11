"""Δοκιμές ολοκλήρωσης: προετοιμασία εικόνας (με ψεύτικο ASTAP), στρώμα sirilpy και φόρτωση των δύο scripts.
Τρέξε:  python tests/test_integration.py        (δεν χρειάζεται internet, ASTAP ή οθόνη)"""
import os, sys, tempfile, threading, time, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "shim"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from astropy.io import fits
from astropy.wcs import WCS

from cosmic3d.prepare import PrepareOptions, prepare
from cosmic3d.runner import CancelToken, Cancelled
import sirilpy

FAKE = os.path.join(HERE, "fakebin", "astap")
fails = 0


def check(name, cond, extra=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not cond:
        fails += 1


tmp = tempfile.mkdtemp(prefix="cosmic3d_test_")
# --- συνθετική εικόνα 16-bit (νεφέλωμα + αστέρια)
rng = np.random.default_rng(1)
H, W = 480, 640
yy, xx = np.mgrid[0:H, 0:W]
img = 0.08 + 0.5 * np.exp(-(((xx - 300) / 120.0) ** 2 + ((yy - 240) / 90.0) ** 2))[..., None] * np.array([0.5, 0.7, 1.0])
for _ in range(200):
    x, y = rng.integers(5, W - 5), rng.integers(5, H - 5)
    img[y - 1:y + 2, x - 1:x + 2] += rng.uniform(0.3, 0.9)
src = os.path.join(tmp, "synthetic.tif")
tifffile.imwrite(src, (np.clip(img, 0, 1) * 65535).astype(np.uint16), photometric="rgb")
work = os.path.join(tmp, "work")

# 1. προετοιμασία με plate solve
logs = []
os.environ.pop("FAKE_ASTAP", None)
r = prepare(src, PrepareOptions(astap_path=FAKE, scale_arcsec=3.5, object_name=""), log=logs.append, workdir=work)
check("plate solve και FITS", r.solved_by == "astap" and os.path.isfile(r.fits_path) and r.width == W and r.height == H, r.summary())
hdr = fits.getheader(r.fits_path)
check("κεφαλίδα WCS στο FITS", WCS(hdr, naxis=[1, 2]).has_celestial and hdr["PLTSOLVD"])
data = fits.getdata(r.fits_path)
check("προσανατολισμός FITS (πρώτη γραμμή = κάτω)", data.shape == (3, H, W) and np.allclose(data[:, 0, :], np.transpose(img, (2, 0, 1))[:, H - 1, :], atol=2e-4))

# 2. αποθηκευμένη λύση / FITS με ήδη WCS
r2 = prepare(src, PrepareOptions(astap_path=FAKE, scale_arcsec=3.5), workdir=work)
check("επαναχρησιμοποίηση αποθηκευμένης λύσης", r2.solved_by == "cache")
r3 = prepare(r.fits_path, PrepareOptions(astap_path=None), workdir=work)
check("FITS με WCS δεν χρειάζεται ASTAP", r3.solved_by == "file", r3.summary())

# 3. σφάλματα
os.environ["FAKE_ASTAP"] = "nosolution"
try:
    prepare(src, PrepareOptions(astap_path=FAKE, force=True), workdir=os.path.join(tmp, "w2"))
    check("σφάλμα χωρίς λύση", False)
except RuntimeError as e:
    check("σφάλμα χωρίς λύση εξηγείται", "Κωδικός ASTAP 1" in str(e), str(e).splitlines()[1][:70])
os.environ.pop("FAKE_ASTAP")
try:
    prepare(src, PrepareOptions(astap_path="/nonexistent"), workdir=os.path.join(tmp, "w3"))
    check("σφάλμα χωρίς ASTAP", False)
except RuntimeError as e:
    check("σφάλμα χωρίς ASTAP εξηγείται", "ASTAP" in str(e))

# 4. ακύρωση
os.environ["FAKE_ASTAP"] = "slow"
tok = CancelToken()
threading.Timer(2.0, tok.cancel).start()
t0 = time.time()
try:
    prepare(src, PrepareOptions(astap_path=FAKE, force=True), cancel=tok, workdir=os.path.join(tmp, "w4"))
    check("ακύρωση", False)
except Cancelled:
    check("ακύρωση σταματά το ASTAP γρήγορα", time.time() - t0 < 15, f"{time.time() - t0:.1f}s")
os.environ.pop("FAKE_ASTAP")

# 5. στρώμα sirilpy
sirilpy.configure(r)
si = sirilpy.SirilInterface()
ra, dec = si.pix2radec(W / 2, H / 2)
x, y = si.radec2pix(ra, dec)
check("pix2radec/radec2pix αντίστροφα", abs(x - W / 2) < 1e-3 and abs(y - H / 2) < 1e-3)
check("API εικόνας", si.get_image().data.shape == (3, H, W) and si.get_image_pixeldata(preview=True).dtype == np.uint8
      and isinstance(si.get_image_fits_header(return_as="dict"), dict) and si.get_image_keywords().pltsolvd)
sirilpy.configure(None)
try:
    sirilpy.SirilInterface().get_image()
    check("NoImageError χωρίς εικόνα", False)
except sirilpy.NoImageError:
    check("NoImageError χωρίς εικόνα", True)
sirilpy.configure(r)

# 6. τα πραγματικά scripts
try:
    from PyQt6.QtWidgets import QApplication
except Exception as e:  # noqa
    print("SKIP scripts (PyQt6 δεν είναι εγκατεστημένο)", e)
else:
    app = QApplication(sys.argv)
    for kind, fname, cls in (("view", "Svenesis-CosmicView3D.py", "CosmicView3DWindow"), ("depth", "Svenesis-CosmicDepth3D.py", "CosmicDepth3DWindow")):
        spec = importlib.util.spec_from_file_location("t_" + kind, os.path.join(ROOT, "scripts", fname))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        w = getattr(mod, cls)(sirilpy.SirilInterface())
        ok = w._load_from_siril()
        check(f"{cls}: φόρτωση εικόνας και WCS από το αρχείο", ok and w._is_plate_solved and w._wcs is not None and (w._img_width, w._img_height) == (W, H),
              f"{w._img_width}x{w._img_height}, {w._pixel_scale:.2f}\"/px")
        w.close()

print("\nΑΠΟΤΥΧΙΕΣ:", fails)
sys.exit(1 if fails else 0)

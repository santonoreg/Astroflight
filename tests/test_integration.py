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
    check("σφάλμα χωρίς λύση εξηγείται", "κωδικός astap 1" in str(e).lower() and "δοκιμές" in str(e), str(e).splitlines()[1][:70])
os.environ.pop("FAKE_ASTAP")
try:
    prepare(src, PrepareOptions(astap_path="/nonexistent"), workdir=os.path.join(tmp, "w3"))
    check("σφάλμα χωρίς ASTAP", False)
except RuntimeError as e:
    check("σφάλμα χωρίς ASTAP εξηγείται", "ASTAP" in str(e))

# 3β. άγνωστη κλίμακα: ο ASTAP λύνει μόνο με σωστή υπόδειξη πεδίου → η σκάλα κλιμάκων πρέπει να τη βρει
os.environ["FAKE_ASTAP"] = "fovonly"
t0 = time.time()
r_l = prepare(src, PrepareOptions(astap_path=FAKE, force=True, allow_approx=False), log=logs.append, workdir=os.path.join(tmp, "w5"))
check("άγνωστη κλίμακα: λύνεται με σκάλα κλιμάκων", r_l.solved_by == "astap" and abs(r_l.scale_arcsec - 3.5) < 0.05, f"{time.time() - t0:.0f}s")
# με λάθος ονομαστική κλίμακα (π.χ. mosaic με resample) πάλι πρέπει να λυθεί
r_n = prepare(src, PrepareOptions(astap_path=FAKE, force=True, allow_approx=False, pixel_um=5.4, focal_mm=300), log=logs.append,
              workdir=os.path.join(tmp, "w6"))
check("λάθος ονομαστική κλίμακα (3.71 αντί 3.5): λύνεται", r_n.solved_by == "astap", f"{r_n.scale_arcsec:.2f}")
os.environ.pop("FAKE_ASTAP")

# 3γ. μεγάλο mosaic με μαύρα περιθώρια: μικραίνει πριν σταλεί και η λύση επιστρέφει στο αρχικό μέγεθος
Hb, Wb = 3000, 4000
big = np.zeros((Hb, Wb, 3), np.uint16)
big[200:2800, 300:3700] = (np.clip(img[0:1, 0:1] * 0 + 0.1, 0, 1) * 65535).astype(np.uint16)
big[1000:1010, 1000:1010] = 60000
bigsrc = os.path.join(tmp, "mosaic.tif")
tifffile.imwrite(bigsrc, big, photometric="rgb")
r_b = prepare(bigsrc, PrepareOptions(astap_path=FAKE, allow_approx=False), log=logs.append, workdir=os.path.join(tmp, "w7"))
f_ds = 2800.0 / 4000.0
check("mosaic: λύση επιστρέφει στην αρχική διάσταση", r_b.width == Wb and r_b.height == Hb and abs(r_b.scale_arcsec - 3.5 * f_ds) < 0.02,
      f"{r_b.scale_arcsec:.3f} (αναμενόμενο {3.5 * f_ds:.3f})")
check("mosaic: το κέντρο της λύσης μένει στο κέντρο", abs(r_b.ra_deg - 97.98) < 0.01 and abs(r_b.dec_deg - 4.93) < 0.01, f"{r_b.ra_deg:.3f} {r_b.dec_deg:.3f}")

# 3δ. αποτυχία ASTAP → προσεγγιστική λύση από όνομα (συντεταγμένες) και κλίμακα
os.environ["FAKE_ASTAP"] = "nosolution"
r_a = prepare(src, PrepareOptions(astap_path=FAKE, force=True, ra_deg=97.98, dec_deg=4.93, object_name="Rosette", scale_arcsec=3.5),
              log=logs.append, workdir=os.path.join(tmp, "w8"))
check("προσεγγιστική λύση όταν αποτύχει ο ASTAP", r_a.solved_by == "approx" and any("ΠΡΟΣΕΓΓΙΣΤΙΚΗ" in w for w in r_a.warnings) and abs(r_a.scale_arcsec - 3.5) < 0.01,
      r_a.summary()[:60])
check("η προσεγγιστική λύση σημαίνεται στο FITS", fits.getheader(r_a.fits_path).get("APPROX") is True and fits.getheader(r_a.fits_path)["PLTSOLVD"] is False)
r_a2 = prepare(src, PrepareOptions(astap_path=FAKE, ra_deg=97.98, dec_deg=4.93, object_name="Rosette", scale_arcsec=3.5), workdir=os.path.join(tmp, "w8"))
check("η προσεγγιστική λύση δεν παρουσιάζεται ως κανονική όταν ξαναχρησιμοποιείται", r_a2.solved_by == "approx" and r_a2.warnings)
os.environ.pop("FAKE_ASTAP")

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

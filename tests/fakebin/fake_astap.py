#!/usr/bin/env python3
"""Ψεύτικος ASTAP για δοκιμές: διαβάζει -f αρχείο.fits και γράφει <όνομα>.wcs με λύση (Rosette, 3.5"/px) – κωδικός 0."""
import sys, os, time
a = sys.argv[1:]
if "-f" not in a:
    print("usage"); sys.exit(16)
f = a[a.index("-f") + 1]
from astropy.io import fits
h = fits.getheader(f)
W, H = h["NAXIS1"], h["NAXIS2"]
print("ASTAP fake: solving", f, W, H, flush=True)
mode = os.environ.get("FAKE_ASTAP", "ok")
time.sleep(1.2)
if mode == "slow":
    time.sleep(60)
if mode == "nosolution":
    print("No solution found", flush=True); sys.exit(1)
if mode == "nodb":
    print("No star database found", flush=True); sys.exit(32)
s = 3.5 / 3600.0
cards = {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 97.98, "CRVAL2": 4.93, "CRPIX1": (W + 1) / 2, "CRPIX2": (H + 1) / 2,
         "CD1_1": -s, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": s, "NAXIS": 2, "NAXIS1": W, "NAXIS2": H}
hdr = fits.Header()
for k, v in cards.items():
    hdr[k] = v
open(os.path.splitext(f)[0] + ".wcs", "w").write(hdr.tostring(sep="\n", padding=False) + "\n")
print("Solution found", flush=True)
sys.exit(0)

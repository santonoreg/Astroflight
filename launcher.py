#!/usr/bin/env python3
"""Cosmic 3D launcher – επιλέγεις μια αστροφωτογραφία και ανοίγεις τα Svenesis CosmicView 3D / CosmicDepth 3D.

Τα δύο scripts (fork του https://github.com/sramuschkat/Siril-Scripts, GPL-3.0) γράφτηκαν ως plugins του Siril και χρειάζονται
μια εικόνα με αστρομετρική λύση (WCS). Αυτό το πρόγραμμα κάνει ό,τι χρειάζεται: διαβάζει το αρχείο που διάλεξες, κάνει
plate solve με ASTAP αν χρειάζεται, και δίνει το αποτέλεσμα στα scripts χωρίς να χρειάζεται Siril.

Χρήση:  python launcher.py                      (γραφικό περιβάλλον)
        python launcher.py --image foto.tif --open view
        python launcher.py --image foto.tif --prepare-only      (χωρίς γραφικά· τυπώνει τη λύση)
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
SHIM_DIR = os.path.join(ROOT, "shim")
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
for p in (ROOT, SHIM_DIR):                 # το shim πρέπει να προηγείται: τα scripts κάνουν «import sirilpy»
    if p not in sys.path:
        sys.path.insert(0, p)

SCRIPTS = {
    "view": ("Svenesis-CosmicView3D.py", "CosmicView3DWindow", "CosmicView 3D"),
    "depth": ("Svenesis-CosmicDepth3D.py", "CosmicDepth3DWindow", "CosmicDepth 3D"),
}


# ---------------------------------------------------------------------------------------------------------------------
def load_script(kind: str):
    """Φορτώνει το script ως module (δεν το τρέχει ως __main__)."""
    fname, _cls, _title = SCRIPTS[kind]
    name = "svenesis_" + kind
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cli_prepare(args) -> int:
    from cosmic3d.prepare import PrepareOptions, prepare
    opts = PrepareOptions(astap_path=args.astap, scale_arcsec=args.scale, pixel_um=args.pixel, focal_mm=args.focal, binning=args.binning,
                          allow_approx=not args.no_approx, object_name=args.object or "", force=args.force)
    try:
        r = prepare(args.image, opts, progress=lambda f, m="": print(f"[{f * 100:3.0f}%] {m}", flush=True), log=lambda m: print("   ", m))
    except Exception as e:  # noqa
        print("ΣΦΑΛΜΑ:", e)
        return 1
    print("Έτοιμο:", r.summary())
    print("FITS:", r.fits_path)
    for w in r.warnings:
        print("Προσοχή:", w)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cosmic 3D launcher (CosmicView 3D / CosmicDepth 3D για οποιαδήποτε αστροφωτογραφία)")
    ap.add_argument("--image", help="αστροφωτογραφία (TIFF/PNG/JPG/FITS)")
    ap.add_argument("--open", choices=["view", "depth", "both"], help="άνοιγμα script μετά την προετοιμασία")
    ap.add_argument("--astap", help="διαδρομή του ASTAP")
    ap.add_argument("--scale", type=float, help="κλίμακα εικόνας σε arcsec/pixel (προαιρετικό)")
    ap.add_argument("--pixel", type=float, help="μέγεθος pixel κάμερας σε µm (π.χ. 5.4 για Atik 383L+)")
    ap.add_argument("--focal", type=float, help="εστιακή απόσταση σε mm")
    ap.add_argument("--binning", type=float, default=1.0)
    ap.add_argument("--no-approx", action="store_true", help="χωρίς εφεδρική προσεγγιστική λύση αν αποτύχει ο ASTAP")
    ap.add_argument("--object", help="όνομα αντικειμένου, π.χ. 'Rosette Nebula' (προαιρετικό)")
    ap.add_argument("--force", action="store_true", help="νέο plate solve ακόμη κι αν υπάρχει αποθηκευμένη λύση")
    ap.add_argument("--prepare-only", action="store_true", help="μόνο προετοιμασία, χωρίς γραφικά")
    args = ap.parse_args(argv)
    if args.prepare_only:
        if not args.image:
            ap.error("--prepare-only χρειάζεται --image")
        return cli_prepare(args)

    # Το QtWebEngine πρέπει να φορτωθεί ΠΡΙΝ δημιουργηθεί το QApplication (αλλιώς τα scripts πέφτουν σε matplotlib)
    try:
        from PyQt6 import QtWebEngineWidgets  # noqa: F401
    except Exception:
        pass
    from PyQt6.QtCore import Qt, QCoreApplication
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    from cosmic3d.gui import run_gui
    return run_gui(args)


if __name__ == "__main__":
    sys.exit(main())

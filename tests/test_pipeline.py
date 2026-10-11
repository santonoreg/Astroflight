"""Δοκιμές του αυτόματου pipeline με ψεύτικα ASTAP/StarNet (tests/fakebin). Τρέξε:  python tests/test_pipeline.py
Ελέγχει: επιτυχία, αποτυχία ASTAP, αποτυχία StarNet, ακύρωση, επαναχρησιμοποίηση αποτελεσμάτων. Δεν χρειάζεται GPU."""
import os, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from astroflight.config import Params
from astroflight.project import Project
from astroflight.pipeline import AutoOptions, Reporter, run_auto
from astroflight.runner import CancelToken, Cancelled

IMG = os.path.join(ROOT, "astrophotography.png")
if not os.path.exists(IMG):
    from make_synthetic import make
    import tempfile
    IMG = make(os.path.join(tempfile.mkdtemp(), "synthetic.tif"))
FB = os.path.join(HERE, "fakebin")


def fresh():
    proj = Project(Params(vol_size=192, vol_layers=24), log=lambda s: None)
    proj.astap_path = os.path.join(FB, "astap")
    proj.starnet_path = os.path.join(FB, "starnet2")
    proj.load_image(IMG)
    return proj


def run(proj, opts=None, token=None):
    states = {}
    rep = Reporter(on_step=lambda i, s, m: states.__setitem__(i, (s, m)))
    res = run_auto(proj, opts or AutoOptions(use_gaia=False), rep, token)
    return res, states


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not cond:
        check.failed += 1
check.failed = 0


# 1. όλα επιτυχή
os.environ.pop("FAKE_ASTAP", None); os.environ.pop("FAKE_STARNET", None)
proj = fresh()
res, st = run(proj)
check("επιτυχής εκτέλεση", proj.scene is not None and res["solved"] and res["starnet"], str(st["solve"]))
check("πτήση δημιουργήθηκε", len(proj.keyframes) > 10 and proj.path_kind == "dive")
check("keyframes έχουν FOV", max(k.state.logfov for k in proj.keyframes) > 1.0)
check("αναγνώριση αντικειμένου", res["object"] and "Rosette" in res["object"], str(res["object"]))

# 2. επαναχρησιμοποίηση
t = time.time()
res2, st2 = run(proj)
check("reuse: δεν ξανατρέχουν τα βήματα 1-5", "προηγούμενη" in st2["starless"][1] and time.time() - t < 20, f"{time.time() - t:.1f}s")

# 3. ο ASTAP δεν βρίσκει λύση
os.environ["FAKE_ASTAP"] = "nosolution"
proj = fresh()
res, st = run(proj)
check("ASTAP χωρίς λύση → συνέχεια με προειδοποίηση", proj.scene is not None and st["solve"][0] == "warn" and not res["solved"], st["solve"][1])
os.environ["FAKE_ASTAP"] = "nodb"
proj = fresh()
res, st = run(proj)
check("ASTAP κωδικός 32 εξηγείται", proj.scene is not None and any("βάση αστεριών" in w for w in res["warnings"]), res["warnings"][0][:90])
os.environ.pop("FAKE_ASTAP")

# 4. το StarNet καταρρέει
os.environ["FAKE_STARNET"] = "crash"
proj = fresh()
res, st = run(proj)
check("StarNet crash → ενσωματωμένη μέθοδος", proj.scene is not None and st["starless"][0] == "warn" and not res["starnet"], st["starless"][1])

# 5. ακύρωση στη μέση του StarNet
os.environ["FAKE_STARNET"] = "slow"
proj = fresh()
tok = CancelToken()
threading.Timer(4.0, tok.cancel).start()
t = time.time()
try:
    run(proj, token=tok)
    check("ακύρωση", False, "δεν σηκώθηκε Cancelled")
except Cancelled:
    check("ακύρωση σταματά το StarNet γρήγορα", time.time() - t < 15, f"{time.time() - t:.1f}s")
os.environ.pop("FAKE_STARNET")

# 6. χωρίς εργαλεία καθόλου
proj = fresh()
proj.astap_path = proj.starnet_path = "/nonexistent"
res, st = run(proj, AutoOptions(use_gaia=False, scale_arcsec=3.5))
check("χωρίς ASTAP/StarNet λειτουργεί με κλίμακα", proj.scene is not None and st["solve"][0] == "skip", str(st["starless"]))

# 7. τοπικό αρχείο Gaia (CSV) – χωρίς internet
import numpy as np, tempfile
proj = fresh()
os.environ.pop("FAKE_ASTAP", None); os.environ.pop("FAKE_STARNET", None)
res, st = run(proj, AutoOptions(use_gaia=False))                 # πρώτα μια λύση για να έχουμε WCS και αστέρια
ra, dec = proj.sky.pix_to_world(proj.stars.x, proj.stars.y)
rng = np.random.default_rng(3)
n = len(proj.stars)
csv = os.path.join(tempfile.mkdtemp(), "gaia.csv")
with open(csv, "w") as f:
    f.write("source_id,ra,dec,parallax,parallax_error,phot_g_mean_mag,bp_rp\n")
    for i in range(n):
        plx = 1000.0 / (1400 + 300 * rng.standard_normal())
        f.write(f"{i},{ra[i]:.7f},{dec[i]:.7f},{plx:.4f},0.02,{proj.stars.mag[i] + 20.0:.3f},0.8\n")
proj.gaia_file = csv
res, st = run(proj, AutoOptions(use_gaia=True))
check("τοπικό αρχείο Gaia αντιστοιχίζεται", res["gaia"] and proj.dist["n_matched"] > 0.7 * n, st["gaia"][1])
check("απόσταση από Gaia ~1400 pc", proj.suggest is not None and 1100 < proj.suggest[0] < 1700 or abs(proj.p.object_distance_pc - 1500) < 1, str(proj.suggest))
proj.gaia_file = None
os.environ["FAKE_ASTAP"] = "nosolution"
proj = fresh()
res, st = run(proj, AutoOptions(use_gaia=True))
check("χωρίς λύση το Gaia εξηγεί γιατί παραλείφθηκε", st["gaia"][0] == "warn" and "plate solve" in " ".join(res["warnings"]), st["gaia"][1])
os.environ.pop("FAKE_ASTAP")

print("\nΑΠΟΤΥΧΙΕΣ:", check.failed)
sys.exit(1 if check.failed else 0)

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

print("\nΑΠΟΤΥΧΙΕΣ:", check.failed)
sys.exit(1 if check.failed else 0)

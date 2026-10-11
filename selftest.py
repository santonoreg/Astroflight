"""Αυτοέλεγχος: φτιάχνει συνθετική αστροφωτογραφία, χτίζει σκηνή, αποδίδει καρέ και συγκρίνει με το πρωτότυπο.
Τρέξε:  python selftest.py        (σε Linux χωρίς οθόνη:  ASTROFLIGHT_BACKEND=egl python selftest.py)"""
import os, sys, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
import numpy as np
import cv2
import moderngl
from make_synthetic import make
from astroflight.config import Params
from astroflight.project import Project
from astroflight.renderer import Renderer
from astroflight.camera import CamState, pose_from_state
from astroflight import imageio_ as io

out = os.path.join(os.getcwd(), "selftest_out")
os.makedirs(out, exist_ok=True)
tif = make(os.path.join(out, "synthetic.tif"))
p = Params(vol_size=384, vol_layers=48)
proj = Project(p, log=print)
proj.load_image(tif)
proj.set_scale(5.0)
t0 = time.time()
sc = proj.build()
print(f"Χτίσιμο σκηνής: {time.time() - t0:.1f} s")

backend = os.environ.get("ASTROFLIGHT_BACKEND") or None
ctx = moderngl.create_standalone_context(backend=backend) if backend else moderngl.create_standalone_context()
print("GL:", ctx.info["GL_RENDERER"])
r = Renderer(ctx)
r.set_scene(sc, proj.tone)
m = sc.meta
W, H = m["W"] // 2, m["H"] // 2
q = Params(**{**p.to_dict(), "bloom": 0.0})
a = r.render_to_array(pose_from_state(CamState(), m["D0"], m["W0"]), q, (W, H)) / 255.0
ref = cv2.resize(proj.img, (W, H), interpolation=cv2.INTER_AREA)
mae = float(np.abs(a - ref).mean())
print(f"Μέσο σφάλμα απόδοσης από το σημείο λήψης vs πρωτότυπο: {mae:.4f}  ({'ΟΚ' if mae < 0.01 else 'ΕΛΕΓΞΕ'})")
io.save_png(os.path.join(out, "origin.png"), a)
io.save_png(os.path.join(out, "reference.png"), ref)
for name, s in (("orbit", CamState(az=12, el=5, logr=np.log(0.8))), ("approach", CamState(az=6, el=3, logr=np.log(0.3)))):
    f = r.render_to_array(pose_from_state(s, m["D0"], m["W0"]), p, (W, H)) / 255.0
    io.save_png(os.path.join(out, f"{name}.png"), f)
print("Τα καρέ αποθηκεύτηκαν στο", out)

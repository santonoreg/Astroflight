"""Γραμμή εντολών (χωρίς GUI): όλη η αυτόματη διαδικασία + εξαγωγή βίντεο.

Παράδειγμα:
    python -m astroflight.cli astrophotography.png flight.mp4 --scale 3.5 --duration 12
Σε Linux χωρίς οθόνη:  ASTROFLIGHT_BACKEND=egl python -m astroflight.cli ...
"""
from __future__ import annotations
import argparse, os, sys, time


def main(argv=None):
    ap = argparse.ArgumentParser(description="AstroFlight – αυτόματη δημιουργία 3D πτήσης από αστροφωτογραφία")
    ap.add_argument("image", help="αστροφωτογραφία (TIFF/PNG/JPG)")
    ap.add_argument("out", nargs="?", default="flight.mp4", help="αρχείο εξόδου .mp4 (ή φάκελος με --png)")
    ap.add_argument("--png", action="store_true", help="εξαγωγή ακολουθίας PNG αντί για MP4")
    ap.add_argument("--scale", type=float, default=None, help="κλίμακα εικόνας σε arcsec/pixel (αν δεν είναι γνωστή, βρίσκεται από τον ASTAP)")
    ap.add_argument("--focal", type=float, default=None, help="εστιακή απόσταση mm (εναλλακτικά της --scale)")
    ap.add_argument("--pixel", type=float, default=None, help="μέγεθος pixel σε µm (με την --focal)")
    ap.add_argument("--object", default="", help="όνομα αντικειμένου (π.χ. 'Rosette Nebula') – βοηθά το plate solve")
    ap.add_argument("--distance", type=float, default=None, help="απόσταση αντικειμένου σε pc (αλλιώς αυτόματα)")
    ap.add_argument("--astap", default=None, help="διαδρομή astap")
    ap.add_argument("--starnet", default=None, help="διαδρομή starnet2")
    ap.add_argument("--gaia-file", default=None, help="τοπικό αρχείο Gaia (CSV/FITS/VOT) αντί για internet")
    ap.add_argument("--check-gaia", action="store_true", help="έλεγχος σύνδεσης με τους servers Gaia και έξοδος")
    ap.add_argument("--no-astap", action="store_true")
    ap.add_argument("--no-starnet", action="store_true")
    ap.add_argument("--no-gaia", action="store_true")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080", help="π.χ. 1280x720")
    ap.add_argument("--ssaa", type=int, default=1)
    ap.add_argument("--vol", type=int, default=512, help="ανάλυση όγκου (128–1024)")
    ap.add_argument("--frames", type=int, default=0, help="δοκιμή: αποδίδει μόνο τα πρώτα Ν καρέ ισαπέχοντα στη διαδρομή")
    args = ap.parse_args(argv)
    if args.check_gaia:
        from .gaia import diagnose_connection
        sys.exit(0 if diagnose_connection() else 1)

    import moderngl
    from .config import Params
    from .project import Project
    from .renderer import Renderer
    from .camera import CameraPath, pose_from_state
    from .pipeline import AutoOptions, Reporter, run_auto, STEP_TITLES
    from . import wcs_solve as ws, export as ex

    p = Params(duration=args.duration, fps=args.fps, ssaa=args.ssaa, vol_size=args.vol)
    p.out_width, p.out_height = (int(v) for v in args.size.lower().split("x"))
    p.steps = max(p.steps, 256)
    if args.distance:
        p.object_distance_pc = args.distance
    proj = Project(p, log=print)
    if args.astap:
        proj.astap_path = args.astap
    if args.starnet:
        proj.starnet_path = args.starnet
    if args.gaia_file:
        proj.gaia_file = args.gaia_file
    scale = args.scale
    if scale is None and args.focal and args.pixel:
        scale = ws.scale_from_optics(args.focal, args.pixel)
    opts = AutoOptions(use_astap=not args.no_astap, use_starnet=not args.no_starnet, use_gaia=not args.no_gaia,
                       scale_arcsec=scale, object_name=args.object, auto_distance=args.distance is None)

    t0 = time.time()
    proj.load_image(args.image)
    icons = dict(running="…", ok="✔", warn="⚠", skip="–", fail="✖")
    last = {}

    def on_step(sid, state, msg):
        if state != "running" and last.get(sid) != (state, msg):
            print(f"[{icons[state]}] {STEP_TITLES[sid]}: {msg}", flush=True)
        last[sid] = (state, msg)

    rep = Reporter(on_step=on_step, on_log=lambda s: print("   ", s, flush=True))
    summary = run_auto(proj, opts, rep)
    print(f"Προετοιμασία: {summary['seconds']:.0f} s" + (f" – προειδοποιήσεις: {len(summary['warnings'])}" if summary["warnings"] else ""))

    backend = os.environ.get("ASTROFLIGHT_BACKEND") or None
    ctx = moderngl.create_standalone_context(backend=backend) if backend else moderngl.create_standalone_context()
    r = Renderer(ctx)
    r.set_scene(proj.scene, proj.tone)
    path = proj.make_path()
    if args.frames:
        import math
        from . import imageio_ as io
        os.makedirs(args.out, exist_ok=True)
        m = proj.scene.meta
        for i in range(args.frames):
            t = p.duration * i / max(args.frames - 1, 1)
            st = path.eval(t)
            fr = r.render_to_array(pose_from_state(st, m["D0"], m["W0"]), p, (p.out_width, p.out_height), p.ssaa,
                                   fov_mul=math.exp(st.logfov), fade=st.fade)
            io.save_png(os.path.join(args.out, f"frame_{i:03d}.png"), fr / 255.0)
            print(f"καρέ {i + 1}/{args.frames}", flush=True)
    else:
        def prog(f):
            print(f"\rΑπόδοση βίντεο {f * 100:5.1f}%", end="", flush=True)
        ex.render_video(r, path, p, args.out, prog, None, print, args.png)
        print()
    print(f"Έτοιμο σε {time.time() - t0:.0f} s → {args.out}")


if __name__ == "__main__":
    main()

"""Εξαγωγή βίντεο (MP4 μέσω ffmpeg) ή ακολουθίας PNG."""
from __future__ import annotations
import os, math, subprocess
import numpy as np
from .camera import CameraPath, pose_from_state
from .config import Params
from . import imageio_ as io


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg")


def render_video(renderer, path_obj: CameraPath, p: Params, out_path: str, progress=None, cancel=None, log=print,
                 png_sequence: bool = False):
    sc = renderer.scene
    m = sc.meta
    w, h = int(p.out_width) // 2 * 2, int(p.out_height) // 2 * 2
    n = max(2, int(round(p.duration * p.fps)))
    proc = None
    if not png_sequence:
        exe = ffmpeg_exe()
        if exe is None:
            raise RuntimeError("Δεν βρέθηκε ffmpeg (pip install imageio-ffmpeg) – χρησιμοποίησε εξαγωγή PNG.")
        cmd = [exe, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
               "-r", str(int(p.fps)), "-i", "-", "-an", "-vcodec", "libx264", "-preset", "slow", "-crf", str(int(p.video_crf)),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
        kw = {"creationflags": 0x08000000} if os.name == "nt" else {}       # χωρίς αναβοσβήσιμο παραθύρου κονσόλας
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, **kw)
    else:
        os.makedirs(out_path, exist_ok=True)
    try:
        for i in range(n):
            if cancel and cancel():
                log("Η εξαγωγή ακυρώθηκε.")
                break
            t = p.duration * i / (n - 1)
            st = path_obj.eval(t)
            pose = pose_from_state(st, m["D0"], m["W0"])
            frame = renderer.render_to_array(pose, p, (w, h), ssaa=max(1, int(p.ssaa)), fov_mul=math.exp(st.logfov))
            if png_sequence:
                io.save_png(os.path.join(out_path, f"frame_{i:05d}.png"), frame / 255.0)
            else:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if progress:
                progress((i + 1) / n)
    finally:
        if proc is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait()
    log(f"Τέλος εξαγωγής: {out_path}")

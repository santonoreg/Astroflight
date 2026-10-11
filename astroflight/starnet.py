"""Ενσωμάτωση StarNet++ (command line) για αφαίρεση αστεριών.

Σύνταξη:
 – StarNet2 (starnet2, επίσημη σελίδα starnetastro.com, v2.6.x):  ``starnet2 -i είσοδος.tif -o έξοδος.tif [-s stride]``
 – StarNet++ v1:  ``starnet++ είσοδος.tif έξοδος.tif [stride]``
Η είσοδος γράφεται ως 16-bit TIFF και ΤΕΝΤΩΜΕΝΗ (non-linear). Το v1 θέλει τα αρχεία βαρών (*.pb) στον ίδιο φάκελο
με το εκτελέσιμο, γι' αυτό το τρέχουμε με cwd = φάκελος του εκτελέσιμου.
Το «πρότυπο ορισμάτων» στο GUI μπορεί να αλλάξει (κενό = αυτόματο ανά έκδοση):
   {in} = αρχείο εισόδου, {out} = αρχείο εξόδου, {stride} = τιμή stride.
"""
from __future__ import annotations
import os, sys, shutil, subprocess, tempfile, glob
import numpy as np

DEFAULT_ARGS = ""          # κενό = αυτόματο ανά έκδοση
ARGS_V1 = "{in} {out} {stride}"
ARGS_V2 = "-i {in} -o {out} -s {stride}"


def default_args(exe: str) -> str:
    n = os.path.basename(exe or "").lower()
    return ARGS_V2 if n.startswith("starnet2") else ARGS_V1
EXE_NAMES = ("starnet2", "starnet++", "starnet", "StarNet2", "StarNet++")
_WIN_DIRS = (r"C:\Program Files\StarNet2\bin", r"C:\Program Files\StarNet++", r"C:\Program Files\StarNet2", r"C:\StarNet++", r"C:\StarNet2",
             r"C:\Starnet", r"C:\Program Files\PixInsight\bin")


def find_starnet() -> str | None:
    for n in EXE_NAMES:
        p = shutil.which(n)
        if p:
            return p
    home = os.path.expanduser("~")
    dirs = list(_WIN_DIRS) + [os.path.join(home, d) for d in ("StarNet++", "StarNet2", "starnet", "Downloads/StarNet++")]
    for d in dirs:
        for n in EXE_NAMES:
            for ext in (".exe", ""):
                p = os.path.join(d, n + ext)
                if os.path.isfile(p):
                    return p
    return None


def _write_tiff16(path: str, rgb01: np.ndarray):
    import tifffile
    a = (np.clip(rgb01, 0, 1) * 65535.0 + 0.5).astype(np.uint16)
    tifffile.imwrite(path, a, photometric="rgb")


def _read_tiff(path: str) -> np.ndarray:
    from . import imageio_ as io
    return io.load_image(path)


def build_cmd(exe: str, fin: str, fout: str, args_template: str = "", stride: int = 0) -> list[str]:
    toks = (args_template or default_args(exe)).split()
    cmd = [exe]
    for t in toks:
        if "{stride}" in t and not stride:
            if len(cmd) > 1 and cmd[-1].startswith("-"):
                cmd.pop()                       # π.χ. το «-s» όταν δεν δίνεται stride
            continue
        cmd.append(t.replace("{in}", fin).replace("{out}", fout).replace("{stride}", str(int(stride))))
    return cmd


def run_starnet(exe: str, stretched01: np.ndarray, args_template: str = "", stride: int = 0,
                timeout: float = 3600.0, log=print) -> np.ndarray:
    """Τρέχει το StarNet στην ΤΕΝΤΩΜΕΝΗ εικόνα (HxWx3, 0..1) και επιστρέφει την starless (ίδιο μέγεθος)."""
    if not exe or not os.path.isfile(exe):
        raise RuntimeError("Δεν βρέθηκε το εκτελέσιμο του StarNet. Όρισε τη διαδρομή (π.χ. C:\\Program Files\\StarNet2\\bin\\starnet2.exe).")
    tmp = tempfile.mkdtemp(prefix="afstar_")           # ASCII διαδρομή: το StarNet δεν χειρίζεται καλά ελληνικά paths
    try:
        fin = os.path.join(tmp, "in.tif")
        fout = os.path.join(tmp, "out.tif")
        if min(stretched01.shape[:2]) < 512:
            raise RuntimeError("Το StarNet2 δέχεται εικόνες τουλάχιστον 512×512 pixel.")
        _write_tiff16(fin, stretched01)
        cmd = build_cmd(exe, fin, fout, args_template, stride)
        log("StarNet: " + " ".join(cmd))
        kw = {}
        if sys.platform.startswith("win"):
            kw["creationflags"] = 0x08000000           # CREATE_NO_WINDOW
        cwd = os.path.dirname(os.path.abspath(exe))
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, errors="replace", **kw)
        out = fout
        if not os.path.isfile(out):
            # προεπιλεγμένο όνομα εξόδου όταν δεν δοθεί
            cands = [os.path.join(cwd, "starless.tiff"), os.path.join(cwd, "starless.tif"),
                     os.path.join(tmp, "starless.tiff"), os.path.join(tmp, "starless.tif")]
            cands += glob.glob(os.path.join(tmp, "*starless*"))
            out = next((c for c in cands if os.path.isfile(c)), None)
        if out is None:
            msg = (r.stdout or "") + "\n" + (r.stderr or "")
            raise RuntimeError(f"Το StarNet δεν δημιούργησε αρχείο εξόδου (κωδικός {r.returncode}).\n{msg.strip()[-800:]}\n"
                               "Έλεγξε ότι το πακέτο είναι πλήρες (τα αρχεία βαρών/DLL μένουν μαζί με το εκτελέσιμο) και ότι το «πρότυπο ορισμάτων» ταιριάζει στην έκδοσή σου.")
        res = _read_tiff(out)
        if res.shape[:2] != stretched01.shape[:2]:
            from . import imageio_ as io
            res = io.resize_to(res, stretched01.shape[1], stretched01.shape[0])
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

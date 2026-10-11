"""Κατασκευή 3D σκηνής: όγκος νεφελώματος (frustum-aligned) + αστέρια με πραγματικές θέσεις.

Σύστημα συντεταγμένων (parsec): παρατηρητής στο (0,0,0), ο άξονας +z δείχνει προς το κέντρο του
πεδίου, +x δεξιά, +y πάνω (όπως φαίνεται η εικόνα). Ο όγκος είναι ευθυγραμμισμένος με το frustum της
αρχικής κάμερας: κάθε στήλη (u,v) του όγκου είναι ακριβώς μία ακτίνα από το σημείο λήψης, άρα από
εκεί η σκηνή αναπαράγει ΑΚΡΙΒΩΣ την αρχική φωτογραφία.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2
from .config import Params
from .depth import compute_depth, compute_thickness, compute_dust_tau
from .stars import Stars

ARCSEC_RAD = 1.0 / 206264.806


@dataclass
class SceneData:
    vol: np.ndarray            # (nz, nyp, nxp, 4) float16: rgb = εκπομπή, a = απορρόφηση (ανά στρώμα)
    detail: np.ndarray         # (Hp, Wp, 3) float16: λόγος full-res / low-res
    bg: np.ndarray             # (3,) φόντο ουρανού (γραμμικό)
    stars: np.ndarray          # (N, 9) float32: pos3, col3, L, F0, fwhm
    meta: dict = field(default_factory=dict)
    depth_preview: np.ndarray | None = None
    star_dist: np.ndarray | None = None


def _fade_profile(n_total: int, pad: int) -> np.ndarray:
    i = np.arange(n_total, dtype=np.float32)
    d = np.maximum(0, np.maximum(pad - 0.5 - i, i - (n_total - 1 - pad) - 0.5)) / max(pad, 1)
    return np.where(d <= 0, 1.0, 0.5 * (1 + np.cos(np.pi * np.clip(d, 0, 1)))).astype(np.float32)


def _pad_fade(img: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return img.astype(np.float32)
    out = cv2.copyMakeBorder(img.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    fy = _fade_profile(out.shape[0], pad)
    fx = _fade_profile(out.shape[1], pad)
    return out * (fy[:, None] * fx[None, :])[..., None]


def build_scene(lin: np.ndarray, starless: np.ndarray, stars: Stars, dist_pc: np.ndarray,
                scale_arcsec: float, p: Params, imported: np.ndarray | None = None,
                ai: np.ndarray | None = None, progress=None, extra: dict | None = None) -> SceneData:
    def prog(f, msg=""):
        if progress:
            progress(f, msg)

    H, W, _ = lin.shape
    s = scale_arcsec * ARCSEC_RAD                      # rad ανά pixel (= εφαπτομενικό επίπεδο)
    D0 = float(p.object_distance_pc)

    # --- φόντο ουρανού και καθαρό νεφέλωμα ---
    sub = starless[::4, ::4].reshape(-1, 3)
    bg = np.percentile(sub, 1.0, axis=0).astype(np.float32)
    neb = np.clip(starless - bg, 0, None).astype(np.float32)

    # --- γεωμετρία ---
    nx0 = int(min(p.vol_size, W))
    k = W / nx0
    ny0 = max(8, int(round(H / k)))
    pad_low = int(round(p.pad_frac * nx0))
    pf = int(round(pad_low * W / nx0))
    low = cv2.resize(neb, (nx0, ny0), interpolation=cv2.INTER_AREA)
    low_p = _pad_fade(low, pad_low)
    full_p = _pad_fade(neb, pf)
    nyp, nxp = low_p.shape[:2]
    Hp, Wp = full_p.shape[:2]
    tx0, ty0 = W / 2.0 * s, H / 2.0 * s
    tx_p, ty_p = (W / 2.0 + pf) * s, (H / 2.0 + pf) * s
    W0 = 2.0 * D0 * tx0
    E = min(p.depth_extent * W0, 1.8 * D0)
    z0 = D0 - E / 2.0
    nz = int(p.vol_layers)
    prog(0.05, "Χάρτης βάθους")

    # --- βάθος / πάχος / σκόνη ---
    depth, lum_n = compute_depth(low_p, p, imported, ai)
    sigma = compute_thickness(lum_n, p)
    tau = compute_dust_tau(low_p, p)
    center = (0.5 + (depth - 0.5) * 0.8).astype(np.float32)
    t = (np.arange(nz, dtype=np.float32) + 0.5) / nz
    use_dust = p.dust_strength > 0 and float(tau.max()) > 1e-4
    dcenter = np.clip(center - p.dust_offset, 0.03, 0.97)
    dsig = np.float32(max(p.dust_thick, 0.5 / nz))

    def prof(kk, c, sg):
        return np.exp(-0.5 * ((t[kk] - c) / sg) ** 2)

    prog(0.15, "Κανονικοποίηση προφίλ")
    # πίσω κέλυφος: καθρεφτισμένο βάθος (1 - βάθος) με βάρος m, ώστε μέσα στο νεφέλωμα να υπάρχει «τοίχωμα» και απέναντι
    mrr = float(np.clip(getattr(p, "rear_shell", 0.0), 0.0, 0.9))
    center2 = np.clip(1.0 - center, 0.03, 0.97).astype(np.float32)
    Wsum = np.zeros((nyp, nxp), np.float32)
    Wsum2 = np.zeros((nyp, nxp), np.float32)
    Dsum = np.zeros((nyp, nxp), np.float32)
    for kk in range(nz):
        Wsum += prof(kk, center, sigma)
        if mrr > 0:
            Wsum2 += prof(kk, center2, sigma)
        if use_dust:
            Dsum += prof(kk, dcenter, dsig)
    Wsum = np.maximum(Wsum, 1e-9)
    Wsum2 = np.maximum(Wsum2, 1e-9)
    Dsum = np.maximum(Dsum, 1e-9)

    def wfun(kk):
        w = (1.0 - mrr) * prof(kk, center, sigma) / Wsum
        if mrr > 0:
            w = w + mrr * prof(kk, center2, sigma) / Wsum2
        return w

    vol = np.zeros((nz, nyp, nxp, 4), np.float16)
    cum = np.zeros((nyp, nxp), np.float32)
    S = np.zeros((nyp, nxp), np.float32)
    for kk in range(nz):
        wk = wfun(kk)
        if use_dust:
            ak = tau * prof(kk, dcenter, dsig) / Dsum
            vol[kk, ..., 3] = ak
            S += wk * np.exp(-(cum + 0.5 * ak))
            cum += ak
        else:
            S += wk
    B = (1.0 / np.maximum(S, 1e-3)).astype(np.float32) if use_dust else np.ones_like(S)
    prog(0.35, "Όγκος εκπομπής")
    for kk in range(nz):
        wk = wfun(kk) * B
        vol[kk, ..., :3] = (low_p * wk[..., None]).astype(np.float16)

    # --- detail (full-res) ---
    prog(0.55, "Detail υψηλής ανάλυσης")
    up = cv2.resize(low_p, (Wp, Hp), interpolation=cv2.INTER_LINEAR)
    eps = 1e-3 + 0.01 * float(low_p.mean())
    ratio = np.clip((full_p + eps) / (up + eps), 0.05, 12.0)
    detail = ratio.astype(np.float16)
    del up, ratio, full_p

    # --- αστέρια ---
    prog(0.7, "Αστέρια")
    sx, sy = stars.x.astype(np.float64), stars.y.astype(np.float64)
    sflux, sfwhm = stars.flux.astype(np.float64), stars.fwhm
    scol = stars.flux_rgb / np.maximum(stars.flux, 1e-12)[:, None]
    rr = np.asarray(dist_pc, np.float64)
    n_main = len(stars)
    if extra is not None and len(extra["x"]):
        sx = np.r_[sx, extra["x"]]
        sy = np.r_[sy, extra["y"]]
        sflux = np.r_[sflux, extra["flux"]]
        sfwhm = np.r_[sfwhm, extra["fwhm"]]
        scol = np.vstack([scol, extra["rgb"]])
        rr = np.r_[rr, extra["dist"]]
    N = len(sx)
    sa = np.zeros((N, 9), np.float32)
    if N:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        xi = (sx - cx) * s
        eta = -(sy - cy) * s
        nrm = np.sqrt(1.0 + xi * xi + eta * eta)
        r = rr
        zs = r / nrm
        pos = np.c_[xi * zs, eta * zs, zs]
        F0 = np.maximum(sflux, 1e-9)
        col = scol
        # τ0 κατά μήκος της ακτίνας προς το αστέρι (ίδια λογική με τον shader)
        tau0 = np.zeros(N, np.float32)
        if use_dust:
            ci = np.clip(((sx + 0.5 + pf) / Wp * nxp).astype(int), 0, nxp - 1)
            ri = np.clip(((sy + 0.5 + pf) / Hp * nyp).astype(int), 0, nyp - 1)
            acol = vol[:, ri, ci, 3].astype(np.float32)               # (nz, N)
            ce = np.vstack([np.zeros((1, N), np.float32), np.cumsum(acol, axis=0)])
            kf = np.clip((zs - z0) / E * nz, 0, nz)
            i0 = np.clip(np.floor(kf).astype(int), 0, nz - 1)
            fr = kf - i0
            ar = np.arange(N)
            tau0 = ce[i0, ar] * (1 - fr) + ce[i0 + 1, ar] * fr
            inside = (sx > -pf) & (sx < W + pf) & (sy > -pf) & (sy < H + pf)
            tau0 = np.where(inside, tau0, 0.0)
        L = F0 * r * r * np.exp(tau0)
        sa[:, 0:3] = pos
        sa[:, 3:6] = col
        sa[:, 6] = L
        sa[:, 7] = F0
        sa[:, 8] = sfwhm
    # --- συνθετικός ουρανός γύρω από το σημείο εξόδου: γεμίζει το κάδρο όταν το οπτικό πεδίο ανοίγει πολύ ---
    n_amb = int(getattr(p, "ambient_stars", 0))
    if n_amb > 0 and len(stars) > 5:
        rng = np.random.default_rng(11)
        C = np.array([0.0, 0.0, D0 + E / 2.0 + 0.4 * W0])                # περίπου εκεί που βγαίνει η κάμερα
        psi = np.radians(70.0)                                            # ημιγωνία κώνου γύρω από τον άξονα +z
        cz = rng.uniform(np.cos(psi), 1.0, n_amb)                         # ομοιόμορφα κατά στερεά γωνία
        ph = rng.uniform(0, 2 * np.pi, n_amb)
        sz = np.sqrt(1 - cz * cz)
        dirs = np.c_[sz * np.cos(ph), sz * np.sin(ph), cz]
        rr_amb = 3.0 * D0 * rng.uniform(0, 1, n_amb) ** 0.45 + 4.0 * W0 / 36.0
        posA = C + dirs * rr_amb[:, None]
        r_o = np.linalg.norm(posA, axis=1)                               # απόσταση από τον αρχικό παρατηρητή
        order = np.argsort(-stars.flux)                                   # μόνο τα φωτεινότερα: στο ανοιχτό κάδρο τα αμυδρά δεν φαίνονται
        bright = order[:max(8, int(0.10 * len(stars)))]
        pick = bright[rng.integers(0, len(bright), n_amb)]
        F0a = np.maximum(stars.flux[pick].astype(np.float64) * np.exp(rng.normal(0, 0.35, n_amb)), 1e-9)
        amb = np.zeros((n_amb, 9), np.float32)
        amb[:, 0:3] = posA
        amb[:, 3:6] = (stars.flux_rgb[pick] / np.maximum(stars.flux[pick], 1e-12)[:, None])
        amb[:, 6] = F0a * r_o * r_o
        amb[:, 7] = F0a
        amb[:, 8] = -np.abs(stars.fwhm[pick])                             # αρνητικό = ουρανός (βλ. renderer)
        sa = np.vstack([sa, amb])
    prog(0.95, "Τέλος")

    meta = dict(nz=nz, nxp=nxp, nyp=nyp, Wp=Wp, Hp=Hp, z0=float(z0), E=float(E), tx_p=float(tx_p), ty_p=float(ty_p),
                tx0=float(tx0), ty0=float(ty0), D0=D0, W0=float(W0), W=W, H=H, s=float(s), pad_low=pad_low, pf=pf,
                use_dust=bool(use_dust), fref=float(np.median(stars.flux)) if len(stars) else 1.0)
    dp = depth[pad_low:nyp - pad_low, pad_low:nxp - pad_low] if pad_low > 0 else depth
    meta['n_main_stars'] = n_main
    return SceneData(vol=vol, detail=detail, bg=bg, stars=sa, meta=meta, depth_preview=dp.copy(), star_dist=rr)


def column_depth_quantile(sc: SceneData, fx: float, fy: float, q: float, half: float = 0.04) -> float:
    """Βάθος z (pc) στο οποίο συσσωρεύεται το ποσοστό q της εκπομπής στη στήλη γύρω από το σημείο (fx, fy)
    (ποσοστά μισού πλάτους/ύψους της εικόνας, +y πάνω). Χρησιμεύει για να τελειώνει η πτήση ΜΕΣΑ στα νέφη."""
    m = sc.meta
    v = sc.vol
    nz, nyp, nxp, _ = v.shape
    pad = m["pad_low"]
    nx0, ny0 = nxp - 2 * pad, nyp - 2 * pad
    cx = pad + (0.5 + 0.5 * fx) * nx0
    cy = pad + (0.5 - 0.5 * fy) * ny0
    r = max(2, int(half * nx0))
    x0, x1 = int(max(0, cx - r)), int(min(nxp, cx + r + 1))
    y0, y1 = int(max(0, cy - r)), int(min(nyp, cy + r + 1))
    e = v[:, y0:y1, x0:x1, :3].astype(np.float32).sum(axis=(1, 2, 3))
    if e.sum() <= 1e-12:
        return m["z0"] + 0.5 * m["E"]
    c = np.cumsum(e) / e.sum()
    k = int(np.searchsorted(c, min(max(q, 0.0), 1.0)))
    k = min(k, nz - 1)
    return float(m["z0"] + (k + 0.5) / nz * m["E"])


def suggest_poi(starless: np.ndarray, max_frac: float = 0.55) -> tuple[float, float]:
    """Προτεινόμενο σημείο-στόχος: η φωτεινότερη περιοχή του νεφελώματος (στο εσωτερικό του κάδρου)."""
    g = starless.mean(axis=2).astype(np.float32)
    H, W = g.shape
    k = max(3, int(0.06 * W)) | 1
    gs = cv2.GaussianBlur(cv2.resize(g, (256, int(256 * H / W))), (0, 0), k / 6.0)
    h2, w2 = gs.shape
    yy, xx = np.mgrid[0:h2, 0:w2]
    fx_ = (xx / (w2 - 1)) * 2 - 1
    fy_ = 1 - (yy / (h2 - 1)) * 2
    mask = (np.abs(fx_) <= max_frac) & (np.abs(fy_) <= max_frac)
    gs = np.where(mask, gs, -1)
    i = int(np.argmax(gs))
    return float(fx_.flat[i]), float(fy_.flat[i])

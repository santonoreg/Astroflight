"""Εντοπισμός αστεριών και φωτομετρία (σε γραμμική εικόνα)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass
class Stars:
    x: np.ndarray          # στήλη (0 = κέντρο 1ου pixel)
    y: np.ndarray          # γραμμή
    flux_rgb: np.ndarray   # (N,3) άθροισμα aperture ανά κανάλι (αφαιρεμένο sky)
    flux: np.ndarray       # μέσος όρος των καναλιών
    fwhm: np.ndarray
    mag: np.ndarray        # -2.5 log10(flux)

    def __len__(self):
        return len(self.x)

    def subset(self, idx):
        return Stars(self.x[idx], self.y[idx], self.flux_rgb[idx], self.flux[idx], self.fwhm[idx], self.mag[idx])


_RING_CACHE = {}


def _grids(h):
    g = _RING_CACHE.get(h)
    if g is None:
        yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float32)
        r2 = xx * xx + yy * yy
        g = (xx, yy, r2, (r2 >= (0.72 * h) ** 2) & (r2 <= h * h))
        _RING_CACHE[h] = g
    return g


def detect_stars(lin: np.ndarray, sigma: float = 6.0, max_stars: int = 6000, progress=None) -> Stars:
    H, W, _ = lin.shape
    lum = lin.mean(axis=2).astype(np.float32)
    dog = cv2.GaussianBlur(lum, (0, 0), 1.4) - cv2.GaussianBlur(lum, (0, 0), 5.0)
    sub = dog[::4, ::4]
    med = float(np.median(sub))
    mad = max(float(np.median(np.abs(sub - med)) * 1.4826), 1e-5)
    thr = med + sigma * mad
    mx = ndimage.maximum_filter(dog, size=5)
    ys, xs = np.nonzero((dog >= mx) & (dog > thr))
    if len(xs) == 0:
        z = np.zeros(0, np.float32)
        return Stars(z, z, np.zeros((0, 3), np.float32), z, z, z)
    order = np.argsort(-dog[ys, xs])[: int(max_stars * 2.2)]
    xs, ys = xs[order], ys[order]

    P = 44
    pad = cv2.copyMakeBorder(lin, P, P, P, P, cv2.BORDER_REFLECT_101)
    N = len(xs)
    fwhm = np.full(N, 3.0, np.float32)
    cxs = xs.astype(np.float32).copy()
    cys = ys.astype(np.float32).copy()
    flux_rgb = np.zeros((N, 3), np.float32)
    ok = np.ones(N, bool)

    def measure(i, h, photometry):
        xx, yy, r2, ring = _grids(h)
        x0, y0 = int(xs[i]), int(ys[i])
        st = pad[y0 + P - h: y0 + P + h + 1, x0 + P - h: x0 + P + h + 1]
        bg = np.median(st[ring], axis=0)
        s3 = np.clip(st - bg, 0, None)
        s = s3.mean(axis=2)
        pk = s[h - 1:h + 2, h - 1:h + 2].max()
        if pk <= 0:
            return False
        n = np.count_nonzero((s > 0.5 * pk) & (r2 <= (0.8 * h) ** 2))
        fwhm[i] = np.clip(2.0 * np.sqrt(n / np.pi), 1.5, 2 * 0.8 * h)
        if photometry:
            m5 = r2 <= max(1.25 * fwhm[i], 2.0) ** 2
            w = s * m5
            tw = w.sum()
            if tw <= 0:
                return False
            cxs[i] = x0 + float((w * xx).sum() / tw)
            cys[i] = y0 + float((w * yy).sum() / tw)
            ra = np.clip(2.0 * fwhm[i], 3.0, 0.72 * h)
            flux_rgb[i] = s3[r2 <= ra * ra].sum(axis=0)
        return True

    # πέρασμα 1: μικρό παράθυρο (καλό για αμυδρά αστέρια πάνω σε νεφέλωμα)
    for i in range(N):
        if not measure(i, 6, False):
            ok[i] = False
        if progress and i % 800 == 0:
            progress(0.25 * i / N)
    # πέρασμα 2: παράθυρο ανάλογο του FWHM (σωστό μέγεθος για μεγάλα/κορεσμένα)
    for i in range(N):
        if ok[i]:
            h = int(np.clip(np.ceil(3.0 * fwhm[i]), 6, 40))
            if not measure(i, h, False):
                ok[i] = False
        if progress and i % 800 == 0:
            progress(0.25 + 0.25 * i / N)
    # πέρασμα 3: τελική φωτομετρία
    for i in range(N):
        if ok[i]:
            h = int(np.clip(np.ceil(3.0 * fwhm[i]), 6, 40))
            if not measure(i, h, True):
                ok[i] = False
        if progress and i % 800 == 0:
            progress(0.5 + 0.5 * i / N)

    # απόρριψη "αστεριών" που είναι πολύ πιο διάχυτα από τα πραγματικά (κόμβοι νεφελώματος)
    good = np.nonzero(ok)[0]
    if len(good) > 20:
        top = good[good < max(40, int(0.12 * N))]          # οι ισχυρότερες υποψήφιες (ταξινομημένες κατά DoG)
        ref = float(np.percentile(fwhm[top], 25)) if len(top) else float(np.percentile(fwhm[good], 10))
        ok &= fwhm <= max(10.0, 6.0 * ref)

    flux = flux_rgb.mean(axis=1)
    ok &= flux > 0
    ok &= (cxs > -0.5) & (cxs < W - 0.5) & (cys > -0.5) & (cys < H - 0.5)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        z = np.zeros(0, np.float32)
        return Stars(z, z, np.zeros((0, 3), np.float32), z, z, z)
    idx = idx[np.argsort(-flux[idx])]

    # αφαίρεση διπλοεντοπισμών (μεγάλα κορεσμένα αστέρια)
    pts = np.c_[cxs[idx], cys[idx]]
    tree = cKDTree(pts)
    removed = np.zeros(len(idx), bool)
    for j in range(len(idx)):
        if removed[j]:
            continue
        r = max(3.0, 0.8 * float(fwhm[idx[j]]))
        for n in tree.query_ball_point(pts[j], r):
            if n > j:                              # ταξινομημένα κατά flux, άρα n>j = πιο αμυδρό
                removed[n] = True
    keep = idx[~removed][:max_stars]
    st = Stars(cxs[keep], cys[keep], flux_rgb[keep], flux[keep], fwhm[keep],
               (-2.5 * np.log10(flux[keep] + 1e-12)).astype(np.float32))
    if progress:
        progress(1.0)
    return st

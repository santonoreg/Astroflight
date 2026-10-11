"""Διαχωρισμός αστεριών / νεφελώματος (starless)."""
from __future__ import annotations
import numpy as np
import cv2
from .stars import Stars


def make_starless(lin: np.ndarray, stars: Stars, grow: float = 1.0) -> np.ndarray:
    """Αφαίρεση αστεριών με κανονικοποιημένη συνέλιξη (γρήγορο, ομαλό).
    Για καλύτερο αποτέλεσμα χρησιμοποίησε starless TIFF από StarXTerminator/StarNet."""
    H, W, _ = lin.shape
    mask = np.zeros((H, W), np.uint8)
    if len(stars):
        fmed = float(np.median(stars.flux))
        for x, y, f, fw in zip(stars.x, stars.y, stars.flux, stars.fwhm):
            boost = 1.8 + 0.5 * max(0.0, np.log10(max(f, 1e-9) / max(fmed, 1e-9)))
            r = int(np.clip(fw * boost * grow, 3, 90))
            cv2.circle(mask, (int(round(x)), int(round(y))), r, 255, -1)
    m = mask > 0
    if not m.any():
        return lin.copy()
    w = (~m).astype(np.float32)
    filled = np.zeros_like(lin)
    have = np.zeros((H, W), bool)
    for sigma in (3.0, 8.0, 20.0, 50.0):
        if (m & ~have).sum() == 0:
            break
        num = cv2.GaussianBlur(lin * w[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(w, (0, 0), sigma)
        est = num / np.maximum(den, 1e-6)[..., None]
        good = (den > 0.35) & ~have
        filled[good] = est[good]
        have |= good
    rest = m & ~have
    if rest.any():
        filled[rest] = np.median(lin.reshape(-1, 3), axis=0)
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.6)[..., None]
    out = lin * (1 - soft) + filled * soft
    return np.minimum(out, lin).astype(np.float32)

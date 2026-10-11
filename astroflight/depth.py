"""Χάρτης βάθους, πάχος και σκόνη για το νεφέλωμα (σε low-res εικόνα)."""
from __future__ import annotations
import numpy as np
import cv2
from .config import Params


def norm01(a: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    l, h = np.percentile(a, [lo, hi])
    return np.clip((a - l) / max(h - l, 1e-9), 0, 1).astype(np.float32)


def _struct(a: np.ndarray) -> np.ndarray:
    k = 6.0 / (np.percentile(a, 99.5) + 1e-6)
    return norm01(np.arcsinh(a * k))


def compute_depth(low: np.ndarray, p: Params, imported: np.ndarray | None = None,
                  ai: np.ndarray | None = None):
    """low: (h,w,3) γραμμικό νεφέλωμα. Επιστρέφει depth (0=κοντά .. 1=μακριά), lum_n."""
    h, w, _ = low.shape
    lum = low.mean(axis=2)
    lum_n = _struct(cv2.GaussianBlur(lum, (0, 0), 1.2))
    comps = []

    if p.w_lum > 0:
        d = lum_n if p.lum_invert else 1.0 - lum_n
        comps.append((p.w_lum, d))
    if p.w_color > 0:
        chans = [_struct(cv2.GaussianBlur(low[..., c], (0, 0), 1.2)) for c in range(3)]
        wsum = abs(p.color_wr) + abs(p.color_wg) + abs(p.color_wb) + 1e-6
        c = (p.color_wr * chans[0] + p.color_wg * chans[1] + p.color_wb * chans[2]) / wsum
        comps.append((p.w_color, norm01(c)))
    if p.w_dome > 0:
        n = float(min(h, w))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - p.dome_cx * w) / (p.dome_radius * n)) ** 2 + ((yy - p.dome_cy * h) / (p.dome_radius * n)) ** 2)
        d = 1.0 - np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))
        comps.append((p.w_dome, (1.0 - d) if p.dome_invert else d))
    if p.w_import > 0 and imported is not None:
        g = cv2.resize(imported.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        comps.append((p.w_import, 1.0 - norm01(g, 0.5, 99.5)))
    if p.w_ai > 0 and ai is not None:
        g = cv2.resize(ai.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        comps.append((p.w_ai, 1.0 - norm01(g, 0.5, 99.5)))

    if comps:
        tw = sum(c[0] for c in comps)
        depth = sum(c[0] * c[1] for c in comps) / tw
        depth = norm01(depth, 0.5, 99.5) if np.ptp(depth) > 1e-6 else np.full((h, w), 0.5, np.float32)
    else:
        depth = np.full((h, w), 0.5, np.float32)
    if p.depth_smooth > 0:
        depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), p.depth_smooth)
    depth = np.clip((depth - 0.5) * p.depth_contrast + 0.5, 0, 1).astype(np.float32)
    return depth, lum_n


def compute_thickness(lum_n: np.ndarray, p: Params) -> np.ndarray:
    s = p.thickness * np.clip(1.0 + p.thick_lum_gain * (lum_n - 0.5) * 2.0, 0.2, 4.0)
    return np.clip(s, 0.015, 0.3).astype(np.float32)


def compute_dust_tau(low: np.ndarray, p: Params) -> np.ndarray:
    """Οπτική πυκνότητα σκόνης από στενά σκοτεινά χαρακτηριστικά μέσα σε φωτεινό αέριο."""
    h, w, _ = low.shape
    if p.dust_strength <= 0:
        return np.zeros((h, w), np.float32)
    lum = cv2.GaussianBlur(low.mean(axis=2), (0, 0), 1.5)
    n = min(h, w)
    k = max(5, int(0.06 * n) | 1)
    base = cv2.dilate(lum, np.ones((k, k), np.uint8))
    base = cv2.GaussianBlur(base, (0, 0), 0.03 * n)
    sky = float(np.percentile(lum, 5))
    span = max(float(np.percentile(base, 95)) - sky, 1e-6)
    sig = np.clip((base - sky) / span * 3.0, 0, 1)
    ratio = np.clip((lum - sky) / np.maximum(base - sky, 1e-6), 1e-3, 1.0)
    tau = np.clip(-np.log(ratio), 0, p.dust_tau_max) * sig * p.dust_strength
    return cv2.GaussianBlur(tau.astype(np.float32), (0, 0), 1.0)

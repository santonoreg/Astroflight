"""Φόρτωση TIFF/PNG/JPG, γραμμικοποίηση και stretch."""
from __future__ import annotations
import os
import numpy as np
import cv2


def _to_hwc(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim == 3 and a.shape[0] in (1, 2, 3, 4) and a.shape[2] > 4:
        a = np.transpose(a, (1, 2, 0))          # channel-first
    c = a.shape[2]
    if c == 2:
        a = a[..., :1]
    elif c >= 4:
        a = a[..., :3]
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return a


def _normalize(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind == "u":
        out = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    elif a.dtype.kind == "i":
        out = (a.astype(np.float32) - np.iinfo(a.dtype).min) / float(np.iinfo(a.dtype).max - np.iinfo(a.dtype).min)
    else:
        out = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mx = float(out.max()) if out.size else 1.0
        if mx > 1.0 + 1e-3:
            out = out / mx
        out = np.clip(out, 0.0, None)
    return np.ascontiguousarray(out)


def load_image(path: str) -> np.ndarray:
    """Επιστρέφει float32 HxWx3 RGB στο [0,1]."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(path)
    else:
        buf = np.fromfile(path, dtype=np.uint8)          # δουλεύει με ελληνικά paths στα Windows
        a = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError(f"Δεν μπόρεσα να διαβάσω την εικόνα: {path}")
        if a.ndim == 3 and a.shape[2] >= 3:
            a = a[..., [2, 1, 0]]
    return _normalize(_to_hwc(a))


def save_png(path: str, rgb01: np.ndarray):
    a = (np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".png", a[..., ::-1])
    buf.tofile(path)


def resize_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def to_linear(img: np.ndarray, stretched: bool, gamma: float) -> np.ndarray:
    return np.power(np.clip(img, 0, 1), gamma, dtype=np.float32) if stretched else img.astype(np.float32)


def mtf(m: float, x):
    x = np.asarray(x, dtype=np.float32)
    return ((m - 1.0) * x) / (((2.0 * m - 1.0) * x) - m + 1e-12)


def tone_params(lin: np.ndarray, stretched: bool, gamma: float) -> dict:
    """Παράμετροι tone mapping που χρησιμοποιεί ο shader (mode 0 = gamma, 1 = MTF auto-stretch)."""
    if stretched:
        return dict(mode=0, gamma=float(gamma), black=0.0, white=1.0, m=0.5)
    lum = lin.mean(axis=2)[::3, ::3]
    med = float(np.median(lum))
    mad = float(np.median(np.abs(lum - med)) * 1.4826)
    black = max(0.0, med - 2.8 * mad)
    white = float(max(np.percentile(lum, 99.9), black + 1e-4))
    xm = (med - black) / (white - black)
    xm = float(np.clip(xm, 1e-5, 0.999))
    m = float(np.clip(mtf(0.25, xm), 1e-4, 0.9999))
    return dict(mode=1, gamma=2.2, black=black, white=white, m=m)


def apply_tone(lin: np.ndarray, tp: dict) -> np.ndarray:
    """CPU εκδοχή του tone mapping (για έλεγχο/preview)."""
    if tp["mode"] == 0:
        return np.power(np.clip(lin, 0, 1), 1.0 / tp["gamma"])
    x = np.clip((lin - tp["black"]) / (tp["white"] - tp["black"]), 0, 1)
    return np.clip(mtf(tp["m"], x), 0, 1)

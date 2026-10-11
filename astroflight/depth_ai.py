"""Προαιρετικός AI χάρτης βάθους (Depth Anything V2). Χρειάζεται torch + transformers + pillow."""
from __future__ import annotations
import numpy as np


def available() -> bool:
    try:
        import torch, transformers  # noqa
        return True
    except Exception:
        return False


def estimate_depth(rgb_display01: np.ndarray, model: str = "depth-anything/Depth-Anything-V2-Small-hf", log=None) -> np.ndarray:
    """Επιστρέφει σχετικό βάθος (μεγαλύτερο = πιο κοντά), ίδιο μέγεθος με την είσοδο.
    ΠΡΟΣΟΧΗ: εκπαιδεύτηκε σε φωτογραφίες καθημερινών σκηνών. Για νεφελώματα είναι μόνο αφετηρία."""
    import torch
    from transformers import pipeline
    from PIL import Image
    dev = 0 if torch.cuda.is_available() else -1
    if log:
        log(f"Depth Anything: φόρτωση μοντέλου ({'GPU' if dev == 0 else 'CPU'})...")
    pipe = pipeline("depth-estimation", model=model, device=dev)
    img = Image.fromarray((np.clip(rgb_display01, 0, 1) * 255).astype(np.uint8))
    out = pipe(img)
    d = np.asarray(out["depth"], dtype=np.float32)
    return d

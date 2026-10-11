"""Μικρός κατάλογος γνωστών νεφελωμάτων/σμηνών: αναγνώριση από τις συντεταγμένες και προσεγγιστική απόσταση.

Οι αποστάσεις είναι στρογγυλεμένες τιμές από δημοσιευμένες μετρήσεις (Gaia κ.ά.)· χρησιμοποιούνται μόνο όταν δεν
υπάρχει καλύτερη εκτίμηση από τα δεδομένα Gaia της ίδιας της φωτογραφίας."""
from __future__ import annotations
import math

# (όνομα, RA°, Dec°, ακτίνα°, απόσταση pc)
CATALOG = [
    ("Orion Nebula (M42/M43)", 83.82, -5.39, 0.9, 412),
    ("Flame Nebula (NGC 2024)", 85.43, -1.84, 0.5, 420),
    ("Horsehead Nebula (B33/IC 434)", 85.25, -2.46, 0.5, 420),
    ("M78", 86.68, 0.05, 0.3, 390),
    ("Rosette Nebula (NGC 2237/2244)", 97.98, 4.93, 1.2, 1500),
    ("Cone Nebula / Christmas Tree (NGC 2264)", 100.24, 9.89, 0.8, 720),
    ("Pleiades (M45)", 56.75, 24.12, 1.5, 136),
    ("California Nebula (NGC 1499)", 60.5, 36.6, 1.6, 470),
    ("Flaming Star Nebula (IC 405)", 79.0, 34.3, 0.5, 450),
    ("Lagoon Nebula (M8)", 270.92, -24.38, 0.9, 1250),
    ("Trifid Nebula (M20)", 270.67, -23.03, 0.4, 1300),
    ("Eagle Nebula (M16)", 274.70, -13.81, 0.5, 1740),
    ("Omega Nebula (M17)", 275.20, -16.18, 0.5, 1900),
    ("North America Nebula (NGC 7000)", 314.7, 44.33, 1.5, 800),
    ("Pelican Nebula (IC 5070)", 312.75, 44.35, 0.8, 800),
    ("Veil Nebula (NGC 6960/6992)", 312.75, 30.7, 2.0, 700),
    ("Dumbbell Nebula (M27)", 299.9, 22.72, 0.2, 400),
    ("Ring Nebula (M57)", 283.4, 33.03, 0.1, 790),
    ("Helix Nebula (NGC 7293)", 337.41, -20.84, 0.3, 200),
    ("Carina Nebula (NGC 3372)", 161.27, -59.87, 1.5, 2350),
    ("Heart Nebula (IC 1805)", 38.2, 61.45, 1.5, 2000),
    ("Soul Nebula (IC 1848)", 42.8, 60.4, 1.2, 2000),
    ("Pacman Nebula (NGC 281)", 13.2, 56.6, 0.5, 2800),
    ("Iris Nebula (NGC 7023)", 315.4, 68.2, 0.3, 430),
]


def _sep_deg(ra1, dec1, ra2, dec2) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def identify(ra_deg: float, dec_deg: float, fov_deg: float = 0.0):
    """Επιστρέφει (όνομα, απόσταση_pc) του πλησιέστερου γνωστού αντικειμένου αν το κέντρο της εικόνας πέφτει μέσα στο
    αντικείμενο (ή το αντικείμενο μέσα στο πεδίο), αλλιώς None."""
    best = None
    for name, ra, dec, rad, d in CATALOG:
        sep = _sep_deg(ra_deg, dec_deg, ra, dec)
        if sep <= max(rad, 0.5 * fov_deg * 0.6):
            if best is None or sep < best[0]:
                best = (sep, name, d)
    return (best[1], float(best[2])) if best else None


def lookup_name(text: str):
    """Αναζήτηση ονόματος: πρώτα στον τοπικό κατάλογο, μετά (αν υπάρχει internet) στο SIMBAD/Sesame.
    Επιστρέφει (ra_deg, dec_deg, όνομα) ή None."""
    t = text.strip().lower()
    if not t:
        return None
    for name, ra, dec, rad, d in CATALOG:
        if t in name.lower():
            return ra, dec, name
    try:
        from astropy.coordinates import SkyCoord
        c = SkyCoord.from_name(text.strip())
        return float(c.ra.deg), float(c.dec.deg), text.strip()
    except Exception:
        return None

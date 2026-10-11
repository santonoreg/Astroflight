"""Κλίμακα εικόνας από τα οπτικά (ονομαστική) και βοηθητικά για υποψήφιες κλίμακες."""
from __future__ import annotations
import math

# Μέγεθος pixel (µm) δημοφιλών καμερών – έλεγξε τις προδιαγραφές της δικής σου κάμερας αν δεν υπάρχει στη λίστα.
CAMERAS: dict[str, float] = {
    "Atik 383L+": 5.4,
    "Atik 460EX": 4.54,
    "ZWO ASI1600MM / MC": 3.8,
    "ZWO ASI294MC / MM": 4.63,
    "ZWO ASI533MC / MM": 3.76,
    "ZWO ASI2600MC / MM": 3.76,
    "ZWO ASI183MC / MM": 2.4,
    "ZWO ASI071MC": 4.78,
    "QHY268M / C": 3.76,
}


def nominal_scale(pixel_um: float, focal_mm: float, binning: float = 1.0, resample: float = 1.0) -> float:
    """Κλίμακα σε arcsec/pixel = 206.265 × pixel(µm) × binning ÷ εστιακή(mm). resample > 1 σημαίνει ότι η εικόνα μίκρυνε
    (λιγότερα pixel) κατά αυτόν τον παράγοντα."""
    return 206.265 * pixel_um * binning * resample / focal_mm


def scale_ladder(lo: float = 0.7, hi: float = 80.0, ratio: float = 1.25) -> list[float]:
    """Γεωμετρική σειρά κλιμάκων (arcsec/pixel) που καλύπτει από πολύ μεγάλα εστιακά μέχρι ευρυγώνια φακό."""
    out, s = [], lo
    while s <= hi:
        out.append(round(s, 3))
        s *= ratio
    return out


def order_candidates(first: list[float], ladder: list[float], tol: float = 0.12) -> list[float]:
    """Πρώτα οι «έξυπνες» υποθέσεις (π.χ. ονομαστική κλίμακα), μετά η σκάλα ταξινομημένη κατά απόσταση από αυτές. Αφαιρεί
    υποψήφιες που απέχουν λιγότερο από tol (σχετικά) από κάποια που ήδη δοκιμάστηκε."""
    chosen: list[float] = []

    def ok(v):
        return all(abs(math.log(v / c)) > tol for c in chosen)
    for v in first:
        if v and v > 0 and ok(v):
            chosen.append(float(v))
    anchor = first[0] if first and first[0] else 3.0
    for v in sorted(ladder, key=lambda x: abs(math.log(x / anchor))):
        if ok(v):
            chosen.append(float(v))
    return chosen

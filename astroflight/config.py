"""Παράμετροι της εφαρμογής + περιγραφή των controls του GUI."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields


@dataclass
class Params:
    # --- εικόνα ---
    stretched_input: bool = True      # το TIFF είναι ήδη "τεντωμένο" (τελική επεξεργασμένη εικόνα)
    gamma: float = 2.2
    # --- αντικείμενο ---
    object_distance_pc: float = 412.0
    # --- αστέρια ---
    star_sigma: float = 6.0
    max_stars: int = 6000
    gaia_glim: float = 15.5
    match_tol_px: float = 3.0
    extra_enable: bool = True
    extra_radius_deg: float = 0.0
    extra_glim: float = 13.5
    # --- βάθος νεφελώματος ---
    w_lum: float = 1.0
    w_color: float = 0.0
    w_dome: float = 0.3
    w_import: float = 0.0
    w_ai: float = 0.0
    color_wr: float = 0.5
    color_wg: float = 0.0
    color_wb: float = -0.5
    lum_invert: bool = False
    dome_cx: float = 0.5
    dome_cy: float = 0.5
    dome_radius: float = 0.6
    dome_invert: bool = False
    depth_smooth: float = 5.0
    depth_contrast: float = 1.0
    depth_extent: float = 0.8
    thickness: float = 0.07
    thick_lum_gain: float = 0.5
    dust_strength: float = 0.7
    dust_offset: float = 0.15
    dust_thick: float = 0.04
    dust_tau_max: float = 2.5
    # --- όγκος ---
    vol_size: int = 512
    vol_layers: int = 64
    pad_frac: float = 0.06
    # --- απόδοση ---
    exposure: float = 1.0
    neb_gain: float = 1.0
    star_gain: float = 1.0
    star_size_exp: float = 0.2
    bloom: float = 0.15
    bloom_thresh: float = 0.7
    bloom_radius: float = 1.0
    spikes: bool = False
    spike_amount: float = 0.04
    steps: int = 128
    limb: float = 0.5
    proc_amt: float = 0.8
    detail_angle: float = 15.0
    preview_scale: float = 0.6
    fov_scale: float = 1.0
    # --- βίντεο ---
    duration: float = 12.0
    fps: int = 30
    out_width: int = 1920
    out_height: int = 1080
    ssaa: int = 1
    ease: bool = True
    # --- πτήση μέσα στο νεφέλωμα (dive) ---
    dive_auto_poi: bool = True        # αυτόματη επιλογή στόχου (φωτεινότερη περιοχή)
    dive_fx: float = 0.0              # στόχος: −1…1 (μισό πλάτος), + = δεξιά
    dive_fy: float = 0.0              # στόχος: −1…1 (μισό ύψος), + = πάνω
    dive_depth: float = 0.35          # σε ποιο ποσοστό της εκπομπής (κατά βάθος) τελειώνει η πτήση: 0 = μπροστά, 1 = πίσω
    dive_drift: float = 0.0           # ήπια πλευρική καμπύλη (× πλάτος νεφελώματος)

    def to_dict(self):
        return asdict(self)

    def update_from(self, d: dict):
        names = {f.name for f in fields(self)}
        for k, v in d.items():
            if k in names:
                setattr(self, k, type(getattr(self, k))(v))


# (key, label, kind, lo, hi, step, tooltip)
UI_GROUPS = {
    "Βάθος νεφελώματος": [
        ("w_lum", "Βάρος: φωτεινότητα", "f", 0, 1, 0.05, "Πιο φωτεινό = πιο κοντά (ή ανάποδα με το 'αντιστροφή')"),
        ("w_color", "Βάρος: χρώμα", "f", 0, 1, 0.05, "Βάθος από τα κανάλια R/G/B (π.χ. στρωματοποίηση ιονισμού SII/Ha/OIII)"),
        ("w_dome", "Βάρος: θόλος/κέλυφος", "f", 0, 1, 0.05, "Ακτινικό σχήμα (πλανητικά, σμήνη, γαλαξίες)"),
        ("w_import", "Βάρος: δικός μου χάρτης", "f", 0, 1, 0.05, "Χάρτης βάθους που φόρτωσες (ασπρόμαυρο, λευκό = κοντά)"),
        ("w_ai", "Βάρος: AI χάρτης", "f", 0, 1, 0.05, "Depth Anything (προαιρετικό, χρειάζεται torch+transformers)"),
        ("color_wr", "Χρώμα: βάρος R", "f", -1, 1, 0.05, "Θετικό = πιο μακριά"),
        ("color_wg", "Χρώμα: βάρος G", "f", -1, 1, 0.05, ""),
        ("color_wb", "Χρώμα: βάρος B", "f", -1, 1, 0.05, ""),
        ("lum_invert", "Αντιστροφή φωτεινότητας", "b", 0, 1, 0, "Φωτεινό = πιο μακριά"),
        ("dome_cx", "Θόλος: κέντρο X", "f", 0, 1, 0.01, ""),
        ("dome_cy", "Θόλος: κέντρο Y", "f", 0, 1, 0.01, ""),
        ("dome_radius", "Θόλος: ακτίνα", "f", 0.1, 1.5, 0.02, ""),
        ("dome_invert", "Θόλος: κοίλος", "b", 0, 1, 0, ""),
        ("depth_smooth", "Εξομάλυνση βάθους (px)", "f", 0, 40, 0.5, ""),
        ("depth_contrast", "Αντίθεση βάθους", "f", 0.2, 3, 0.05, ""),
        ("depth_extent", "Συνολικό βάθος (× πλάτος)", "f", 0.05, 3, 0.05, "Πόσο 'βαθύ' είναι το νεφέλωμα σε σχέση με το πλάτος του"),
        ("thickness", "Πάχος στρώματος", "f", 0.02, 0.4, 0.01, ""),
        ("thick_lum_gain", "Πάχος ∝ φωτεινότητα", "f", -1, 2, 0.05, ""),
        ("dust_strength", "Σκόνη: ένταση", "f", 0, 1.5, 0.05, "Σκοτεινές λωρίδες = σκόνη μπροστά από το αέριο"),
        ("dust_offset", "Σκόνη: απόσταση μπροστά", "f", 0, 0.5, 0.01, ""),
        ("dust_thick", "Σκόνη: πάχος", "f", 0.01, 0.2, 0.01, ""),
        ("dust_tau_max", "Σκόνη: μέγιστη οπτική πυκνότητα", "f", 0.5, 4, 0.1, ""),
    ],
    "Όγκος (ποιότητα)": [
        ("vol_size", "Ανάλυση όγκου (μακριά πλευρά)", "i", 128, 1024, 64, "Μεγαλύτερο = περισσότερη μνήμη GPU"),
        ("vol_layers", "Στρώματα βάθους", "i", 16, 128, 8, ""),
        ("pad_frac", "Περιθώριο γύρω από την εικόνα", "f", 0.0, 0.3, 0.01, "Για να μη φαίνεται κόψιμο όταν η κάμερα κινείται"),
    ],
    "Απόδοση": [
        ("exposure", "Έκθεση", "f", 0.1, 8, 0.05, ""),
        ("neb_gain", "Φωτεινότητα νεφελώματος", "f", 0.1, 4, 0.05, ""),
        ("star_gain", "Φωτεινότητα αστεριών", "f", 0.1, 4, 0.05, ""),
        ("star_size_exp", "Μεγένθυνση αστεριών κοντά", "f", 0, 0.5, 0.01, ""),
        ("bloom", "Bloom (λάμψη)", "f", 0, 1, 0.01, ""),
        ("bloom_thresh", "Bloom: κατώφλι", "f", 0, 4, 0.05, ""),
        ("bloom_radius", "Bloom: ακτίνα", "f", 0.3, 3, 0.05, ""),
        ("spikes", "Ακτίνες περίθλασης", "b", 0, 1, 0, ""),
        ("spike_amount", "Ένταση ακτίνων", "f", 0, 0.3, 0.005, ""),
        ("steps", "Βήματα raymarch", "i", 32, 384, 16, ""),
        ("limb", "Φωτεινότερο σε πλάγια θέαση", "f", 0, 1, 0.05, "0 = όχι, 1 = φυσικό"),
        ("proc_amt", "Τεχνητή λεπτομέρεια (κοντά)", "f", 0, 2, 0.05, "Διαδικαστικός 3D θόρυβος που αντικαθιστά τη λεπτομέρεια της φωτογραφίας όταν η κάμερα απομακρύνεται από το σημείο λήψης ή πλησιάζει πολύ"),
        ("detail_angle", "Γωνία διατήρησης λεπτομέρειας (°)", "f", 2, 45, 1, "Μέχρι ποια γωνία παράλλαξης κρατιέται η λεπτομέρεια της φωτογραφίας"),
        ("preview_scale", "Ανάλυση preview", "f", 0.25, 1, 0.05, ""),
        ("fov_scale", "Zoom (FOV)", "f", 0.3, 2.5, 0.05, ""),
    ],
}

"""Κάμερα σε τροχιά γύρω από το νεφέλωμα + keyframes με Catmull-Rom."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np

FIELDS = ["az", "el", "roll", "logr", "tx", "ty", "tz", "yaw", "pitch"]


@dataclass
class CamState:
    az: float = 0.0      # μοίρες
    el: float = 0.0
    roll: float = 0.0
    logr: float = 0.0    # ln(R / D0): 0 = στο σημείο λήψης
    tx: float = 0.0      # μετατόπιση στόχου, σε μονάδες πλάτους νεφελώματος
    ty: float = 0.0
    tz: float = 0.0
    yaw: float = 0.0     # ελεύθερη στροφή βλέμματος (μοίρες, + = δεξιά)
    pitch: float = 0.0   # (μοίρες, + = πάνω)

    def vec(self):
        return np.array([getattr(self, f) for f in FIELDS], dtype=np.float64)

    @staticmethod
    def from_vec(v):
        return CamState(**{f: float(x) for f, x in zip(FIELDS, v)})

    def to_dict(self):
        return asdict(self)


def pose_from_state(s: CamState, D0: float, W0: float):
    az, el, roll = math.radians(s.az), math.radians(max(-85.0, min(85.0, s.el))), math.radians(s.roll)
    R = D0 * math.exp(s.logr)
    target = np.array([s.tx * W0, s.ty * W0, D0 + s.tz * W0])
    off = R * np.array([math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el)])
    pos = target + off
    fwd = -off / max(np.linalg.norm(off), 1e-12)
    right = np.cross([0.0, 1.0, 0.0], fwd)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(fwd, right)
    yw, pt = math.radians(s.yaw), math.radians(s.pitch)
    if yw != 0.0:
        c, sn = math.cos(yw), math.sin(yw)
        fwd, right = fwd * c + right * sn, right * c - fwd * sn
    if pt != 0.0:
        c, sn = math.cos(pt), math.sin(pt)
        fwd, up = fwd * c + up * sn, up * c - fwd * sn
    if roll != 0.0:
        c, sn = math.cos(roll), math.sin(roll)
        right, up = right * c + up * sn, -right * sn + up * c
    return pos.astype(np.float32), right.astype(np.float32), up.astype(np.float32), fwd.astype(np.float32)


def _catmull(p0, p1, p2, p3, u):
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


@dataclass
class Keyframe:
    t: float
    state: CamState


class CameraPath:
    def __init__(self, keys: list[Keyframe], duration: float, ease: bool = True):
        self.keys = sorted(keys, key=lambda k: k.t)
        self.duration = max(duration, 1e-3)
        self.ease = ease

    def eval(self, time_s: float) -> CamState:
        ks = self.keys
        if not ks:
            return CamState()
        if len(ks) == 1:
            return ks[0].state
        x = min(max(time_s / self.duration, 0.0), 1.0)
        if self.ease:
            x = x * x * (3 - 2 * x)
        t = x * self.duration
        t = min(max(t, ks[0].t), ks[-1].t)
        i = 0
        while i < len(ks) - 2 and t > ks[i + 1].t:
            i += 1
        k1, k2 = ks[i], ks[i + 1]
        u = 0.0 if k2.t == k1.t else (t - k1.t) / (k2.t - k1.t)
        v1, v2 = k1.state.vec(), k2.state.vec()
        v0 = ks[i - 1].state.vec() if i > 0 else 2 * v1 - v2
        v3 = ks[i + 2].state.vec() if i + 2 < len(ks) else 2 * v2 - v1
        return CamState.from_vec(_catmull(v0, v1, v2, v3, u))


def make_flythrough(D0: float, W0: float, end_dist_w: float = 0.6, az_end: float = 20.0,
                    el_end: float = 8.0, duration: float = 12.0) -> list[Keyframe]:
    """Προεπιλεγμένη πτήση: από το σημείο λήψης προς το νεφέλωμα με ήπια περιστροφή."""
    r_end = max(end_dist_w * W0, 1e-6)
    lr_end = math.log(r_end / D0)
    return [
        Keyframe(0.0, CamState()),
        Keyframe(duration * 0.55, CamState(az=az_end * 0.3, el=el_end * 0.3, logr=lr_end * 0.62)),
        Keyframe(duration, CamState(az=az_end, el=el_end, logr=lr_end)),
    ]


def _smooth(a: float, b: float, x: float) -> float:
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def make_dive(D0: float, W0: float, E: float, tx0: float, ty0: float, fx: float = 0.0, fy: float = 0.0,
              z_end: float | None = None, drift: float = 0.0, duration: float = 12.0, n_keys: int = 25,
              frac_approach: float = 0.42) -> list[Keyframe]:
    """Ευθεία πτήση ΜΕΣΑ στο νεφέλωμα, από το σημείο λήψης προς ένα σημείο του κάδρου.

    fx, fy:  σημείο-στόχος ως ποσοστό του μισού πλάτους/ύψους της εικόνας (−1…1, 0 = κέντρο, +x δεξιά, +y πάνω)
    z_end:   βάθος (pc) όπου τελειώνει η πτήση (None = κέντρο του όγκου)
    drift:   ήπια πλευρική καμπύλη (σε πλάτη νεφελώματος), 0 = καθαρά ευθεία
    tx0, ty0: εφαπτομένη μισού πεδίου της εικόνας
    frac_approach: ποσοστό του χρόνου για την προσέγγιση (εκθετική, μέσα από τα αστέρια) μέχρι την είσοδο στο νεφέλωμα·
                   το υπόλοιπο ο χρόνος περνά ΜΕΣΑ στα νέφη με σχεδόν σταθερή ταχύτητα.

    Η κάμερα ξεκινά ΑΚΡΙΒΩΣ από το σημείο λήψης (κοιτάζει +z) και κινείται σε ευθεία γραμμή προς τον στόχο.
    Παραμετροποίηση: az=el=0, pos = z_cam·(fx·tx0, fy·ty0, 1) = s·P_end, με R = z_far − z_cam (logr = ln(R/D0))."""
    z0 = D0 - E / 2.0
    z_far = D0 + E / 2.0
    z_end = D0 if z_end is None else float(z_end)
    z_end = min(max(z_end, z0 + 0.12 * E), z_far - 0.05 * E)
    P_end = np.array([fx * tx0 * z_end, fy * ty0 * z_end, z_end])
    yaw_T = math.degrees(math.atan(fx * tx0))
    pitch_T = math.degrees(math.atan(fy * ty0))
    tz = (z_far - D0) / W0
    # --- χαρακτηριστικές αποστάσεις
    z_in = z0 - 0.06 * E                 # σημείο «ακριβώς πριν την είσοδο»
    a0 = max(z_in, 1e-3)                 # απόσταση από το σημείο εισόδου στην αρχή
    a1 = 0.06 * E * 0.0 + 0.02 * E       # τελική απόσταση της φάσης προσέγγισης (μικρή)
    d_in = z_end - (z0 - a1)             # απόσταση που διανύεται μέσα στο νεφέλωμα
    f1 = min(max(frac_approach, 0.1), 0.8)
    keys = []
    N = n_keys
    for i in range(N):
        x = i / (N - 1)
        if x <= f1:
            u = x / f1
            a = a0 * (a1 / a0) ** u                       # εκθετική προσέγγιση προς το σημείο εισόδου
            z_cam = z0 - a
        else:
            u = (x - f1) / (1.0 - f1)
            e = 1.0 - (1.0 - u) ** 1.35                   # σχεδόν σταθερή ταχύτητα, ήπια επιβράδυνση στο τέλος
            z_cam = (z0 - a1) + d_in * e
        z_cam = min(max(z_cam, 0.0), z_end)
        s_ = z_cam / z_end
        R = max(z_far - z_cam, 0.02 * W0)
        g = _smooth(0.0, 0.85, s_)
        bell = math.sin(math.pi * x) ** 2
        keys.append(Keyframe(duration * x, CamState(
            az=0.0, el=0.0, roll=0.0, logr=math.log(R / D0),
            tx=s_ * P_end[0] / W0 + drift * bell, ty=s_ * P_end[1] / W0, tz=tz,
            yaw=yaw_T * g, pitch=pitch_T * g)))
    return keys

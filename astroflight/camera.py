"""Κάμερα σε τροχιά γύρω από το νεφέλωμα + keyframes με Catmull-Rom."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np

FIELDS = ["az", "el", "roll", "logr", "tx", "ty", "tz", "yaw", "pitch", "logfov", "fade"]


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
    fade: float = 1.0    # φωτεινότητα καρέ (1 = κανονική, 0 = μαύρο) – για σβήσιμο στο τέλος
    logfov: float = 0.0  # ln(πολλαπλασιαστής οπτικού πεδίου): 0 = όπως στη φωτογραφία, >0 = ευρύτερο

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


def _dive_profile(a0: float, a_c: float, L_in: float, duration: float, n: int, t_start: float = 0.9,
                  t_end: float = 1.6, v_in: float = 0.55, zoom_rate: float = 0.25, hold: float = 0.0) -> np.ndarray:
    """Θέση της κάμερας στη διαδρομή: επιστρέφει την ΥΠΟΛΟΙΠΗ απόσταση a(t) από την είσοδο του νεφελώματος (a>0 = έξω,
    a<0 = μέσα) για n ισαπέχοντα χρόνους στο [0, duration].

    Ταχύτητα  v = min(k·√(a + a_c), zoom_rate·(a + a_c))  έξω: το δεύτερο όριο εγγυάται ότι το νεφέλωμα δεν μεγαλώνει στο
    κάδρο πιο γρήγορα από zoom_rate (σε ln ανά δευτερόλεπτο, π.χ. 0.25 = ×1.28 το δευτερόλεπτο). Αν η διάρκεια είναι πολύ
    μικρή για να τηρηθεί, το όριο αυξάνεται αυτόματα. Το k είναι η «βασική» ταχύτητα. Μέσα στα νέφη σταθερή (v_in × την
    ταχύτητα εισόδου). Ομαλή εκκίνηση (t_start s).

    hold = 0: η κάμερα «προσγειώνεται» ομαλά στο τέλος του μήκους L_in (μέσα στα νέφη).
    hold > 0: το L_in είναι το μήκος που πρέπει να διανυθεί ΜΕΧΡΙ να βγει η κάμερα από το νεφέλωμα· αυτό γίνεται ακριβώς τη
    στιγμή duration − hold, και τα τελευταία hold δευτερόλεπτα η κάμερα συνεχίζει αργά (και επιβραδύνει) σε πεδίο αστεριών."""
    duration = max(float(duration), 1.0)
    hold = min(max(float(hold), 0.0), 0.5 * duration)      # ποτέ περισσότερο από το μισό βίντεο στο πεδίο αστεριών
    dt = duration / 4000.0
    ts = np.arange(4001) * dt
    zr = max(zoom_rate, 1e-3)
    T1 = duration - hold if hold > 0 else duration

    def sim(k):
        a = a0
        out = np.empty_like(ts)
        t_cross = math.inf
        v_c = 0.0
        for i, t in enumerate(ts):
            out[i] = a
            ease = _smooth(0.0, t_start, t)
            if hold > 0 and a <= -L_in:
                if t_cross == math.inf:
                    t_cross = t
                v = v_c * (1.0 - 0.7 * _smooth(0.0, hold, t - t_cross))
            elif a > 0:
                v = min(k * math.sqrt(a + a_c), zr * (a + a_c)) * ease
                v_c = v
            else:
                v = min(k * math.sqrt(a_c), zr * a_c) * v_in * ease
                if hold == 0:
                    v *= 0.3 + 0.7 * _smooth(0.0, 1.0, (a + L_in) / max(0.2 * L_in, 1e-9))
                v_c = v
            a -= v * dt
            if hold == 0 and a < -L_in:
                a = -L_in
        return out, t_cross

    def reached(k):
        out, tc = sim(k)
        return (tc <= T1) if hold > 0 else (out[-1] <= -L_in * 0.995)

    for _ in range(80):                         # αν δεν φτάνει στον χρόνο ακόμη και με πολύ μεγάλο k, χαλάρωσε το όριο ζουμ
        if reached(1e3):
            break
        zr *= 1.25
    lo, hi = 1e-4, 1e3
    for _ in range(60):
        k = 0.5 * (lo + hi)
        if reached(k):
            hi = k
        else:
            lo = k
    traj, _tc = sim(hi)
    return np.interp(np.linspace(0.0, duration, n), ts, traj)


def make_dive(D0: float, W0: float, E: float, tx0: float, ty0: float, fx: float = 0.0, fy: float = 0.0,
              z_end: float | None = None, drift: float = 0.0, duration: float = 12.0, n_keys: int = 0,
              frac_approach: float = 0.42, zoom_end: float = 2.5, fov_max_deg: float = 55.0,
              zoom_rate: float = 0.25, hold: float = 0.0, fade_s: float = 0.0) -> list[Keyframe]:
    """Ευθεία πτήση ΜΕΣΑ στο νεφέλωμα, από το σημείο λήψης προς ένα σημείο του κάδρου.

    fx, fy:  σημείο-στόχος ως ποσοστό του μισού πλάτους/ύψους της εικόνας (−1…1, 0 = κέντρο, +x δεξιά, +y πάνω)
    z_end:   βάθος (pc) όπου τελειώνει η πτήση (None = κέντρο του όγκου)
    drift:   ήπια πλευρική καμπύλη (σε πλάτη νεφελώματος), 0 = καθαρά ευθεία
    tx0, ty0: εφαπτομένη μισού πεδίου της εικόνας
    frac_approach: (δεν χρησιμοποιείται πλέον για το ρυθμό· διατηρείται για συμβατότητα)
    zoom_end:    πόσες φορές «μεγαλύτερο» φαίνεται το νεφέλωμα όταν πλησιάζει την είσοδο σε σχέση με τη φωτογραφία
                 (το οπτικό πεδίο ανοίγει ώστε το νεφέλωμα να μεγαλώνει ομαλά και όχι να βγαίνει απότομα από το κάδρο)
    hold:        αν > 0 (s) η κάμερα περνά ΠΕΡΑ από την πίσω πλευρά του νεφελώματος και μένει τόσα δευτερόλεπτα σε πεδίο μόνο με
                 αστέρια μέχρι το τέλος του βίντεο (το z_end αγνοείται).
    fade_s:      σβήσιμο στο μαύρο τα τελευταία fade_s δευτερόλεπτα.
    fov_max_deg: μέγιστο ΚΑΤΑΚΟΡΥΦΟ οπτικό πεδίο της κάμερας (μοίρες) – όσο μπαίνεις στα νέφη ανοίγει μέχρι εδώ.

    Η κάμερα ξεκινά ΑΚΡΙΒΩΣ από το σημείο λήψης (κοιτάζει +z) και κινείται σε ευθεία γραμμή προς τον στόχο.
    Παραμετροποίηση: az=el=0, pos = z_cam·(fx·tx0, fy·ty0, 1) = s·P_end, με R = z_far − z_cam (logr = ln(R/D0))."""
    z0 = D0 - E / 2.0
    z_far = D0 + E / 2.0
    z_end = D0 if z_end is None else float(z_end)
    hold = min(max(float(hold), 0.0), 0.5 * duration)
    fade_s = min(max(float(fade_s), 0.0), 0.5 * duration)
    if hold > 0:
        z_end = z_far                                   # προσωρινά· υπολογίζεται από το προφίλ παρακάτω
    else:
        z_end = min(max(z_end, z0 + 0.12 * E), z_far - 0.05 * E)
    z_ref = max(z_far, z_end) + 0.05 * E              # σημείο αναφοράς (target) πίσω από την κάμερα ώστε R > 0 πάντα
    P_end = np.array([fx * tx0 * z_end, fy * ty0 * z_end, z_end])
    yaw_T = math.degrees(math.atan(fx * tx0))
    pitch_T = math.degrees(math.atan(fy * ty0))
    tz = (z_ref - D0) / W0
    a_front = 0.02 * E                       # σταματά η «προσέγγιση» λίγο πριν το μέτωπο του όγκου
    z_in = z0 - a_front
    a_c = max(0.6 * E, 1e-3)                 # κλίμακα όπου η ταχύτητα αρχίζει να πέφτει
    N = n_keys if n_keys else max(61, int(duration * 4) + 1)
    if hold > 0:
        L_in = z_far - z_in                  # μήκος μέχρι να βγει από το νεφέλωμα
        a_traj = _dive_profile(z_in, a_c, L_in, duration, N, zoom_rate=zoom_rate, hold=hold)
        z_end = z_in - float(a_traj[-1])
        z_ref = z_end + 0.05 * E
        P_end = np.array([fx * tx0 * z_end, fy * ty0 * z_end, z_end])
    else:
        L_in = z_end - z_in
        a_traj = _dive_profile(z_in, a_c, L_in, duration, N, zoom_rate=zoom_rate)
    tz = (z_ref - D0) / W0
    ln_ref = math.log(max(z_in, 1.0) / max(3.0 * E, 1e-3))
    keys = []
    for i in range(N):
        x = i / (N - 1)
        a = float(a_traj[i])
        z_cam = min(max(z_in - a, 0.0), z_end)
        s_ = z_cam / z_end
        R = max(z_ref - z_cam, 0.02 * W0)
        g = _smooth(0.0, 0.85, s_)
        bell = math.sin(math.pi * x) ** 2
        # οπτικό πεδίο: το νεφέλωμα (κέντρο στο D0) φαίνεται zoom_end φορές μεγαλύτερο από τη φωτογραφία όταν φτάνει στην
        # είσοδο· T = εφαπτομένη μισού κατακόρυφου πεδίου. Όταν μπαίνεις στα νέφη ανοίγει μέχρι το fov_max.
        d_ref = max(D0 - z_cam, 0.15 * E)
        tan_neb = ty0 * D0 / d_ref
        prog = min(max(math.log(max(z_in, 1.0) / max(a + 3.0 * E * 0.0 + 1e-6, 1e-6)) / max(math.log(max(z_in, 1.0) / (3.0 * E)), 1e-6), 0.0), 1.0) if a > 0 else 1.0
        S = math.exp(math.log(max(zoom_end, 1.0)) * _smooth(0.0, 1.0, prog))
        T = min(max(tan_neb / S, ty0), math.tan(math.radians(fov_max_deg) / 2.0))
        keys.append(Keyframe(duration * x, CamState(
            az=0.0, el=0.0, roll=0.0, logr=math.log(R / D0),
            tx=s_ * P_end[0] / W0 + drift * bell, ty=s_ * P_end[1] / W0, tz=tz,
            yaw=yaw_T * g, pitch=pitch_T * g, logfov=math.log(T / ty0),
            fade=1.0 - _smooth(duration - fade_s, duration, duration * x) if fade_s > 0 else 1.0)))
    return keys

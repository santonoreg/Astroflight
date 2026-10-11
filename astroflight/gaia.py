"""Gaia DR3: λήψη καταλόγου, αντιστοίχιση με αστέρια της εικόνας, απόσταση ανά αστέρι."""
from __future__ import annotations
import os, hashlib
import numpy as np
from scipy.spatial import cKDTree
from .stars import Stars
from .wcs_solve import SkyWCS

ADQL = """
SELECT TOP {top} g.source_id, g.ra, g.dec, g.parallax, g.parallax_error, g.phot_g_mean_mag AS g_mag,
       g.bp_rp, d.r_med_geo
FROM gaiadr3.gaia_source AS g
LEFT JOIN external.gaiaedr3_distance AS d ON g.source_id = d.source_id
WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {rad:.5f}))
  AND g.phot_g_mean_mag < {glim:.2f}
"""
# Εναλλακτικό (χωρίς join με τον πίνακα αποστάσεων Bailer-Jones): πολύ ταχύτερο, αποστάσεις από παράλλαξη
ADQL_NOJOIN = """
SELECT TOP {top} g.source_id, g.ra, g.dec, g.parallax, g.parallax_error, g.phot_g_mean_mag AS g_mag,
       g.bp_rp
FROM gaiadr3.gaia_source AS g
WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {rad:.5f}))
  AND g.phot_g_mean_mag < {glim:.2f}
"""

COLS = ["ra", "dec", "parallax", "parallax_error", "g_mag", "bp_rp", "r_med_geo"]


def _run_query(Gaia, q: str, timeout: float, log):
    import threading, time
    box = {}

    def work():
        try:
            box["t"] = Gaia.launch_job_async(q, dump_to_file=False).get_results()
        except Exception as e:  # noqa
            box["err"] = e
    th = threading.Thread(target=work, daemon=True)
    t0 = time.time()
    th.start()
    last = 0.0
    while th.is_alive():
        th.join(2.0)
        el = time.time() - t0
        if log and el - last >= 15:
            last = el
            log(f"Gaia: αναμονή απάντησης… {el:.0f} s")
        if el > timeout:
            raise RuntimeError(f"καμία απάντηση σε {timeout:.0f} s")
    if "err" in box:
        raise RuntimeError(f"σφάλμα: {box['err']}")
    return box["t"]


def fetch_gaia(ra: float, dec: float, radius_deg: float, glim: float = 18.0, top: int = 120000,
               cache_dir: str | None = None, log=None, timeout: float = 300.0) -> dict:
    key = hashlib.md5(f"{ra:.4f}{dec:.4f}{radius_deg:.4f}{glim:.2f}{top}".encode()).hexdigest()[:12]
    cache = os.path.join(cache_dir, f"gaia_{key}.npz") if cache_dir else None
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if log:
            log(f"Gaia: φορτώθηκε από cache ({len(z['ra'])} πηγές)")
        return {k: z[k] for k in COLS}
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    t = None
    # 1η προσπάθεια: με αποστάσεις Bailer-Jones (join). Αν αργεί, 2η: μόνο παράλλαξη (πολύ ταχύτερη).
    for join, tmo in ((True, timeout * 0.45), (False, timeout * 0.55)):
        q = (ADQL if join else ADQL_NOJOIN).format(top=top, ra=ra, dec=dec, rad=radius_deg, glim=glim)
        if log:
            log(f"Gaia: ερώτημα {'με αποστάσεις Bailer-Jones' if join else 'μόνο παράλλαξη (ταχύτερο)'} – ακτίνα {radius_deg:.2f}°, G<{glim:.1f}")
        try:
            t = _run_query(Gaia, q, tmo, log)
            break
        except Exception as e:  # noqa
            if log:
                log(f"Gaia: {e}")
            err = e
    if t is None:
        raise RuntimeError(f"Το Gaia δεν απάντησε ({err}). Δοκίμασε αργότερα ή χρησιμοποίησε συνθετικές αποστάσεις.")
    out = {}
    for k in COLS:
        if k not in t.colnames:
            out[k] = np.full(len(t), np.nan)
            continue
        col = t[k]
        out[k] = np.asarray(col.filled(np.nan) if hasattr(col, "filled") else col, dtype=np.float64)
    if log:
        log(f"Gaia: {len(out['ra'])} πηγές")
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, **out)
    return out


SYN_REF = 1000.0   # οι συνθετικές αποστάσεις παράγονται για αντικείμενο στα 1000 pc και κλιμακώνονται αργότερα


def _synthetic_distances(mag: np.ndarray, rng) -> np.ndarray:
    """Χωρίς Gaia: εύλογες αποστάσεις (ΔΕΝ είναι μετρήσεις). Λογαριθμοκανονική κατανομή γύρω από το αντικείμενο·
    τα αμυδρότερα αστέρια τείνουν να είναι πιο μακριά. Επιστρέφει pc για αντικείμενο στα SYN_REF pc."""
    from scipy.special import ndtri
    n = len(mag)
    if n == 0:
        return np.zeros(0)
    rank = (np.argsort(np.argsort(mag + rng.normal(0, 0.6, n))) + 0.5) / n          # 0 = φωτεινότερο
    q = np.clip(rank, 0.01, 0.99)
    return np.clip(SYN_REF * 0.9 * np.exp(0.85 * ndtri(q)), 0.05 * SYN_REF, 6.0 * SYN_REF)


def assign_distances(stars: Stars, sky: SkyWCS | None, gaia: dict | None, W: int, H: int,
                     tol_px: float = 3.0, seed: int = 1) -> dict:
    """Επιστρέφει dist_pc (N), matched (bool N), zp, n_matched."""
    rng = np.random.default_rng(seed)
    N = len(stars)
    dist = np.full(N, np.nan)
    matched = np.zeros(N, bool)
    info = {"zp": None, "n_matched": 0}
    if gaia is not None and sky is not None and N > 0:
        gx, gy = sky.world_to_pix(gaia["ra"], gaia["dec"])
        inside = np.isfinite(gx) & np.isfinite(gy) & (gx > -10) & (gx < W + 10) & (gy > -10) & (gy < H + 10)
        gd = gaia["r_med_geo"].copy()
        plx, pe = gaia["parallax"], gaia["parallax_error"]
        fb = np.isfinite(plx) & (plx > 0) & (plx / np.maximum(pe, 1e-6) > 3)
        gd = np.where(np.isfinite(gd), gd, np.where(fb, 1000.0 / np.maximum(plx + 0.017, 1e-3), np.nan))
        sel = inside & np.isfinite(gaia["g_mag"])
        G = gaia["g_mag"][sel]
        gpx = np.c_[gx[sel], gy[sel]]
        gdist = gd[sel]
        if len(G) > 5:
            tree = cKDTree(gpx)
            spx = np.c_[stars.x, stars.y]
            d1, i1 = tree.query(spx, distance_upper_bound=tol_px)
            ok1 = np.isfinite(d1)
            if ok1.sum() >= 5:
                br = ok1 & (stars.mag <= np.percentile(stars.mag[ok1], 40))
                zp = float(np.median(G[i1[br]] - stars.mag[br])) if br.sum() >= 3 else float(np.median(G[i1[ok1]] - stars.mag[ok1]))
                info["zp"] = zp
                for i in np.nonzero(ok1)[0]:
                    cands = tree.query_ball_point(spx[i], tol_px)
                    if not cands:
                        continue
                    cands = np.asarray(cands)
                    dm = np.abs(G[cands] - (stars.mag[i] + zp))
                    j = cands[np.argmin(dm + 0.05 * np.linalg.norm(gpx[cands] - spx[i], axis=1))]
                    if abs(G[j] - (stars.mag[i] + zp)) < 1.6 and np.isfinite(gdist[j]):
                        dist[i] = gdist[j]
                        matched[i] = True
    info["n_matched"] = int(matched.sum())
    um = ~matched
    if matched.sum() >= 12 and um.any():
        mm = stars.mag[matched]
        md = dist[matched]
        order = np.argsort(mm)
        mm_s, md_s = mm[order], md[order]
        for i in np.nonzero(um)[0]:
            k = int(np.searchsorted(mm_s, stars.mag[i]))
            lo, hi = max(0, k - 8), min(len(mm_s), k + 8)
            dist[i] = rng.choice(md_s[lo:hi])
    elif um.any():
        dist[um] = _synthetic_distances(stars.mag[um], rng)
        info["syn_all"] = True
    return {"dist_pc": np.clip(dist, 1.0, 50000.0), "matched": matched, **info}


def suggest_cluster_distance(dist: np.ndarray, matched: np.ndarray) -> tuple[float, float] | None:
    """Βρίσκει κορυφή στην κατανομή αποστάσεων (π.χ. ανοιχτό σμήνος). Επιστρέφει (pc, ένταση κορυφής)."""
    d = dist[matched]
    d = d[np.isfinite(d) & (d > 20)]
    if len(d) < 20:
        return None
    from scipy.stats import gaussian_kde
    ld = np.log10(d)
    kde = gaussian_kde(ld, bw_method=0.12)
    grid = np.linspace(ld.min(), ld.max(), 400)
    dens = kde(grid)
    k = int(np.argmax(dens))
    prom = float(dens[k] / np.median(dens))
    return float(10 ** grid[k]), prom


# ---------------------------------------------------------------------------------------------
# Επιπλέον αστέρια ΕΞΩ από το πεδίο της φωτογραφίας (ώστε ο ουρανός να μην είναι άδειος όταν η κάμερα
# κινείται και βλέπει από άλλη γωνία)
# ---------------------------------------------------------------------------------------------
def bp_rp_to_rgb(ci: np.ndarray) -> np.ndarray:
    """Προσεγγιστικό χρώμα αστεριού από Gaia BP-RP (κανονικοποιημένο ώστε mean(RGB)=1)."""
    ci = np.nan_to_num(np.asarray(ci, float), nan=0.8)
    ci = np.clip(ci, -0.4, 3.5)
    T = 4600.0 * (1.0 / (0.92 * (0.78 * ci + 0.1) + 1.7) + 1.0 / (0.92 * (0.78 * ci + 0.1) + 0.62))
    T = np.clip(T, 2500, 20000) / 100.0
    r = np.where(T <= 66, 255.0, 329.698727446 * np.power(np.maximum(T - 60, 1e-3), -0.1332047592))
    g = np.where(T <= 66, 99.4708025861 * np.log(np.maximum(T, 1e-3)) - 161.1195681661,
                 288.1221695283 * np.power(np.maximum(T - 60, 1e-3), -0.0755148492))
    b = np.where(T >= 66, 255.0, np.where(T <= 19, 0.0, 138.5177312231 * np.log(np.maximum(T - 10, 1e-3)) - 305.0447927307))
    rgb = np.clip(np.c_[r, g, b], 20, 255) / 255.0
    return (rgb / rgb.mean(axis=1, keepdims=True)).astype(np.float32)


def extra_stars_gaia(gaia: dict, sky: SkyWCS, W: int, H: int, zp: float, fwhm: float, margin_px: float = 2.0) -> dict:
    """Αστέρια Gaia ΕΞΩ από το κάδρο, με φωτεινότητα βαθμονομημένη στο flux της εικόνας."""
    gx, gy = sky.world_to_pix(gaia["ra"], gaia["dec"])
    G = gaia["g_mag"]
    d = gaia["r_med_geo"].copy()
    plx, pe = gaia["parallax"], gaia["parallax_error"]
    fb = np.isfinite(plx) & (plx > 0) & (plx / np.maximum(pe, 1e-6) > 3)
    d = np.where(np.isfinite(d), d, np.where(fb, 1000.0 / np.maximum(plx + 0.017, 1e-3), np.nan))
    outside = (gx < -margin_px) | (gx > W - 1 + margin_px) | (gy < -margin_px) | (gy > H - 1 + margin_px)
    ok = outside & np.isfinite(gx) & np.isfinite(gy) & np.isfinite(G) & np.isfinite(d)
    flux = 10 ** (-0.4 * (G[ok] - zp))
    rgb = bp_rp_to_rgb(gaia["bp_rp"][ok])
    return dict(x=gx[ok].astype(np.float32), y=gy[ok].astype(np.float32), flux=flux.astype(np.float32),
                rgb=rgb, fwhm=np.full(ok.sum(), fwhm, np.float32), dist=np.clip(d[ok], 1.0, 50000.0), synthetic=False)


def extra_stars_synthetic(stars: Stars, dist: np.ndarray, W: int, H: int, radius_px: float, n: int, seed: int = 5) -> dict:
    """Χωρίς Gaia: ξαναχρησιμοποιεί στατιστικά τα εντοπισμένα αστέρια σε μεγαλύτερο κώνο (ΔΕΝ είναι πραγματικά αστέρια)."""
    rng = np.random.default_rng(seed)
    N = len(stars)
    if N == 0 or n <= 0:
        return dict(x=np.zeros(0, np.float32), y=np.zeros(0, np.float32), flux=np.zeros(0, np.float32),
                    rgb=np.zeros((0, 3), np.float32), fwhm=np.zeros(0, np.float32), dist=np.zeros(0), synthetic=True)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    ang = rng.uniform(0, 2 * np.pi, n * 2)
    rad = radius_px * np.sqrt(rng.uniform(0, 1, n * 2))
    x, y = cx + rad * np.cos(ang), cy + rad * np.sin(ang)
    out = (x < -2) | (x > W + 1) | (y < -2) | (y > H + 1)
    x, y = x[out][:n], y[out][:n]
    k = rng.integers(0, N, len(x))
    return dict(x=x.astype(np.float32), y=y.astype(np.float32), flux=stars.flux[k], rgb=stars.flux_rgb[k] / np.maximum(stars.flux[k], 1e-12)[:, None],
                fwhm=stars.fwhm[k], dist=np.asarray(dist)[k], synthetic=True)

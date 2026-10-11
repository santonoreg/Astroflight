"""Gaia DR3: λήψη καταλόγου, αντιστοίχιση με αστέρια της εικόνας, απόσταση ανά αστέρι."""
from __future__ import annotations
import os, hashlib
import numpy as np
from scipy.spatial import cKDTree
from .stars import Stars
from .wcs_solve import SkyWCS

ADQL = """
SELECT TOP {top} g.ra, g.dec, g.parallax, g.parallax_error, g.phot_g_mean_mag AS g_mag,
       g.bp_rp, d.r_med_geo
FROM gaiadr3.gaia_source AS g
LEFT JOIN external.gaiaedr3_distance AS d ON g.source_id = d.source_id
WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {rad:.5f}))
  AND g.phot_g_mean_mag < {glim:.2f}
"""
# Εναλλακτικό (χωρίς join με τον πίνακα αποστάσεων Bailer-Jones): πολύ ταχύτερο, αποστάσεις από παράλλαξη
ADQL_NOJOIN = """
SELECT TOP {top} g.ra, g.dec, g.parallax, g.parallax_error, g.phot_g_mean_mag AS g_mag,
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


COL_ALIASES = {
    "ra": ("ra", "ra_icrs", "raj2000", "_raj2000"),
    "dec": ("dec", "de_icrs", "dej2000", "_dej2000"),
    "parallax": ("parallax", "plx"),
    "parallax_error": ("parallax_error", "e_plx", "plx_error", "parallax_err"),
    "g_mag": ("phot_g_mean_mag", "gmag", "g_mag", "g"),
    "bp_rp": ("bp_rp", "bp-rp", "bprp"),
    "r_med_geo": ("r_med_geo", "rgeo", "r_geo"),
}


def gaia_query_text(ra: float, dec: float, radius_deg: float, glim: float = 15.5, top: int = 120000) -> str:
    """Ερώτημα ADQL που μπορείς να επικολλήσεις στο https://gea.esac.esa.int/archive/ (Search → Advanced ADQL) και να
    κατεβάσεις το αποτέλεσμα ως CSV, αν η εφαρμογή δεν μπορεί να συνδεθεί μόνη της."""
    return ADQL_NOJOIN.format(top=top, ra=ra, dec=dec, rad=radius_deg, glim=glim).strip()


def _table_to_dict(t) -> dict:
    names = {c.lower(): c for c in t.colnames}
    out = {}
    for k in COLS:
        col = next((names[a] for a in COL_ALIASES[k] if a in names), None)
        if col is None:
            out[k] = np.full(len(t), np.nan)
            continue
        c = t[col]
        out[k] = np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, dtype=np.float64)
    return out


def _sep_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = (np.radians(x) for x in (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def load_gaia_file(path: str, ra: float, dec: float, radius_deg: float, glim: float = 99.0, log=None) -> dict:
    """Διαβάζει αρχείο Gaia που κατέβασες εσύ (CSV / FITS / VOTable από το Gaia archive ή το VizieR).
    Χρειάζεται στήλες ra, dec, parallax (+ parallax_error, phot_g_mean_mag· προαιρετικά bp_rp, r_med_geo)."""
    from astropy.table import Table
    t = Table.read(path)
    out = _table_to_dict(t)
    if not np.isfinite(out["ra"]).any() or not np.isfinite(out["dec"]).any():
        raise RuntimeError(f"Το αρχείο {os.path.basename(path)} δεν έχει στήλες ra/dec (βρέθηκαν: {', '.join(t.colnames[:12])}).")
    if not np.isfinite(out["g_mag"]).any():
        raise RuntimeError("Το αρχείο δεν έχει στήλη μεγέθους G (phot_g_mean_mag).")
    sep = _sep_deg(out["ra"], out["dec"], ra, dec)
    keep = np.isfinite(sep) & (sep <= radius_deg * 1.02)      # δεν φιλτράρουμε με μέγεθος: το αρχείο το διάλεξε ο χρήστης
    out = {k: v[keep] for k, v in out.items()}
    if log:
        log(f"Gaia: φορτώθηκαν {int(keep.sum())} πηγές από το αρχείο {os.path.basename(path)}")
        if keep.sum() and np.percentile(sep[keep], 95) < 0.6 * radius_deg:
            log("Gaia: ΠΡΟΣΟΧΗ – το αρχείο φαίνεται να καλύπτει μικρότερη περιοχή από το πεδίο της φωτογραφίας.")
    if keep.sum() < 20:
        raise RuntimeError(f"Μόνο {int(keep.sum())} πηγές του αρχείου πέφτουν μέσα στο πεδίο (RA {ra:.3f}°, Dec {dec:.3f}°, ακτίνα {radius_deg:.2f}°). "
                           "Έλεγξε ότι το αρχείο αφορά την ίδια περιοχή του ουρανού.")
    return out


def _query_tap(tap, ra, dec, radius_deg, glim, top, timeout, log, label):
    """Ερώτημα ADQL σε TAP υπηρεσία: πρώτα με αποστάσεις Bailer-Jones (αν υποστηρίζεται), μετά μόνο παράλλαξη."""
    err = None
    for join, tmo in ((True, timeout * 0.4), (False, timeout * 0.6)):
        if label != "ESA" and join:
            continue
        q = (ADQL if join else ADQL_NOJOIN).format(top=top, ra=ra, dec=dec, rad=radius_deg, glim=glim)
        if log:
            log(f"Gaia ({label}): ερώτημα {'με αποστάσεις Bailer-Jones' if join else 'μόνο παράλλαξη'} – ακτίνα {radius_deg:.2f}°, G<{glim:.1f}")
        try:
            return _run_query(tap, q, tmo, log)
        except Exception as e:  # noqa
            err = e
            if log:
                log(f"Gaia ({label}): {e}")
    raise RuntimeError(str(err))


def _query_vizier(ra, dec, radius_deg, glim, top, timeout, log):
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if log:
        log(f"Gaia (VizieR): ερώτημα I/355/gaiadr3 – ακτίνα {radius_deg:.2f}°, G<{glim:.1f}")
    v = Vizier(columns=["RA_ICRS", "DE_ICRS", "Plx", "e_Plx", "Gmag", "BP-RP"], row_limit=top,
               column_filters={"Gmag": f"<{glim:.2f}"}, timeout=timeout)
    res = v.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius_deg * u.deg, catalog="I/355/gaiadr3")
    if len(res) == 0 or len(res[0]) == 0:
        raise RuntimeError("το VizieR δεν επέστρεψε πηγές")
    return res[0]


_BACKEND_STATE = {"ok": None, "bad": set()}


def _backend_order() -> list[str]:
    """Σειρά δοκιμής: ό,τι δούλεψε τελευταίο, μετά VizieR → ARI → ESA (ο ESA είναι συχνά αργός), και όσοι απέτυχαν στο τέλος."""
    base = ["VizieR (CDS)", "ARI Heidelberg", "ESA Gaia archive"]
    ok = _BACKEND_STATE["ok"]
    order = ([ok] if ok else []) + [b for b in base if b != ok]
    good = [b for b in order if b not in _BACKEND_STATE["bad"]]
    return good + [b for b in order if b in _BACKEND_STATE["bad"]]


def fetch_gaia(ra: float, dec: float, radius_deg: float, glim: float = 18.0, top: int = 120000,
               cache_dir: str | None = None, log=None, timeout: float = 300.0, local_file: str | None = None) -> dict:
    """Πηγές Gaia DR3 γύρω από (ra, dec). Σειρά προσπαθειών: τοπικό αρχείο (αν δόθηκε) → cache → ESA → ARI Heidelberg → VizieR.
    Σε περίπτωση αποτυχίας το μήνυμα λέει τι ακριβώς απέτυχε σε κάθε υπηρεσία."""
    if local_file:
        return load_gaia_file(local_file, ra, dec, radius_deg, glim, log)
    key = hashlib.md5(f"{ra:.4f}{dec:.4f}{radius_deg:.4f}{glim:.2f}{top}".encode()).hexdigest()[:12]
    cache = os.path.join(cache_dir, f"gaia_{key}.npz") if cache_dir else None
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if log:
            log(f"Gaia: φορτώθηκε από cache ({len(z['ra'])} πηγές)")
        return {k: z[k] for k in COLS}
    t = None
    errors = []
    backends = []

    def esa():
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = -1
        return _query_tap(Gaia, ra, dec, radius_deg, glim, top, timeout * 0.4, log, "ESA")

    def ari():
        from astroquery.utils.tap.core import TapPlus
        return _query_tap(TapPlus(url="https://gaia.ari.uni-heidelberg.de/tap"), ra, dec, radius_deg, glim, top, timeout * 0.3, log, "ARI")

    def viz():
        return _query_vizier(ra, dec, radius_deg, glim, top, timeout * 0.3, log)

    fns = {"ESA Gaia archive": esa, "ARI Heidelberg": ari, "VizieR (CDS)": viz}
    for name in _backend_order():
        try:
            t = fns[name]()
            if t is not None and len(t):
                _BACKEND_STATE["ok"] = name
                if log:
                    log(f"Gaia: δεδομένα από {name}")
                break
            t = None
        except Exception as e:  # noqa
            _BACKEND_STATE["bad"].add(name)                 # μέχρι να κλείσει η εφαρμογή δεν ξαναδοκιμάζεται πρώτος
            errors.append(f"{name}: {str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__}")
            if log:
                log(f"Gaia: {name} απέτυχε – {errors[-1]}")
    if t is None:
        q = gaia_query_text(ra, dec, radius_deg, min(glim, 15.5), top)
        raise RuntimeError(
            "Δεν μπόρεσα να πάρω δεδομένα Gaia από καμία υπηρεσία:\n  • " + "\n  • ".join(errors) +
            "\nΈλεγξε τη σύνδεση (firewall/antivirus/proxy) ή κατέβασε τα δεδομένα χειροκίνητα: άνοιξε https://gea.esac.esa.int/archive/ "
            "→ Search → Advanced (ADQL), επικόλλησε το ερώτημα παρακάτω, τρέξε το, κατέβασε το αποτέλεσμα ως CSV και φόρτωσέ το "
            "στην εφαρμογή ως «Τοπικό αρχείο Gaia».\n\n" + q)
    out = _table_to_dict(t)
    if log:
        log(f"Gaia: {len(out['ra'])} πηγές")
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, **out)
    return out


def diagnose_connection(log=print, timeout: float = 8.0) -> bool:
    """Ελέγχει αν φτάνουμε στις υπηρεσίες Gaia και εξηγεί τι δεν δουλεύει. Επιστρέφει True αν τουλάχιστον μία απαντά."""
    import urllib.request, ssl, socket
    ok_any = False
    tests = [("ESA Gaia archive", "https://gea.esac.esa.int/tap-server/tap/availability"),
             ("ARI Heidelberg", "https://gaia.ari.uni-heidelberg.de/tap/availability"),
             ("VizieR (CDS)", "https://vizier.cds.unistra.fr/viz-bin/votable?-source=I/355/gaiadr3&-c=83.8,-5.4&-c.rs=0.01&-out.max=1")]
    proxies = {k: v for k, v in os.environ.items() if k.lower() in ("http_proxy", "https_proxy")}
    log("Έλεγχος σύνδεσης" + (f" (proxy από το περιβάλλον: {', '.join(proxies.values())})" if proxies else "") + "…")
    import urllib.error
    for name, url in tests:
        host = url.split("/")[2]
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                log(f"  ✔ {name}: απαντά (HTTP {r.status})")
                ok_any = True
        except urllib.error.HTTPError as e:
            # ο server απάντησε (π.χ. 404 σε διαφορετικό path) → υπάρχει σύνδεση
            log(f"  ✔ {name}: προσιτός (HTTP {e.code})")
            ok_any = True
        except Exception as e:  # noqa
            reason = getattr(e, "reason", e)
            if isinstance(reason, ssl.SSLError) or isinstance(e, ssl.SSLError):
                log(f"  ✖ {name}: σφάλμα πιστοποιητικού SSL ({reason}). Συχνά το προκαλεί antivirus/εταιρικό proxy που «ανοίγει» τις συνδέσεις HTTPS.")
            elif isinstance(reason, socket.gaierror):
                log(f"  ✖ {name}: δεν βρέθηκε η διεύθυνση {host} – δεν υπάρχει internet ή το DNS μπλοκάρεται.")
            elif isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason):
                log(f"  ✖ {name}: καμία απάντηση σε {timeout:.0f} s – πιθανό firewall ή αργή σύνδεση.")
            else:
                log(f"  ✖ {name}: {type(e).__name__}: {reason}")
    if not ok_any:
        log("Καμία υπηρεσία δεν απάντησε: δοκίμασε από browser το https://gea.esac.esa.int/archive/ . Αν ανοίγει εκεί αλλά όχι εδώ, "
            "φταίει proxy/antivirus/firewall για την Python· τότε χρησιμοποίησε «Τοπικό αρχείο Gaia».")
    return ok_any


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

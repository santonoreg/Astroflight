"""Δημιουργεί συνθετική αστροφωτογραφία (16-bit TIFF) για δοκιμές."""
import sys, os
import numpy as np, cv2, tifffile


def make(path, W=1600, H=1200, seed=3, n_stars=700):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # νεφέλωμα: blobs + fractal θόρυβος
    neb = np.zeros((H, W, 3), np.float32)
    for _ in range(14):
        cx, cy = rng.uniform(0.25, 0.75) * W, rng.uniform(0.25, 0.75) * H
        sx, sy = rng.uniform(0.05, 0.2) * W, rng.uniform(0.05, 0.2) * H
        a = rng.uniform(0.2, 1.0)
        col = np.array([rng.uniform(0.6, 1.0), rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.9)], np.float32)
        g = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
        neb += a * g[..., None] * col
    noise = np.zeros((H, W), np.float32)
    for o in range(4):
        n = rng.standard_normal((H // 2 ** (5 - o) + 2, W // 2 ** (5 - o) + 2)).astype(np.float32)
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
        noise += n / 2 ** (o * 0.6)
    noise = (noise - noise.min()) / np.ptp(noise)
    neb *= (0.35 + 0.9 * noise)[..., None]
    # σκοτεινή λωρίδα σκόνης
    lane = np.exp(-((yy - (0.55 * H + 0.15 * H * np.sin(xx / W * 6.0))) / (0.025 * H)) ** 2)
    neb *= (1 - 0.8 * lane)[..., None]
    neb *= 0.55
    neb += 0.004
    # αστέρια (Moffat)
    stars = np.zeros((H, W, 3), np.float32)
    truth = []
    for i in range(n_stars):
        x, y = rng.uniform(10, W - 10), rng.uniform(10, H - 10)
        mag = 6.0 * rng.random() ** 0.6
        flux = 40.0 * 10 ** (-0.4 * mag)
        fw = 2.6 + 0.35 * max(0, 3 - mag)
        al = fw / 1.02
        r = int(min(60, 8 * al)) + 1
        x0, y0 = int(x), int(y)
        xs = np.arange(max(0, x0 - r), min(W, x0 + r + 1))
        ys = np.arange(max(0, y0 - r), min(H, y0 + r + 1))
        X, Y = np.meshgrid(xs, ys)
        prof = (1 + ((X - x) ** 2 + (Y - y) ** 2) / al ** 2) ** -3.0
        prof *= 2.0 / (np.pi * al ** 2)
        t = rng.uniform(0.0, 1.0)
        col = np.array([1.0, 0.8 + 0.2 * (1 - t), 0.6 + 0.4 * t], np.float32) if rng.random() < 0.5 else np.array([0.7 + 0.3 * t, 0.85, 1.0], np.float32)
        col /= col.mean()
        stars[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] += (flux * prof)[..., None] * col
        truth.append((x, y, flux))
    lin = np.clip(neb + stars, 0, 1)
    lin = lin + rng.normal(0, 0.0015, lin.shape).astype(np.float32)
    lin = np.clip(lin, 0, 1)
    disp = lin ** (1 / 2.2)
    tifffile.imwrite(path, (disp * 65535).astype(np.uint16), photometric="rgb")
    starless = np.clip(neb + rng.normal(0, 0.0015, neb.shape), 0, 1) ** (1 / 2.2)
    tifffile.imwrite(path.replace(".tif", "_starless.tif"), (starless * 65535).astype(np.uint16), photometric="rgb")
    np.save(path.replace(".tif", "_truth.npy"), np.array(truth, np.float32))
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic.tif"
    print(make(out))

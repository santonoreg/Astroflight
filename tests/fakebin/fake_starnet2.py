#!/usr/bin/env python3
"""Ψεύτικο starnet2 για δοκιμές: -i είσοδος -o έξοδος. «Αφαιρεί» αστέρια με μορφολογικό άνοιγμα + median."""
import sys, time
import numpy as np, cv2, tifffile
a = sys.argv[1:]
fin, fout = a[a.index("-i") + 1], a[a.index("-o") + 1]
import os
mode = os.environ.get("FAKE_STARNET", "ok")
if mode == "crash":
    print("Segmentation fault (fake)", flush=True); sys.exit(139)
if mode == "slow":
    time.sleep(120)
im = tifffile.imread(fin)
print("StarNet fake: processing", im.shape, flush=True)
for i in range(3):
    time.sleep(0.5); print(f"tile {i + 1}/3", flush=True)
f = im.astype(np.float32)
op = cv2.morphologyEx(f, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
out = np.minimum(f, cv2.medianBlur(np.clip(op, 0, 65535).astype(np.uint16), 5).astype(np.float32))
tifffile.imwrite(fout, np.clip(out, 0, 65535).astype(np.uint16), photometric="rgb")
sys.exit(0)

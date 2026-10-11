# Προέλευση και τροποποιήσεις

Τα αρχεία στον φάκελο `scripts/` είναι **αντίγραφα** των παρακάτω scripts του έργου
[Svenesis Siril Scripts](https://github.com/sramuschkat/Siril-Scripts) (συγγραφέας: Sven Ramuschkat), κατά το commit
`e891d18299e62ab7cad467d0a1fbd1fe2a2342e6`:

| Αρχείο | Πηγή |
|---|---|
| `scripts/Svenesis-CosmicView3D.py` | `Svenesis-CosmicView3D.py` (v1.0.0) |
| `scripts/Svenesis-CosmicDepth3D.py` | `Svenesis-CosmicDepth3D.py` (v1.0.1) |
| `docs/*Instructions*.md`, `docs/screenshots/` | `Instructions/` και `screenshots/` του upstream |
| `LICENSE` | GNU GPL v3 (όπως στο upstream) |

Αντιγράφηκαν μόνο αυτά τα δύο scripts, όχι ολόκληρο το upstream repository.

## Τι τροποποιήθηκε

Τα δύο scripts παραμένουν **αυτούσια, χωρίς καμία αλλαγή στον κώδικά τους** (έτσι ενημερώσεις του upstream μπορούν να αντικατασταθούν
απευθείας). Η προσαρμογή, ώστε να δουλεύουν με επιλογή αρχείου εικόνας αντί για ανοιχτό Siril, έγινε **γύρω** τους:

* `shim/sirilpy/` – στρώμα συμβατότητας που μιμείται το υποσύνολο του `sirilpy` που χρησιμοποιούν τα scripts
  (εικόνα, κεφαλίδα FITS/WCS, `pix2radec`/`radec2pix`, φάκελος εργασίας) πάνω σε ένα προετοιμασμένο αρχείο.
* `cosmic3d/` – προετοιμασία εικόνας (ανάγνωση TIFF/PNG/JPG/FITS, plate solve με ASTAP, εγγραφή FITS με WCS) και γραφικό περιβάλλον.
* `launcher.py` – σημείο εκκίνησης.

## Άδεια

Όλο το έργο διανέμεται υπό **GNU General Public License v3.0 ή νεότερη** (`SPDX-License-Identifier: GPL-3.0-or-later`),
όπως επιβάλλει η άδεια του upstream. Βλ. `LICENSE`.

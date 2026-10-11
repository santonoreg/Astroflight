"""Εκτέλεση εξωτερικών προγραμμάτων (ASTAP, StarNet) με ζωντανή πρόοδο και δυνατότητα ακύρωσης."""
from __future__ import annotations
import subprocess, sys, threading, time, queue


class Cancelled(Exception):
    """Ο χρήστης ακύρωσε την επεξεργασία."""


class CancelToken:
    def __init__(self):
        self._ev = threading.Event()

    def cancel(self):
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def check(self):
        if self._ev.is_set():
            raise Cancelled("Ακυρώθηκε από τον χρήστη.")


def run_cmd(cmd: list[str], cwd: str | None = None, timeout: float = 600.0, tick=None, cancel: CancelToken | None = None,
            on_line=None) -> tuple[int, str]:
    """Τρέχει την εντολή και επιστρέφει (κωδικός εξόδου, έξοδος stdout+stderr).

    tick(elapsed_s, last_line): καλείται ~3 φορές το δευτερόλεπτο όσο τρέχει το πρόγραμμα (για ένδειξη προόδου).
    on_line(line): καλείται για κάθε γραμμή που τυπώνει το πρόγραμμα.
    Αν ακυρωθεί ή λήξει ο χρόνος, το πρόγραμμα τερματίζεται."""
    import os
    if os.sep in cmd[0] or "/" in cmd[0]:
        cmd = [os.path.abspath(cmd[0])] + list(cmd[1:])      # το cwd αλλάζει, άρα η διαδρομή πρέπει να είναι απόλυτη
    kw = {}
    if sys.platform.startswith("win"):
        kw["creationflags"] = 0x08000000                # CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace", bufsize=1, **kw)
    q: queue.Queue = queue.Queue()

    def reader():
        try:
            for line in proc.stdout:                    # type: ignore[union-attr]
                q.put(line.rstrip("\r\n"))
        finally:
            q.put(None)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    t0 = time.time()
    lines: list[str] = []
    last = ""
    done_reading = False
    try:
        while True:
            try:
                item = q.get(timeout=0.3)
                if item is None:
                    done_reading = True
                else:
                    lines.append(item)
                    if item.strip():
                        last = item.strip()
                        if on_line:
                            on_line(last)
            except queue.Empty:
                pass
            el = time.time() - t0
            if tick:
                tick(el, last)
            if cancel is not None and cancel.cancelled:
                proc.kill()
                proc.wait()
                raise Cancelled("Ακυρώθηκε από τον χρήστη.")
            if el > timeout:
                proc.kill()
                proc.wait()
                raise TimeoutError(f"Το πρόγραμμα δεν τελείωσε σε {timeout:.0f} s: {' '.join(cmd[:1])}")
            if done_reading and proc.poll() is not None:
                break
            if done_reading:
                proc.wait()
                break
    finally:
        if proc.poll() is None:
            proc.kill()
    th.join(timeout=1.0)
    return proc.returncode, "\n".join(lines)

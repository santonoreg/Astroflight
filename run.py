"""Εκκίνηση AstroFlight. Εμφανίζει αμέσως οθόνη φόρτωσης (οι βιβλιοθήκες αστρονομίας αργούν να φορτώσουν)."""
import sys


def start():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QPixmap, QSurfaceFormat
    from PySide6.QtWidgets import QApplication, QSplashScreen

    fmt = QSurfaceFormat()               # πρέπει να οριστεί ΠΡΙΝ δημιουργηθεί το QApplication
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    pm = QPixmap(520, 180)
    pm.fill(QColor(30, 31, 38))
    splash = QSplashScreen(pm)
    f = QFont()
    f.setPointSize(14)
    splash.setFont(f)
    splash.showMessage("AstroFlight\nΦόρτωση βιβλιοθηκών… (λίγα δευτερόλεπτα)", Qt.AlignCenter, QColor(220, 222, 232))
    splash.show()
    app.processEvents()
    from astroflight.gui import main
    main(app, splash)


if __name__ == "__main__":
    start()

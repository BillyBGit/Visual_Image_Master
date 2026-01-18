"""
Image Sorter Application - Main Entry Point
"""

import os
import sys

if __name__ == "__main__":
    # Suppress Qt ICC profile warnings
    os.environ['QT_LOGGING_RULES'] = 'qt.gui.icc=false'

    # Import Qt ASAP to show splash before heavy imports
    from PySide6.QtWidgets import QApplication, QSplashScreen
    from PySide6.QtGui import QPixmap
    from PySide6.QtCore import Qt

    app = QApplication(sys.argv)

    # Show splash screen IMMEDIATELY before any other imports
    splash_path = os.path.join(os.path.dirname(__file__), 'splash.jpg')
    splash = None
    if os.path.exists(splash_path):
        splash_pix = QPixmap(splash_path)
        splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        splash.showMessage("Loading modules...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
        app.processEvents()

    # NOW do the heavy imports after splash is visible
    from image_sorter import ImageSorter

    # Create main window
    if splash:
        splash.showMessage("Initializing application...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
        app.processEvents()

    window = ImageSorter()

    if splash:
        splash.showMessage("Ready!", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
        app.processEvents()

    # Close splash and show main window
    if splash:
        splash.finish(window)

    window.show()
    sys.exit(app.exec())

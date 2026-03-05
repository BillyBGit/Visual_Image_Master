"""Pytest configuration for PyQt6 tests."""

import sys
from PyQt6.QtWidgets import QApplication


# Global QApplication instance for PyQt6 tests
if not QApplication.instance():
    app = QApplication(sys.argv)
else:
    app = QApplication.instance()

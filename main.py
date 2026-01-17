"""
Image Sorter Application - Main Entry Point
"""

from PySide6.QtWidgets import QApplication
from image_sorter import ImageSorter


if __name__ == "__main__":
    app = QApplication([])
    window = ImageSorter()
    window.show()
    app.exec()

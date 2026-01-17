"""
Utility functions for the Image Sorter application.
"""

from PySide6.QtWidgets import QApplication


def find_main_window():
    """Helper function to find the ImageSorter main window instance."""
    from image_sorter import ImageSorter

    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, ImageSorter):
            return widget
    return None

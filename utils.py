"""
Utility functions for the Image Sorter application.
"""

from PySide6.QtWidgets import QApplication, QMessageBox


def find_main_window():
    """Helper function to find the ImageSorter main window instance."""
    from image_sorter import ImageSorter

    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, ImageSorter):
            return widget
    return None


def reset_link_list(parent, text):
    """
    Start a fresh clipboard-based "link list" containing only `text`.

    The link list lives entirely in the system clipboard rather than in any
    app-side buffer, so if the clipboard already holds content this confirms
    with the user before overwriting it.
    """
    clipboard = QApplication.clipboard()
    existing = clipboard.text()
    if existing.strip():
        reply = QMessageBox.question(
            parent,
            "Start Link List",
            "The clipboard already has content. Clear it and start a new link list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
    clipboard.setText(text)


def append_to_link_list(text):
    """Append `text` to the clipboard-based link list on a new line."""
    clipboard = QApplication.clipboard()
    existing = clipboard.text()
    if existing.strip():
        clipboard.setText(existing + "\n" + text)
    else:
        clipboard.setText(text)

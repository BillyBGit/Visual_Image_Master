"""Tests for AdvancedFilterDialog.setup_ui method."""

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QScrollArea,
    QCheckBox,
    QPushButton,
)
from advanced_filter_dialog import AdvancedFilterDialog


# Qt application instance for testing
@pytest.fixture(scope="module")
def qt_app():
    """Qt application fixture for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # app.quit()  # Don't quit, as other tests might need it


class TestAdvancedFilterDialogSetupUi:
    """Test suite for AdvancedFilterDialog.setup_ui method."""

    def test_setup_ui_creates_main_layout(self, qt_app):
        """Test that the dialog has a QVBoxLayout as its main layout."""
        dialog = AdvancedFilterDialog()
        assert dialog.layout() is not None
        assert isinstance(dialog.layout(), QVBoxLayout)

    def test_setup_ui_creates_scroll_area(self, qt_app):
        """Test that the dialog contains a QScrollArea widget."""
        dialog = AdvancedFilterDialog()
        # Find the scroll area in the layout
        scroll_area = None
        for i in range(dialog.layout().count()):
            widget = dialog.layout().itemAt(i).widget()
            if isinstance(widget, QScrollArea):
                scroll_area = widget
                break
        assert scroll_area is not None

    def test_setup_ui_creates_filter_sections(self, qt_app):
        """Test that the scroll area contains all filter sections."""
        dialog = AdvancedFilterDialog()
        # Find the scroll area
        scroll_area = None
        for i in range(dialog.layout().count()):
            widget = dialog.layout().itemAt(i).widget()
            if isinstance(widget, QScrollArea):
                scroll_area = widget
                break
        assert scroll_area is not None

        # Get the scroll widget and its layout
        scroll_widget = scroll_area.widget()
        scroll_layout = scroll_widget.layout()

        # Check that all filter sections are present
        assert scroll_layout.count() == 5, "Should have 5 filter sections"

    def test_setup_ui_creates_global_invert_checkbox(self, qt_app):
        """Test that the dialog contains the global invert checkbox."""
        dialog = AdvancedFilterDialog()
        # Find the global invert checkbox
        global_invert_cb = None
        for i in range(dialog.layout().count()):
            widget = dialog.layout().itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                global_invert_cb = widget
                break
        assert global_invert_cb is not None
        assert (
            global_invert_cb.text() == "Invert entire filter (show what does NOT match)"
        )

    def test_setup_ui_creates_buttons(self, qt_app):
        """Test that the dialog contains all three buttons."""
        dialog = AdvancedFilterDialog()
        # Find the button layout (should be the last item in main layout)
        button_layout = None
        for i in range(dialog.layout().count()):
            item = dialog.layout().itemAt(i)
            if item.layout() is not None:
                button_layout = item.layout()

        assert button_layout is not None

        # Check that all three buttons are present
        button_texts = []
        for i in range(button_layout.count()):
            widget = button_layout.itemAt(i).widget()
            if isinstance(widget, QPushButton):
                button_texts.append(widget.text())

        assert "Apply Filters" in button_texts
        assert "Clear All" in button_texts
        assert "Cancel" in button_texts

    def test_setup_ui_connects_signals(self, qt_app):
        """Test that all signal connections are properly set up."""
        dialog = AdvancedFilterDialog()

        # Check global_invert_cb connection
        assert dialog.global_invert_cb.stateChanged.connect is not None

        # Check button connections
        assert dialog.apply_btn.clicked.connect is not None
        assert dialog.clear_btn.clicked.connect is not None
        assert dialog.cancel_btn.clicked.connect is not None

    def test_setup_ui_layout_structure(self, qt_app):
        """Test the layout hierarchy structure."""
        dialog = AdvancedFilterDialog()
        main_layout = dialog.layout()

        # Find scroll area
        scroll_area = None
        global_invert_cb = None
        button_layout = None

        for i in range(main_layout.count()):
            item = main_layout.itemAt(i)
            widget = item.widget() if item.widget() else None
            layout = item.layout() if item.layout() else None

            if isinstance(widget, QScrollArea):
                scroll_area = widget
            elif isinstance(widget, QCheckBox):
                global_invert_cb = widget
            elif layout is not None:
                button_layout = layout

        assert scroll_area is not None, "Scroll area should be in main layout"
        assert global_invert_cb is not None, (
            "Global invert checkbox should be in main layout"
        )
        assert button_layout is not None, "Button layout should be in main layout"

        # Check button layout contains the three buttons
        button_widgets = []
        for i in range(button_layout.count()):
            widget = button_layout.itemAt(i).widget()
            if isinstance(widget, QPushButton):
                button_widgets.append(widget)

        assert len(button_widgets) == 3, "Button layout should contain 3 buttons"

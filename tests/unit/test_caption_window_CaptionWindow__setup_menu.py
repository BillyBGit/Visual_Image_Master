"""Tests for CaptionWindow._setup_menu method."""

from unittest.mock import patch
from PySide6.QtWidgets import QMenuBar, QMenu
from PySide6.QtGui import QAction
from caption_window import CaptionWindow


class TestCaptionWindowSetupMenu:
    """Test suite for CaptionWindow._setup_menu method."""

    def test_menu_bar_created(self):
        """Test that menuBar() returns a QMenuBar instance."""
        window = CaptionWindow([], 0)
        menu_bar = window.menuBar()
        assert isinstance(menu_bar, QMenuBar), (
            "menuBar() should return a QMenuBar instance"
        )

    def test_options_menu_created(self):
        """Test that options menu is created in the menu bar."""
        window = CaptionWindow([], 0)
        menu_bar = window.menuBar()
        # Check that the menu bar has menu children
        menu_children = [
            child for child in menu_bar.children() if isinstance(child, QMenu)
        ]
        assert len(menu_children) > 0, "Should have at least one menu"
        # Check that the first menu has the correct title
        assert menu_children[0].title() == "&Options", "First menu should be '&Options'"

    def test_fit_to_window_action_created(self):
        """Test that fit to window action is created correctly."""
        window = CaptionWindow([], 0)
        assert window.fit_act is not None, "fit_act should be created"
        assert isinstance(window.fit_act, QAction), "fit_act should be a QAction"
        assert window.fit_act.text() == "&Fit to Window", (
            "Action text should be '&Fit to Window'"
        )
        assert window.fit_act.isCheckable(), "Action should be checkable"
        assert window.fit_act.isChecked() == window.fit_to_window, (
            "Checked state should match fit_to_window"
        )

    def test_native_size_action_created(self):
        """Test that native size action is created correctly."""
        window = CaptionWindow([], 0)
        assert window.native_act is not None, "native_act should be created"
        assert isinstance(window.native_act, QAction), "native_act should be a QAction"
        assert window.native_act.text() == "&Native Size", (
            "Action text should be '&Native Size'"
        )
        assert window.native_act.isCheckable(), "Action should be checkable"
        assert window.native_act.isChecked() == (not window.fit_to_window), (
            "Checked state should be opposite of fit_to_window"
        )

    def test_fit_action_connected_to_toggle_fit_mode(self):
        """Test that fit action is connected to toggle_fit_mode method."""
        window = CaptionWindow([], 0)
        # Mock the toggle_fit_mode method to verify it gets called
        with patch.object(window, "toggle_fit_mode") as mock_toggle:
            window.fit_act.trigger()
            mock_toggle.assert_called_once()

    def test_native_action_connected_to_set_native_size(self):
        """Test that native size action is connected to set_native_size method."""
        window = CaptionWindow([], 0)
        # Mock the set_native_size method to verify it gets called
        with patch.object(window, "set_native_size") as mock_set_native:
            window.native_act.trigger()
            mock_set_native.assert_called_once()

    def test_actions_added_to_menu(self):
        """Test that actions are added to the options menu."""
        window = CaptionWindow([], 0)
        menu_bar = window.menuBar()
        # Get the options menu by checking menu children
        menu_children = [
            child for child in menu_bar.children() if isinstance(child, QMenu)
        ]
        assert len(menu_children) > 0, "Should have at least one menu"
        options_menu = menu_children[0]
        actions = options_menu.actions()
        assert window.fit_act in actions, "fit_act should be in options menu actions"
        assert window.native_act in actions, (
            "native_act should be in options menu actions"
        )

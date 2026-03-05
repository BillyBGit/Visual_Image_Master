"""Tests for AdvancedFilterDialog.create_quality_filter_section method."""

from PySide6.QtWidgets import QGroupBox, QCheckBox, QRadioButton, QSlider, QLabel
from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogCreateQualityFilterSection:
    """Test suite for AdvancedFilterDialog.create_quality_filter_section method."""

    def test_create_quality_filter_section_returns_groupbox(self):
        """Test that create_quality_filter_section returns QGroupBox with title 'Image Quality'."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_quality_filter_section()
        assert isinstance(group, QGroupBox)
        assert group.title() == "Image Quality"

    def test_quality_enable_checkbox_exists(self):
        """Test that layout contains QCheckBox with text 'Filter by quality metrics'."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_quality_filter_section()
        layout = group.layout()

        # Find the checkbox in the layout
        checkbox = None
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                checkbox = widget
                break

        assert checkbox is not None
        assert checkbox.text() == "Filter by quality metrics"

    def test_quality_mode_radio_buttons_exist(self):
        """Test that layout contains QRadioButton with text 'Meets criteria' and 'Does NOT meet criteria'."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_quality_filter_section()
        layout = group.layout()

        # Find radio buttons in the layout
        meets_rb = None
        not_meets_rb = None

        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.layout():
                sub_layout = item.layout()
                for j in range(sub_layout.count()):
                    widget = sub_layout.itemAt(j).widget()
                    if isinstance(widget, QRadioButton):
                        if widget.text() == "Meets criteria":
                            meets_rb = widget
                        elif widget.text() == "Does NOT meet criteria":
                            not_meets_rb = widget

        assert meets_rb is not None
        assert meets_rb.text() == "Meets criteria"
        assert not_meets_rb is not None
        assert not_meets_rb.text() == "Does NOT meet criteria"

    def test_quality_score_slider_exists(self):
        """Test that layout contains QSlider with range 0-100 and default value 60."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_quality_filter_section()
        layout = group.layout()

        # Find the score slider (first slider with range 0-100)
        score_slider = None
        slider_count = 0
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.layout():
                sub_layout = item.layout()
                for j in range(sub_layout.count()):
                    widget = sub_layout.itemAt(j).widget()
                    if isinstance(widget, QSlider):
                        if widget.minimum() == 0 and widget.maximum() == 100:
                            if (
                                slider_count == 0
                            ):  # First slider with 0-100 range is score
                                score_slider = widget
                            slider_count += 1

        assert score_slider is not None
        assert score_slider.minimum() == 0
        assert score_slider.maximum() == 100
        assert score_slider.value() == 60

    def test_quality_sharpness_slider_exists(self):
        """Test that layout contains QSlider with range 0-100 and default value 40."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_quality_filter_section()
        layout = group.layout()

        # Find the sharpness slider (second slider with range 0-100)
        sharp_slider = None
        slider_count = 0
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.layout():
                sub_layout = item.layout()
                for j in range(sub_layout.count()):
                    widget = sub_layout.itemAt(j).widget()
                    if isinstance(widget, QSlider):
                        if widget.minimum() == 0 and widget.maximum() == 100:
                            if (
                                slider_count == 1
                            ):  # Second slider with 0-100 range is sharpness
                                sharp_slider = widget
                            slider_count += 1

        assert sharp_slider is not None
        assert sharp_slider.minimum() == 0
        assert sharp_slider.maximum() == 100
        assert sharp_slider.value() == 40

    def test_quality_brightness_sliders_exist(self):
        """Test that layout contains two QSliders with range 0-255 and default values 50 and 200 respectively."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_quality_filter_section()
        layout = group.layout()

        # Find the brightness sliders (third and fourth sliders in the layout)
        bright_sliders = []
        slider_count = 0
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.layout():
                sub_layout = item.layout()
                for j in range(sub_layout.count()):
                    widget = sub_layout.itemAt(j).widget()
                    if isinstance(widget, QSlider):
                        if slider_count >= 2:  # Third and fourth sliders are brightness
                            bright_sliders.append(widget)
                        slider_count += 1

        assert len(bright_sliders) == 2
        assert bright_sliders[0].minimum() == 0
        assert bright_sliders[0].maximum() == 255
        assert bright_sliders[0].value() == 50
        assert bright_sliders[1].minimum() == 0
        assert bright_sliders[1].maximum() == 255
        assert bright_sliders[1].value() == 200

    def test_quality_label_updates_on_slider_change(self):
        """Test that labels update when slider values change."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_quality_filter_section()
        layout = group.layout()

        # Find sliders and their corresponding labels
        score_slider = None
        score_label = None
        sharp_slider = None
        sharp_label = None
        bright_min_slider = None
        bright_min_label = None
        bright_max_slider = None
        bright_max_label = None

        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.layout():
                sub_layout = item.layout()
                for j in range(sub_layout.count()):
                    widget = sub_layout.itemAt(j).widget()
                    if isinstance(widget, QSlider):
                        if score_slider is None:
                            score_slider = widget
                        elif sharp_slider is None:
                            sharp_slider = widget
                        elif bright_min_slider is None:
                            bright_min_slider = widget
                        elif bright_max_slider is None:
                            bright_max_slider = widget
                    elif isinstance(widget, QLabel):
                        if score_label is None and score_slider is not None:
                            score_label = widget
                        elif sharp_label is None and sharp_slider is not None:
                            sharp_label = widget
                        elif bright_min_label is None and bright_min_slider is not None:
                            bright_min_label = widget
                        elif bright_max_label is None and bright_max_slider is not None:
                            bright_max_label = widget

        # Test score slider label update
        assert score_label is not None
        score_slider.setValue(75)
        assert score_label.text() == "75"

        # Test sharpness slider label update
        assert sharp_label is not None
        sharp_slider.setValue(50)
        assert sharp_label.text() == "50"

        # Test brightness min slider label update
        assert bright_min_label is not None
        bright_min_slider.setValue(60)
        assert bright_min_label.text() == "60"

        # Test brightness max slider label update
        assert bright_max_label is not None
        bright_max_slider.setValue(210)
        assert bright_max_label.text() == "210"

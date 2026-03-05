"""
Face detection settings dialog for configuring detection parameters.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QGroupBox,
)
from face_detection import FACE_DETECTION_PRESETS


class FaceDetectionDialog(QDialog):
    """Dialog for configuring face detection settings."""

    def __init__(self, parent, current_preset="balanced"):
        super().__init__(parent)
        self.setWindowTitle("Face Detection Settings")
        self.setModal(True)
        self.resize(500, 350)

        self.selected_preset = current_preset

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title = QLabel("Choose Face Detection Method")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        main_layout.addWidget(title)

        # Description
        desc = QLabel(
            "Select how face detection should work when filtering images. "
            "More accurate methods may take longer to process."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; margin-bottom: 10px;")
        main_layout.addWidget(desc)

        # Preset options group
        preset_group = QGroupBox("Detection Presets")
        preset_layout = QVBoxLayout()

        self.button_group = QButtonGroup()
        self.preset_buttons = {}

        for preset_key in ["fast", "balanced", "thorough"]:
            preset_info = FACE_DETECTION_PRESETS[preset_key]

            # Create radio button
            radio = QRadioButton(preset_key.capitalize())
            radio.setChecked(preset_key == current_preset)

            # Create description label
            desc_label = QLabel(f"   {preset_info['description']}")
            desc_label.setStyleSheet(
                "color: #666; margin-left: 20px; margin-bottom: 8px;"
            )
            desc_label.setWordWrap(True)

            # Add method details
            method_text = f"   Method: {preset_info['method'].upper()}"
            if preset_info["method"] == "haar":
                method_text += " (Haar Cascade only)"
            elif preset_info["method"] == "dnn":
                method_text += " (Deep Neural Network)"
            elif preset_info["method"] == "both":
                method_text += " (Haar + DNN combined)"

            method_label = QLabel(method_text)
            method_label.setStyleSheet(
                "color: #888; margin-left: 20px; margin-bottom: 12px; font-size: 9pt;"
            )

            self.button_group.addButton(radio)
            self.preset_buttons[preset_key] = radio

            preset_layout.addWidget(radio)
            preset_layout.addWidget(desc_label)
            preset_layout.addWidget(method_label)

        preset_group.setLayout(preset_layout)
        main_layout.addWidget(preset_group)

        # Info box
        info_box = QLabel(
            "💡 Note: DNN-based detection requires model files. "
            "If models are not found, the system will automatically fall back to Haar Cascade detection."
        )
        info_box.setWordWrap(True)
        info_box.setStyleSheet("""
            background-color: #E3F2FD;
            color: #1565C0;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #90CAF9;
        """)
        main_layout.addWidget(info_box)

        main_layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("Apply")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def get_selected_preset(self):
        """Get the selected preset name."""
        for preset_key, button in self.preset_buttons.items():
            if button.isChecked():
                return preset_key
        return "balanced"

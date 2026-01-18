"""
Advanced filtering dialog for AI-powered image filtering.
Combines face detection, object detection, scene recognition, and quality assessment.
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QCheckBox, QLabel, QPushButton, QSlider, QComboBox,
                               QScrollArea, QWidget, QRadioButton, QButtonGroup,
                               QFileDialog, QLineEdit)
from PySide6.QtCore import Qt

from object_detection import OBJECT_CATEGORIES, COCO_CLASSES
from scene_recognition import SCENE_GROUPS


class AdvancedFilterDialog(QDialog):
    """Dialog for configuring advanced AI-powered filters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced AI Filters")
        self.resize(700, 600)

        # Store filter settings
        self.filter_settings = {
            'global_invert': False,  # Invert entire filter result

            'face_detection': False,
            'face_count_min': 0,
            'face_inverse': False,  # True = does NOT contain

            'object_detection': False,
            'object_categories': [],
            'object_confidence': 0.5,
            'object_inverse': False,  # True = does NOT contain
            'custom_model_path': None,  # Path to custom YOLO model

            'scene_recognition': False,
            'scene_groups': [],
            'scene_confidence': 0.3,
            'scene_inverse': False,  # True = does NOT contain

            'quality_filter': False,
            'quality_min_score': 60,
            'quality_sharpness_min': 40,
            'quality_brightness_min': 50,
            'quality_brightness_max': 200,
            'quality_inverse': False,  # True = does NOT meet criteria

            'filesize_filter': False,
            'filesize_min_kb': 0,  # Minimum file size in KB
            'filesize_max_kb': 10240,  # Maximum file size in KB (10MB default)
        }

        self.setup_ui()

    def setup_ui(self):
        """Setup the dialog UI."""
        main_layout = QVBoxLayout()

        # Create scroll area for filters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Add filter sections
        scroll_layout.addWidget(self.create_face_filter_section())
        scroll_layout.addWidget(self.create_object_filter_section())
        scroll_layout.addWidget(self.create_scene_filter_section())
        scroll_layout.addWidget(self.create_quality_filter_section())
        scroll_layout.addWidget(self.create_filesize_filter_section())

        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        # Global invert option
        self.global_invert_cb = QCheckBox("Invert entire filter (show what does NOT match)")
        self.global_invert_cb.setStyleSheet("font-weight: bold; color: #FF5722;")
        self.global_invert_cb.stateChanged.connect(self.update_filter_settings)
        main_layout.addWidget(self.global_invert_cb)

        # Buttons
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Filters")
        self.apply_btn.clicked.connect(self.accept)
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all_filters)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.clear_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.apply_btn)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def create_face_filter_section(self):
        """Create face detection filter section."""
        group = QGroupBox("Face Detection")
        layout = QVBoxLayout()

        self.face_enable_cb = QCheckBox("Filter by faces")
        self.face_enable_cb.stateChanged.connect(self.update_filter_settings)
        layout.addWidget(self.face_enable_cb)

        # Contains/Does not contain
        mode_layout = QHBoxLayout()
        self.face_contains_rb = QRadioButton("Contains")
        self.face_contains_rb.setChecked(True)
        self.face_contains_rb.toggled.connect(self.update_filter_settings)
        self.face_not_contains_rb = QRadioButton("Does NOT contain")
        self.face_not_contains_rb.toggled.connect(self.update_filter_settings)
        mode_layout.addWidget(self.face_contains_rb)
        mode_layout.addWidget(self.face_not_contains_rb)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Minimum face count
        face_count_layout = QHBoxLayout()
        face_count_layout.addWidget(QLabel("Minimum faces:"))
        self.face_count_slider = QSlider(Qt.Orientation.Horizontal)
        self.face_count_slider.setRange(0, 10)
        self.face_count_slider.setValue(0)
        self.face_count_slider.valueChanged.connect(self.update_filter_settings)
        self.face_count_label = QLabel("0")
        face_count_layout.addWidget(self.face_count_slider)
        face_count_layout.addWidget(self.face_count_label)
        layout.addLayout(face_count_layout)

        self.face_count_slider.valueChanged.connect(
            lambda v: self.face_count_label.setText(str(v))
        )

        group.setLayout(layout)
        return group

    def create_object_filter_section(self):
        """Create object detection filter section."""
        group = QGroupBox("Object Detection (YOLO)")
        layout = QVBoxLayout()

        self.object_enable_cb = QCheckBox("Filter by objects")
        self.object_enable_cb.stateChanged.connect(self.update_filter_settings)
        layout.addWidget(self.object_enable_cb)

        # Contains/Does not contain
        mode_layout = QHBoxLayout()
        self.object_contains_rb = QRadioButton("Contains")
        self.object_contains_rb.setChecked(True)
        self.object_contains_rb.toggled.connect(self.update_filter_settings)
        self.object_not_contains_rb = QRadioButton("Does NOT contain")
        self.object_not_contains_rb.toggled.connect(self.update_filter_settings)
        mode_layout.addWidget(self.object_contains_rb)
        mode_layout.addWidget(self.object_not_contains_rb)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Object categories
        layout.addWidget(QLabel("Select object categories:"))
        self.object_category_cbs = {}
        for category in OBJECT_CATEGORIES.keys():
            cb = QCheckBox(category.replace('_', ' ').title())
            cb.stateChanged.connect(self.on_object_category_changed)
            self.object_category_cbs[category] = cb
            layout.addWidget(cb)

        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.object_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.object_conf_slider.setRange(10, 90)
        self.object_conf_slider.setValue(50)
        self.object_conf_slider.valueChanged.connect(self.update_filter_settings)
        self.object_conf_label = QLabel("0.50")
        conf_layout.addWidget(self.object_conf_slider)
        conf_layout.addWidget(self.object_conf_label)
        layout.addLayout(conf_layout)

        self.object_conf_slider.valueChanged.connect(
            lambda v: self.object_conf_label.setText(f"{v/100:.2f}")
        )

        # Custom model path
        layout.addWidget(QLabel("Custom YOLO Model (optional):"))
        help_label = QLabel("Leave categories unchecked to detect ANY object from custom model")
        help_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(help_label)
        model_layout = QHBoxLayout()
        self.custom_model_path_edit = QLineEdit()
        self.custom_model_path_edit.setPlaceholderText("Path to .pt, .pth, .safetensors, .weights, or .onnx model")
        self.custom_model_path_edit.textChanged.connect(self.update_filter_settings)
        model_layout.addWidget(self.custom_model_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_custom_model)
        model_layout.addWidget(browse_btn)

        clear_model_btn = QPushButton("Clear")
        clear_model_btn.clicked.connect(lambda: self.custom_model_path_edit.clear())
        model_layout.addWidget(clear_model_btn)
        layout.addLayout(model_layout)

        group.setLayout(layout)
        return group

    def create_scene_filter_section(self):
        """Create scene recognition filter section."""
        group = QGroupBox("Scene Recognition (Places365)")
        layout = QVBoxLayout()

        self.scene_enable_cb = QCheckBox("Filter by scene type")
        self.scene_enable_cb.stateChanged.connect(self.update_filter_settings)
        layout.addWidget(self.scene_enable_cb)

        # Contains/Does not contain
        mode_layout = QHBoxLayout()
        self.scene_contains_rb = QRadioButton("Contains")
        self.scene_contains_rb.setChecked(True)
        self.scene_contains_rb.toggled.connect(self.update_filter_settings)
        self.scene_not_contains_rb = QRadioButton("Does NOT contain")
        self.scene_not_contains_rb.toggled.connect(self.update_filter_settings)
        mode_layout.addWidget(self.scene_contains_rb)
        mode_layout.addWidget(self.scene_not_contains_rb)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Scene groups
        layout.addWidget(QLabel("Select scene types:"))
        self.scene_group_cbs = {}
        for group_name in SCENE_GROUPS.keys():
            cb = QCheckBox(group_name.replace('_', ' ').title())
            cb.stateChanged.connect(self.on_scene_group_changed)
            self.scene_group_cbs[group_name] = cb
            layout.addWidget(cb)

        # Add indoor/outdoor shortcuts
        layout.addWidget(QLabel("Quick selections:"))
        quick_layout = QHBoxLayout()
        self.indoor_btn = QPushButton("Indoor Scenes")
        self.indoor_btn.clicked.connect(self.select_indoor_scenes)
        self.outdoor_btn = QPushButton("Outdoor Scenes")
        self.outdoor_btn.clicked.connect(self.select_outdoor_scenes)
        quick_layout.addWidget(self.indoor_btn)
        quick_layout.addWidget(self.outdoor_btn)
        layout.addLayout(quick_layout)

        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.scene_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.scene_conf_slider.setRange(10, 90)
        self.scene_conf_slider.setValue(30)
        self.scene_conf_slider.valueChanged.connect(self.update_filter_settings)
        self.scene_conf_label = QLabel("0.30")
        conf_layout.addWidget(self.scene_conf_slider)
        conf_layout.addWidget(self.scene_conf_label)
        layout.addLayout(conf_layout)

        self.scene_conf_slider.valueChanged.connect(
            lambda v: self.scene_conf_label.setText(f"{v/100:.2f}")
        )

        group.setLayout(layout)
        return group

    def create_quality_filter_section(self):
        """Create quality assessment filter section."""
        group = QGroupBox("Image Quality")
        layout = QVBoxLayout()

        self.quality_enable_cb = QCheckBox("Filter by quality metrics")
        self.quality_enable_cb.stateChanged.connect(self.update_filter_settings)
        layout.addWidget(self.quality_enable_cb)

        # Meets/Does not meet criteria
        mode_layout = QHBoxLayout()
        self.quality_meets_rb = QRadioButton("Meets criteria")
        self.quality_meets_rb.setChecked(True)
        self.quality_meets_rb.toggled.connect(self.update_filter_settings)
        self.quality_not_meets_rb = QRadioButton("Does NOT meet criteria")
        self.quality_not_meets_rb.toggled.connect(self.update_filter_settings)
        mode_layout.addWidget(self.quality_meets_rb)
        mode_layout.addWidget(self.quality_not_meets_rb)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Overall quality score
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("Min overall score:"))
        self.quality_score_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_score_slider.setRange(0, 100)
        self.quality_score_slider.setValue(60)
        self.quality_score_slider.valueChanged.connect(self.update_filter_settings)
        self.quality_score_label = QLabel("60")
        score_layout.addWidget(self.quality_score_slider)
        score_layout.addWidget(self.quality_score_label)
        layout.addLayout(score_layout)

        self.quality_score_slider.valueChanged.connect(
            lambda v: self.quality_score_label.setText(str(v))
        )

        # Sharpness
        sharp_layout = QHBoxLayout()
        sharp_layout.addWidget(QLabel("Min sharpness:"))
        self.quality_sharp_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_sharp_slider.setRange(0, 100)
        self.quality_sharp_slider.setValue(40)
        self.quality_sharp_slider.valueChanged.connect(self.update_filter_settings)
        self.quality_sharp_label = QLabel("40")
        sharp_layout.addWidget(self.quality_sharp_slider)
        sharp_layout.addWidget(self.quality_sharp_label)
        layout.addLayout(sharp_layout)

        self.quality_sharp_slider.valueChanged.connect(
            lambda v: self.quality_sharp_label.setText(str(v))
        )

        # Brightness range
        layout.addWidget(QLabel("Brightness range:"))
        bright_layout = QHBoxLayout()
        bright_layout.addWidget(QLabel("Min:"))
        self.quality_bright_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_bright_min_slider.setRange(0, 255)
        self.quality_bright_min_slider.setValue(50)
        self.quality_bright_min_slider.valueChanged.connect(self.update_filter_settings)
        self.quality_bright_min_label = QLabel("50")
        bright_layout.addWidget(self.quality_bright_min_slider)
        bright_layout.addWidget(self.quality_bright_min_label)

        bright_layout.addWidget(QLabel("Max:"))
        self.quality_bright_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_bright_max_slider.setRange(0, 255)
        self.quality_bright_max_slider.setValue(200)
        self.quality_bright_max_slider.valueChanged.connect(self.update_filter_settings)
        self.quality_bright_max_label = QLabel("200")
        bright_layout.addWidget(self.quality_bright_max_slider)
        bright_layout.addWidget(self.quality_bright_max_label)
        layout.addLayout(bright_layout)

        self.quality_bright_min_slider.valueChanged.connect(
            lambda v: self.quality_bright_min_label.setText(str(v))
        )
        self.quality_bright_max_slider.valueChanged.connect(
            lambda v: self.quality_bright_max_label.setText(str(v))
        )

        group.setLayout(layout)
        return group

    def create_filesize_filter_section(self):
        """Create file size filter section."""
        group = QGroupBox("File Size")
        layout = QVBoxLayout()

        self.filesize_enable_cb = QCheckBox("Filter by file size")
        self.filesize_enable_cb.stateChanged.connect(self.update_filter_settings)
        layout.addWidget(self.filesize_enable_cb)

        # Min file size
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min size (KB):"))
        self.filesize_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.filesize_min_slider.setRange(0, 10240)  # 0 to 10MB in KB
        self.filesize_min_slider.setValue(0)
        self.filesize_min_slider.valueChanged.connect(self.update_filter_settings)
        self.filesize_min_label = QLabel("0")
        min_layout.addWidget(self.filesize_min_slider)
        min_layout.addWidget(self.filesize_min_label)
        layout.addLayout(min_layout)

        self.filesize_min_slider.valueChanged.connect(
            lambda v: self.filesize_min_label.setText(self._format_size_label(v))
        )

        # Max file size
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max size (KB):"))
        self.filesize_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.filesize_max_slider.setRange(0, 10240)  # 0 to 10MB in KB
        self.filesize_max_slider.setValue(10240)
        self.filesize_max_slider.valueChanged.connect(self.update_filter_settings)
        self.filesize_max_label = QLabel(self._format_size_label(10240))
        max_layout.addWidget(self.filesize_max_slider)
        max_layout.addWidget(self.filesize_max_label)
        layout.addLayout(max_layout)

        self.filesize_max_slider.valueChanged.connect(
            lambda v: self.filesize_max_label.setText(self._format_size_label(v))
        )

        group.setLayout(layout)
        return group

    def _format_size_label(self, kb):
        """Format file size label from KB value."""
        if kb < 1024:
            return f"{kb}KB"
        else:
            return f"{kb / 1024:.1f}MB"

    def browse_custom_model(self):
        """Browse for custom YOLO model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom YOLO Model",
            "",
            "YOLO Models (*.pt *.pth *.safetensors *.weights *.onnx);;All Files (*)"
        )
        if file_path:
            self.custom_model_path_edit.setText(file_path)

    def select_indoor_scenes(self):
        """Quick select indoor scene types."""
        indoor_groups = ["indoor_home", "indoor_commercial", "entertainment"]
        for group_name, cb in self.scene_group_cbs.items():
            cb.setChecked(group_name in indoor_groups)
        self.update_filter_settings()

    def select_outdoor_scenes(self):
        """Quick select outdoor scene types."""
        outdoor_groups = ["outdoor_nature", "outdoor_urban"]
        for group_name, cb in self.scene_group_cbs.items():
            cb.setChecked(group_name in outdoor_groups)
        self.update_filter_settings()

    def on_object_category_changed(self):
        """Auto-enable object detection when a category is selected."""
        any_checked = any(cb.isChecked() for cb in self.object_category_cbs.values())
        if any_checked and not self.object_enable_cb.isChecked():
            self.object_enable_cb.setChecked(True)
        self.update_filter_settings()

    def on_scene_group_changed(self):
        """Auto-enable scene recognition when a group is selected."""
        any_checked = any(cb.isChecked() for cb in self.scene_group_cbs.values())
        if any_checked and not self.scene_enable_cb.isChecked():
            self.scene_enable_cb.setChecked(True)
        self.update_filter_settings()

    def update_filter_settings(self):
        """Update filter settings from UI."""
        # Global invert
        self.filter_settings['global_invert'] = self.global_invert_cb.isChecked()

        # Face detection
        self.filter_settings['face_detection'] = self.face_enable_cb.isChecked()
        self.filter_settings['face_count_min'] = self.face_count_slider.value()
        self.filter_settings['face_inverse'] = self.face_not_contains_rb.isChecked()

        # Object detection
        self.filter_settings['object_detection'] = self.object_enable_cb.isChecked()
        self.filter_settings['object_categories'] = [
            cat for cat, cb in self.object_category_cbs.items() if cb.isChecked()
        ]
        self.filter_settings['object_confidence'] = self.object_conf_slider.value() / 100
        self.filter_settings['object_inverse'] = self.object_not_contains_rb.isChecked()
        custom_path = self.custom_model_path_edit.text().strip()
        self.filter_settings['custom_model_path'] = custom_path if custom_path else None

        # Scene recognition
        self.filter_settings['scene_recognition'] = self.scene_enable_cb.isChecked()
        self.filter_settings['scene_groups'] = [
            group for group, cb in self.scene_group_cbs.items() if cb.isChecked()
        ]
        self.filter_settings['scene_confidence'] = self.scene_conf_slider.value() / 100
        self.filter_settings['scene_inverse'] = self.scene_not_contains_rb.isChecked()

        # Quality
        self.filter_settings['quality_filter'] = self.quality_enable_cb.isChecked()
        self.filter_settings['quality_min_score'] = self.quality_score_slider.value()
        self.filter_settings['quality_sharpness_min'] = self.quality_sharp_slider.value()
        self.filter_settings['quality_brightness_min'] = self.quality_bright_min_slider.value()
        self.filter_settings['quality_brightness_max'] = self.quality_bright_max_slider.value()
        self.filter_settings['quality_inverse'] = self.quality_not_meets_rb.isChecked()

        # File size
        self.filter_settings['filesize_filter'] = self.filesize_enable_cb.isChecked()
        self.filter_settings['filesize_min_kb'] = self.filesize_min_slider.value()
        self.filter_settings['filesize_max_kb'] = self.filesize_max_slider.value()

    def clear_all_filters(self):
        """Clear all filter selections."""
        # Face detection
        self.face_enable_cb.setChecked(False)
        self.face_count_slider.setValue(0)

        # Object detection
        self.object_enable_cb.setChecked(False)
        for cb in self.object_category_cbs.values():
            cb.setChecked(False)
        self.object_conf_slider.setValue(50)

        # Scene recognition
        self.scene_enable_cb.setChecked(False)
        for cb in self.scene_group_cbs.values():
            cb.setChecked(False)
        self.scene_conf_slider.setValue(30)

        # Quality
        self.quality_enable_cb.setChecked(False)
        self.quality_score_slider.setValue(60)
        self.quality_sharp_slider.setValue(40)
        self.quality_bright_min_slider.setValue(50)
        self.quality_bright_max_slider.setValue(200)

        # File size
        self.filesize_enable_cb.setChecked(False)
        self.filesize_min_slider.setValue(0)
        self.filesize_max_slider.setValue(10240)

        self.update_filter_settings()

    def get_filter_settings(self):
        """Get current filter settings."""
        return self.filter_settings.copy()

    def has_active_filters(self):
        """Check if any filters are enabled."""
        return (self.filter_settings['face_detection'] or
                self.filter_settings['object_detection'] or
                self.filter_settings['scene_recognition'] or
                self.filter_settings['quality_filter'] or
                self.filter_settings['filesize_filter'])

    def set_filter_settings(self, settings):
        """Restore filter settings from a previous session."""
        if not settings:
            return

        # Global invert
        self.global_invert_cb.setChecked(settings.get('global_invert', False))

        # Face detection
        if settings.get('face_detection', False):
            self.face_enable_cb.setChecked(True)
            self.face_count_slider.setValue(settings.get('face_count_min', 0))
            if settings.get('face_inverse', False):
                self.face_not_contains_rb.setChecked(True)
            else:
                self.face_contains_rb.setChecked(True)

        # Object detection
        if settings.get('object_detection', False):
            self.object_enable_cb.setChecked(True)
            categories = settings.get('object_categories', [])
            for category, cb in self.object_category_cbs.items():
                cb.setChecked(category in categories)
            conf = int(settings.get('object_confidence', 0.5) * 100)
            self.object_conf_slider.setValue(conf)
            if settings.get('object_inverse', False):
                self.object_not_contains_rb.setChecked(True)
            else:
                self.object_contains_rb.setChecked(True)
            custom_path = settings.get('custom_model_path')
            if custom_path:
                self.custom_model_path_edit.setText(custom_path)

        # Scene recognition
        if settings.get('scene_recognition', False):
            self.scene_enable_cb.setChecked(True)
            groups = settings.get('scene_groups', [])
            for group, cb in self.scene_group_cbs.items():
                cb.setChecked(group in groups)
            conf = int(settings.get('scene_confidence', 0.3) * 100)
            self.scene_conf_slider.setValue(conf)
            if settings.get('scene_inverse', False):
                self.scene_not_contains_rb.setChecked(True)
            else:
                self.scene_contains_rb.setChecked(True)

        # Quality
        if settings.get('quality_filter', False):
            self.quality_enable_cb.setChecked(True)
            self.quality_score_slider.setValue(settings.get('quality_min_score', 60))
            self.quality_sharp_slider.setValue(settings.get('quality_sharpness_min', 40))
            self.quality_bright_min_slider.setValue(settings.get('quality_brightness_min', 50))
            self.quality_bright_max_slider.setValue(settings.get('quality_brightness_max', 200))
            if settings.get('quality_inverse', False):
                self.quality_not_meets_rb.setChecked(True)
            else:
                self.quality_meets_rb.setChecked(True)

        # File size
        if settings.get('filesize_filter', False):
            self.filesize_enable_cb.setChecked(True)
            self.filesize_min_slider.setValue(settings.get('filesize_min_kb', 0))
            self.filesize_max_slider.setValue(settings.get('filesize_max_kb', 10240))

        # Trigger update to sync filter_settings dict
        self.update_filter_settings()

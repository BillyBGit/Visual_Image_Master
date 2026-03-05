"""
YOLO label panel widget for managing class labels and output settings.
"""

import os
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QLineEdit,
    QScrollArea,
    QColorDialog,
    QFileDialog,
    QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor


# Default colors for new labels (will cycle through these)
DEFAULT_COLORS = [
    QColor(255, 0, 0),  # Red
    QColor(0, 255, 0),  # Green
    QColor(0, 0, 255),  # Blue
    QColor(255, 255, 0),  # Yellow
    QColor(255, 0, 255),  # Magenta
    QColor(0, 255, 255),  # Cyan
    QColor(255, 128, 0),  # Orange
    QColor(128, 0, 255),  # Purple
    QColor(0, 128, 255),  # Sky Blue
    QColor(255, 0, 128),  # Pink
]


class LabelRow(QFrame):
    """A single row representing a class label with color, ID, and name."""

    # Signals
    selected = Signal(int)  # Emitted when this label is selected (passes class_id)
    color_changed = Signal(int, QColor)  # Emitted when color changes
    name_changed = Signal(int, str)  # Emitted when name changes
    delete_requested = Signal(int)  # Emitted when delete is requested

    def __init__(self, class_id, color, name="", parent=None):
        super().__init__(parent)
        self.class_id = class_id
        self.color = color
        self.is_selected = False

        self._setup_ui(name)
        self._update_style()

    def _setup_ui(self, name):
        """Setup the row UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Color box (clickable)
        self.color_box = QLabel()
        self.color_box.setFixedSize(24, 24)
        self.color_box.setCursor(Qt.CursorShape.PointingHandCursor)
        self.color_box.setToolTip("Click to select, or use color button to change")
        self._update_color_box()
        layout.addWidget(self.color_box)

        # Color picker button
        self.color_btn = QPushButton("⚙")
        self.color_btn.setFixedSize(20, 20)
        self.color_btn.setToolTip("Change color")
        self.color_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.color_btn.clicked.connect(self._open_color_picker)
        layout.addWidget(self.color_btn)

        # Class ID (read-only)
        self.id_label = QLabel(str(self.class_id))
        self.id_label.setFixedWidth(24)
        self.id_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.id_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.id_label)

        # Label name (editable)
        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText(f"Class {self.class_id}")
        self.name_edit.textChanged.connect(self._on_name_changed)
        layout.addWidget(self.name_edit, 1)

        # Delete button (red X)
        self.delete_btn = QPushButton("×")
        self.delete_btn.setFixedSize(20, 20)
        self.delete_btn.setToolTip("Delete this label")
        self.delete_btn.setStyleSheet(
            "QPushButton { color: #cc0000; font-weight: bold; font-size: 14px; }"
            "QPushButton:hover { background-color: #ffcccc; color: #990000; }"
        )
        self.delete_btn.clicked.connect(
            lambda: self.delete_requested.emit(self.class_id)
        )
        layout.addWidget(self.delete_btn)

    def _update_color_box(self):
        """Update the color box appearance."""
        self.color_box.setStyleSheet(
            f"background-color: {self.color.name()}; "
            f"border: 2px solid {'white' if self.is_selected else 'gray'}; "
            f"border-radius: 3px;"
        )

    def _update_style(self):
        """Update the row style based on selection state."""
        if self.is_selected:
            self.setStyleSheet(
                "LabelRow { background-color: rgba(100, 100, 255, 50); "
                "border: 1px solid #6666ff; border-radius: 4px; }"
            )
        else:
            self.setStyleSheet(
                "LabelRow { background-color: transparent; "
                "border: 1px solid transparent; border-radius: 4px; }"
                "LabelRow:hover { background-color: rgba(100, 100, 100, 30); }"
            )
        self._update_color_box()

    def set_selected(self, selected):
        """Set the selection state of this row."""
        self.is_selected = selected
        self._update_style()

    def set_color(self, color):
        """Set the color for this label."""
        self.color = color
        self._update_color_box()

    def get_name(self):
        """Get the label name."""
        name = self.name_edit.text().strip()
        return name if name else f"class_{self.class_id}"

    def _on_name_changed(self, text):
        """Handle name text changes."""
        self.name_changed.emit(self.class_id, text)

    def _open_color_picker(self):
        """Open color picker dialog."""
        new_color = QColorDialog.getColor(
            self.color,
            self.window(),
            f"Choose color for {self.get_name()}",
            QColorDialog.ColorDialogOption.DontUseNativeDialog,
        )
        if new_color.isValid():
            self.color = new_color
            self._update_color_box()
            self.color_changed.emit(self.class_id, new_color)

    def mousePressEvent(self, event):
        """Handle mouse press for selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Left-click anywhere on the row (except buttons) selects it
            self.selected.emit(self.class_id)
        elif event.button() == Qt.MouseButton.RightButton:
            # Right-click on color box opens color picker (for discoverability)
            color_box_rect = self.color_box.geometry()
            if color_box_rect.contains(event.pos()):
                self._open_color_picker()

        super().mousePressEvent(event)


class YoloLabelPanel(QWidget):
    """Panel for managing YOLO labels and output settings."""

    # Signals
    label_selected = Signal(int, QColor, str)  # class_id, color, name
    label_deleted = Signal(int)  # class_id - emitted when a label is deleted
    labels_renumbered = Signal(dict)  # old_id -> new_id mapping after renumber
    labels_changed = Signal()  # Emitted when labels list changes
    output_path_changed = Signal(str)  # Emitted when output path changes
    save_requested = Signal()  # Emitted when save button clicked
    load_directory_requested = Signal()  # Emitted when load directory button clicked
    refresh_files_requested = Signal()  # Emitted when refresh files button clicked

    def __init__(self, parent=None):
        super().__init__(parent)

        # Label data: list of dicts with 'class_id', 'color', 'name'
        self.labels = []
        self.selected_class_id = None
        self.label_rows = {}  # class_id -> LabelRow widget

        # Output path
        self.output_path = ""

        self._setup_ui()

    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Labels group
        labels_group = QGroupBox("Labels")
        labels_layout = QVBoxLayout(labels_group)

        # Scroll area for label rows
        self.labels_scroll = QScrollArea()
        self.labels_scroll.setWidgetResizable(True)
        self.labels_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.labels_scroll.setMinimumHeight(150)

        # Container for label rows
        self.labels_container = QWidget()
        self.labels_container_layout = QVBoxLayout(self.labels_container)
        self.labels_container_layout.setContentsMargins(4, 4, 4, 4)
        self.labels_container_layout.setSpacing(4)
        self.labels_container_layout.addStretch()

        self.labels_scroll.setWidget(self.labels_container)
        labels_layout.addWidget(self.labels_scroll)

        # Add/Delete buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("+ Add Label")
        self.add_btn.clicked.connect(lambda: self.add_label())
        btn_layout.addWidget(self.add_btn)

        self.delete_all_btn = QPushButton("Clear All")
        self.delete_all_btn.clicked.connect(self.clear_labels)
        btn_layout.addWidget(self.delete_all_btn)

        labels_layout.addLayout(btn_layout)
        layout.addWidget(labels_group)

        # Output path group
        path_group = QGroupBox("Output Path")
        path_layout = QVBoxLayout(path_group)

        # Path entry with browse button
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select output directory...")
        self.path_edit.textChanged.connect(self._on_path_changed)
        path_row.addWidget(self.path_edit, 1)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_output_path)
        path_row.addWidget(self.browse_btn)

        path_layout.addLayout(path_row)
        layout.addWidget(path_group)

        # Save and utility buttons row
        save_row = QHBoxLayout()

        # Save button (larger, takes 2x space)
        self.save_btn = QPushButton("Save Annotation")
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14px; }"
            "QPushButton:enabled { background-color: #4CAF50; color: white; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.save_btn.clicked.connect(self._on_save_clicked)
        save_row.addWidget(self.save_btn, 2)  # Stretch factor 2

        # Load Directory button
        self.load_dir_btn = QPushButton("Load Dir")
        self.load_dir_btn.setMinimumHeight(40)
        self.load_dir_btn.setToolTip("Reload all annotations from the output directory")
        self.load_dir_btn.clicked.connect(self._on_load_directory_clicked)
        save_row.addWidget(self.load_dir_btn, 1)  # Stretch factor 1

        # Refresh Files button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setMinimumHeight(40)
        self.refresh_btn.setToolTip("Update all label files to match current class IDs")
        self.refresh_btn.clicked.connect(self._on_refresh_files_clicked)
        save_row.addWidget(self.refresh_btn, 1)  # Stretch factor 1

        layout.addLayout(save_row)

        # Initial state
        self._update_button_states()

    def add_label(self, name="", pick_color=True):
        """Add a new label with the next available class ID."""
        # Ensure name is a string (in case boolean is passed from signal)
        if not isinstance(name, str):
            name = ""

        # Find next available class ID
        existing_ids = set(label["class_id"] for label in self.labels)
        class_id = 0
        while class_id in existing_ids:
            class_id += 1

        # Get color from default list (cycle through) or let user pick
        default_color = DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)]

        if pick_color:
            # Open color picker for new label using static method for reliability
            color = QColorDialog.getColor(
                default_color,
                self.window(),
                f"Choose color for new label (class {class_id})",
                QColorDialog.ColorDialogOption.DontUseNativeDialog,
            )
            if not color.isValid():
                # User cancelled - don't add the label
                return None
        else:
            color = default_color

        # Create label data
        label_data = {"class_id": class_id, "color": color, "name": name}
        self.labels.append(label_data)

        # Create and add row widget
        row = LabelRow(class_id, color, name)
        row.selected.connect(self._on_label_selected)
        row.color_changed.connect(self._on_label_color_changed)
        row.name_changed.connect(self._on_label_name_changed)
        row.delete_requested.connect(self.delete_label)

        self.label_rows[class_id] = row

        # Insert before the stretch at the end
        self.labels_container_layout.insertWidget(
            self.labels_container_layout.count() - 1, row
        )

        # Ensure the row is visible
        row.show()

        # Auto-select the new label
        self._on_label_selected(class_id)

        self._update_button_states()
        self.labels_changed.emit()

        return class_id

    def delete_label(self, class_id):
        """Delete a label by class ID and renumber remaining labels."""
        # Remove from labels list
        self.labels = [lbl for lbl in self.labels if lbl["class_id"] != class_id]

        # Remove widget
        if class_id in self.label_rows:
            row = self.label_rows.pop(class_id)
            self.labels_container_layout.removeWidget(row)
            row.deleteLater()

        # Clear selection if this was selected
        if self.selected_class_id == class_id:
            self.selected_class_id = None

        self._update_button_states()
        self.label_deleted.emit(class_id)  # Notify to remove annotations

        # Renumber remaining labels to maintain 0-indexed sequence
        self._renumber_labels()

        # Select first label if available
        if self.labels and self.selected_class_id is None:
            self._on_label_selected(self.labels[0]["class_id"])

        self.labels_changed.emit()

    def _renumber_labels(self):
        """Renumber all labels to maintain 0-indexed sequence with no gaps.

        Returns:
            dict mapping old_class_id -> new_class_id for any IDs that changed
        """
        # Sort labels by current class_id
        sorted_labels = sorted(self.labels, key=lambda x: x["class_id"])

        # Build mapping of old -> new IDs
        id_mapping = {}
        for new_id, label in enumerate(sorted_labels):
            old_id = label["class_id"]
            if old_id != new_id:
                id_mapping[old_id] = new_id

        if not id_mapping:
            return {}  # No renumbering needed

        # Update labels list
        for new_id, label in enumerate(sorted_labels):
            label["class_id"] = new_id

        self.labels = sorted_labels

        # Update label rows - need to rebuild with new IDs
        old_rows = self.label_rows.copy()
        self.label_rows = {}

        for label in self.labels:
            new_id = label["class_id"]
            # Find the row that had the original ID for this label
            old_id = None
            for oid, nid in id_mapping.items():
                if nid == new_id:
                    old_id = oid
                    break
            if old_id is None:
                old_id = new_id  # ID didn't change

            if old_id in old_rows:
                row = old_rows[old_id]
                # Update the row's class_id
                row.class_id = new_id
                row.id_label.setText(str(new_id))
                row.name_edit.setPlaceholderText(f"Class {new_id}")
                self.label_rows[new_id] = row

        # Update selected class ID if it was renumbered
        if self.selected_class_id in id_mapping:
            self.selected_class_id = id_mapping[self.selected_class_id]

        # Emit signal so yolo_window can update label files
        if id_mapping:
            self.labels_renumbered.emit(id_mapping)

        return id_mapping

    def get_id_mapping_for_renumber(self):
        """Get what the ID mapping would be if labels were renumbered now.

        This is useful for previewing changes before committing.

        Returns:
            dict mapping old_class_id -> new_class_id
        """
        sorted_labels = sorted(self.labels, key=lambda x: x["class_id"])
        id_mapping = {}
        for new_id, label in enumerate(sorted_labels):
            old_id = label["class_id"]
            if old_id != new_id:
                id_mapping[old_id] = new_id
        return id_mapping

    def clear_labels(self):
        """Clear all labels."""
        # Emit deletion signal for each label so annotations are removed
        class_ids_to_delete = list(self.label_rows.keys())

        for class_id in class_ids_to_delete:
            row = self.label_rows.pop(class_id)
            self.labels_container_layout.removeWidget(row)
            row.deleteLater()
            self.label_deleted.emit(class_id)  # Remove annotations for this class

        self.labels = []
        self.selected_class_id = None
        self._update_button_states()
        self.labels_changed.emit()

    def _on_label_selected(self, class_id):
        """Handle label selection."""
        # Deselect previous
        if (
            self.selected_class_id is not None
            and self.selected_class_id in self.label_rows
        ):
            self.label_rows[self.selected_class_id].set_selected(False)

        # Select new
        self.selected_class_id = class_id
        if class_id in self.label_rows:
            row = self.label_rows[class_id]
            row.set_selected(True)

            # Find label data
            for label in self.labels:
                if label["class_id"] == class_id:
                    self.label_selected.emit(class_id, label["color"], row.get_name())
                    break

    def _on_label_color_changed(self, class_id, color):
        """Handle label color change."""
        for label in self.labels:
            if label["class_id"] == class_id:
                label["color"] = color
                break

        # Re-emit selected signal if this is the selected label
        if self.selected_class_id == class_id:
            row = self.label_rows[class_id]
            self.label_selected.emit(class_id, color, row.get_name())

        self.labels_changed.emit()

    def _on_label_name_changed(self, class_id, name):
        """Handle label name change."""
        for label in self.labels:
            if label["class_id"] == class_id:
                label["name"] = name
                break

        self.labels_changed.emit()

    def _on_path_changed(self, path):
        """Handle output path change."""
        self.output_path = path
        self._update_button_states()
        self.output_path_changed.emit(path)

    def _browse_output_path(self):
        """Open directory browser for output path."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select YOLO Output Directory",
            self.output_path or os.path.expanduser("~"),
        )
        if path:
            self.path_edit.setText(path)

    def _on_save_clicked(self):
        """Handle save button click."""
        self.save_requested.emit()

    def _on_load_directory_clicked(self):
        """Handle load directory button click."""
        self.load_directory_requested.emit()

    def _on_refresh_files_clicked(self):
        """Handle refresh files button click."""
        self.refresh_files_requested.emit()

    def _update_button_states(self):
        """Update button enabled states based on current state."""
        has_labels = len(self.labels) > 0
        has_path = bool(self.output_path and os.path.isdir(self.output_path))

        self.delete_all_btn.setEnabled(has_labels)
        self.save_btn.setEnabled(has_labels and has_path)
        self.load_dir_btn.setEnabled(has_path)
        self.refresh_btn.setEnabled(has_labels and has_path)

        # Update save button tooltip
        if not has_labels:
            self.save_btn.setToolTip("Add at least one label first")
        elif not has_path:
            self.save_btn.setToolTip("Select a valid output directory first")
        else:
            self.save_btn.setToolTip("Save annotation (Ctrl+S)")

    def get_selected_label(self):
        """Get the currently selected label data, or None."""
        if self.selected_class_id is None:
            return None

        for label in self.labels:
            if label["class_id"] == self.selected_class_id:
                row = self.label_rows.get(self.selected_class_id)
                return {
                    "class_id": label["class_id"],
                    "color": label["color"],
                    "name": row.get_name() if row else label.get("name", ""),
                }
        return None

    def get_all_labels(self):
        """Get all labels with their current names."""
        result = []
        for label in self.labels:
            row = self.label_rows.get(label["class_id"])
            result.append(
                {
                    "class_id": label["class_id"],
                    "color": label["color"],
                    "name": row.get_name() if row else label.get("name", ""),
                }
            )
        return result

    def set_labels(self, labels):
        """Set labels from saved data."""
        self.clear_labels()

        for label in labels:
            class_id = label["class_id"]
            color = label.get("color", DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)])
            if isinstance(color, str):
                color = QColor(color)
            name = label.get("name", "")

            # Create label data
            label_data = {"class_id": class_id, "color": color, "name": name}
            self.labels.append(label_data)

            # Create row widget
            row = LabelRow(class_id, color, name)
            row.selected.connect(self._on_label_selected)
            row.color_changed.connect(self._on_label_color_changed)
            row.name_changed.connect(self._on_label_name_changed)
            row.delete_requested.connect(self.delete_label)

            self.label_rows[class_id] = row
            self.labels_container_layout.insertWidget(
                self.labels_container_layout.count() - 1, row
            )

        # Select first label if available
        if self.labels:
            self._on_label_selected(self.labels[0]["class_id"])

        self._update_button_states()

    def set_output_path(self, path):
        """Set the output path."""
        self.path_edit.setText(path or "")

    def select_label_by_id(self, class_id):
        """Select a label by its class ID."""
        if class_id in self.label_rows:
            self._on_label_selected(class_id)
            return True
        return False

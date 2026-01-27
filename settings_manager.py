"""
Settings manager for persisting application state.
"""

import os
import json
from PySide6.QtCore import QSettings


class SettingsManager:
    """Handles reading and writing application settings."""

    def __init__(self, app_instance):
        self.app = app_instance
        self.settings = QSettings("MyCompany", "ImageSorter")

    def write_settings(self):
        """Persist all application settings to disk."""
        # Don't write settings during initial load
        if not self.app._settings_loaded:
            return

        # Save toggle states
        self.settings.setValue("remember_settings", self.app.remember_settings_act.isChecked())
        self.settings.setValue("preview_fit", self.app.preview_fit_default)
        self.settings.setValue("preview_autoname_extract", self.app.preview_autoname_extract)
        self.settings.setValue("confirm_delete", self.app.conf_delete_act.isChecked())
        self.settings.setValue("show_names", self.app.show_names_act.isChecked())
        self.settings.setValue("show_dimensions", self.app.show_dimensions_act.isChecked())
        self.settings.setValue("show_counts", self.app.show_counts_act.isChecked())

        # Save sort settings
        self.settings.setValue("sort_by", self.app.sort_by)
        self.settings.setValue("sort_ascending", self.app.sort_ascending)

        # Save filter settings
        self.settings.setValue("filter_active", self.app.filter_active)
        self.settings.setValue("filter_mode", self.app.filter_mode if self.app.filter_mode else "")

        # Save face detection preset
        self.settings.setValue("face_detection_preset", self.app.face_detection_preset)

        # Save folder tree zoom levels
        self.settings.setValue("src_tree_zoom", self.app.src_tree.zoom_level)
        self.settings.setValue("dest_tree_zoom", self.app.dest_tree.zoom_level)

        # Save main window geometry and position
        self.settings.setValue("main_geometry", self.app.saveGeometry())
        self.settings.setValue("main_window_state", self.app.saveState())

        # Save splitter state (pane dimensions)
        self.settings.setValue("main_splitter_state", self.app.main_splitter.saveState())

        # Save preview window geometry if it exists
        if hasattr(self.app, 'preview_geometry') and self.app.preview_geometry:
            self.settings.setValue("preview_geometry", self.app.preview_geometry)

        # Note: YOLO window geometry not saved - always opens maximized

        # Save YOLO labels and output path
        if hasattr(self.app, 'yolo_labels') and self.app.yolo_labels:
            # Convert QColor to string for storage
            labels_to_save = []
            for label in self.app.yolo_labels:
                labels_to_save.append({
                    'class_id': label['class_id'],
                    'color': label['color'].name() if hasattr(label['color'], 'name') else label['color'],
                    'name': label['name']
                })
            self.settings.setValue("yolo_labels", labels_to_save)

        if hasattr(self.app, 'yolo_output_path'):
            self.settings.setValue("yolo_output_path", self.app.yolo_output_path or "")

        if hasattr(self.app, 'yolo_resolution_index'):
            self.settings.setValue("yolo_resolution_index", self.app.yolo_resolution_index)

        if hasattr(self.app, 'yolo_size_mode'):
            self.settings.setValue("yolo_size_mode", self.app.yolo_size_mode)

        # Save caption settings
        if hasattr(self.app, 'caption_backend'):
            self.settings.setValue("caption_backend", self.app.caption_backend or "florence2")

        if hasattr(self.app, 'caption_ollama_url'):
            self.settings.setValue("caption_ollama_url", self.app.caption_ollama_url or "")

        if hasattr(self.app, 'caption_ollama_model'):
            self.settings.setValue("caption_ollama_model", self.app.caption_ollama_model or "")

        if hasattr(self.app, 'caption_florence_model'):
            self.settings.setValue("caption_florence_model", self.app.caption_florence_model or "base")

        if hasattr(self.app, 'caption_florence_detail'):
            self.settings.setValue("caption_florence_detail", self.app.caption_florence_detail or "detailed")

        if hasattr(self.app, 'caption_florence_custom_path'):
            self.settings.setValue("caption_florence_custom_path", self.app.caption_florence_custom_path or "")

        if hasattr(self.app, 'caption_wd14_model'):
            self.settings.setValue("caption_wd14_model", self.app.caption_wd14_model or "vit-v2")

        if hasattr(self.app, 'caption_wd14_threshold'):
            self.settings.setValue("caption_wd14_threshold", self.app.caption_wd14_threshold or 0.55)

        if hasattr(self.app, 'caption_wd14_custom_paths'):
            # Convert dict to JSON string for storage
            import json
            paths_json = json.dumps(self.app.caption_wd14_custom_paths or {})
            self.settings.setValue("caption_wd14_custom_paths", paths_json)

        if hasattr(self.app, 'caption_blend_instructions'):
            self.settings.setValue("caption_blend_instructions", self.app.caption_blend_instructions or "")

        if hasattr(self.app, 'caption_blend_model'):
            self.settings.setValue("caption_blend_model", self.app.caption_blend_model or "")

        # Enhancement settings
        if hasattr(self.app, 'enhance_model_dir'):
            self.settings.setValue("enhance_model_dir", self.app.enhance_model_dir or "")
        if hasattr(self.app, 'enhance_save_mode'):
            self.settings.setValue("enhance_save_mode", self.app.enhance_save_mode or "overwrite")
        if hasattr(self.app, 'enhance_copy_suffix'):
            self.settings.setValue("enhance_copy_suffix", self.app.enhance_copy_suffix or "_enhanced")
        if hasattr(self.app, 'enhance_last_operation'):
            self.settings.setValue("enhance_last_operation", self.app.enhance_last_operation or "sharpen")
        if hasattr(self.app, 'enhance_last_params'):
            self.settings.setValue("enhance_last_params", json.dumps(self.app.enhance_last_params or {}))

        # Save current folder paths
        self.settings.setValue("src_root_path", self.app.src_root_path if self.app.src_root_path else "/")
        self.settings.setValue("dest_root_path", self.app.dest_root_path if self.app.dest_root_path else "/")

        # Save the currently selected folder in source tree (for thumbnails)
        current_src_index = self.app.src_tree.currentIndex()
        if current_src_index.isValid():
            self.settings.setValue("src_selected_path", self.app.src_model.filePath(current_src_index))

        # Save the currently selected folder in dest tree
        current_dest_index = self.app.dest_tree.currentIndex()
        if current_dest_index.isValid():
            self.settings.setValue("dest_selected_path", self.app.dest_model.filePath(current_dest_index))

        self.settings.sync()

    def read_settings(self):
        """Restore application settings from disk."""
        # 1. Check the "Remember Settings" toggle first
        remember = self.settings.value("remember_settings", True, type=bool)
        self.app.remember_settings_act.setChecked(remember)

        if not remember:
            return

        # 2. Block signals during restoration to prevent premature saves
        self.app.src_tree.blockSignals(True)
        self.app.dest_tree.blockSignals(True)

        # 3. Restore Window Geometry and State
        main_geo = self.settings.value("main_geometry")
        main_state = self.settings.value("main_window_state")
        main_splitter_state = self.settings.value("main_splitter_state")

        if main_geo:
            self.app.restoreGeometry(main_geo)
        if main_state:
            self.app.restoreState(main_state)
        if main_splitter_state:
            self.app.main_splitter.restoreState(main_splitter_state)

        # 4. Restore Boolean Settings
        self.app.conf_delete_act.setChecked(self.settings.value("confirm_delete", True, type=bool))
        self.app.show_names_act.setChecked(self.settings.value("show_names", False, type=bool))
        self.app.show_dimensions_act.setChecked(self.settings.value("show_dimensions", False, type=bool))
        self.app.preview_fit_default = self.settings.value("preview_fit", False, type=bool)
        self.app.preview_autoname_extract = self.settings.value("preview_autoname_extract", False, type=bool)

        # Restore folder counts setting
        show_counts = self.settings.value("show_counts", False, type=bool)
        self.app.show_counts_act.setChecked(show_counts)
        self.app.src_tree.set_show_counts(show_counts)
        self.app.dest_tree.set_show_counts(show_counts)

        # Restore sort settings
        self.app.sort_by = self.settings.value("sort_by", "name")
        self.app.sort_ascending = self.settings.value("sort_ascending", True, type=bool)
        if hasattr(self.app, 'sort_actions'):
            for key, action in self.app.sort_actions.items():
                action.setChecked(key == (self.app.sort_by, self.app.sort_ascending))

        # Restore filter settings
        self.app.filter_active = self.settings.value("filter_active", False, type=bool)
        filter_mode_str = self.settings.value("filter_mode", "")
        self.app.filter_mode = filter_mode_str if filter_mode_str else None
        if hasattr(self.app, 'filter_actions'):
            for mode, action in self.app.filter_actions.items():
                action.setChecked(mode == self.app.filter_mode and self.app.filter_active)
        # Update visual indicators
        if hasattr(self.app, 'filter_btn'):
            self.app.update_filter_visuals()

        # Restore face detection preset
        self.app.face_detection_preset = self.settings.value("face_detection_preset", "balanced")

        # Restore folder tree zoom levels
        src_zoom = self.settings.value("src_tree_zoom", 1.0, type=float)
        dest_zoom = self.settings.value("dest_tree_zoom", 1.0, type=float)
        self.app.src_tree.zoom_level = max(0.8, min(2.0, src_zoom))
        self.app.dest_tree.zoom_level = max(0.8, min(2.0, dest_zoom))
        self.app.src_tree.apply_zoom()
        self.app.dest_tree.apply_zoom()

        # 5. Restore Preview Window Geometry
        self.app.preview_geometry = self.settings.value("preview_geometry")

        # Note: YOLO window geometry not restored - always opens maximized

        # 5b. Restore YOLO labels and output path
        self.app.yolo_labels = self.settings.value("yolo_labels", [])
        self.app.yolo_output_path = self.settings.value("yolo_output_path", "")
        self.app.yolo_resolution_index = self.settings.value("yolo_resolution_index", 0, type=int)
        self.app.yolo_size_mode = self.settings.value("yolo_size_mode", "image")

        # 5c. Restore Caption settings
        self.app.caption_backend = self.settings.value("caption_backend", "florence2")
        self.app.caption_ollama_url = self.settings.value("caption_ollama_url", "http://localhost:11434")
        self.app.caption_ollama_model = self.settings.value("caption_ollama_model", "llava")
        self.app.caption_florence_model = self.settings.value("caption_florence_model", "base")
        self.app.caption_florence_detail = self.settings.value("caption_florence_detail", "detailed")
        self.app.caption_florence_custom_path = self.settings.value("caption_florence_custom_path", "")
        self.app.caption_wd14_model = self.settings.value("caption_wd14_model", "vit-v2")
        self.app.caption_wd14_threshold = self.settings.value("caption_wd14_threshold", 0.55, type=float)

        # Load WD14 custom paths from JSON
        import json
        paths_json = self.settings.value("caption_wd14_custom_paths", "{}")
        try:
            self.app.caption_wd14_custom_paths = json.loads(paths_json)
        except (json.JSONDecodeError, TypeError):
            self.app.caption_wd14_custom_paths = {}

        # Load blend instructions
        self.app.caption_blend_instructions = self.settings.value("caption_blend_instructions", "")

        # Load blend model
        self.app.caption_blend_model = self.settings.value("caption_blend_model", "")

        # Load enhancement settings
        self.app.enhance_model_dir = self.settings.value("enhance_model_dir", "")
        self.app.enhance_save_mode = self.settings.value("enhance_save_mode", "overwrite")
        self.app.enhance_copy_suffix = self.settings.value("enhance_copy_suffix", "_enhanced")
        self.app.enhance_last_operation = self.settings.value("enhance_last_operation", "sharpen")
        params_json = self.settings.value("enhance_last_params", "{}")
        try:
            self.app.enhance_last_params = json.loads(params_json)
        except (json.JSONDecodeError, TypeError):
            self.app.enhance_last_params = {}

        # 6. Restore Folder Paths
        src_root = self.settings.value("src_root_path", os.path.expanduser("~"))
        dest_root = self.settings.value("dest_root_path", os.path.expanduser("~"))
        src_selected = self.settings.value("src_selected_path", src_root)
        dest_selected = self.settings.value("dest_selected_path", dest_root)

        # Validate and fallback for source root
        if not os.path.isdir(src_root):
            src_root = os.path.expanduser("~")
            if not os.path.isdir(src_root):
                src_root = "/"

        # Validate and fallback for destination root
        if not os.path.isdir(dest_root):
            dest_root = os.path.expanduser("~")
            if not os.path.isdir(dest_root):
                dest_root = "/"

        # Validate selected paths
        if not os.path.isdir(src_selected):
            src_selected = src_root
        if not os.path.isdir(dest_selected):
            dest_selected = dest_root

        # Set source tree root and selected path
        self.app.src_root_path = src_root
        self.app.src_path_display.setText(src_root)
        src_root_index = self.app.src_model.index(src_root)
        if src_root_index.isValid():
            self.app.src_tree.setRootIndex(src_root_index)

            src_sel_index = self.app.src_model.index(src_selected)
            if src_sel_index.isValid():
                self.app.src_tree.setCurrentIndex(src_sel_index)
                self.app.src_tree.scrollTo(src_sel_index)
        else:
            self.app.src_tree.setRootIndex(self.app.src_model.index("/"))
            self.app.src_path_display.setText("/")
            self.app.src_root_path = "/"

        # Set destination tree root and selected path
        self.app.dest_root_path = dest_root
        self.app.dest_path_display.setText(dest_root)
        dest_root_index = self.app.dest_model.index(dest_root)
        if dest_root_index.isValid():
            self.app.dest_tree.setRootIndex(dest_root_index)

            dest_sel_index = self.app.dest_model.index(dest_selected)
            if dest_sel_index.isValid():
                self.app.dest_tree.setCurrentIndex(dest_sel_index)
                self.app.dest_tree.scrollTo(dest_sel_index)
        else:
            self.app.dest_tree.setRootIndex(self.app.dest_model.index("/"))
            self.app.dest_path_display.setText("/")
            self.app.dest_root_path = "/"

        # 7. Unblock signals
        self.app.src_tree.blockSignals(False)
        self.app.dest_tree.blockSignals(False)

"""
FilteringMixin - Sort, filter, and dedup menu logic for ImageSorter.
"""

import os
from PySide6.QtWidgets import QMessageBox, QApplication, QListWidgetItem, QMenu
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QIcon
from PIL import Image

from advanced_filter_dialog import AdvancedFilterDialog
from composition_similarity import order_by_diversity
from object_detection import get_detector as get_object_detector
from quality_assessment import get_assessor as get_quality_assessor
from scene_recognition import get_recognizer as get_scene_recognizer

VALID_IMAGE_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".avif",
    ".tiff",
    ".bmp",
    ".tn3",
)


class FilteringMixin:
    """Mixin providing sort, filter, dedup menus, thumbnail refresh, and sorting logic."""

    def setup_sort_menu(self):
        """Create the sort menu with all sorting options."""
        name_asc_act = QAction("Name (A-Z)", self, checkable=True)
        name_asc_act.triggered.connect(lambda: self.set_sort("name", True))
        self.sort_menu.addAction(name_asc_act)

        name_desc_act = QAction("Name (Z-A)", self, checkable=True)
        name_desc_act.triggered.connect(lambda: self.set_sort("name", False))
        self.sort_menu.addAction(name_desc_act)

        self.sort_menu.addSeparator()

        modified_asc_act = QAction("Modified Date (Oldest First)", self, checkable=True)
        modified_asc_act.triggered.connect(lambda: self.set_sort("modified", True))
        self.sort_menu.addAction(modified_asc_act)

        modified_desc_act = QAction(
            "Modified Date (Newest First)", self, checkable=True
        )
        modified_desc_act.triggered.connect(lambda: self.set_sort("modified", False))
        self.sort_menu.addAction(modified_desc_act)

        self.sort_menu.addSeparator()

        res_asc_act = QAction("Resolution (Low to High)", self, checkable=True)
        res_asc_act.triggered.connect(lambda: self.set_sort("resolution", True))
        self.sort_menu.addAction(res_asc_act)

        res_desc_act = QAction("Resolution (High to Low)", self, checkable=True)
        res_desc_act.triggered.connect(lambda: self.set_sort("resolution", False))
        self.sort_menu.addAction(res_desc_act)

        self.sort_menu.addSeparator()

        width_asc_act = QAction("Width (Narrow to Wide)", self, checkable=True)
        width_asc_act.triggered.connect(lambda: self.set_sort("width", True))
        self.sort_menu.addAction(width_asc_act)

        width_desc_act = QAction("Width (Wide to Narrow)", self, checkable=True)
        width_desc_act.triggered.connect(lambda: self.set_sort("width", False))
        self.sort_menu.addAction(width_desc_act)

        self.sort_menu.addSeparator()

        height_asc_act = QAction("Height (Short to Tall)", self, checkable=True)
        height_asc_act.triggered.connect(lambda: self.set_sort("height", True))
        self.sort_menu.addAction(height_asc_act)

        height_desc_act = QAction("Height (Tall to Short)", self, checkable=True)
        height_desc_act.triggered.connect(lambda: self.set_sort("height", False))
        self.sort_menu.addAction(height_desc_act)

        self.sort_menu.addSeparator()

        # Quality/Blur sorting
        quality_asc_act = QAction("Quality (Blurry to Sharp)", self, checkable=True)
        quality_asc_act.triggered.connect(lambda: self.set_sort("quality", True))
        self.sort_menu.addAction(quality_asc_act)

        quality_desc_act = QAction("Quality (Sharp to Blurry)", self, checkable=True)
        quality_desc_act.triggered.connect(lambda: self.set_sort("quality", False))
        self.sort_menu.addAction(quality_desc_act)

        self.sort_menu.addSeparator()

        # Face area sorting (largest detected face / image area) — useful for
        # finding tightly-framed portrait candidates
        face_ratio_asc_act = QAction("Face Area % (Low to High)", self, checkable=True)
        face_ratio_asc_act.triggered.connect(lambda: self.set_sort("face_ratio", True))
        self.sort_menu.addAction(face_ratio_asc_act)

        face_ratio_desc_act = QAction("Face Area % (High to Low)", self, checkable=True)
        face_ratio_desc_act.triggered.connect(
            lambda: self.set_sort("face_ratio", False)
        )
        self.sort_menu.addAction(face_ratio_desc_act)

        self.sort_menu.addSeparator()

        # Composition diversity sorting — orders by structural (colour-excluded)
        # dissimilarity via greedy farthest-point sampling, so taking the first
        # N images gives a spread-out, non-redundant training subset
        comp_div_desc_act = QAction(
            "Composition Diversity (Most Diverse First)", self, checkable=True
        )
        comp_div_desc_act.triggered.connect(
            lambda: self.set_sort("composition_diversity", False)
        )
        self.sort_menu.addAction(comp_div_desc_act)

        comp_div_asc_act = QAction(
            "Composition Diversity (Most Redundant First)", self, checkable=True
        )
        comp_div_asc_act.triggered.connect(
            lambda: self.set_sort("composition_diversity", True)
        )
        self.sort_menu.addAction(comp_div_asc_act)

        self.sort_menu.addSeparator()

        # File size sorting
        size_asc_act = QAction("File Size (Smallest First)", self, checkable=True)
        size_asc_act.triggered.connect(lambda: self.set_sort("filesize", True))
        self.sort_menu.addAction(size_asc_act)

        size_desc_act = QAction("File Size (Largest First)", self, checkable=True)
        size_desc_act.triggered.connect(lambda: self.set_sort("filesize", False))
        self.sort_menu.addAction(size_desc_act)

        self.sort_menu.addSeparator()

        # Portrait quality: a weighted composite of sharpness, resolution,
        # and face area % - all higher is better for a portrait
        portrait_asc_act = QAction(
            "Portrait Quality (Worst First)", self, checkable=True
        )
        portrait_asc_act.triggered.connect(
            lambda: self.set_sort("portrait_quality", True)
        )
        self.sort_menu.addAction(portrait_asc_act)

        portrait_desc_act = QAction(
            "Portrait Quality (Best First)", self, checkable=True
        )
        portrait_desc_act.triggered.connect(
            lambda: self.set_sort("portrait_quality", False)
        )
        self.sort_menu.addAction(portrait_desc_act)

        self.sort_menu.addSeparator()

        # Identity grouping: cluster thumbnails by detected face so the same
        # person's images sit together, via face_embedding.py
        identity_act = QAction("Group by Identity", self, checkable=True)
        identity_act.setToolTip(
            "Group images together by detected face - the same person's "
            "photos end up adjacent to each other"
        )
        identity_act.triggered.connect(lambda: self.set_sort("identity", True))
        self.sort_menu.addAction(identity_act)

        # Similarity to a chosen reference image: closest facial match first
        similarity_act = QAction("Sort by Similarity to Image...", self, checkable=True)
        similarity_act.setToolTip(
            "Pick a reference image and sort so the closest facial matches "
            "to it appear first"
        )
        similarity_act.triggered.connect(self.prompt_similarity_sort)
        self.sort_menu.addAction(similarity_act)

        # Stronger, multi-reference version: nearest-neighbor match against
        # a whole folder of reference images instead of just one
        similarity_folder_act = QAction(
            "Sort by Similarity to Reference Folder...", self, checkable=True
        )
        similarity_folder_act.setToolTip(
            "Pick a folder of reference images and sort so the closest "
            "facial match to ANY of them appears first - more robust than "
            "a single reference image (no one reference photo's angle/"
            "lighting/expression can throw it off), at the cost of a "
            "slower initial scan"
        )
        similarity_folder_act.triggered.connect(self.prompt_similarity_folder_sort)
        self.sort_menu.addAction(similarity_folder_act)

        self.sort_actions = {
            ("name", True): name_asc_act,
            ("name", False): name_desc_act,
            ("modified", True): modified_asc_act,
            ("modified", False): modified_desc_act,
            ("resolution", True): res_asc_act,
            ("resolution", False): res_desc_act,
            ("width", True): width_asc_act,
            ("width", False): width_desc_act,
            ("height", True): height_asc_act,
            ("height", False): height_desc_act,
            ("quality", True): quality_asc_act,
            ("quality", False): quality_desc_act,
            ("face_ratio", True): face_ratio_asc_act,
            ("face_ratio", False): face_ratio_desc_act,
            ("composition_diversity", True): comp_div_asc_act,
            ("composition_diversity", False): comp_div_desc_act,
            ("filesize", True): size_asc_act,
            ("filesize", False): size_desc_act,
            ("portrait_quality", True): portrait_asc_act,
            ("portrait_quality", False): portrait_desc_act,
            ("identity", True): identity_act,
            ("similarity", False): similarity_act,
            ("similarity_multi", False): similarity_folder_act,
        }

        self.sort_actions[("name", True)].setChecked(True)

    def setup_filter_menu(self):
        """Create the filter menu with face detection options."""
        self.filter_menu.clear()

        has_face_act = QAction("Contains Face", self, checkable=True)
        has_face_act.triggered.connect(lambda: self.set_filter("has_face"))
        self.filter_menu.addAction(has_face_act)

        no_face_act = QAction("Does Not Contain Face", self, checkable=True)
        no_face_act.triggered.connect(lambda: self.set_filter("no_face"))
        self.filter_menu.addAction(no_face_act)

        self.filter_menu.addSeparator()

        advanced_act = QAction("Advanced AI Filters...", self)
        advanced_act.triggered.connect(self.show_advanced_filters)
        self.filter_menu.addAction(advanced_act)

        self.filter_menu.addSeparator()

        # Invert filter option
        self.invert_filter_act = QAction("Invert Filter", self, checkable=True)
        self.invert_filter_act.triggered.connect(self.toggle_filter_inversion)
        self.filter_menu.addAction(self.invert_filter_act)

        self.filter_actions = {
            "has_face": has_face_act,
            "no_face": no_face_act,
        }

    def setup_dedup_menu(self):
        """Create the deduplication menu with different modes."""
        self.dedup_menu.clear()

        # Create action group for mutual exclusivity
        dedup_group = QActionGroup(self)
        dedup_group.setExclusive(True)

        exact_act = QAction("Exact (MD5 checksum)", self, checkable=True)
        exact_act.triggered.connect(lambda: self.set_dedup_mode("exact"))
        dedup_group.addAction(exact_act)
        self.dedup_menu.addAction(exact_act)

        visual_largest_act = QAction("Visual - Retain Largest", self, checkable=True)
        visual_largest_act.triggered.connect(
            lambda: self.set_dedup_mode("visual_largest")
        )
        dedup_group.addAction(visual_largest_act)
        self.dedup_menu.addAction(visual_largest_act)

        visual_smallest_act = QAction("Visual - Retain Smallest", self, checkable=True)
        visual_smallest_act.triggered.connect(
            lambda: self.set_dedup_mode("visual_smallest")
        )
        dedup_group.addAction(visual_smallest_act)
        self.dedup_menu.addAction(visual_smallest_act)

        # Set default checked
        exact_act.setChecked(True)

        self.dedup_actions = {
            "exact": exact_act,
            "visual_largest": visual_largest_act,
            "visual_smallest": visual_smallest_act,
        }

    def set_dedup_mode(self, mode):
        """Set the deduplication mode."""
        self.dedup_mode = mode
        self.dedup_actions[mode].setChecked(True)

    def show_dedup_menu(self, pos):
        """Show the deduplication menu on right-click."""
        self.dedup_menu.exec(self.dedup_btn.mapToGlobal(pos))

    def setup_folder_dedup_menu(self):
        """Create the folder deduplication menu with extended options."""
        self.folder_dedup_menu = QMenu(self)

        # Create action group for mutual exclusivity of modes
        folder_dedup_group = QActionGroup(self)
        folder_dedup_group.setExclusive(True)

        exact_act = QAction("Exact (MD5 checksum)", self, checkable=True)
        exact_act.triggered.connect(lambda: self.set_folder_dedup_mode("exact"))
        folder_dedup_group.addAction(exact_act)
        self.folder_dedup_menu.addAction(exact_act)

        visual_largest_act = QAction("Visual - Retain Largest", self, checkable=True)
        visual_largest_act.triggered.connect(
            lambda: self.set_folder_dedup_mode("visual_largest")
        )
        folder_dedup_group.addAction(visual_largest_act)
        self.folder_dedup_menu.addAction(visual_largest_act)

        visual_smallest_act = QAction("Visual - Retain Smallest", self, checkable=True)
        visual_smallest_act.triggered.connect(
            lambda: self.set_folder_dedup_mode("visual_smallest")
        )
        folder_dedup_group.addAction(visual_smallest_act)
        self.folder_dedup_menu.addAction(visual_smallest_act)

        visual_deepest_act = QAction("Visual - Retain Deepest", self, checkable=True)
        visual_deepest_act.triggered.connect(
            lambda: self.set_folder_dedup_mode("visual_deepest")
        )
        folder_dedup_group.addAction(visual_deepest_act)
        self.folder_dedup_menu.addAction(visual_deepest_act)

        visual_shallow_act = QAction("Visual - Retain Shallow", self, checkable=True)
        visual_shallow_act.triggered.connect(
            lambda: self.set_folder_dedup_mode("visual_shallow")
        )
        folder_dedup_group.addAction(visual_shallow_act)
        self.folder_dedup_menu.addAction(visual_shallow_act)

        # Separator before recursive option
        self.folder_dedup_menu.addSeparator()

        # Recursive toggle (independent, not mutually exclusive)
        self.folder_recursive_act = QAction(
            "Recursive (include subdirectories)", self, checkable=True
        )
        self.folder_recursive_act.triggered.connect(self.toggle_folder_dedup_recursive)
        self.folder_dedup_menu.addAction(self.folder_recursive_act)

        # Set default checked
        exact_act.setChecked(True)

        self.folder_dedup_actions = {
            "exact": exact_act,
            "visual_largest": visual_largest_act,
            "visual_smallest": visual_smallest_act,
            "visual_deepest": visual_deepest_act,
            "visual_shallow": visual_shallow_act,
        }

    def set_folder_dedup_mode(self, mode):
        """Set the folder deduplication mode."""
        self.folder_dedup_mode = mode
        self.folder_dedup_actions[mode].setChecked(True)

    def toggle_folder_dedup_recursive(self):
        """Toggle recursive folder deduplication."""
        self.folder_dedup_recursive = self.folder_recursive_act.isChecked()

    def show_folder_dedup_menu(self, pos):
        """Show the folder deduplication menu on right-click."""
        # Ensure menu is set up
        if not hasattr(self, "folder_dedup_menu"):
            self.setup_folder_dedup_menu()
        # Get the sender button
        sender = self.sender()
        if sender:
            self.folder_dedup_menu.exec(sender.mapToGlobal(pos))

    def show_filter_menu(self, pos):
        """Show the filter menu on right-click."""
        # Enable/disable invert filter based on whether a filter is active
        filter_is_active = self.filter_active and self.filter_mode is not None
        self.invert_filter_act.setEnabled(filter_is_active)

        # Update invert filter checkmark based on current advanced filter settings
        if self.filter_mode == "advanced":
            is_inverted = self.advanced_filter_settings.get("global_invert", False)
            self.invert_filter_act.setChecked(is_inverted)
        else:
            self.invert_filter_act.setChecked(False)

        self.filter_menu.exec(self.filter_btn.mapToGlobal(pos))

    def toggle_filter_inversion(self):
        """Toggle the global filter inversion for advanced filters."""
        if self.filter_mode == "advanced":
            # Toggle the global_invert flag
            current_invert = self.advanced_filter_settings.get("global_invert", False)
            self.advanced_filter_settings["global_invert"] = not current_invert

            # Refresh thumbnails to apply the inverted filter
            self.refresh_thumbnails()

            if self._settings_loaded:
                self.write_settings()
        else:
            QMessageBox.information(
                self,
                "Advanced Filters Only",
                "Filter inversion only applies to Advanced AI Filters.\n\n"
                "Please activate Advanced AI Filters first by right-clicking "
                "the filter button and selecting 'Advanced AI Filters...'",
            )

    def toggle_filter(self):
        """Toggle filter on/off with left click."""
        self.filter_active = not self.filter_active
        self.update_filter_visuals()
        self.refresh_thumbnails()

        if self._settings_loaded:
            self.write_settings()

    def set_filter(self, filter_mode):
        """Set the filter mode and activate filtering."""
        if self.filter_mode == filter_mode:
            self.filter_mode = None
            self.filter_active = False
        else:
            self.filter_mode = filter_mode
            self.filter_active = True

        for mode, action in self.filter_actions.items():
            action.setChecked(mode == self.filter_mode and self.filter_active)

        self.update_filter_visuals()
        self.refresh_thumbnails()

        if self._settings_loaded:
            self.write_settings()

    def update_filter_visuals(self):
        """Update visual indicators for filter state."""
        if self.filter_active and self.filter_mode:
            self.filter_btn.setStyleSheet("""
                QToolButton {
                    background-color: #007FFF;
                    color: white;
                    border: 2px solid #005FCC;
                    border-radius: 3px;
                    padding: 3px;
                }
                QToolButton:hover {
                    background-color: #0066CC;
                }
            """)
            self.thumb_list.set_filter_active(True)

            if self.filter_mode == "has_face":
                filter_name = "Contains Face"
            elif self.filter_mode == "no_face":
                filter_name = "Does Not Contain Face"
            else:
                filter_name = "Advanced AI Filters"
            self.status_bar.showMessage(f"Filter active: {filter_name}", 3000)
        else:
            if hasattr(self, "_filter_btn_original_style"):
                self.filter_btn.setStyleSheet(self._filter_btn_original_style)
            else:
                self.filter_btn.setStyleSheet("")
            self.thumb_list.set_filter_active(False)

            if not self.filter_active:
                self.status_bar.showMessage("Filter disabled", 3000)

    def show_advanced_filters(self):
        """Show the advanced AI filters dialog."""
        dialog = AdvancedFilterDialog(self)

        # Pre-populate with current settings if advanced mode was active
        if self.filter_mode == "advanced" and self.advanced_filter_settings:
            dialog.set_filter_settings(self.advanced_filter_settings)

        if dialog.exec():
            self.advanced_filter_settings = dialog.get_filter_settings()

            # Debug output
            print(f"Advanced filter settings: {self.advanced_filter_settings}")
            print(f"Has active filters: {dialog.has_active_filters()}")

            if dialog.has_active_filters():
                self.filter_mode = "advanced"
                self.filter_active = True

                # Reset debug counter
                if hasattr(self, "_filter_debug_count"):
                    delattr(self, "_filter_debug_count")

                # Uncheck simple filter actions
                for action in self.filter_actions.values():
                    action.setChecked(False)

                self.update_filter_visuals()
                self.refresh_thumbnails()
                print(
                    f"Applied advanced filters - mode: {self.filter_mode}, active: {self.filter_active}"
                )
            else:
                # No filters enabled, clear advanced mode
                self.filter_mode = None
                self.filter_active = False
                self.advanced_filter_settings = {}
                self.update_filter_visuals()
                self.refresh_thumbnails()

            if self._settings_loaded:
                self.write_settings()

    def get_advanced_detectors(self):
        """Lazy-load and return advanced AI detectors."""
        if self._object_detector is None:
            conf = self.advanced_filter_settings.get("object_confidence", 0.5)
            self._object_detector = get_object_detector(conf)

            # Set custom model if specified
            custom_model_path = self.advanced_filter_settings.get("custom_model_path")
            if custom_model_path:
                try:
                    self._object_detector.set_custom_model(custom_model_path)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Custom Model Error",
                        f"Failed to load custom YOLO model:\n{e}\n\nFalling back to default model.",
                    )

        if self._quality_assessor is None:
            self._quality_assessor = get_quality_assessor()

        if self._scene_recognizer is None:
            conf = self.advanced_filter_settings.get("scene_confidence", 0.3)
            self._scene_recognizer = get_scene_recognizer(conf)

        return self._object_detector, self._quality_assessor, self._scene_recognizer

    def matches_advanced_filters(self, image_path):
        """Check if image matches advanced filter criteria."""
        if not self.advanced_filter_settings:
            return True

        obj_detector, quality_assessor, scene_recognizer = self.get_advanced_detectors()

        # Debug: Track which filters are being evaluated
        if not hasattr(self, "_filter_debug_count"):
            self._filter_debug_count = 0
            print("\n=== Starting advanced filter evaluation ===")
            print(f"Filter settings: {self.advanced_filter_settings}")

        # Face detection filter
        if self.advanced_filter_settings.get("face_detection", False):
            min_faces = self.advanced_filter_settings.get("face_count_min", 0)
            face_count = len(self.face_detector.detect_faces(image_path))
            inverse = self.advanced_filter_settings.get("face_inverse", False)

            if inverse:
                # Does NOT contain: face count must be less than minimum
                if face_count >= min_faces:
                    return False
            else:
                # Contains: face count must be at least minimum
                if face_count < min_faces:
                    return False

        # Object detection filter
        if self.advanced_filter_settings.get("object_detection", False):
            categories = self.advanced_filter_settings.get("object_categories", [])
            inverse = self.advanced_filter_settings.get("object_inverse", False)
            custom_model_path = self.advanced_filter_settings.get("custom_model_path")

            self._filter_debug_count += 1
            if self._filter_debug_count <= 3:  # Debug first 3 images
                print(
                    f"\n[Image {self._filter_debug_count}] {os.path.basename(image_path)}"
                )
                print(f"  Categories: {categories}")
                print(f"  Inverse: {inverse}")
                print(f"  Custom model: {custom_model_path}")

            # For custom models: if no categories selected, check for ANY detection
            if custom_model_path and not categories:
                detections = obj_detector.detect_objects(image_path)
                has_detections = len(detections) > 0

                if self._filter_debug_count <= 3:
                    print(f"  Custom model detections: {detections}")

                if inverse:
                    # Does NOT contain: must have no detections
                    if has_detections:
                        if self._filter_debug_count <= 3:
                            print("  REJECTED: has detections (inverse mode)")
                        return False
                else:
                    # Contains: must have at least one detection
                    if not has_detections:
                        if self._filter_debug_count <= 3:
                            print("  REJECTED: no detections")
                        return False
            # For standard models or when categories are selected
            elif categories:
                matches_any = False
                for category in categories:
                    if obj_detector.has_category(image_path, category):
                        matches_any = True
                        if self._filter_debug_count <= 3:
                            print(f"  Found category: {category}")
                        break

                if self._filter_debug_count <= 3:
                    print(f"  Matches any category: {matches_any}")

                if inverse:
                    # Does NOT contain: must not match any category
                    if matches_any:
                        if self._filter_debug_count <= 3:
                            print("  REJECTED: matches category (inverse mode)")
                        return False
                else:
                    # Contains: must match at least one category
                    if not matches_any:
                        if self._filter_debug_count <= 3:
                            print("  REJECTED: no category match")
                        return False

            if self._filter_debug_count <= 3:
                print("  PASSED object detection filter")
            # If no categories selected and no custom model, skip this filter (don't exclude anything)

        # Scene recognition filter
        if self.advanced_filter_settings.get("scene_recognition", False):
            scene_groups = self.advanced_filter_settings.get("scene_groups", [])
            inverse = self.advanced_filter_settings.get("scene_inverse", False)

            # Only apply filter if scene groups are selected
            if scene_groups:
                matches_any = False
                for group in scene_groups:
                    if scene_recognizer.has_scene_group(image_path, group):
                        matches_any = True
                        break

                if inverse:
                    # Does NOT contain: must not match any scene group
                    if matches_any:
                        return False
                else:
                    # Contains: must match at least one scene group
                    if not matches_any:
                        return False
            # If no scene groups selected, skip this filter (don't exclude anything)

        # Quality filter
        if self.advanced_filter_settings.get("quality_filter", False):
            quality = quality_assessor.assess_quality(image_path)
            inverse = self.advanced_filter_settings.get("quality_inverse", False)

            if quality is None:
                # If we can't assess quality, treat as not meeting criteria
                return inverse  # Return True if inverse (looking for bad quality), False otherwise

            min_score = self.advanced_filter_settings.get("quality_min_score", 60)
            min_sharpness = self.advanced_filter_settings.get(
                "quality_sharpness_min", 40
            )
            min_bright = self.advanced_filter_settings.get("quality_brightness_min", 50)
            max_bright = self.advanced_filter_settings.get(
                "quality_brightness_max", 200
            )

            meets_criteria = (
                quality["overall_score"] >= min_score
                and quality["sharpness"] >= min_sharpness
                and min_bright <= quality["brightness"] <= max_bright
            )

            if inverse:
                # Does NOT meet: must fail criteria
                if meets_criteria:
                    return False
            else:
                # Meets: must pass criteria
                if not meets_criteria:
                    return False

        # File size filter
        if self.advanced_filter_settings.get("filesize_filter", False):
            min_kb = self.advanced_filter_settings.get("filesize_min_kb", 0)
            max_kb = self.advanced_filter_settings.get("filesize_max_kb", 10240)

            try:
                file_size_bytes = os.path.getsize(image_path)
                file_size_kb = file_size_bytes / 1024

                # Check if file size is within range
                if file_size_kb < min_kb or file_size_kb > max_kb:
                    return False
            except Exception:
                # If we can't get file size, exclude the file
                return False

        # Apply global invert if enabled
        result = True
        if self.advanced_filter_settings.get("global_invert", False):
            result = not result

        return result

    def set_sort(self, sort_by, ascending):
        """Update the sort criteria and refresh thumbnails."""
        self.sort_by = sort_by
        self.sort_ascending = ascending

        for key, action in self.sort_actions.items():
            action.setChecked(key == (sort_by, ascending))

        self.refresh_thumbnails()

        if self._settings_loaded:
            self.write_settings()

    def prompt_similarity_sort(self):
        """Prompt for a reference image, then sort by facial similarity to it."""
        from PySide6.QtWidgets import QFileDialog

        start_dir = self.get_current_source_path() or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Reference Image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.gif *.webp *.avif *.tiff *.bmp)",
        )
        if not path:
            # Cancelled - the action already flipped to checked on click
            # (Qt does this before triggered fires), so restore whichever
            # action actually matches the current sort mode.
            for key, action in self.sort_actions.items():
                action.setChecked(key == (self.sort_by, self.sort_ascending))
            return
        self.set_similarity_sort(path)

    def set_similarity_sort(self, reference_path):
        """Sort the currently displayed thumbnails by facial similarity to
        a chosen reference image, closest match first."""
        self.similarity_reference_path = reference_path
        self.set_sort("similarity", False)
        self.status_bar.showMessage(
            f"Sorting by similarity to: {os.path.basename(reference_path)}", 4000
        )

    def prompt_similarity_folder_sort(self):
        """Prompt for a folder of reference images, then sort by
        nearest-neighbor facial similarity to the whole set - stronger than
        a single reference image since no one reference photo's particular
        angle, lighting, or expression can throw off the match."""
        from PySide6.QtWidgets import QFileDialog

        start_dir = self.get_current_source_path() or ""
        folder = QFileDialog.getExistingDirectory(
            self, "Choose Reference Image Folder", start_dir
        )
        if not folder:
            for key, action in self.sort_actions.items():
                action.setChecked(key == (self.sort_by, self.sort_ascending))
            return

        try:
            reference_paths = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(VALID_IMAGE_EXTS)
            ]
        except OSError as e:
            QMessageBox.warning(self, "Error", f"Could not read folder: {e}")
            for key, action in self.sort_actions.items():
                action.setChecked(key == (self.sort_by, self.sort_ascending))
            return

        if not reference_paths:
            QMessageBox.warning(self, "No Images", "That folder has no images in it.")
            for key, action in self.sort_actions.items():
                action.setChecked(key == (self.sort_by, self.sort_ascending))
            return

        self.set_similarity_sort_multi(reference_paths)

    def set_similarity_sort_multi(self, reference_paths):
        """Sort the currently displayed thumbnails by nearest-neighbor
        facial similarity to a SET of reference images, closest match
        first."""
        self.similarity_reference_paths = list(reference_paths)
        self.set_sort("similarity_multi", False)
        self.status_bar.showMessage(
            f"Sorting by similarity to {len(reference_paths)} reference image(s)",
            4000,
        )

    def refresh_thumbnails(self):
        # Disabled for the duration of the reload so a double-click during a
        # slow directory load (large/high-res folders can take a while, and
        # QApplication.processEvents() below keeps the UI responsive enough
        # for that click to land) can't open a preview window built from a
        # partial mid-load snapshot of the list - previously, doing so froze
        # the preview's navigation at whatever count had loaded so far,
        # making it look like it "only had the first N images" and wrapped
        # around early.
        try:
            self._refresh_thumbnails_impl()
        finally:
            self.thumb_list.setEnabled(True)

    def _refresh_thumbnails_impl(self):
        self.thumb_list.setEnabled(False)
        # Each call gets a unique generation token. Any loop that sees a newer
        # token knows it has been superseded and must stop appending items.
        self._refresh_generation = getattr(self, "_refresh_generation", 0) + 1
        my_gen = self._refresh_generation

        # Show progress bar immediately
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.start_loading_spinner("Loading directory...")
        QApplication.processEvents()  # Force UI update
        if self._refresh_generation != my_gen:
            self.stop_loading_spinner()
            return

        self.thumb_list.clear()
        path = self.get_current_source_path()
        if not os.path.isdir(path):
            self.progress_bar.setVisible(False)
            self.stop_loading_spinner()
            return

        self._update_source_watcher(path)

        files = [f for f in os.listdir(path) if f.lower().endswith(VALID_IMAGE_EXTS)]

        # Apply face filter if active
        if self.filter_active and self.filter_mode:
            filtered_files = []
            total = len(files)

            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.set_loading_spinner_message(f"Filtering {total} images...")

            for i, f in enumerate(files):
                full_path = os.path.join(path, f)

                # Apply appropriate filter
                include = False
                if self.filter_mode == "has_face":
                    include = self.face_detector.has_face(full_path)
                elif self.filter_mode == "no_face":
                    include = not self.face_detector.has_face(full_path)
                elif self.filter_mode == "advanced":
                    include = self.matches_advanced_filters(full_path)

                if include:
                    filtered_files.append(f)

                self.progress_bar.setValue(i + 1)
                if i % 5 == 0:
                    QApplication.processEvents()
                    if self._refresh_generation != my_gen:
                        self.stop_loading_spinner()
                        return

            files = filtered_files
            self.progress_bar.setVisible(False)

        files = self.sort_files(path, files)
        total_files = len(files)

        if total_files == 0:
            self.stop_loading_spinner()
            if self.filter_active and self.filter_mode:
                if self.filter_mode == "has_face":
                    filter_name = "with faces"
                elif self.filter_mode == "no_face":
                    filter_name = "without faces"
                else:
                    filter_name = "matching AI filters"
                self.status_bar.showMessage(
                    f"No images {filter_name} found in this directory.", 3000
                )
            else:
                self.status_bar.showMessage("No images found in this directory.", 3000)
            return

        show_names = self.show_names_act.isChecked()
        show_dimensions = self.show_dimensions_act.isChecked()
        show_filesize = self.show_filesize_act.isChecked()

        from rating_manager import get_rating_manager, combine_display_text

        full_paths = [os.path.join(path, f) for f in files]
        ratings = get_rating_manager().get_ratings_bulk(full_paths)

        self.progress_bar.setRange(0, total_files)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.set_loading_spinner_message(f"Loading {total_files} images...")

        for i, f in enumerate(files):
            full_p = full_paths[i]

            # Determine display text (mutually exclusive)
            if show_names:
                display_text = f
            elif show_dimensions:
                display_text = self.get_image_dimensions_str(full_p)
            elif show_filesize:
                display_text = self.get_file_size_str(full_p)
            else:
                display_text = ""

            item = QListWidgetItem(
                QIcon(full_p),
                combine_display_text(display_text, ratings.get(full_p)),
            )
            item.setData(Qt.ItemDataRole.UserRole, full_p)
            item.setData(Qt.ItemDataRole.UserRole + 1, display_text)
            self.thumb_list.addItem(item)

            self.progress_bar.setValue(i + 1)

            if i % 10 == 0:
                QApplication.processEvents()
                if self._refresh_generation != my_gen:
                    self.stop_loading_spinner()
                    return

        self.progress_bar.setVisible(False)
        self.stop_loading_spinner()

        if self.filter_active and self.filter_mode:
            if self.filter_mode == "has_face":
                filter_name = "with faces"
            elif self.filter_mode == "no_face":
                filter_name = "without faces"
            else:
                filter_name = "matching AI filters"
            self.status_bar.showMessage(
                f"Showing {total_files} images {filter_name}", 3000
            )
        else:
            self.status_bar.showMessage("Ready", 3000)

        if self.show_counts_act.isChecked():
            self.src_tree.count_delegate._count_cache.clear()
            self.dest_tree.count_delegate._count_cache.clear()
            self.src_tree.viewport().update()
            self.dest_tree.viewport().update()

    # ── Live-watching the source folder for externally added files ─────────

    def _update_source_watcher(self, path):
        """Point the QFileSystemWatcher at whichever folder is now displayed."""
        watched = self.source_watcher.directories()
        if watched:
            self.source_watcher.removePaths(watched)
        if path and os.path.isdir(path):
            self.source_watcher.addPath(path)
        self._pending_new_files = {}

    def on_source_directory_changed(self, path):
        """Fired by the OS when the watched folder's contents change.

        Only reacts to the folder currently being displayed - a signal for a
        folder we've since navigated away from is stale and ignored. Debounced
        via a short timer since a single external file write can fire this
        more than once in quick succession.
        """
        if os.path.normpath(path) != os.path.normpath(self.get_current_source_path()):
            return
        self._new_file_check_timer.start(400)

    def _check_for_new_thumbnails(self):
        """Add any newly-appeared image file(s) to the grid without a full reload.

        A file is only added once its size has been stable across two checks,
        so a partially-written file (an external app still saving) doesn't
        get picked up and thumbnailed mid-write.
        """
        path = self.get_current_source_path()
        if not os.path.isdir(path):
            self._pending_new_files = {}
            return

        try:
            current_files = {
                f for f in os.listdir(path) if f.lower().endswith(VALID_IMAGE_EXTS)
            }
        except OSError:
            return

        shown_paths = set()
        for i in range(self.thumb_list.count()):
            item_path = self.thumb_list.item(i).data(Qt.ItemDataRole.UserRole)
            if item_path:
                shown_paths.add(os.path.normpath(item_path))

        still_pending = {}
        needs_recheck = False
        added_any = False

        for f in current_files:
            full_path = os.path.normpath(os.path.join(path, f))
            if full_path in shown_paths:
                continue

            try:
                size = os.path.getsize(full_path)
            except OSError:
                continue

            prev_size = self._pending_new_files.get(full_path)
            if prev_size is not None and prev_size == size:
                if self._add_new_thumbnail(path, f, full_path):
                    added_any = True
            else:
                still_pending[full_path] = size
                needs_recheck = True

        self._pending_new_files = still_pending
        if needs_recheck:
            self._new_file_check_timer.start(400)

        if added_any and self.show_counts_act.isChecked():
            self.src_tree.count_delegate._count_cache.clear()
            self.src_tree.viewport().update()

    def _add_new_thumbnail(self, dir_path, filename, full_path):
        """Insert a single new file into the thumbnail grid at its correct
        sorted position, respecting whichever filter is currently active.

        Returns True if the thumbnail was added.
        """
        target = os.path.realpath(full_path)
        for i in range(self.thumb_list.count()):
            item_path = self.thumb_list.item(i).data(Qt.ItemDataRole.UserRole)
            if item_path and os.path.realpath(item_path) == target:
                return False  # already added by a refresh that ran meanwhile

        if self.filter_active and self.filter_mode:
            include = False
            if self.filter_mode == "has_face":
                include = self.face_detector.has_face(full_path)
            elif self.filter_mode == "no_face":
                include = not self.face_detector.has_face(full_path)
            elif self.filter_mode == "advanced":
                include = self.matches_advanced_filters(full_path)
            if not include:
                return False

        from rating_manager import get_rating_manager, combine_display_text

        if self.show_names_act.isChecked():
            display_text = filename
        elif self.show_dimensions_act.isChecked():
            display_text = self.get_image_dimensions_str(full_path)
        elif self.show_filesize_act.isChecked():
            display_text = self.get_file_size_str(full_path)
        else:
            display_text = ""

        rating = get_rating_manager().get_ratings_bulk([full_path]).get(full_path)
        item = QListWidgetItem(
            QIcon(full_path), combine_display_text(display_text, rating)
        )
        item.setData(Qt.ItemDataRole.UserRole, full_path)
        item.setData(Qt.ItemDataRole.UserRole + 1, display_text)

        current_basenames = [
            os.path.basename(self.thumb_list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self.thumb_list.count())
            if self.thumb_list.item(i).data(Qt.ItemDataRole.UserRole)
        ]
        combined = self.sort_files(dir_path, current_basenames + [filename])
        try:
            insert_row = combined.index(filename)
        except ValueError:
            insert_row = self.thumb_list.count()

        self.thumb_list.insertItem(insert_row, item)
        self.status_bar.showMessage(f"Added: {filename}", 3000)
        return True

    def sort_files(self, path, files):
        """Sort files based on current sort criteria."""
        if self.sort_by == "name":
            files = sorted(files, reverse=not self.sort_ascending)
        elif self.sort_by == "modified":
            files = sorted(
                files,
                key=lambda f: os.path.getmtime(os.path.join(path, f)),
                reverse=not self.sort_ascending,
            )
        elif self.sort_by == "resolution":
            files = sorted(
                files,
                key=lambda f: self.get_image_resolution(os.path.join(path, f)),
                reverse=not self.sort_ascending,
            )
        elif self.sort_by == "width":
            files = sorted(
                files,
                key=lambda f: self.get_image_width(os.path.join(path, f)),
                reverse=not self.sort_ascending,
            )
        elif self.sort_by == "height":
            files = sorted(
                files,
                key=lambda f: self.get_image_height(os.path.join(path, f)),
                reverse=not self.sort_ascending,
            )
        elif self.sort_by == "quality":
            files = sorted(
                files,
                key=lambda f: self.get_image_quality(os.path.join(path, f)),
                reverse=not self.sort_ascending,
            )
        elif self.sort_by == "face_ratio":
            files = sorted(
                files,
                key=lambda f: self.get_face_area_ratio(os.path.join(path, f)),
                reverse=not self.sort_ascending,
            )
        elif self.sort_by == "composition_diversity":
            files = self.order_by_composition_diversity(path, files)
            if self.sort_ascending:
                files = list(reversed(files))
        elif self.sort_by == "filesize":
            files = sorted(
                files,
                key=lambda f: self.get_file_size(os.path.join(path, f)),
                reverse=not self.sort_ascending,
            )
        elif self.sort_by == "portrait_quality":
            files = sorted(
                files,
                key=lambda f: self.get_portrait_quality(os.path.join(path, f)),
                reverse=not self.sort_ascending,
            )
        elif self.sort_by == "identity":
            files = self.order_by_identity(path, files)
        elif self.sort_by == "similarity":
            files = self.order_by_similarity(path, files)
        elif self.sort_by == "similarity_multi":
            files = self.order_by_similarity_multi(path, files)
        return files

    def order_by_composition_diversity(self, path, files):
        """Order files by structural (colour-excluded) diversity.

        Uses greedy farthest-point sampling over each image's HOG composition
        descriptor, so the returned order starts with a representative seed
        image and then always adds whichever remaining image is least like
        anything already placed. Taking the first N files from this order
        gives a diverse, non-redundant subset (e.g. for LoRA training),
        instead of several near-identical framings.
        """
        full_paths = [os.path.join(path, f) for f in files]
        my_gen = self._refresh_generation
        total = len(full_paths)

        # Computing a HOG descriptor per image is the expensive part (disk +
        # decode + resize + HOG) - keep the UI responsive and interruptible,
        # since this can be slow on folders with hundreds+ of images.
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.set_loading_spinner_message(
            f"Computing composition descriptors for {total} images..."
        )

        descriptors = []
        valid_files = []
        for i, (f, full_path) in enumerate(zip(files, full_paths)):
            desc = self.composition_analyzer.get_descriptor(full_path)
            if desc is not None:
                descriptors.append(desc)
                valid_files.append(f)

            self.progress_bar.setValue(i + 1)
            if i % 10 == 0:
                QApplication.processEvents()
                if self._refresh_generation != my_gen:
                    # Folder/sort changed mid-computation - the caller's
                    # generation check will discard this result shortly
                    # anyway, so just stop doing further work now.
                    self.progress_bar.setVisible(False)
                    self.stop_loading_spinner()
                    return files

        self.progress_bar.setVisible(False)

        if len(descriptors) <= 2:
            return files

        self.set_loading_spinner_message("Ordering by composition diversity...")
        QApplication.processEvents()

        ordered = order_by_diversity(valid_files, descriptors)

        # Files whose descriptor couldn't be computed (unreadable) are kept
        # in their existing relative order, appended at the end.
        skipped = [f for f in files if f not in valid_files]
        return ordered + skipped

    def order_by_identity(self, path, files):
        """Group files by detected face identity - the same person's photos
        end up adjacent to each other.

        Computing a face embedding per image is the expensive part (same
        progress/interruption pattern as order_by_composition_diversity);
        clustering itself is fast once embeddings are cached.
        """
        from face_embedding import get_embedder

        embedder = get_embedder()
        full_paths = [os.path.join(path, f) for f in files]
        my_gen = self._refresh_generation
        total = len(full_paths)

        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.set_loading_spinner_message(f"Detecting faces in {total} images...")

        for i, full_path in enumerate(full_paths):
            embedder.get_embedding(full_path)  # populates the cache

            self.progress_bar.setValue(i + 1)
            if i % 10 == 0:
                QApplication.processEvents()
                if self._refresh_generation != my_gen:
                    self.progress_bar.setVisible(False)
                    self.stop_loading_spinner()
                    return files

        self.progress_bar.setVisible(False)
        self.stop_loading_spinner()

        clusters = embedder.cluster_by_identity(full_paths)
        return sorted(files, key=lambda f: (clusters[os.path.join(path, f)], f))

    def order_by_similarity(self, path, files):
        """Order files by facial similarity to self.similarity_reference_path,
        closest match first. Images with no comparable face sort last."""
        from face_embedding import get_embedder

        reference = getattr(self, "similarity_reference_path", None)
        if not reference:
            return files

        embedder = get_embedder()
        full_paths = [os.path.join(path, f) for f in files]
        my_gen = self._refresh_generation
        total = len(full_paths)

        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.set_loading_spinner_message(f"Comparing faces in {total} images...")

        similarities = {}
        for i, full_path in enumerate(full_paths):
            similarities[full_path] = embedder.compute_similarity(reference, full_path)

            self.progress_bar.setValue(i + 1)
            if i % 10 == 0:
                QApplication.processEvents()
                if self._refresh_generation != my_gen:
                    self.progress_bar.setVisible(False)
                    self.stop_loading_spinner()
                    return files

        self.progress_bar.setVisible(False)
        self.stop_loading_spinner()

        return sorted(
            files,
            key=lambda f: (
                similarities[os.path.join(path, f)]
                if similarities[os.path.join(path, f)] is not None
                else -1.0
            ),
            reverse=True,
        )

    def order_by_similarity_multi(self, path, files):
        """Order files by nearest-neighbor facial similarity to a SET of
        reference images (self.similarity_reference_paths) - closest match
        to any one of them first. Images with no comparable face sort last.

        Slower than order_by_similarity since every reference image also
        needs a face embedding computed, but more robust: it isn't thrown
        off by any single reference photo's particular angle, lighting, or
        expression the way comparing against just one image can be.
        """
        from face_embedding import get_embedder

        reference_paths = getattr(self, "similarity_reference_paths", None)
        if not reference_paths:
            return files

        embedder = get_embedder()
        full_paths = [os.path.join(path, f) for f in files]
        my_gen = self._refresh_generation
        total = len(reference_paths) + len(full_paths)
        done = 0

        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.set_loading_spinner_message(
            f"Analyzing {len(reference_paths)} reference image(s)..."
        )

        reference_embeddings = []
        for ref_path in reference_paths:
            reference_embeddings.append(embedder.get_embedding(ref_path))
            done += 1
            self.progress_bar.setValue(done)
            if done % 10 == 0:
                QApplication.processEvents()
                if self._refresh_generation != my_gen:
                    self.progress_bar.setVisible(False)
                    self.stop_loading_spinner()
                    return files

        reference_embeddings = [e for e in reference_embeddings if e is not None]
        if not reference_embeddings:
            self.progress_bar.setVisible(False)
            self.stop_loading_spinner()
            self.status_bar.showMessage(
                "No detectable faces found in the reference images.", 4000
            )
            return files

        self.set_loading_spinner_message(
            f"Comparing faces in {len(full_paths)} images..."
        )

        similarities = {}
        for full_path in full_paths:
            similarities[full_path] = embedder.compute_similarity_to_set(
                full_path, reference_embeddings
            )
            done += 1
            self.progress_bar.setValue(done)
            if done % 10 == 0:
                QApplication.processEvents()
                if self._refresh_generation != my_gen:
                    self.progress_bar.setVisible(False)
                    self.stop_loading_spinner()
                    return files

        self.progress_bar.setVisible(False)
        self.stop_loading_spinner()

        return sorted(
            files,
            key=lambda f: (
                similarities[os.path.join(path, f)]
                if similarities[os.path.join(path, f)] is not None
                else -1.0
            ),
            reverse=True,
        )

    def get_image_resolution(self, file_path):
        """Get total pixel count (width * height) of an image."""
        try:
            with Image.open(file_path) as img:
                return img.width * img.height
        except Exception:
            return 0

    def get_image_width(self, file_path):
        """Get width of an image."""
        try:
            with Image.open(file_path) as img:
                return img.width
        except Exception:
            return 0

    def get_image_height(self, file_path):
        """Get height of an image."""
        try:
            with Image.open(file_path) as img:
                return img.height
        except Exception:
            return 0

    def get_image_quality(self, file_path):
        """Get image quality/sharpness score."""
        return self.quality_analyzer.get_blur_score(file_path)

    def get_face_area_ratio(self, file_path):
        """Get the ratio of image area occupied by the largest detected face."""
        return self.face_detector.get_face_area_ratio(file_path)

    def get_portrait_quality(self, file_path):
        """Composite portrait-quality score (0-100ish, higher is better):
        sharpness (40%) + resolution (30%) + face area % (30%), each
        normalized to roughly 0-100 first so no single signal dominates
        purely due to scale."""
        quality = get_quality_assessor().assess_quality(file_path)
        sharpness = min(100.0, quality["sharpness"]) if quality else 0.0

        resolution = self.get_image_resolution(file_path)
        # 100 points at 1 megapixel and up; scales down linearly below that
        resolution_score = min(100.0, (resolution / (1024 * 1024)) * 100.0)

        face_score = min(100.0, self.get_face_area_ratio(file_path) * 100.0)

        return sharpness * 0.4 + resolution_score * 0.3 + face_score * 0.3

    def get_file_size(self, file_path):
        """Get file size in bytes."""
        try:
            return os.path.getsize(file_path)
        except Exception:
            return 0

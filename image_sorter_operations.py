"""
FileOperationsMixin - File operations, dedup execution, undo, tool windows, captioning,
and help/about dialogs for ImageSorter.
"""

import csv
import os
import shutil
from PySide6.QtWidgets import (
    QMessageBox,
    QApplication,
    QDialog,
    QProgressDialog,
    QMenu,
    QLabel,
    QVBoxLayout,
    QTextBrowser,
    QDialogButtonBox,
    QFileDialog,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup


class FileOperationsMixin:
    """Mixin providing file operations, dedup execution, undo, tool windows, and help."""

    def remove_thumbnails_by_path(self, paths_to_remove):
        """Removes specific items from the thumb_list without a full reload.

        Compares paths via realpath (not just normpath) so this still
        matches when the caller's path string reaches the same file through
        a different symlink than however it was originally added to the
        grid - e.g. a tool window with its own independently browsed/typed
        folder path, rather than the main window's own tree selection.
        """
        paths_to_remove = {os.path.realpath(p) for p in paths_to_remove}

        # Find the lowest index of items being removed to select next item
        first_removed_index = None
        for i in range(self.thumb_list.count() - 1, -1, -1):
            item = self.thumb_list.item(i)
            item_path = os.path.realpath(item.data(Qt.ItemDataRole.UserRole))

            if item_path in paths_to_remove:
                if first_removed_index is None or i < first_removed_index:
                    first_removed_index = i
                self.thumb_list.takeItem(i)

        # Select the next thumbnail after the removed items
        if first_removed_index is not None and self.thumb_list.count() > 0:
            # Select the item at the same index (which is now the next item)
            # or the last item if we removed from the end
            next_index = min(first_removed_index, self.thumb_list.count() - 1)
            next_item = self.thumb_list.item(next_index)
            if next_item:
                self.thumb_list.setCurrentItem(next_item)
                self.thumb_list.scrollToItem(next_item)

        self.status_bar.showMessage(f"Removed {len(paths_to_remove)} items.", 3000)

        if self.show_counts_act.isChecked():
            self.src_tree.count_delegate._count_cache.clear()
            self.dest_tree.count_delegate._count_cache.clear()
            self.src_tree.viewport().update()
            self.dest_tree.viewport().update()

    def start_operation_batch(self, operation_type):
        """Start a new operation batch for undo tracking."""
        self.last_operation = {"type": operation_type, "files": []}

    def smart_transfer(self, src, dst_folder, is_copy):
        """Handles file transfer with automatic renaming to avoid conflicts.

        If src has already vanished by the time this runs - a stale
        thumbnail whose file was moved/deleted by something other than
        this app's own tracked operations (another app, a script, or just
        weaker external-drive change notification) - this reports it
        quietly on the status bar instead of popping up an alarming error
        dialog. The caller still cleans up that thumbnail the same way as
        a normal successful move either way (FolderTree.dropEvent and
        ThumbnailList.transfer_files both add every attempted source to
        their remove-from-grid list regardless of outcome), so nothing is
        actually left in a broken state - just no reason to interrupt the
        user over it.
        """
        if not os.path.exists(src):
            self.status_bar.showMessage(
                f"Skipped {os.path.basename(src)} - no longer exists", 4000
            )
            return

        dest = os.path.join(dst_folder, os.path.basename(src))
        name, ext = os.path.splitext(dest)
        counter = 1
        while os.path.exists(dest):
            dest = f"{name}_{counter}{ext}"
            counter += 1
        try:
            if is_copy:
                shutil.copy2(src, dest)
                # Track copy operation for undo (only if operation batch is active)
                if self.last_operation and self.last_operation["type"] == "copy":
                    self.last_operation["files"].append((src, dest))
            else:
                shutil.move(src, dest)
                # Track move operation for undo (only if operation batch is active)
                if self.last_operation and self.last_operation["type"] == "move":
                    self.last_operation["files"].append((src, dest))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not transfer file: {e}")

    def remove_duplicates(self):
        """Remove duplicate images based on the selected deduplication mode."""

        # Get files to process
        selected_items = self.thumb_list.selectedItems()
        if selected_items:
            # Use selected files
            file_paths = [
                item.data(Qt.ItemDataRole.UserRole) for item in selected_items
            ]
            scope_msg = f"{len(file_paths)} selected file(s)"
        else:
            # Use all files in thumbnail list
            file_paths = [
                self.thumb_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.thumb_list.count())
            ]
            scope_msg = f"all {len(file_paths)} file(s) in directory"

        if len(file_paths) < 2:
            QMessageBox.information(
                self, "Remove Duplicates", "Need at least 2 images to find duplicates."
            )
            return

        # Confirm operation
        mode_descriptions = {
            "exact": "MD5 checksum comparison",
            "visual_largest": "visual similarity (retain largest)",
            "visual_smallest": "visual similarity (retain smallest)",
        }
        mode_desc = mode_descriptions.get(self.dedup_mode, self.dedup_mode)

        reply = QMessageBox.question(
            self,
            "Remove Duplicates",
            f"Find and remove duplicates from {scope_msg} using {mode_desc}?\n\n"
            "This will permanently delete duplicate files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Show progress
        self.status_bar.showMessage("Finding duplicates...", 0)
        QApplication.processEvents()

        try:
            if self.dedup_mode == "exact":
                duplicates = self.find_exact_duplicates(file_paths)
            else:
                duplicates = self.find_visual_duplicates(file_paths)

            if not duplicates:
                self.status_bar.showMessage("No duplicates found", 3000)
                QMessageBox.information(
                    self, "Remove Duplicates", "No duplicate images found."
                )
                return

            # Delete duplicates
            deleted_count = self.delete_duplicate_files(duplicates)

            # Refresh thumbnail view
            self.refresh_thumbnails()

            self.status_bar.showMessage(
                f"Removed {deleted_count} duplicate file(s)", 3000
            )
            QMessageBox.information(
                self,
                "Remove Duplicates",
                f"Successfully removed {deleted_count} duplicate file(s).",
            )

        except Exception as e:
            self.status_bar.showMessage("Error during deduplication", 3000)
            QMessageBox.critical(self, "Error", f"Failed to remove duplicates:\n{e}")

    def find_exact_duplicates(self, file_paths):
        """Find exact duplicates using MD5 checksums. Returns dict of {hash: [files]}."""
        import hashlib

        hash_dict = {}
        for path in file_paths:
            try:
                with open(path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash not in hash_dict:
                        hash_dict[file_hash] = []
                    hash_dict[file_hash].append(path)
            except Exception as e:
                print(f"Error hashing {path}: {e}")

        # Return only groups with duplicates (more than 1 file)
        return {h: files for h, files in hash_dict.items() if len(files) > 1}

    def find_visual_duplicates(self, file_paths, threshold=10):
        """Find visually similar images using perceptual hashing (pHash).

        Uses a 64-bit DCT-based hash computed via numpy — no external tools needed.
        threshold: max Hamming distance (out of 64) to call two images duplicates.
        10 is a good default: catches re-saves, JPEG re-compression, minor crops.
        Lower values (e.g. 4) require a closer match; 0 means byte-identical pixels.

        Returns dict of {group_id: [file_paths]} for groups with >= 2 files.
        """
        import numpy as np
        from PIL import Image

        HASH_SIZE = 8  # produces a 64-bit hash

        def _dct1d(x):
            n = len(x)
            v = np.concatenate([x[::2], x[1::2][::-1]])
            V = np.fft.rfft(v)
            k = np.arange(n) * (-1j * np.pi / (2 * n))
            return (np.exp(k) * np.concatenate([V, np.conj(V[-2:0:-1])])).real[:n]

        def compute_phash(path):
            img = (
                Image.open(path)
                .convert("L")
                .resize((HASH_SIZE * 4, HASH_SIZE * 4), Image.LANCZOS)
            )
            arr = np.array(img, dtype=np.float64)
            dct2d = np.apply_along_axis(_dct1d, 0, np.apply_along_axis(_dct1d, 1, arr))
            low = dct2d[:HASH_SIZE, :HASH_SIZE]
            bits = (low > np.median(low)).flatten()
            result = 0
            for bit in bits:
                result = (result << 1) | int(bit)
            return result

        # --- Hash all images, reporting progress ---
        hashes = []
        n = len(file_paths)
        self.status_bar.showMessage(f"Computing hashes for {n} images…", 0)
        for i, path in enumerate(file_paths):
            try:
                hashes.append(compute_phash(path))
            except Exception:
                hashes.append(None)
            if i % 20 == 0:
                self.status_bar.showMessage(f"Hashing {i + 1}/{n}…", 0)
                QApplication.processEvents()

        # --- Union-Find to group transitively linked duplicates ---
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        self.status_bar.showMessage("Comparing hashes…", 0)
        QApplication.processEvents()
        for i in range(n):
            if hashes[i] is None:
                continue
            for j in range(i + 1, n):
                if hashes[j] is None:
                    continue
                if bin(hashes[i] ^ hashes[j]).count("1") <= threshold:
                    union(i, j)

        # --- Collect groups ---
        groups: dict[int, list] = {}
        for i, path in enumerate(file_paths):
            root = find(i)
            groups.setdefault(root, []).append(path)

        duplicates = {}
        group_id = 0
        for files in groups.values():
            if len(files) >= 2:
                duplicates[group_id] = files
                group_id += 1

        return duplicates

    def delete_duplicate_files(self, duplicates):
        """Delete duplicate files, keeping one per group based on dedup_mode.

        Args:
            duplicates: dict of {group_id: [file_paths]}

        Returns:
            Number of files deleted
        """
        deleted_count = 0

        for group_id, files in duplicates.items():
            if len(files) < 2:
                continue

            # Determine which file to keep
            if self.dedup_mode == "exact":
                # For exact duplicates, keep the first one (arbitrary)
                _keep_file = files[0]
                delete_files = files[1:]
            elif self.dedup_mode == "visual_largest":
                # Keep the largest file
                files_with_size = [(f, os.path.getsize(f)) for f in files]
                files_with_size.sort(key=lambda x: x[1], reverse=True)
                _keep_file = files_with_size[0][0]
                delete_files = [f for f, _ in files_with_size[1:]]
            elif self.dedup_mode == "visual_smallest":
                # Keep the smallest file
                files_with_size = [(f, os.path.getsize(f)) for f in files]
                files_with_size.sort(key=lambda x: x[1])
                _keep_file = files_with_size[0][0]
                delete_files = [f for f, _ in files_with_size[1:]]
            else:
                continue

            # Delete the duplicate files
            for file_path in delete_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        return deleted_count

    def remove_folder_duplicates(self, tree):
        """Remove duplicate images in selected folders based on folder dedup settings.

        Args:
            tree: The FolderTree widget (source or destination)
        """

        # Ensure folder dedup menu is set up
        if not hasattr(self, "folder_dedup_menu"):
            self.setup_folder_dedup_menu()

        # Get selected folder paths
        selected_indexes = tree.selectedIndexes()
        # Filter to column 0 only to avoid duplicates from multiple columns
        selected_paths = []
        seen_rows = set()
        for idx in selected_indexes:
            if idx.column() == 0 and idx.row() not in seen_rows:
                seen_rows.add(idx.row())
                path = tree.model().filePath(idx)
                if os.path.isdir(path):
                    selected_paths.append(path)

        if not selected_paths:
            # Use root path if nothing selected
            if tree == self.src_tree:
                selected_paths = [self.src_root_path]
            else:
                selected_paths = [self.dest_root_path]

        if not selected_paths or not selected_paths[0]:
            QMessageBox.information(
                self, "Remove Duplicates", "Please select one or more folders first."
            )
            return

        # Collect all image files from selected folders
        valid_extensions = (
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
        file_paths = []

        for folder_path in selected_paths:
            if self.folder_dedup_recursive:
                # Recursive: walk all subdirectories
                for root, dirs, files in os.walk(folder_path):
                    for f in files:
                        if f.lower().endswith(valid_extensions):
                            file_paths.append(os.path.join(root, f))
            else:
                # Non-recursive: only immediate directory
                try:
                    for f in os.listdir(folder_path):
                        if f.lower().endswith(valid_extensions):
                            full_path = os.path.join(folder_path, f)
                            if os.path.isfile(full_path):
                                file_paths.append(full_path)
                except (PermissionError, OSError) as e:
                    print(f"Error accessing {folder_path}: {e}")

        if len(file_paths) < 2:
            QMessageBox.information(
                self, "Remove Duplicates", "Need at least 2 images to find duplicates."
            )
            return

        # Build scope message
        folder_names = [os.path.basename(p) for p in selected_paths]
        if len(folder_names) > 3:
            scope_msg = f"{len(folder_names)} folders"
        else:
            scope_msg = ", ".join(folder_names)
        if self.folder_dedup_recursive:
            scope_msg += " (recursive)"

        # Confirm operation
        mode_descriptions = {
            "exact": "MD5 checksum comparison",
            "visual_largest": "visual similarity (retain largest)",
            "visual_smallest": "visual similarity (retain smallest)",
            "visual_deepest": "visual similarity (retain deepest path)",
            "visual_shallow": "visual similarity (retain shallowest path)",
        }
        mode_desc = mode_descriptions.get(
            self.folder_dedup_mode, self.folder_dedup_mode
        )

        reply = QMessageBox.question(
            self,
            "Remove Duplicates",
            f"Find and remove duplicates from {scope_msg} ({len(file_paths)} files) using {mode_desc}?\n\n"
            "This will permanently delete duplicate files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Show progress
        self.status_bar.showMessage("Finding duplicates...", 0)
        QApplication.processEvents()

        try:
            if self.folder_dedup_mode == "exact":
                duplicates = self.find_exact_duplicates(file_paths)
            else:
                duplicates = self.find_visual_duplicates(file_paths)

            if not duplicates:
                self.status_bar.showMessage("No duplicates found", 3000)
                QMessageBox.information(
                    self, "Remove Duplicates", "No duplicate images found."
                )
                return

            # Delete duplicates using folder-specific mode
            deleted_count = self.delete_folder_duplicate_files(duplicates)

            # Refresh thumbnail view if it shows affected folder
            self.refresh_thumbnails()

            # Update folder counts
            if self.show_counts_act.isChecked():
                self.src_tree.count_delegate._count_cache.clear()
                self.dest_tree.count_delegate._count_cache.clear()
                self.src_tree.viewport().update()
                self.dest_tree.viewport().update()

            self.status_bar.showMessage(
                f"Removed {deleted_count} duplicate file(s)", 3000
            )
            QMessageBox.information(
                self,
                "Remove Duplicates",
                f"Successfully removed {deleted_count} duplicate file(s).",
            )

        except Exception as e:
            self.status_bar.showMessage("Error during deduplication", 3000)
            QMessageBox.critical(self, "Error", f"Failed to remove duplicates:\n{e}")

    def delete_folder_duplicate_files(self, duplicates):
        """Delete duplicate files using folder dedup mode.

        Args:
            duplicates: dict of {group_id: [file_paths]}

        Returns:
            Number of files deleted
        """
        deleted_count = 0

        for group_id, files in duplicates.items():
            if len(files) < 2:
                continue

            # Determine which file to keep based on folder_dedup_mode
            if self.folder_dedup_mode == "exact":
                # For exact duplicates, keep the first one (arbitrary)
                _keep_file = files[0]
                delete_files = files[1:]
            elif self.folder_dedup_mode == "visual_largest":
                # Keep the largest file
                files_with_size = [(f, os.path.getsize(f)) for f in files]
                files_with_size.sort(key=lambda x: x[1], reverse=True)
                _keep_file = files_with_size[0][0]
                delete_files = [f for f, _ in files_with_size[1:]]
            elif self.folder_dedup_mode == "visual_smallest":
                # Keep the smallest file
                files_with_size = [(f, os.path.getsize(f)) for f in files]
                files_with_size.sort(key=lambda x: x[1])
                _keep_file = files_with_size[0][0]
                delete_files = [f for f, _ in files_with_size[1:]]
            elif self.folder_dedup_mode == "visual_deepest":
                # Keep the file with the deepest path (most directory separators)
                files_with_depth = [(f, f.count(os.sep)) for f in files]
                files_with_depth.sort(key=lambda x: x[1], reverse=True)
                _keep_file = files_with_depth[0][0]
                delete_files = [f for f, _ in files_with_depth[1:]]
            elif self.folder_dedup_mode == "visual_shallow":
                # Keep the file with the shallowest path (fewest directory separators)
                files_with_depth = [(f, f.count(os.sep)) for f in files]
                files_with_depth.sort(key=lambda x: x[1])
                _keep_file = files_with_depth[0][0]
                delete_files = [f for f, _ in files_with_depth[1:]]
            else:
                continue

            # Delete the duplicate files
            for file_path in delete_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        return deleted_count

    def undo_last_operation(self):
        """Undo the last move, copy, or delete operation."""
        # First check if there's a delete to undo (takes priority as it's the most recent)
        if hasattr(self, "delete_undo_manager") and self.delete_undo_manager.can_undo():
            self.undo_delete()
            return

        if not self.last_operation:
            self.status_bar.showMessage("Nothing to undo", 3000)
            return

        operation_type = self.last_operation["type"]
        files = self.last_operation["files"]

        if not files:
            self.last_operation = None
            return

        # Save current scroll position
        scroll_bar = self.thumb_list.verticalScrollBar()
        saved_scroll_pos = scroll_bar.value()

        success_count = 0
        failed_count = 0

        for src, dest in files:
            try:
                if operation_type == "copy":
                    # For copy, delete the copied file
                    if os.path.exists(dest):
                        os.remove(dest)
                        success_count += 1
                    else:
                        failed_count += 1
                elif operation_type == "move":
                    # For move, move the file back to original location
                    if os.path.exists(dest):
                        shutil.move(dest, src)
                        success_count += 1
                    else:
                        failed_count += 1
            except Exception as e:
                failed_count += 1
                QMessageBox.warning(
                    self, "Undo Error", f"Could not undo {os.path.basename(dest)}:\n{e}"
                )

        # Clear the operation history after undo
        self.last_operation = None

        # Refresh thumbnails and folder counts
        self.refresh_thumbnails()

        # Restore scroll position after thumbnails are reloaded
        QTimer.singleShot(100, lambda: scroll_bar.setValue(saved_scroll_pos))

        # Update folder count caches
        if hasattr(self.src_tree, "count_delegate") and hasattr(
            self.dest_tree, "count_delegate"
        ):
            self.src_tree.count_delegate._count_cache.clear()
            self.dest_tree.count_delegate._count_cache.clear()
            self.src_tree.viewport().update()
            self.dest_tree.viewport().update()

        self.status_bar.showMessage(
            f"Undid {operation_type}: {success_count} file(s)", 3000
        )

    def undo_delete(self):
        """Undo the last file deletion."""
        if (
            not hasattr(self, "delete_undo_manager")
            or not self.delete_undo_manager.can_undo()
        ):
            self.status_bar.showMessage("Nothing to undo", 3000)
            return

        # Save current scroll position
        scroll_bar = self.thumb_list.verticalScrollBar()
        saved_scroll_pos = scroll_bar.value()

        # Restore the deleted files
        restored_paths = self.delete_undo_manager.undo()

        if restored_paths:
            # Refresh thumbnails to show restored files
            self.refresh_thumbnails()

            # Restore scroll position after thumbnails are reloaded
            QTimer.singleShot(100, lambda: scroll_bar.setValue(saved_scroll_pos))

            # Update folder count caches
            if hasattr(self.src_tree, "count_delegate") and hasattr(
                self.dest_tree, "count_delegate"
            ):
                self.src_tree.count_delegate._count_cache.clear()
                self.dest_tree.count_delegate._count_cache.clear()
                self.src_tree.viewport().update()
                self.dest_tree.viewport().update()

            self.status_bar.showMessage(f"Restored {len(restored_paths)} file(s)", 3000)
        else:
            self.status_bar.showMessage("Could not restore deleted file(s)", 3000)

    def open_yolo_window(self):
        """Open the YOLO labeling window with current thumbnails."""
        from yolo_window import YoloWindow

        # Get the current image list from thumbnails
        image_list = []
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                image_list.append(path)

        if not image_list:
            QMessageBox.information(
                self, "No Images", "Please select a source folder with images first."
            )
            return

        # Get current selection index, default to 0
        current_index = 0
        selected_items = self.thumb_list.selectedItems()
        if selected_items:
            current_index = self.thumb_list.row(selected_items[0])

        # Open YOLO window - use showMaximized() for immediate full-screen
        self.yolo_window = YoloWindow(image_list, current_index, self)
        self.yolo_window.showMaximized()

    def open_reference_consistency_window(self):
        """Prompt for a reference-image folder, then open a one-shot audit
        showing every image's leave-one-out best facial match against the
        rest of that same folder - a photo that's mislabeled, wrong-
        identity, or just a poor/ambiguous crop stands out with a lower
        score (and a red border once below the adjustable threshold),
        making it easy to spot what to remove or replace before using that
        folder with the reference-folder similarity sorts."""
        from reference_consistency_window import ReferenceConsistencyWindow

        start_dir = self.get_current_source_path() or ""
        folder = QFileDialog.getExistingDirectory(
            self, "Choose Reference Folder to Check", start_dir
        )
        if not folder:
            return

        self.reference_consistency_window = ReferenceConsistencyWindow(folder, self)
        self.reference_consistency_window.show()

    def open_slideshow(self):
        """Open a slideshow of the currently displayed thumbnails.

        If 2 or more thumbnails are selected, the slideshow is restricted to
        just that selection; otherwise (0 or 1 selected) it falls back to
        every thumbnail currently shown.
        """
        from slideshow_window import SlideshowWindow

        selected_rows = sorted(
            self.thumb_list.row(item) for item in self.thumb_list.selectedItems()
        )

        image_list = []
        current_index = 0
        if len(selected_rows) >= 2:
            for row in selected_rows:
                path = self.thumb_list.item(row).data(Qt.ItemDataRole.UserRole)
                if path:
                    image_list.append(path)
        else:
            for i in range(self.thumb_list.count()):
                item = self.thumb_list.item(i)
                path = item.data(Qt.ItemDataRole.UserRole)
                if path:
                    image_list.append(path)
            if selected_rows:
                current_index = selected_rows[0]

        if not image_list:
            QMessageBox.information(
                self, "No Images", "Please select a source folder with images first."
            )
            return

        self.slideshow_window = SlideshowWindow(image_list, current_index, self)
        self.slideshow_window.showMaximized()

    def open_caption_window(self):
        """Open the Auto Caption window with current thumbnails."""
        from caption_window import CaptionWindow

        # Get the current image list from thumbnails
        image_list = []
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                image_list.append(path)

        if not image_list:
            QMessageBox.information(
                self, "No Images", "Please select a source folder with images first."
            )
            return

        # Get current selection index, default to 0
        current_index = 0
        selected_items = self.thumb_list.selectedItems()
        if selected_items:
            current_index = self.thumb_list.row(selected_items[0])

        # Open Caption window
        self.caption_window = CaptionWindow(image_list, current_index, self)
        self.caption_window.showMaximized()

    def open_enhance_dialog(self):
        """Open the Image Enhance dialog for selected thumbnails."""
        from enhance_dialog import EnhanceDialog

        # Get selected image paths
        selected_items = self.thumb_list.selectedItems()
        if selected_items:
            image_paths = [
                item.data(Qt.ItemDataRole.UserRole)
                for item in selected_items
                if item.data(Qt.ItemDataRole.UserRole)
            ]
        else:
            # No selection - gather all and ask
            image_paths = []
            for i in range(self.thumb_list.count()):
                item = self.thumb_list.item(i)
                path = item.data(Qt.ItemDataRole.UserRole)
                if path:
                    image_paths.append(path)

            if not image_paths:
                QMessageBox.information(
                    self,
                    "No Images",
                    "Please select a source folder with images first.",
                )
                return

            reply = QMessageBox.question(
                self,
                "Enhance All?",
                f"No images selected. Enhance all {len(image_paths)} images in the current folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        dialog = EnhanceDialog(image_paths, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_thumbnails()

    def open_letterbox_dialog(self):
        """Open the Remove Letterbox dialog for selected thumbnails."""
        from letterbox_dialog import LetterboxDialog

        selected_items = self.thumb_list.selectedItems()
        if selected_items:
            image_paths = [
                item.data(Qt.ItemDataRole.UserRole)
                for item in selected_items
                if item.data(Qt.ItemDataRole.UserRole)
            ]
        else:
            image_paths = []
            for i in range(self.thumb_list.count()):
                item = self.thumb_list.item(i)
                path = item.data(Qt.ItemDataRole.UserRole)
                if path:
                    image_paths.append(path)

            if not image_paths:
                QMessageBox.information(
                    self,
                    "No Images",
                    "Please select a source folder with images first.",
                )
                return

            reply = QMessageBox.question(
                self,
                "Remove Letterbox — All Images?",
                f"No thumbnails selected. Process all {len(image_paths)} images"
                " in the current folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        dialog = LetterboxDialog(image_paths, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_thumbnails()

    def open_pair_selector(self):
        """Open the Pair Selector window pre-populated from current app state.

        LHS (dir_a / files_a):
          - Selected thumbnails → pass those specific files as files_a
          - No selection       → pass the currently active source folder as dir_a

        RHS (dir_b):
          - Currently selected folder in the destination tree, or dest_root_path
        """
        from pair_selector_window import PairSelectorWindow

        # ── LHS ───────────────────────────────────────────────────────────────
        selected_items = self.thumb_list.selectedItems()
        if selected_items:
            files_a = [
                item.data(Qt.ItemDataRole.UserRole)
                for item in selected_items
                if item.data(Qt.ItemDataRole.UserRole)
            ]
            dir_a = None
        else:
            files_a = None
            dir_a = self.get_current_source_path()

        # ── RHS ───────────────────────────────────────────────────────────────
        dest_index = self.dest_tree.currentIndex()
        if dest_index.isValid():
            dir_b = self.dest_model.filePath(self.dest_proxy.mapToSource(dest_index))
        else:
            dir_b = self.dest_root_path

        win = PairSelectorWindow(parent=self, dir_a=dir_a, dir_b=dir_b, files_a=files_a)
        win.show()

    def open_filename_matcher(self):
        """Open the Filename Matcher window, pre-populated from current app state.

        Left pane:  currently active source folder.
        Right pane: currently selected destination folder, or dest_root_path.
        """
        from filename_matcher_window import FilenameMatcherWindow

        dir_a = self.get_current_source_path()

        dest_index = self.dest_tree.currentIndex()
        if dest_index.isValid():
            dir_b = self.dest_model.filePath(self.dest_proxy.mapToSource(dest_index))
        else:
            dir_b = self.dest_root_path

        win = FilenameMatcherWindow(parent=self, dir_a=dir_a, dir_b=dir_b)
        win.show()

    def open_re_extract_matcher(self):
        """Open the Re-Extract Matcher window, pre-populated from current app state.

        Originals pane: currently active source folder.
        Extracts pane:  currently selected destination folder, or dest_root_path.
        """
        from re_extract_matcher_window import ReExtractMatcherWindow

        dir_originals = self.get_current_source_path()

        dest_index = self.dest_tree.currentIndex()
        if dest_index.isValid():
            dir_extracts = self.dest_model.filePath(
                self.dest_proxy.mapToSource(dest_index)
            )
        else:
            dir_extracts = self.dest_root_path

        win = ReExtractMatcherWindow(
            parent=self, dir_originals=dir_originals, dir_extracts=dir_extracts
        )
        win.show()

    def open_rating_analysis(self):
        """Correlate ratings of the currently displayed images against their
        embedded ComfyUI LoRAs/strengths and prompt keywords.

        Scoped to whatever's currently shown in the thumbnail grid (respects
        any active filter), not the whole dataset - matches "rate a batch,
        then analyze that batch."
        """
        from rating_manager import get_rating_manager
        from rating_analysis import analyze_ratings

        paths = [
            self.thumb_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.thumb_list.count())
        ]
        if not paths:
            QMessageBox.information(
                self, "Analyze Ratings", "No images currently displayed."
            )
            return

        ratings = get_rating_manager().get_ratings_bulk(paths)
        if len(ratings) < 3:
            QMessageBox.information(
                self,
                "Analyze Ratings",
                f"Only {len(ratings)} rated image(s) in the current view. "
                "Rate at least a handful first (press 1-5 on a thumbnail, "
                "or 0 to clear a rating) before analyzing.",
            )
            return

        result = analyze_ratings(ratings)
        if result["num_with_metadata"] == 0:
            QMessageBox.information(
                self,
                "Analyze Ratings",
                f"{len(ratings)} image(s) rated, but none had embedded "
                "ComfyUI metadata to analyze. Enable Options > Show ComfyUI "
                "Info to check whether these images carry generation data.",
            )
            return

        box = QMessageBox(self)
        box.setWindowTitle("Rating Analysis")
        box.setIcon(QMessageBox.Icon.Information)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(self._format_rating_analysis(result))
        box.exec()

    def _format_rating_analysis(self, result):
        lines = [
            f"{result['num_rated']} image(s) rated, "
            f"{result['num_with_metadata']} had ComfyUI metadata "
            f"(avg rating {result['overall_avg']:.2f}/5)",
            "",
            "LoRA strength correlation with rating:",
            "(closer to +1.0 = higher strength -> higher rating; "
            "closer to -1.0 = higher strength -> lower rating)",
        ]

        if result["lora_stats"]:
            for stat in result["lora_stats"][:10]:
                corr = stat["strength_correlation"]
                corr_str = (
                    f"{corr:+.2f}"
                    if corr is not None
                    else "n/a (not enough strength variation)"
                )
                lines.append(
                    f"  - {stat['name']}: avg {stat['avg_rating']:.2f}/5 "
                    f"over {stat['count']} use(s), correlation {corr_str}"
                )
        else:
            lines.append("  (no LoRAs found in rated images)")

        lines.append("")
        lines.append("Prompt keywords associated with HIGHER ratings:")
        if result["keyword_stats_high"]:
            for kw in result["keyword_stats_high"]:
                lines.append(
                    f"  - '{kw['keyword']}': avg {kw['avg_rating']:.2f}/5 "
                    f"({kw['delta']:+.2f} vs overall, {kw['count']} occurrence(s))"
                )
        else:
            lines.append("  (none found)")

        lines.append("")
        lines.append("Prompt keywords associated with LOWER ratings:")
        if result["keyword_stats_low"]:
            for kw in result["keyword_stats_low"]:
                lines.append(
                    f"  - '{kw['keyword']}': avg {kw['avg_rating']:.2f}/5 "
                    f"({kw['delta']:+.2f} vs overall, {kw['count']} occurrence(s))"
                )
        else:
            lines.append("  (none found)")

        return "\n".join(lines)

    def export_ratings_csv(self):
        """Export the currently displayed rated images' ComfyUI metadata to
        a CSV, one row per (image, LoRA) pair, for flexible slicing outside
        the app (Excel, pandas, etc.).

        Scoped the same way as Analyze Ratings - whatever's currently shown
        in the thumbnail grid. Images with no LoRAs still get one row (with
        the lora columns blank) so every rated image is represented.
        """
        from rating_manager import get_rating_manager
        from comfyui_metadata import extract_comfyui_info

        paths = [
            self.thumb_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.thumb_list.count())
        ]
        if not paths:
            QMessageBox.information(
                self, "Export Ratings CSV", "No images currently displayed."
            )
            return

        ratings = get_rating_manager().get_ratings_bulk(paths)
        if not ratings:
            QMessageBox.information(
                self,
                "Export Ratings CSV",
                "No rated images in the current view. Rate at least one "
                "(press 1-5 on a thumbnail) before exporting.",
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Ratings CSV", "ratings_export.csv", "CSV Files (*.csv)"
        )
        if not save_path:
            return

        fieldnames = [
            "image_path",
            "image_filename",
            "rating",
            "model_name",
            "lora_name",
            "lora_strength_model",
            "lora_strength_clip",
            "positive_prompt",
            "negative_prompt",
        ]

        row_count = 0
        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for path in sorted(ratings.keys()):
                    info = extract_comfyui_info(path)
                    loras = info["loras"] if info else []
                    base_row = {
                        "image_path": path,
                        "image_filename": os.path.basename(path),
                        "rating": ratings[path],
                        "model_name": (info["model_name"] if info else "") or "",
                        "positive_prompt": (info["positive_prompt"] if info else "")
                        or "",
                        "negative_prompt": (info["negative_prompt"] if info else "")
                        or "",
                    }

                    if loras:
                        for lora in loras:
                            writer.writerow(
                                {
                                    **base_row,
                                    "lora_name": lora["name"],
                                    "lora_strength_model": (
                                        lora["strength_model"]
                                        if lora["strength_model"] is not None
                                        else ""
                                    ),
                                    "lora_strength_clip": (
                                        lora["strength_clip"]
                                        if lora["strength_clip"] is not None
                                        else ""
                                    ),
                                }
                            )
                            row_count += 1
                    else:
                        writer.writerow(
                            {
                                **base_row,
                                "lora_name": "",
                                "lora_strength_model": "",
                                "lora_strength_clip": "",
                            }
                        )
                        row_count += 1
        except OSError as e:
            QMessageBox.warning(
                self, "Export Failed", f"Could not write CSV file:\n{e}"
            )
            return

        self.status_bar.showMessage(
            f"Exported {row_count} row(s) for {len(ratings)} rated image(s) "
            f"to {os.path.basename(save_path)}",
            5000,
        )

    def show_caption_settings_menu(self, pos):
        """Show right-click settings menu for caption button."""
        menu = QMenu(self)

        # Current settings
        current_backend = getattr(self, "caption_backend", "florence2")

        # --- Backend selection ---
        backend_label = QAction("Caption Backend:", self)
        backend_label.setEnabled(False)
        menu.addAction(backend_label)

        backend_group = QActionGroup(self)
        backend_group.setExclusive(True)

        backends = [
            ("Florence2 (Local)", "florence2"),
            ("WD14 Tagger (Local)", "wd14"),
            ("Ollama (Server)", "ollama"),
        ]
        for label, key in backends:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(current_backend == key)
            act.setData(key)
            backend_group.addAction(act)
            menu.addAction(act)

        backend_group.triggered.connect(self._on_caption_backend_changed)

        menu.addSeparator()

        # --- Model selection (depends on backend) ---
        if current_backend == "florence2":
            model_label = QAction("Florence2 Model:", self)
            model_label.setEnabled(False)
            menu.addAction(model_label)

            current_model = getattr(self, "caption_florence_model", "base")
            model_group = QActionGroup(self)
            model_group.setExclusive(True)

            florence_models = [
                ("Base (~460MB)", "base"),
                ("Large (~1.5GB)", "large"),
                ("Base Fine-tuned (~460MB)", "base-ft"),
                ("Large Fine-tuned (~1.5GB)", "large-ft"),
            ]
            for label, key in florence_models:
                act = QAction(label, self)
                act.setCheckable(True)
                act.setChecked(current_model == key)
                act.setData(key)
                model_group.addAction(act)
                menu.addAction(act)

            model_group.triggered.connect(self._on_caption_florence_model_changed)

            menu.addSeparator()

            # Detail level
            detail_label = QAction("Detail Level:", self)
            detail_label.setEnabled(False)
            menu.addAction(detail_label)

            current_detail = getattr(self, "caption_florence_detail", "detailed")
            detail_group = QActionGroup(self)
            detail_group.setExclusive(True)

            details = [
                ("Brief", "brief"),
                ("Standard", "standard"),
                ("Detailed", "detailed"),
            ]
            for label, key in details:
                act = QAction(label, self)
                act.setCheckable(True)
                act.setChecked(current_detail == key)
                act.setData(key)
                detail_group.addAction(act)
                menu.addAction(act)

            detail_group.triggered.connect(self._on_caption_florence_detail_changed)

        elif current_backend == "wd14":
            model_label = QAction("WD14 Model:", self)
            model_label.setEnabled(False)
            menu.addAction(model_label)

            current_model = getattr(self, "caption_wd14_model", "vit-v2")
            model_group = QActionGroup(self)
            model_group.setExclusive(True)

            wd14_models = [
                ("ViT v2 (~350MB)", "vit-v2"),
                ("SwinV2 v2 (~430MB)", "swinv2-v2"),
                ("ConvNeXtV2 v2 (~680MB)", "convnextv2-v2"),
                ("MOAT v2 (~840MB)", "moat-v2"),
                ("ViT v3 (~350MB)", "vit-v3"),
            ]
            for label, key in wd14_models:
                act = QAction(label, self)
                act.setCheckable(True)
                act.setChecked(current_model == key)
                act.setData(key)
                model_group.addAction(act)
                menu.addAction(act)

            model_group.triggered.connect(self._on_caption_wd14_model_changed)

        elif current_backend == "ollama":
            model_label = QAction("Ollama Model:", self)
            model_label.setEnabled(False)
            menu.addAction(model_label)

            current_model = getattr(self, "caption_ollama_model", "llava")
            current_url = getattr(self, "caption_ollama_url", "http://localhost:11434")

            # Try to fetch available models
            try:
                import urllib.request
                import json

                api_url = f"{current_url.rstrip('/')}/api/tags"
                request = urllib.request.Request(api_url)
                with urllib.request.urlopen(request, timeout=2) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    models = sorted(
                        [m.get("name", "") for m in result.get("models", [])]
                    )
            except Exception:
                models = []

            if models:
                model_group = QActionGroup(self)
                model_group.setExclusive(True)

                for model_name in models:
                    act = QAction(model_name, self)
                    act.setCheckable(True)
                    act.setChecked(current_model == model_name)
                    act.setData(model_name)
                    model_group.addAction(act)
                    menu.addAction(act)

                model_group.triggered.connect(self._on_caption_ollama_model_changed)
            else:
                no_models = QAction("(No models found - is Ollama running?)", self)
                no_models.setEnabled(False)
                menu.addAction(no_models)

        menu.addSeparator()

        # Open full caption window
        open_window_act = QAction("Open Caption Window...", self)
        open_window_act.triggered.connect(self.open_caption_window)
        menu.addAction(open_window_act)

        menu.exec(self.batch_caption_btn.mapToGlobal(pos))

    def _on_caption_backend_changed(self, action):
        """Handle caption backend change from context menu."""
        self.caption_backend = action.data()
        self.write_settings()
        self.status_bar.showMessage(f"Caption backend: {action.text()}", 3000)

    def _on_caption_florence_model_changed(self, action):
        """Handle Florence2 model change from context menu."""
        self.caption_florence_model = action.data()
        self.write_settings()
        self.status_bar.showMessage(f"Florence2 model: {action.text()}", 3000)

    def _on_caption_florence_detail_changed(self, action):
        """Handle Florence2 detail level change from context menu."""
        self.caption_florence_detail = action.data()
        self.write_settings()
        self.status_bar.showMessage(f"Detail level: {action.text()}", 3000)

    def _on_caption_wd14_model_changed(self, action):
        """Handle WD14 model change from context menu."""
        self.caption_wd14_model = action.data()
        self.write_settings()
        self.status_bar.showMessage(f"WD14 model: {action.text()}", 3000)

    def _on_caption_ollama_model_changed(self, action):
        """Handle Ollama model change from context menu."""
        self.caption_ollama_model = action.data()
        self.write_settings()
        self.status_bar.showMessage(f"Ollama model: {action.text()}", 3000)

    def batch_generate_captions(self):
        """Generate captions for selected or all thumbnail images."""
        from caption_generator import get_caption_generator

        # Get images to process - selected items or all if none selected
        selected_items = self.thumb_list.selectedItems()
        if selected_items:
            image_paths = []
            for item in selected_items:
                path = item.data(Qt.ItemDataRole.UserRole)
                if path:
                    image_paths.append(path)
        else:
            # No selection - use all thumbnails
            image_paths = []
            for i in range(self.thumb_list.count()):
                item = self.thumb_list.item(i)
                path = item.data(Qt.ItemDataRole.UserRole)
                if path:
                    image_paths.append(path)

        if not image_paths:
            QMessageBox.information(
                self, "No Images", "No images available to caption."
            )
            return

        # Filter out images that already have captions
        images_without_captions = []
        images_with_captions = []
        for path in image_paths:
            caption_path = os.path.splitext(path)[0] + ".txt"
            if os.path.exists(caption_path):
                images_with_captions.append(path)
            else:
                images_without_captions.append(path)

        # Ask what to do if some already have captions
        if images_with_captions and images_without_captions:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Existing Captions")
            msg_box.setText(
                f"{len(images_with_captions)} image(s) already have captions.\n"
                f"{len(images_without_captions)} need captions.\n\n"
                "What would you like to do?"
            )
            skip_btn = msg_box.addButton(
                "Skip existing", QMessageBox.ButtonRole.AcceptRole
            )
            overwrite_btn = msg_box.addButton(
                "Overwrite all", QMessageBox.ButtonRole.DestructiveRole
            )
            cancel_btn = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg_box.setDefaultButton(skip_btn)
            msg_box.exec()

            clicked = msg_box.clickedButton()
            if clicked == cancel_btn:
                return
            elif clicked == skip_btn:
                # Skip existing - only process images without captions
                image_paths = images_without_captions
            # overwrite_btn = process all (overwrite existing)
        elif images_with_captions and not images_without_captions:
            # All have captions
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("All Have Captions")
            msg_box.setText(
                f"All {len(images_with_captions)} image(s) already have captions.\n"
                "Do you want to regenerate them?"
            )
            overwrite_btn = msg_box.addButton(
                "Overwrite", QMessageBox.ButtonRole.AcceptRole
            )
            cancel_btn = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg_box.setDefaultButton(cancel_btn)
            msg_box.exec()

            if msg_box.clickedButton() != overwrite_btn:
                return

        if not image_paths:
            QMessageBox.information(self, "Nothing to Do", "No images need captions.")
            return

        # Create progress dialog
        progress = QProgressDialog(
            "Generating captions...", "Cancel", 0, len(image_paths), self
        )
        progress.setWindowTitle("Batch Caption Generation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Get caption settings
        backend = getattr(self, "caption_backend", "florence2")
        ollama_url = getattr(self, "caption_ollama_url", "http://localhost:11434")
        ollama_model = getattr(self, "caption_ollama_model", "llava")
        florence_model = getattr(self, "caption_florence_model", "base")
        florence_detail = getattr(self, "caption_florence_detail", "detailed")
        wd14_model = getattr(self, "caption_wd14_model", "vit-v2")
        wd14_threshold = getattr(self, "caption_wd14_threshold", 0.55)

        # Get generator instance
        generator = get_caption_generator()

        # Process images
        successful = 0
        failed = 0
        errors = []

        for i, image_path in enumerate(image_paths):
            if progress.wasCanceled():
                break

            # Update progress
            filename = os.path.basename(image_path)
            progress.setLabelText(
                f"Generating caption for:\n{filename}\n\n({i + 1} of {len(image_paths)})"
            )
            progress.setValue(i)
            QApplication.processEvents()

            try:
                # Generate caption
                if backend == "florence2":
                    caption = generator.generate_florence(
                        image_path,
                        model_key=florence_model,
                        detail_level=florence_detail,
                    )
                elif backend == "wd14":
                    caption = generator.generate_wd14(
                        image_path, model_key=wd14_model, threshold=wd14_threshold
                    )
                else:  # ollama
                    caption = generator.generate_ollama(
                        image_path, ollama_url, ollama_model
                    )

                # Save caption to txt file
                caption_path = os.path.splitext(image_path)[0] + ".txt"
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(caption)

                successful += 1

            except Exception as e:
                failed += 1
                errors.append(f"{filename}: {str(e)[:50]}")

            QApplication.processEvents()

        progress.setValue(len(image_paths))
        progress.close()

        # Show results
        if progress.wasCanceled():
            msg = f"Batch captioning cancelled.\n\nCompleted: {successful}\nFailed: {failed}"
        else:
            msg = f"Batch captioning complete.\n\nSuccessful: {successful}\nFailed: {failed}"

        if errors and len(errors) <= 5:
            msg += "\n\nErrors:\n" + "\n".join(errors)
        elif errors:
            msg += "\n\nFirst 5 errors:\n" + "\n".join(errors[:5])
            msg += f"\n... and {len(errors) - 5} more"

        QMessageBox.information(self, "Batch Caption Results", msg)

    def show_about(self):
        """Show the About dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("About Visual Image Master")
        dialog.setMinimumSize(450, 350)

        layout = QVBoxLayout(dialog)

        # Application title
        title_label = QLabel("<h1>Visual Image Master 2026</h1>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Version and description
        info_text = """
        <p style='text-align: center;'>
        <b>Version 1.0.0</b><br><br>
        A powerful desktop application for browsing, sorting,<br>
        filtering, and organizing image collections.<br><br>
        <b>Features:</b><br>
        Three-pane interface with drag-and-drop support<br>
        Face detection and AI-powered filtering<br>
        Visual duplicate detection and removal<br>
        Full-screen preview with crop functionality<br>
        Batch file operations with undo support<br><br>
        <b>Technologies:</b><br>
        Python 3 | PySide6 (Qt6) | OpenCV | Pillow<br><br>
        <hr>
        <small>
        Copyright 2026. All rights reserved.<br>
        Licensed under the MIT License.
        </small>
        </p>
        """
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # OK button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)

        dialog.exec()

    def show_manual(self):
        """Show the Manual/Help dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Visual Image Master - Manual")
        dialog.setMinimumSize(700, 600)

        layout = QVBoxLayout(dialog)

        # Create text browser for markdown-style content
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setReadOnly(True)

        manual_text = self.get_manual_text()
        text_browser.setMarkdown(manual_text)

        layout.addWidget(text_browser)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.exec()

    def get_manual_text(self):
        """Return the manual/help text content."""
        return """
# Visual Image Master 2026

A powerful desktop application for browsing, sorting, filtering, and organizing large image collections with an intuitive three-pane interface.

## Overview

Visual Image Master provides a streamlined workflow for managing images through:
- **Source Tree (Left)**: Browse and select source directories
- **Thumbnail Grid (Center)**: View and interact with images
- **Destination Tree (Right)**: Select target folders for organizing

## Getting Started

### Installation

```bash
pip install -r requirements.txt
python main.py
```

### Requirements
- Python 3.8+
- PySide6 (Qt6 bindings)
- Pillow (image processing)
- OpenCV (face detection and quality analysis)
- findimagedupes (optional, for visual duplicate detection)

## Interface Guide

### Navigation

- **Source/Destination Trees**: Click folders to navigate; use path entry box to type paths directly
- **Browse Button (...)**: Open folder selection dialog
- **Thumbnail Grid**: Click to select, Ctrl+click for multi-select, Shift+click for range select

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Delete` | Delete selected files (with undo) |
| `Shift+Delete` | Permanently delete (no undo) |
| `Ctrl+Z` | Undo last operation |
| `Ctrl+R` | Rotate image 90° clockwise |
| `Ctrl+S` | Save cropped image (in preview) |
| `Enter` | Open preview window |
| `Ctrl+Q` | Exit application |
| `F1` | Open this manual |
| `Ctrl+Wheel` | Zoom thumbnails/trees/preview |

### Mouse Controls

- **Double-click thumbnail**: Open full-screen preview
- **Drag thumbnails**: Move/copy to destination folder
- **Ctrl+Drag**: Copy instead of move
- **Right-click thumbnail**: Context menu with file operations
- **Right-click folder**: Create, rename, or delete folders

## Features

### Filtering Images

Click the filter button (🔽) in the toolbar:
- **Left-click**: Toggle filter on/off
- **Right-click**: Select filter mode

Filter modes:
- **Has Face**: Show only images with detected faces
- **No Face**: Show only images without faces
- **Advanced**: Configure AI-powered filters (objects, quality, scenes)

### Sorting Images

Click the sort button (⇅) to access sort options:
- Name (A-Z or Z-A)
- Date Modified
- Resolution
- Width / Height
- Quality Score
- File Size

### Duplicate Removal

**Thumbnail Pane** (🗑️ button):
- Works on selected images or all images in view
- Right-click for mode options

**Folder Trees** (🗑️ button next to path):
- Works on selected folders (multi-select with Ctrl/Shift)
- Supports recursive processing

Deduplication modes:
- **Exact (MD5)**: Remove byte-identical files
- **Visual - Retain Largest**: Keep highest resolution
- **Visual - Retain Smallest**: Keep smallest file size
- **Visual - Retain Deepest**: Keep file deepest in folder structure
- **Visual - Retain Shallow**: Keep file closest to root

### Preview Window

Double-click any thumbnail or press Enter to open the preview:
- **Arrow keys**: Navigate between images
- **Escape**: Close preview
- **Scroll wheel**: Zoom in/out (when not in fit mode)

**Crop Functionality** (View > Crop menu):
- Rectangle: Free-form resizable crop
- Fixed sizes: 1024×1024, 768×768, 640×640, 512×512
- Drag to reposition, drag corners to resize
- `Ctrl+S` to save cropped image

### File Operations

**Moving/Copying Files**:
1. Select thumbnails in the grid
2. Drag to destination folder, or
3. Right-click > Move/Copy to Selected, or
4. Right-click > Move/Copy to New (enter path)

**Undo Support**:
- `Ctrl+Z` undoes the last move, copy, or delete
- Delete operations store files temporarily for recovery
- Shift+Delete bypasses undo (permanent deletion)

## Options Menu

- **Remember Settings**: Persist window layout and preferences
- **Show Filenames**: Display names below thumbnails
- **Show Dimensions**: Display image dimensions
- **Show File Size**: Display file sizes
- **Show Folder Counts**: Show image counts in folder trees
- **Confirm Deletions**: Prompt before deleting files
- **Face Detection Settings**: Configure detection preset (Fast/Balanced/Thorough)

## Face Detection

Three detection presets:
- **Fast**: Haar Cascade only (quickest, less accurate)
- **Balanced**: DNN-based detection (recommended)
- **Thorough**: Both methods combined (most accurate)

For best results with DNN detection, download the model files:
```bash
cd models
python download_models.py
```

## Tips & Tricks

1. **Quick Navigation**: Type or paste paths directly in the path entry boxes
2. **Batch Operations**: Select multiple files with Ctrl+click before moving/copying
3. **Zoom Levels**: Use Ctrl+wheel to zoom thumbnails, folder trees, or preview
4. **Filter Workflow**: Enable face filter to quickly sort portrait vs. landscape photos
5. **Duplicate Cleanup**: Use recursive mode on parent folders to scan entire hierarchies

## Troubleshooting

**"findimagedupes not found"**: Install the tool for visual duplicate detection:
```bash
# Ubuntu/Debian
sudo apt install findimagedupes

# Fedora
sudo dnf install findimagedupes

# Arch
sudo pacman -S findimagedupes
```

**Slow face detection**: Switch to "Fast" preset in Options > Face Detection Settings

**Missing DNN models**: Run `python models/download_models.py` to download detection models

---

For bug reports and feature requests, visit the project repository.
"""

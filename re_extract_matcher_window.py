"""
Re-Extract Matcher — pairs full-size originals with the (resized) extracts
already made from them, so a poorly-aligned extract can be redone manually.

Left pane (Originals) is filtered to show only images that have a fuzzy
filename match among the Extracts on the right — everything else is hidden,
since there's nothing to re-extract for an original with no derived crop.
Flip "Only show unmatched originals" to invert that filter instead — just
the originals with no extract yet, multi-selectable, so a batch of them can
be moved/copied out for extraction in one go. The Extracts pane is cleared
in that view since none of the shown originals have a match by definition.

Selecting a thumbnail in either pane selects and scrolls to its paired
thumbnail in the other pane. Originals can be dragged out to an external
editor (real file URLs, not just an internal reorder), or right-clicked to
move/copy them to the folder currently selected in the main window, delete
them, copy their path, or reveal them in the file manager.
"""

import os
import subprocess

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QSplitter,
    QStatusBar,
    QFileDialog,
    QAbstractItemView,
    QApplication,
    QMessageBox,
)
from PySide6.QtCore import Qt, QSize, QMimeData, QUrl, QThread, Signal
from PySide6.QtGui import QIcon, QAction

from filename_similarity import scan_sorted, pair_files
from utils import find_main_window

THUMB_SIZE = 150


class ContentMatchWorker(QThread):
    """Runs SIFT/RANSAC content matching off the UI thread.

    Filename similarity is fast but meaningless when extracts use an
    unrelated naming scheme (e.g. sequential "c__0014.jpg" crop numbers) —
    this compares actual pixel content instead, at the cost of taking
    tens of seconds to minutes depending on library size.
    """

    progress = Signal(int, int, str)
    finished_matching = Signal(list)  # [(original_path, extract_path, score), ...]
    error = Signal(str)

    def __init__(self, originals, extracts, cache_dir=None, parent=None):
        super().__init__(parent)
        self._originals = originals
        self._extracts = extracts
        self._cache_dir = cache_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from content_matcher import match_extracts_to_originals

            results = match_extracts_to_originals(
                self._originals,
                self._extracts,
                cache_dir=self._cache_dir,
                progress_callback=lambda done, total, msg: self.progress.emit(
                    done, total, msg
                ),
                should_cancel=lambda: self._cancelled,
            )
            if not self._cancelled:
                self.finished_matching.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class DraggableThumbList(QListWidget):
    """Thumbnail grid that drags real file URLs out to external applications."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)

    def mimeData(self, items):
        mime = QMimeData()
        urls = [
            QUrl.fromLocalFile(item.data(Qt.ItemDataRole.UserRole))
            for item in items
            if item.data(Qt.ItemDataRole.UserRole)
        ]
        mime.setUrls(urls)
        return mime


class ReExtractMatcherWindow(QMainWindow):
    """Two-pane tool: originals (matched-only) vs. their derived extracts."""

    def __init__(self, parent=None, dir_originals=None, dir_extracts=None):
        super().__init__(parent)
        self.setWindowTitle("Re-Extract Matcher")
        self.resize(1200, 750)

        self._pair_by_original = {}
        self._pair_by_extract = {}
        self._syncing = False
        self._match_mode = "filename"
        self._content_worker = None

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._build_ui()

        if dir_originals:
            self.orig_path_edit.setText(dir_originals)
        if dir_extracts:
            self.extract_path_edit.setText(dir_extracts)
        if dir_originals or dir_extracts:
            self.reload()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Originals:"))
        self.orig_path_edit = QLineEdit()
        self.orig_path_edit.setPlaceholderText("Folder of full-size originals...")
        self.orig_path_edit.returnPressed.connect(self.reload)
        top_row.addWidget(self.orig_path_edit, 1)
        orig_browse = QPushButton("...")
        orig_browse.setMaximumWidth(30)
        orig_browse.clicked.connect(self._browse_originals)
        top_row.addWidget(orig_browse)

        top_row.addSpacing(12)
        top_row.addWidget(QLabel("Extracts:"))
        self.extract_path_edit = QLineEdit()
        self.extract_path_edit.setPlaceholderText("Folder of resized extracts...")
        self.extract_path_edit.returnPressed.connect(self.reload)
        top_row.addWidget(self.extract_path_edit, 1)
        extract_browse = QPushButton("...")
        extract_browse.setMaximumWidth(30)
        extract_browse.clicked.connect(self._browse_extracts)
        top_row.addWidget(extract_browse)

        refresh_btn = QToolButton()
        refresh_btn.setText("⟳")
        refresh_btn.setToolTip("Refresh")
        refresh_btn.clicked.connect(self.reload)
        top_row.addWidget(refresh_btn)
        root.addLayout(top_row)

        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Match by:"))
        self.mode_filename_radio = QRadioButton("Filename similarity (fast)")
        self.mode_content_radio = QRadioButton("Image content (SIFT, slower)")
        self.mode_filename_radio.setChecked(True)
        self.mode_filename_radio.setToolTip(
            "Pairs by filename similarity. Fast, but only works when extract\n"
            "filenames are actually derived from the original's filename."
        )
        self.mode_content_radio.setToolTip(
            "Pairs by comparing actual image content (SIFT keypoints + RANSAC).\n"
            "Works regardless of naming convention, but takes time to run —\n"
            "roughly a couple of minutes for a few hundred images."
        )
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.mode_filename_radio)
        mode_group.addButton(self.mode_content_radio)
        self.mode_filename_radio.toggled.connect(self._on_mode_changed)
        self.mode_content_radio.toggled.connect(self._on_mode_changed)
        options_row.addWidget(self.mode_filename_radio)
        options_row.addWidget(self.mode_content_radio)

        self.run_content_match_btn = QPushButton("Run Content Match…")
        self.run_content_match_btn.setVisible(False)
        self.run_content_match_btn.clicked.connect(self._start_content_match)
        options_row.addWidget(self.run_content_match_btn)

        self.cancel_content_match_btn = QPushButton("Cancel")
        self.cancel_content_match_btn.setVisible(False)
        self.cancel_content_match_btn.clicked.connect(self._cancel_content_match)
        options_row.addWidget(self.cancel_content_match_btn)

        options_row.addSpacing(16)
        self.show_unmatched_check = QCheckBox("Show unmatched originals too")
        self.show_unmatched_check.setToolTip(
            "Check this to fall back to showing every original, not just the\n"
            "ones a match was found for."
        )
        self.show_unmatched_check.toggled.connect(self._on_show_unmatched_toggled)
        options_row.addWidget(self.show_unmatched_check)

        self.unmatched_only_check = QCheckBox("Only show unmatched originals")
        self.unmatched_only_check.setToolTip(
            "Check this to hide originals that already have a matching extract,\n"
            "leaving only the ones that still need one. The Extracts pane is\n"
            "cleared while this is on, since none of the shown originals have\n"
            "a match by definition. Enables multi-select on the Originals pane."
        )
        self.unmatched_only_check.toggled.connect(self._on_unmatched_only_toggled)
        options_row.addWidget(self.unmatched_only_check)
        options_row.addStretch()
        root.addLayout(options_row)

        self.match_progress = QProgressBar()
        self.match_progress.setVisible(False)
        root.addWidget(self.match_progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(QLabel("Originals (matched only)"))
        self.orig_list = DraggableThumbList()
        self._configure_list(self.orig_list)
        left_layout.addWidget(self.orig_list, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(QLabel("Extracts"))
        self.extract_list = DraggableThumbList()
        self._configure_list(self.extract_list)
        right_layout.addWidget(self.extract_list, 1)
        splitter.addWidget(right)

        splitter.setSizes([600, 600])
        root.addWidget(splitter, 1)

        self.orig_list.itemSelectionChanged.connect(self._on_orig_selection)
        self.extract_list.itemSelectionChanged.connect(self._on_extract_selection)
        self.orig_list.itemDoubleClicked.connect(self._on_orig_double_clicked)

        hint = QLabel(
            "Double-click an original to open it for re-extraction  ·  "
            "Drag a thumbnail out to another app  ·  "
            "Right-click for move / copy / delete / path actions"
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #777; font-size: 10px;")
        root.addWidget(hint)

    def _configure_list(self, list_widget):
        list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        list_widget.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        list_widget.setWordWrap(True)
        list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        list_widget.customContextMenuRequested.connect(
            lambda pos, lw=list_widget: self._show_context_menu(lw, pos)
        )

    # ── Directory pickers ────────────────────────────────────────────────

    def _browse_originals(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Originals Folder",
            self.orig_path_edit.text() or os.path.expanduser("~"),
        )
        if path:
            self.orig_path_edit.setText(path)
            self.reload()

    def _browse_extracts(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Extracts Folder",
            self.extract_path_edit.text() or os.path.expanduser("~"),
        )
        if path:
            self.extract_path_edit.setText(path)
            self.reload()

    # ── Loading ───────────────────────────────────────────────────────────

    def reload(self):
        dir_o = self.orig_path_edit.text().strip()
        dir_e = self.extract_path_edit.text().strip()

        self._files_o = scan_sorted(dir_o) if os.path.isdir(dir_o) else []
        self._files_e = scan_sorted(dir_e) if os.path.isdir(dir_e) else []

        if self._match_mode == "filename":
            matches = (
                pair_files(self._files_o, self._files_e)
                if self._files_o and self._files_e
                else []
            )
            self._pair_by_original = {po: (pe, score) for po, pe, score in matches}
            self._pair_by_extract = {pe: (po, score) for po, pe, score in matches}
        else:
            # Content matching is expensive — only (re)run it when the user
            # explicitly asks via "Run Content Match...", not on every reload.
            self._pair_by_original = {}
            self._pair_by_extract = {}

        self._populate_originals()
        self._populate_extracts()
        self._update_status()

    def _on_mode_changed(self, _checked):
        self._match_mode = (
            "content" if self.mode_content_radio.isChecked() else "filename"
        )
        self.run_content_match_btn.setVisible(self._match_mode == "content")
        self.reload()

    def _start_content_match(self):
        if self._content_worker is not None:
            return
        if not self._files_o or not self._files_e:
            self.status_bar.showMessage(
                "Load both an originals and an extracts folder first.", 4000
            )
            return

        cache_dir = os.path.join(
            self.orig_path_edit.text().strip(), ".img_sort_feature_cache"
        )

        self.run_content_match_btn.setVisible(False)
        self.cancel_content_match_btn.setVisible(True)
        self.match_progress.setVisible(True)
        self.match_progress.setValue(0)
        self.match_progress.setMaximum(len(self._files_o) + len(self._files_e))

        self._content_worker = ContentMatchWorker(
            self._files_o, self._files_e, cache_dir=cache_dir
        )
        self._content_worker.progress.connect(self._on_content_progress)
        self._content_worker.finished_matching.connect(self._on_content_finished)
        self._content_worker.error.connect(self._on_content_error)
        self._content_worker.start()

    def _cancel_content_match(self):
        if self._content_worker is not None:
            self._content_worker.cancel()
            self.status_bar.showMessage("Cancelling content match…")

    def _on_content_progress(self, done, total, message):
        self.match_progress.setMaximum(total)
        self.match_progress.setValue(done)
        self.status_bar.showMessage(message)

    def _on_content_finished(self, results):
        self._pair_by_original = {po: (pe, score) for po, pe, score in results}
        self._pair_by_extract = {pe: (po, score) for po, pe, score in results}
        self._teardown_content_worker()
        self._populate_originals()
        self._populate_extracts()
        self._update_status()

    def _on_content_error(self, message):
        self.status_bar.showMessage(f"Content match failed: {message}", 6000)
        self._teardown_content_worker()

    def _teardown_content_worker(self):
        if self._content_worker is not None:
            self._content_worker.wait()
            self._content_worker.deleteLater()
        self._content_worker = None
        self.run_content_match_btn.setVisible(self._match_mode == "content")
        self.cancel_content_match_btn.setVisible(False)
        self.match_progress.setVisible(False)

    def closeEvent(self, event):
        if self._content_worker is not None:
            self._content_worker.cancel()
            self._content_worker.wait()
            self._content_worker.deleteLater()
            self._content_worker = None
        super().closeEvent(event)

    def _populate_originals(self):
        self.orig_list.clear()
        unmatched_only = self.unmatched_only_check.isChecked()
        show_all = self.show_unmatched_check.isChecked() or (
            self._match_mode == "content" and not self._pair_by_original
        )
        for path in getattr(self, "_files_o", []):
            match = self._pair_by_original.get(path)
            if unmatched_only:
                if match is not None:
                    continue
            elif match is None and not show_all:
                continue
            item = QListWidgetItem(QIcon(path), os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            if match is not None:
                extract_path, score = match
                item.setToolTip(
                    f"{path}\nmatches {os.path.basename(extract_path)}  (score {score:.2f})"
                )
            else:
                item.setToolTip(f"{path}\n(no matching extract found)")
            self.orig_list.addItem(item)
        self._update_status()

    def _populate_extracts(self):
        self.extract_list.clear()
        if self.unmatched_only_check.isChecked():
            # Every original shown while this filter is on has no match by
            # definition, so there's nothing meaningful to pair against here.
            return
        for path in getattr(self, "_files_e", []):
            item = QListWidgetItem(QIcon(path), os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            match = self._pair_by_extract.get(path)
            if match is not None:
                orig_path, score = match
                item.setToolTip(
                    f"{path}\nmatches {os.path.basename(orig_path)}  (score {score:.2f})"
                )
            else:
                item.setToolTip(path)
            self.extract_list.addItem(item)

    def _update_status(self):
        n_o = len(getattr(self, "_files_o", []))
        n_e = len(getattr(self, "_files_e", []))
        n_matched = len(self._pair_by_original)
        if self._match_mode == "content" and not self._pair_by_original and n_o and n_e:
            self.status_bar.showMessage(
                f"{n_o} originals, {n_e} extracts loaded — "
                "click 'Run Content Match…' to pair them by image content."
            )
            return
        if self.unmatched_only_check.isChecked():
            self.status_bar.showMessage(
                f"{n_o - n_matched} of {n_o} originals have no matching extract"
            )
            return
        self.status_bar.showMessage(
            f"{n_matched} of {n_o} originals have a matching extract  ·  "
            f"{n_e} extracts total"
        )

    # ── Filter toggles ───────────────────────────────────────────────────

    def _on_show_unmatched_toggled(self, checked):
        if checked and self.unmatched_only_check.isChecked():
            self.unmatched_only_check.blockSignals(True)
            self.unmatched_only_check.setChecked(False)
            self.unmatched_only_check.blockSignals(False)
            self.orig_list.setSelectionMode(
                QAbstractItemView.SelectionMode.SingleSelection
            )
        self._populate_originals()
        self._populate_extracts()

    def _on_unmatched_only_toggled(self, checked):
        if checked and self.show_unmatched_check.isChecked():
            self.show_unmatched_check.blockSignals(True)
            self.show_unmatched_check.setChecked(False)
            self.show_unmatched_check.blockSignals(False)
        self.orig_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
            if checked
            else QAbstractItemView.SelectionMode.SingleSelection
        )
        self._populate_originals()
        self._populate_extracts()

    # ── Selection linking ────────────────────────────────────────────────

    def _on_orig_selection(self):
        if self._syncing:
            return
        items = self.orig_list.selectedItems()
        if len(items) != 1:
            return
        path = items[0].data(Qt.ItemDataRole.UserRole)
        match = self._pair_by_original.get(path)
        if match is None:
            return
        extract_path, _score = match
        self._select_in_list(self.extract_list, extract_path)

    def _on_extract_selection(self):
        if self._syncing:
            return
        items = self.extract_list.selectedItems()
        if len(items) != 1:
            return
        path = items[0].data(Qt.ItemDataRole.UserRole)
        match = self._pair_by_extract.get(path)
        if match is None:
            return
        orig_path, _score = match
        self._select_in_list(self.orig_list, orig_path)

    def _select_in_list(self, list_widget, path):
        self._syncing = True
        try:
            list_widget.clearSelection()
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == path:
                    item.setSelected(True)
                    list_widget.setCurrentItem(item)
                    list_widget.scrollToItem(item)
                    break
        finally:
            self._syncing = False

    def _on_orig_double_clicked(self, item):
        """Open the original in the preview window for manual re-extraction."""
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return

        from preview_window import ImagePreviewWindow

        # Resolve the real ImageSorter main window (not this tool window) so
        # ImagePreviewWindow's settings persistence (crop align mode, save
        # folder, etc.) reads/writes the actual app settings rather than a
        # dead-end attribute on ReExtractMatcherWindow.
        main_win = self.parentWidget() or find_main_window()
        win = ImagePreviewWindow([path], 0, parent=main_win)
        win.show()

    # ── Context menu ─────────────────────────────────────────────────────

    def _show_context_menu(self, list_widget, pos):
        item = list_widget.itemAt(pos)
        if not item:
            return

        selected_items = list_widget.selectedItems()
        if item not in selected_items:
            list_widget.clearSelection()
            item.setSelected(True)
            selected_items = [item]

        paths = [
            it.data(Qt.ItemDataRole.UserRole)
            for it in selected_items
            if it.data(Qt.ItemDataRole.UserRole)
        ]
        if not paths:
            return

        menu = QMenu(self)

        move_act = QAction("Move to Selected", self)
        move_act.triggered.connect(lambda: self._transfer_paths(paths, is_copy=False))
        menu.addAction(move_act)

        copy_to_act = QAction("Copy to Selected", self)
        copy_to_act.triggered.connect(lambda: self._transfer_paths(paths, is_copy=True))
        menu.addAction(copy_to_act)

        menu.addSeparator()

        copy_act = QAction("Copy Path to Clipboard", self)
        copy_act.triggered.connect(lambda: self._copy_paths(paths))
        menu.addAction(copy_act)

        reveal_act = QAction("Reveal in File Manager", self)
        reveal_act.triggered.connect(lambda: self._reveal_in_file_manager(paths[0]))
        menu.addAction(reveal_act)

        menu.addSeparator()
        delete_label = (
            "Delete File..." if len(paths) == 1 else f"Delete {len(paths)} Files..."
        )
        delete_act = QAction(delete_label, self)
        delete_act.triggered.connect(lambda: self._delete_files(paths))
        menu.addAction(delete_act)

        menu.exec(list_widget.viewport().mapToGlobal(pos))

    def _copy_paths(self, paths):
        QApplication.clipboard().setText("\n".join(paths))
        if len(paths) == 1:
            self.status_bar.showMessage(f"Copied path: {paths[0]}", 4000)
        else:
            self.status_bar.showMessage(f"Copied {len(paths)} paths", 4000)

    def _reveal_in_file_manager(self, path):
        try:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
        except OSError as e:
            self.status_bar.showMessage(f"Could not open file manager: {e}", 4000)

    def _get_selected_dest_folder(self, main_win):
        """Currently selected destination folder in the main window's RHS tree."""
        current_index = main_win.dest_tree.currentIndex()
        if current_index.isValid():
            return main_win.dest_model.filePath(
                main_win.dest_proxy.mapToSource(current_index)
            )
        return main_win.dest_root_path

    def _transfer_paths(self, paths, is_copy):
        """Move/copy files to the destination folder currently selected in
        the main window, mirroring the main thumbnail view's Move/Copy to
        Selected actions."""
        main_win = self.parentWidget() or find_main_window()
        if not main_win:
            return

        dest_folder = self._get_selected_dest_folder(main_win)
        if not dest_folder:
            QMessageBox.warning(
                self, "No Destination", "Please select a destination folder first."
            )
            return

        if hasattr(main_win, "start_operation_batch"):
            main_win.start_operation_batch("copy" if is_copy else "move")

        transferred = []
        failed = 0
        for path in paths:
            try:
                main_win.smart_transfer(path, dest_folder, is_copy)
                if not is_copy:
                    transferred.append(path)
            except Exception as e:  # noqa: BLE001
                failed += 1
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Could not transfer {os.path.basename(path)}:\n{e}",
                )

        if not is_copy and transferred:
            if hasattr(main_win, "remove_thumbnails_by_path"):
                main_win.remove_thumbnails_by_path(transferred)
            for path in transferred:
                self._remove_path_from_lists(path)
            self._populate_originals()
            self._populate_extracts()
            self._update_status()

        operation = "Copied" if is_copy else "Moved"
        success = len(paths) - failed
        if success > 0:
            self.status_bar.showMessage(
                f"{operation} {success} file(s) to {os.path.basename(dest_folder)}",
                4000,
            )

    def _delete_files(self, paths):
        """Delete files, respecting the app's confirm-delete setting and undo manager."""
        main_win = self.parentWidget() or find_main_window()

        should_confirm = True
        if main_win and hasattr(main_win, "conf_delete_act"):
            should_confirm = main_win.conf_delete_act.isChecked()

        if should_confirm:
            question = (
                f"Are you sure you want to delete {os.path.basename(paths[0])}?"
                if len(paths) == 1
                else f"Are you sure you want to delete {len(paths)} files?"
            )
            reply = QMessageBox.question(
                self,
                "Delete File" if len(paths) == 1 else "Delete Files",
                question,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        if main_win and hasattr(main_win, "delete_undo_manager"):
            main_win.delete_undo_manager.start_new_deletion()

        deleted = []
        for path in paths:
            try:
                if main_win and hasattr(main_win, "delete_undo_manager"):
                    main_win.delete_undo_manager.store_for_undo(path)
                elif os.path.exists(path):
                    os.remove(path)
                deleted.append(path)
            except OSError as e:
                QMessageBox.critical(
                    self, "Error", f"Could not delete {os.path.basename(path)}:\n{e}"
                )

        if not deleted:
            return

        if main_win and hasattr(main_win, "remove_thumbnails_by_path"):
            main_win.remove_thumbnails_by_path(deleted)

        for path in deleted:
            self._remove_path_from_lists(path)
        self._populate_originals()
        self._populate_extracts()
        self._update_status()

        message = f"Deleted {len(deleted)} file(s)"
        if main_win and hasattr(main_win, "delete_undo_manager"):
            message += " — Ctrl+Z in main window to undo"
        self.status_bar.showMessage(message, 4000)

    def _remove_path_from_lists(self, path):
        """Drop a moved/deleted file from the in-memory file/pair lists.

        Does not repaint — callers batch-remove several paths, then call
        _populate_originals()/_populate_extracts()/_update_status() once.
        """
        if hasattr(self, "_files_o") and path in self._files_o:
            self._files_o.remove(path)
        if hasattr(self, "_files_e") and path in self._files_e:
            self._files_e.remove(path)

        match = self._pair_by_original.pop(path, None)
        if match is not None:
            self._pair_by_extract.pop(match[0], None)

        match = self._pair_by_extract.pop(path, None)
        if match is not None:
            self._pair_by_original.pop(match[0], None)

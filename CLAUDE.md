# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Visual Image Master 2026 is a PySide6-based desktop application for browsing, sorting, filtering, and organizing images. It provides a three-pane interface: source folder tree (left), thumbnail grid (center), and destination folder tree (right). Users can drag-and-drop images between folders, filter by face detection, sort by various criteria, and preview images in a full-screen viewer.

## Development Commands

### Running the Application
```bash
python main.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

The application requires:
- PySide6 (Qt6 bindings)
- Pillow (image processing)
- opencv-python (face detection and quality analysis)

### Setting Up Face Detection Models
The application supports DNN-based face detection for improved accuracy. Download the models:
```bash
cd models
python download_models.py
```

If models are not downloaded, the application automatically falls back to Haar Cascade detection.

## Architecture

### Component Structure

The application follows a modular architecture with clear separation of concerns:

**Core Components:**
- `image_sorter.py` - Main application window (ImageSorter class), orchestrates all components
- `thumbnail_list.py` - Image thumbnail grid view with drag-and-drop support and context menu
- `folder_tree.py` - Dual folder tree views (source/destination) with custom delegate for image counts
- `preview_window.py` - Full-screen image preview window with navigation and crop functionality
- `crop_overlay.py` - Overlay widget for interactive image cropping with visual feedback
- `face_detection.py` - OpenCV-based face detection (Haar Cascade + DNN) and image quality analysis
- `face_detection_dialog.py` - Settings dialog for configuring face detection presets
- `settings_manager.py` - Persistent settings storage using QSettings
- `utils.py` - Utility functions (currently only main window finder)
- `models/` - Directory containing DNN face detection model files

### Key Architectural Patterns

**Settings Persistence:**
The application uses a centralized settings manager that saves/loads all user preferences to disk. Settings are only written when `_settings_loaded` is True to prevent premature saves during initialization. The main window is the "source of truth" for all settings, and other components read from it.

**Face Detection System:**
Face detection is computationally expensive, so `FaceDetector` maintains an in-memory cache (`_cache` dict) of detection results keyed by `image_path_preset`. The detector supports three presets:
- **Fast**: Haar Cascade only (fastest, less accurate)
- **Balanced**: DNN-based detection (good balance, default)
- **Thorough**: Both Haar + DNN (most accurate, slowest)

The system is CPU-only and modular - if DNN models aren't available, it automatically falls back to Haar Cascade. Users can change presets via Options → Face Detection Settings. Cache is cleared when preset changes. Similarly, `ImageQualityAnalyzer` caches blur scores. These caches are cleared when folder counts are refreshed.

**Drag-and-Drop Architecture:**
- `ThumbnailList` initiates drags with QMimeData containing file URLs
- `FolderTree` accepts drops and calls `ImageSorter.smart_transfer()`
- Ctrl modifier during drop triggers copy mode instead of move
- After moves, `remove_thumbnails_by_path()` updates the UI without full reload

**Context Menu Operations:**
Thumbnails support right-click context menu with file operations:
- **Move/Copy to Selected**: Transfer to currently selected destination folder
- **Move/Copy to New**: Prompt for path (absolute or relative to current destination), create directory if needed
- **Delete**: Uses existing deletion functionality with confirmation dialog
All operations use `smart_transfer()` for conflict resolution and update UI appropriately

**Preview Window Sync:**
Preview windows must stay synchronized with the main thumbnail list:
- When files are deleted from thumbnail list, it notifies all open `ImagePreviewWindow` instances via `handle_external_deletion()`
- Preview windows also call `remove_thumbnails_by_path()` when they delete files
- Settings (fit mode, geometry) are synced back to main window on changes

**Crop Functionality:**
Preview windows support interactive image cropping:
- **Crop Menu**: Mutually exclusive crop modes (Rectangle, 1024x1024, 768x768, 640x640, 512x512)
- **Rectangle Mode**: Resizable crop box (~30% of image, can resize via corner/edge handles)
- **Fixed Size Modes**: Fixed dimension crop boxes that can be moved but not resized
- **Visual Overlay**: Red border with semi-transparent dark overlay outside crop area (`CropOverlay` widget)
- **Coordinate Mapping**: Correctly maps displayed image coordinates to original image pixels accounting for zoom/scale
- **Save with Ctrl+S**: Saves cropped area to original file, respects confirmation setting
- **Auto-disable Fit Mode**: Cropping automatically switches to native size mode for accurate pixel selection

**Filter/Sort System:**
- Filter state managed by `filter_active` (bool) and `filter_mode` (str: "has_face"/"no_face"/None)
- Sort state managed by `sort_by` (str) and `sort_ascending` (bool)
- Both trigger `refresh_thumbnails()` which rebuilds the thumbnail list
- Visual indicators: filter button changes style when active, thumbnail list shows border

### Data Flow

1. **Initialization:** `main.py` → `ImageSorter.__init__()` → loads settings → populates trees → `refresh_thumbnails()`
2. **Thumbnail Loading:** User selects source folder → `on_source_folder_changed()` → `refresh_thumbnails()` → reads directory → optionally filters faces → sorts → populates `ThumbnailList`
3. **File Operations:** User drags thumbnails → `FolderTree.dropEvent()` → `smart_transfer()` → moves/copies file → `remove_thumbnails_by_path()` updates UI
4. **Settings Changes:** User modifies UI → event handler updates state → `write_settings()` → `SettingsManager` persists to QSettings

### Important Implementation Details

**Signal Blocking:**
When restoring settings, trees use `blockSignals(True/False)` to prevent triggering change events that would cause premature settings writes or unnecessary refreshes.

**Smart Transfer:**
The `smart_transfer()` method in `ImageSorter` handles file conflicts by appending `_N` to filenames (e.g., `image.jpg` → `image_1.jpg`) until a unique name is found.

**Zoom Functionality:**
- Thumbnail list: Ctrl+wheel zooms thumbnails (64-512px), reloads icons at new size
- Folder trees: Ctrl+wheel zooms font/icons (0.8x-2.0x multiplier), saves zoom level in settings
- Preview window: Ctrl+wheel zooms image (0.1x-10.0x) when not in fit-to-window mode

**Folder Count Delegate:**
`FolderCountDelegate` overrides `displayText()` to append image counts to folder names. It caches counts in `_count_cache` dict to avoid repeated directory scans. Cache is cleared on folder operations or when toggled off.

**Progress Bar:**
Long operations (face filtering, thumbnail loading) show progress bar in status bar. `QApplication.processEvents()` is called periodically to keep UI responsive.

**Thumbnail Display Options:**
The application supports displaying filenames or dimensions below thumbnails. These are mutually exclusive - only one can be active at a time, or neither. When a user enables one, the other is automatically disabled. The display is updated in three places:
- `toggle_filenames()` and `toggle_dimensions()` update existing thumbnails
- `refresh_thumbnails()` respects the setting when loading new thumbnails
- Settings are persisted via `SettingsManager`

## Common Development Patterns

When modifying this codebase:

**Adding New Sort Criteria:**
1. Add menu action in `setup_sort_menu()`
2. Add sort logic in `sort_files()` method
3. Add helper method like `get_image_*()` if needed
4. Update `sort_actions` dict with the new action

**Adding New Filter Criteria:**
1. Add menu action in `setup_filter_menu()`
2. Add detection logic (similar to face detection)
3. Add filter application in `refresh_thumbnails()`
4. Update `filter_actions` dict

**Adding New Settings:**
1. Add instance variable in `ImageSorter.__init__()`
2. Add save logic in `SettingsManager.write_settings()`
3. Add restore logic in `SettingsManager.read_settings()`
4. Ensure `write_settings()` is called when setting changes

**Working with Preview Windows:**
Preview windows are independent `QMainWindow` instances but must sync state with the main window. Always update both the preview's local state and the main window's corresponding setting, then call `main_win.write_settings()`.

## File Operations

The application handles file operations through Qt's drag-and-drop system. Files are never deleted by the OS file manager integration - only through the Delete key within the application UI. Deletions respect the "Confirm Deletions" setting.

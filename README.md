# Visual Image Master 2026

A powerful desktop application for browsing, sorting, filtering, and organizing large image collections with an intuitive three-pane interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PySide6](https://img.shields.io/badge/PySide6-Qt6-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Three-Pane Interface**: Source tree, thumbnail grid, and destination tree for efficient workflow
- **Drag-and-Drop**: Move or copy files between folders with visual feedback
- **Face Detection**: Filter images by face presence using Haar Cascade or DNN detection
- **AI-Powered Filtering**: Advanced filters for objects, quality scores, and scene recognition
- **Duplicate Removal**: Find and remove exact (MD5) or visual duplicates
- **Full-Screen Preview**: Navigate images with zoom, pan, and crop functionality
- **Batch Operations**: Select multiple files for bulk move, copy, or delete
- **Undo Support**: Recover from accidental deletions and moves with Ctrl+Z
- **Persistent Settings**: Remember window layout, paths, and preferences

## Screenshots

*Coming soon*

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/visual-image-master.git
cd visual-image-master

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Dependencies

The application requires the following Python packages:

```
PySide6>=6.0.0
Pillow>=9.0.0
opencv-python>=4.5.0
```

### Optional Dependencies

For visual duplicate detection, install `findimagedupes`:

```bash
# Ubuntu/Debian
sudo apt install findimagedupes

# Fedora
sudo dnf install findimagedupes

# Arch Linux
sudo pacman -S findimagedupes

# macOS (via Homebrew)
brew install findimagedupes
```

### Face Detection Models (Optional)

For improved face detection accuracy using DNN, download the model files:

```bash
cd models
python download_models.py
```

If models are not available, the application automatically falls back to Haar Cascade detection.

## Usage

### Interface Overview

```
+------------------+------------------+------------------+
|   Source Tree    |  Thumbnail Grid  | Destination Tree |
|                  |                  |                  |
|  [Path Entry]    |  [Filter] [Sort] |  [Path Entry]    |
|  [...] [Dedup]   |  [Dedup]         |  [...] [Dedup]   |
|                  |                  |                  |
|  /home/user/     |  +----+ +----+   |  /home/user/     |
|  ├── Photos/     |  |img1| |img2|   |  ├── Sorted/     |
|  │   ├── 2024/   |  +----+ +----+   |  │   ├── Keep/   |
|  │   └── 2025/   |  +----+ +----+   |  │   └── Delete/ |
|  └── Downloads/  |  |img3| |img4|   |  └── Archive/    |
|                  |  +----+ +----+   |                  |
+------------------+------------------+------------------+
|                      Status Bar                        |
+--------------------------------------------------------+
```

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
| `F1` | Open manual |
| `Ctrl+Wheel` | Zoom thumbnails/trees/preview |
| `Left/Right` | Navigate images (in preview) |
| `Escape` | Close preview window |

### Mouse Controls

- **Double-click thumbnail**: Open full-screen preview
- **Drag thumbnails**: Move to destination folder
- **Ctrl+Drag**: Copy instead of move
- **Right-click thumbnail**: Context menu with file operations
- **Right-click folder**: Create, rename, or delete folders
- **Ctrl+Scroll**: Zoom thumbnails, folder trees, or preview image

### Filtering Images

Click the filter button (🔽) in the toolbar:
- **Left-click**: Toggle filter on/off
- **Right-click**: Select filter mode

Available filter modes:
- **Has Face**: Show only images containing detected faces
- **No Face**: Show only images without detected faces
- **Advanced**: Configure AI-powered filters for objects, quality, and scenes

### Sorting Images

Click the sort button (⇅) to access sort options:
- **Name**: Alphabetical sorting (A-Z or Z-A)
- **Date Modified**: By file modification time
- **Resolution**: By total pixel count (width × height)
- **Width**: By image width
- **Height**: By image height
- **Quality**: By sharpness/blur score
- **File Size**: By file size in bytes

### Duplicate Removal

**From Thumbnail Pane** (🗑️ button):
1. Select specific images, or leave unselected to process all
2. Left-click to remove duplicates
3. Right-click to change deduplication mode

**From Folder Trees** (🗑️ button next to path):
1. Select one or more folders (Ctrl+click for multiple)
2. Left-click to remove duplicates in selected folders
3. Right-click to configure mode and recursive option

Deduplication modes:
- **Exact (MD5)**: Remove byte-identical files
- **Visual - Retain Largest**: Keep the file with largest dimensions
- **Visual - Retain Smallest**: Keep the file with smallest size
- **Visual - Retain Deepest**: Keep file deepest in folder hierarchy
- **Visual - Retain Shallow**: Keep file closest to root folder

### Preview Window

Double-click any thumbnail or press Enter to open full-screen preview:

- **Navigation**: Left/Right arrow keys or click edges
- **Zoom**: Ctrl+Scroll wheel (when not in fit-to-window mode)
- **Pan**: Click and drag when zoomed in
- **Close**: Escape key or close button

**Crop Functionality** (View > Crop menu):
- **Rectangle**: Free-form resizable crop area
- **Fixed Sizes**: 1024×1024, 768×768, 640×640, 512×512
- **Save**: Ctrl+S to save cropped region to original file

### File Operations

**Moving/Copying Files**:
1. Select thumbnails in the grid
2. Choose one of:
   - Drag to destination folder tree
   - Right-click > Move/Copy to Selected
   - Right-click > Move/Copy to New (enter custom path)

**Undo Support**:
- Press `Ctrl+Z` to undo the last operation
- Delete operations store files temporarily for recovery
- Use `Shift+Delete` for permanent deletion (bypasses undo)

## Configuration

### Options Menu

- **Remember Settings**: Save and restore window layout, paths, and preferences
- **Show Filenames**: Display filenames below thumbnails
- **Show Dimensions**: Display image dimensions below thumbnails
- **Show File Size**: Display file sizes below thumbnails
- **Show Folder Counts**: Display image counts next to folder names
- **Confirm Deletions**: Show confirmation dialog before deleting files
- **Face Detection Settings**: Choose detection preset (Fast/Balanced/Thorough)

### Face Detection Presets

| Preset | Method | Speed | Accuracy |
|--------|--------|-------|----------|
| Fast | Haar Cascade | Fastest | Good |
| Balanced | DNN | Medium | Better |
| Thorough | Both | Slowest | Best |

## Project Structure

```
visual-image-master/
├── main.py                 # Application entry point
├── image_sorter.py         # Main window and core logic
├── thumbnail_list.py       # Thumbnail grid widget
├── folder_tree.py          # Folder tree widgets
├── preview_window.py       # Full-screen image preview
├── crop_overlay.py         # Crop selection overlay
├── face_detection.py       # Face detection and quality analysis
├── face_detection_dialog.py # Face detection settings dialog
├── advanced_filter_dialog.py # Advanced filter configuration
├── object_detection.py     # Object detection module
├── quality_assessment.py   # Image quality scoring
├── scene_recognition.py    # Scene classification
├── settings_manager.py     # Settings persistence
├── utils.py                # Utility functions
├── models/                 # DNN model files
│   └── download_models.py  # Model download script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Troubleshooting

### "findimagedupes not found"

Install the findimagedupes tool for visual duplicate detection. See [Optional Dependencies](#optional-dependencies).

### Slow face detection

Switch to the "Fast" preset in Options > Face Detection Settings. This uses only Haar Cascade detection which is significantly faster.

### Missing DNN models

If face detection isn't working well, you may need to download the DNN models:

```bash
cd models
python download_models.py
```

### High memory usage

For very large image collections:
- Process images in smaller batches
- Reduce thumbnail size with Ctrl+Scroll wheel
- Close preview windows when not in use

### Application won't start

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Check Python version (requires 3.8+):

```bash
python --version
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PySide6](https://wiki.qt.io/Qt_for_Python) - Qt6 Python bindings
- [OpenCV](https://opencv.org/) - Computer vision library
- [Pillow](https://python-pillow.org/) - Python Imaging Library
- [findimagedupes](https://github.com/jhnc/findimagedupes) - Visual duplicate detection

## Changelog

### Version 1.0.0
- Initial release
- Three-pane interface with drag-and-drop
- Face detection filtering (Haar + DNN)
- Advanced AI filtering (objects, quality, scenes)
- Duplicate removal (exact and visual)
- Full-screen preview with crop
- Undo support for file operations
- Persistent settings

---

**Visual Image Master 2026** - Organize your images with ease.

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
- **Image Enhancement**: Built-in adjustments (brightness, contrast, sharpen, blur, denoise, etc.) with live split-view preview
- **AI Model Enhancement**: Load .pth/.safetensors super-resolution models via Spandrel for upscaling, denoising, and restoration
- **Auto Captioning**: Generate image captions using Florence2, WD14, or Ollama backends
- **Batch Captioning**: Generate captions for entire folders in one operation
- **Caption Blending**: Use Ollama LLMs to intelligently merge and refine captions
- **YOLO Labeling**: Create object detection labels in YOLO format for ML training
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
git clone https://github.com/BillyBGit/visual-image-master.git
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

**For Auto Captioning** (Florence2/WD14):
```bash
pip install transformers torch torchvision
pip install onnxruntime  # For WD14 tagger
```

**For Ollama Integration**:
- Install Ollama from https://ollama.com
- Pull a vision model: `ollama pull llava`
- Start the server: `ollama serve`

**For AI Model Enhancement** (super-resolution, denoising, restoration):
```bash
pip install spandrel torch
```
Download `.pth` or `.safetensors` model files from [OpenModelDB](https://openmodeldb.info) and place them in the `models/` directory. Supports 50+ architectures including ESRGAN, SwinIR, HAT, SPAN, Real-CUGAN, and SCUNet.

**For visual duplicate detection**, install `findimagedupes`:

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

**Enhancement** (Enhance menu):
- Apply built-in adjustments or AI models directly from the preview window
- Opens the Enhancement dialog with split-view preview

### Image Enhancement

Access via the Enhance menu in the preview window, or right-click a thumbnail and select Enhance.

**Built-In Operations** (no extra dependencies):
- Brightness, Contrast, Saturation, Sharpness
- Gaussian Blur, Bilateral Denoise
- Auto Levels, Histogram Equalization (CLAHE)

**AI Model Enhancement** (requires `spandrel` and `torch`):
- Load any `.pth` or `.safetensors` model from the `models/` directory
- Supports 50+ architectures: ESRGAN, SwinIR, HAT, SPAN, Real-CUGAN, SCUNet, etc.
- Use for super-resolution upscaling, denoising, and image restoration

**Split-View Preview**:
- Draggable vertical divider shows original (left) vs enhanced (right) side-by-side
- Colored labels ("Original" in cyan, "Enhanced" in green) for clarity
- Apply to save the enhanced result, or Cancel to discard

### Auto Caption Tool

Access via View > Auto Caption or right-click context menu on thumbnails.

**Caption Backends**:

| Backend | Type | Description |
|---------|------|-------------|
| Florence2 | Local | Microsoft's vision-language model for detailed captions |
| WD14 | Local | Danbooru-style tag generation for anime/illustration |
| Ollama | Server | Use any Ollama vision model (llava, bakllava, etc.) |

**Two-Box Interface**:
- **Generated Caption**: Shows AI-generated caption, editable before use
- **Saved Caption**: Shows caption from .txt file, saved alongside image

**Action Buttons**:
- **Replace (⬇)**: Copy generated caption to saved caption box
- **Append (+)**: Add generated caption after saved caption
- **Blend (⚡)**: Use Ollama to intelligently merge both captions

**Caption Blending**:
1. Generate a caption using any backend
2. Edit the blend instructions if needed (or use defaults)
3. Select an Ollama model for blending (text-only models work fine)
4. Click Blend to merge generated and saved captions
5. Result appears in Generated Caption box for review
6. Use Ctrl+Z/Ctrl+Y to toggle between original and blended text
7. Click Replace when satisfied to update the saved caption

**Keyboard Shortcuts** (in Auto Caption window):
- `Ctrl+S`: Save caption to .txt file
- `Ctrl+Z`: Undo (restore previous generated caption)
- `Ctrl+Y`: Redo (restore blended caption)
- `Left/Right`: Navigate between images
- `Escape`: Close window

### YOLO Labeling Window

Access via View > YOLO Labels for creating object detection training data.

- Draw bounding boxes around objects
- Assign class labels to each box
- Export labels in YOLO format (.txt files)
- Supports multiple classes per image

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
├── main.py                    # Application entry point
├── image_sorter.py            # Main window (core init, settings, navigation)
├── image_sorter_filters.py    # FilteringMixin (sort, filter, dedup menus)
├── image_sorter_operations.py # FileOperationsMixin (file ops, tools, undo)
├── thumbnail_list.py          # Thumbnail grid widget
├── folder_tree.py             # Folder tree widgets
├── preview_window.py          # Full-screen image preview with crop/enhance
├── crop_overlay.py            # Crop selection overlay
├── enhance_dialog.py          # Image enhancement dialog (split-view preview)
├── image_enhance.py           # Enhancement backends (OpenCV + Spandrel AI)
├── caption_window.py          # Auto Caption tool window
├── caption_generator.py       # Caption generation backends (Florence2, WD14, Ollama)
├── autocaption_dialog.py      # Batch captioning progress dialog
├── yolo_window.py             # YOLO label editor window
├── yolo_label_panel.py        # YOLO label panel widget
├── yolo_overlay.py            # YOLO bounding box overlay
├── yolo_file_handler.py       # YOLO file I/O (labels, yaml)
├── face_detection.py          # Face detection and quality analysis
├── face_detection_dialog.py   # Face detection settings dialog
├── advanced_filter_dialog.py  # Advanced filter configuration
├── object_detection.py        # Object detection module
├── quality_assessment.py      # Image quality scoring
├── scene_recognition.py       # Scene classification
├── settings_manager.py        # Settings persistence
├── utils.py                   # Utility functions
├── models/                    # DNN and AI enhancement model files
│   └── download_models.py     # Model download script
├── requirements.txt           # Python dependencies
├── MANUAL.md                  # User manual
└── README.md                  # This file
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

### Ollama not detected

If the caption blending feature can't find Ollama:

1. **Check if Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Start Ollama if not running**:
   ```bash
   ollama serve
   ```

3. **If using Open-WebUI with Docker**, ensure Ollama port is exposed or run Ollama standalone.

4. **Enable debug mode** to see connection attempts:
   ```bash
   IMG_SORT_DEBUG=1 python main.py
   ```

### ONNX CUDAExecutionProvider warning

If you see this warning when using WD14:
```
UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.
```

This means WD14 is falling back to CPU. To enable GPU acceleration:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

Requires an NVIDIA GPU with CUDA installed. The warning is harmless if you only have CPU available.

### AI Model enhancement not available

If the AI Model option shows install instructions instead of model controls:

```bash
pip install spandrel torch
```

Place `.pth` or `.safetensors` model files in the `models/` directory (or browse to any folder via the dialog). Models can be downloaded from [OpenModelDB](https://openmodeldb.info).

### Caption generation slow or unresponsive

- Florence2 and WD14 models load on first use (may take 30-60 seconds initially)
- Ollama response time depends on model size and hardware
- The UI shows "Blending captions with Ollama..." during processing
- Check terminal for debug output if issues persist

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
- [Spandrel](https://github.com/chaiNNer-org/spandrel) - Universal PyTorch model loader for image enhancement
- [OpenModelDB](https://openmodeldb.info) - Community database of AI enhancement models
- [findimagedupes](https://github.com/jhnc/findimagedupes) - Visual duplicate detection

## Changelog

### Version 1.2.0
- Image Enhancement dialog with live split-view preview (draggable before/after divider)
- Built-in OpenCV operations: brightness, contrast, saturation, sharpness, blur, denoise, auto levels, CLAHE
- AI Model enhancement via Spandrel: load .pth/.safetensors models for upscaling, denoising, and restoration
- Batch caption generation for entire folders (Tools > Batch Generate Captions)
- Enhance menu in preview window (Enhance > operation submenu)
- AI Model option always visible with install instructions when Spandrel not available
- Auto-detection of models/ directory for AI model files
- Refactored image_sorter.py into modular mixin architecture (FilteringMixin, FileOperationsMixin)
- User manual added (MANUAL.md, accessible via Help > Manual or F1)

### Version 1.1.0
- Auto Caption Tool with multiple backends (Florence2, WD14, Ollama)
- Caption blending using Ollama LLMs
- Two-box caption interface (Generated/Saved)
- Undo/redo support for caption blending (Ctrl+Z/Ctrl+Y)
- YOLO labeling window for object detection training data
- Automatic Ollama endpoint detection (standalone and Docker)
- Debug mode for troubleshooting (`IMG_SORT_DEBUG=1`)
- Progress feedback during long-running operations

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

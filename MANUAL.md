# Visual Image Master 2026 - User Manual

This manual provides step-by-step procedures for every feature in Visual Image Master 2026. It is organised by task so you can quickly find how to do what you need.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Main Interface Overview](#main-interface-overview)
3. [Browsing and Navigating Folders](#browsing-and-navigating-folders)
4. [Working with Thumbnails](#working-with-thumbnails)
5. [Moving and Copying Files](#moving-and-copying-files)
6. [Deleting Files and Undo](#deleting-files-and-undo)
7. [Filtering Images](#filtering-images)
8. [Sorting Images](#sorting-images)
9. [Removing Duplicates](#removing-duplicates)
10. [Preview Window](#preview-window)
11. [Cropping Images](#cropping-images)
12. [Image Enhancement](#image-enhancement)
13. [AI Model Enhancement](#ai-model-enhancement)
14. [Auto Captioning](#auto-captioning)
15. [Batch Caption Generation](#batch-caption-generation)
16. [Caption Blending with Ollama](#caption-blending-with-ollama)
17. [YOLO Labeling](#yolo-labeling)
18. [Settings and Preferences](#settings-and-preferences)
19. [Keyboard Shortcuts Reference](#keyboard-shortcuts-reference)
20. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Launching the Application

1. Activate the virtual environment:
   ```bash
   source ~/scripts/venv/bin/activate
   ```
2. Run the application:
   ```bash
   python main.py
   ```
3. To enable debug output (useful for troubleshooting):
   ```bash
   python main.py --debug
   ```

### First Launch

On first launch, the application opens with both folder trees pointing to your home directory. You will need to:

1. Set a **source folder** (left pane) -- this is where your images live
2. Set a **destination folder** (right pane) -- this is where you want to organise images into

The application remembers your last session state, so subsequent launches restore your previous folder paths, window size, and all settings.

### Installing Optional Features

| Feature | Install Command |
|---------|----------------|
| Auto Caption (Florence2) | `pip install transformers torch torchvision` |
| Auto Caption (WD14) | `pip install onnxruntime` (or `onnxruntime-gpu` for NVIDIA) |
| Auto Caption (Ollama) | Install from https://ollama.com, then `ollama pull llava` |
| AI Model Enhancement | `pip install spandrel torch` |
| Visual Duplicate Detection | `sudo apt install findimagedupes` |
| DNN Face Detection | `cd models && python download_models.py` |

---

## Main Interface Overview

The application uses a three-pane layout:

```
+-------------------+--------------------+--------------------+
| SOURCE TREE       | THUMBNAIL GRID     | DESTINATION TREE   |
|                   |                    |                    |
| [Path Entry][...] | [Filter][Sort]     | [Path Entry][...]  |
| [Dedup]           | [Dedup][Caption]   | [Dedup]            |
|                   |                    |                    |
| /home/user/       |  +----+  +----+   | /home/user/        |
| +-- Photos/       |  |img1|  |img2|   | +-- Sorted/        |
| |   +-- 2024/     |  +----+  +----+   | |   +-- Keep/      |
| |   +-- 2025/     |  +----+  +----+   | |   +-- Delete/    |
| +-- Downloads/    |  |img3|  |img4|   | +-- Archive/       |
|                   |  +----+  +----+   |                    |
+-------------------+--------------------+--------------------+
| Status Bar                                    [Progress]    |
+-------------------------------------------------------------+
```

**Menu Bar**: File, Options, Tools, Help

**Toolbar Buttons** (above thumbnail grid):
- **Filter** (funnel icon) -- left-click toggles, right-click opens filter menu
- **Sort** (arrows icon) -- click to open sort menu
- **Dedup** (bin icon) -- left-click runs dedup, right-click for mode options
- **Caption** (notepad icon) -- left-click runs batch caption, right-click for backend settings

**Status Bar**: Shows current operation status and a progress bar for long operations.

---

## Browsing and Navigating Folders

### Setting a Source Folder

**Method 1 -- Type a path:**
1. Click the path entry box at the top of the left pane
2. Type or paste the full path (e.g. `/home/user/Photos`)
3. Press **Enter**

**Method 2 -- Browse button:**
1. Click the **...** button next to the source path entry
2. Navigate to the desired folder in the dialog
3. Click **Select Folder**

**Method 3 -- Click in the tree:**
1. Click any folder in the source tree to select it
2. Thumbnails update immediately to show images in that folder

### Setting a Destination Folder

Follow the same steps using the right pane's path entry, browse button, or tree.

### Zooming Folder Trees

Hold **Ctrl** and scroll the mouse wheel while hovering over a folder tree to zoom the font and icons. The zoom level (0.8x to 2.0x) is saved per tree.

---

## Working with Thumbnails

### Viewing Thumbnails

Thumbnails load automatically when you select a source folder. The status bar shows loading progress for large directories.

### Selecting Thumbnails

- **Single select**: Click a thumbnail
- **Multi-select**: Hold **Ctrl** and click additional thumbnails
- **Range select**: Hold **Shift** and click to select a range

### Zooming Thumbnails

Hold **Ctrl** and scroll the mouse wheel over the thumbnail grid to resize thumbnails (64px to 512px). Icons reload at the new size for crisp display.

### Displaying Information Below Thumbnails

From the **Options** menu, enable one of (these are mutually exclusive):

- **Show Filenames** -- displays the file name
- **Show Dimensions** -- displays width x height (e.g. `1920x1080`)
- **Show File Size** -- displays human-readable size (e.g. `2.4MB`)

### Opening a Preview

- **Double-click** a thumbnail, or
- **Select** a thumbnail and press **Enter**

This opens a full-screen preview window (see [Preview Window](#preview-window)).

### Right-Click Context Menu

Right-click on a thumbnail (or selection of thumbnails) to access:

| Menu Item | Action |
|-----------|--------|
| **Move to Selected** | Move file(s) to the currently selected destination folder |
| **Move to New...** | Move to a new path (enter absolute or relative path, created if needed) |
| **Copy to Selected** | Copy file(s) to the currently selected destination folder |
| **Copy to New...** | Copy to a new path |
| **Auto Caption...** | Open the Auto Caption window for selected image(s) |
| **Enhance...** | Open the Image Enhancement dialog for selected image(s) |
| **Delete** | Delete selected file(s) (respects Confirm Deletions setting) |

---

## Moving and Copying Files

### Drag and Drop

1. Select one or more thumbnails
2. Drag them onto a folder in the **destination tree** (right pane)
3. Release to **move** the files
4. Hold **Ctrl** while dropping to **copy** instead of move

### Context Menu

1. Select the destination folder by clicking it in the right tree
2. Right-click the thumbnail(s) you want to transfer
3. Choose **Move to Selected** or **Copy to Selected**

### Move/Copy to New Path

1. Right-click thumbnail(s)
2. Choose **Move to New...** or **Copy to New...**
3. Enter a path:
   - **Absolute path**: e.g. `/home/user/sorted/landscapes`
   - **Relative path**: e.g. `landscapes` (created relative to current destination root)
4. The folder is created automatically if it doesn't exist

### Conflict Resolution

If a file with the same name exists at the destination, the application automatically appends `_1`, `_2`, etc. to the filename (e.g. `photo.jpg` becomes `photo_1.jpg`).

### Undoing Transfers

Press **Ctrl+Z** immediately after a move or copy to reverse the operation. Only the most recent operation can be undone.

---

## Deleting Files and Undo

### Standard Delete (with Undo)

1. Select one or more thumbnails
2. Press **Delete**
3. If **Confirm Deletions** is enabled (Options menu), confirm the dialog
4. Files are moved to a temporary location for potential recovery

### Permanent Delete (no Undo)

1. Select one or more thumbnails
2. Press **Shift+Delete**
3. Files are permanently removed from disk

### Restoring Deleted Files

Press **Ctrl+Z** to restore the most recently deleted file(s). Only the last deletion batch can be restored. Performing another delete clears the previous undo history.

---

## Filtering Images

### Quick Toggle

**Left-click** the filter button to toggle the current filter on/off. When active, the button turns blue and the thumbnail list shows a coloured border.

### Selecting a Filter Mode

**Right-click** the filter button to open the filter menu:

| Filter | Effect |
|--------|--------|
| **Contains Face** | Show only images where a face is detected |
| **Does Not Contain Face** | Show only images with no faces detected |
| **Advanced AI Filters...** | Opens dialog for object/quality/scene filters |
| **Invert Filter** | Reverses the current filter result (advanced mode only) |

### Using Advanced AI Filters

1. Right-click the filter button
2. Select **Advanced AI Filters...**
3. Configure one or more filter criteria:

   **Face Detection**: Set minimum face count and toggle contains/does-not-contain mode.

   **Object Detection**: Select object categories (person, car, animal, etc.) to filter by. Optionally load a custom YOLO model for specialised detection.

   **Scene Recognition**: Select scene groups (outdoor, indoor, buildings, etc.).

   **Quality Filter**: Set minimum thresholds for overall score, sharpness, and brightness range.

   **File Size Filter**: Set minimum and maximum file size in KB.

4. Click **OK** to apply
5. The filter evaluates each image against all enabled criteria

### Face Detection Presets

Configure via **Options > Face Detection Settings...**:

| Preset | Method | Speed | Accuracy |
|--------|--------|-------|----------|
| Fast | Haar Cascade only | Fastest | Good |
| Balanced | DNN-based | Medium | Better |
| Thorough | Both methods combined | Slowest | Best |

Changing the preset clears the detection cache.

---

## Sorting Images

Click the **sort button** (double arrow icon) in the toolbar to open the sort menu. Each option sorts in either ascending or descending order:

| Sort Criteria | Ascending | Descending |
|--------------|-----------|------------|
| **Name** | A to Z | Z to A |
| **Modified Date** | Oldest first | Newest first |
| **Resolution** | Low to high pixel count | High to low |
| **Width** | Narrow to wide | Wide to narrow |
| **Height** | Short to tall | Tall to short |
| **Quality** | Blurry to sharp | Sharp to blurry |
| **File Size** | Smallest first | Largest first |

The current sort option is checked in the menu. Sorting is applied immediately and persisted across sessions.

---

## Removing Duplicates

### From the Thumbnail Grid

1. Optionally select specific thumbnails (or leave none selected to process all)
2. **Left-click** the dedup button (bin icon) in the toolbar
3. Confirm the operation when prompted
4. Duplicates are permanently deleted

**Right-click** the dedup button to choose a mode:
- **Exact (MD5 checksum)** -- removes byte-identical files
- **Visual - Retain Largest** -- keeps the largest file from each duplicate group
- **Visual - Retain Smallest** -- keeps the smallest file

### From Folder Trees

1. Select one or more folders in the source or destination tree (Ctrl+click for multi-select)
2. Click the **dedup button** next to the path entry
3. Confirm the operation

**Right-click** the folder dedup button for additional options:
- All modes above, plus:
- **Visual - Retain Deepest** -- keeps the file deepest in the directory hierarchy
- **Visual - Retain Shallow** -- keeps the file closest to the root
- **Recursive (include subdirectories)** -- toggle to process all subfolders

### Requirements

Visual duplicate detection requires `findimagedupes` to be installed on your system. Exact (MD5) duplicate detection works without any extra tools.

---

## Preview Window

### Opening the Preview

Double-click any thumbnail or select one and press **Enter**.

### Navigation

| Action | Control |
|--------|---------|
| Next image | **Right arrow** or click right edge |
| Previous image | **Left arrow** or click left edge |
| Close | **Escape** |

### Zoom and Pan

| Action | Control |
|--------|---------|
| Zoom in/out | **Ctrl+Scroll wheel** (disabled in Fit to Window mode) |
| Pan | Click and drag when zoomed in |
| Fit to Window | Options > Fit to Window |
| Native Size (1:1) | Options > Native Size |

### Preview Menu Bar

**Options menu:**
- **Fit to Window** -- scale image to fit the window
- **Native Size** -- display at 1:1 pixel ratio
- **Autoname Extract** -- extract filenames for naming

**Enhance menu** (applies to the currently viewed image):
- Sharpen, Denoise, Brightness/Contrast, Auto Levels, Color Saturation, Resize
- AI Model (opens Enhancement dialog for current image)

**Crop menu** (see [Cropping Images](#cropping-images))

---

## Cropping Images

### Activating Crop Mode

In the preview window, go to **Crop** menu and select a mode:

| Mode | Behaviour |
|------|-----------|
| **Rectangle** | Free-form resizable crop box (starts at ~30% of image) |
| **1024x1024** | Fixed square crop, draggable but not resizable |
| **768x768** | Fixed square crop |
| **640x640** | Fixed square crop |
| **512x512** | Fixed square crop |

### Using the Crop Overlay

- A red-bordered crop area appears with a semi-transparent dark overlay outside
- **Drag** the crop box to reposition
- **Drag corners/edges** to resize (Rectangle mode only)
- Crop mode automatically switches to Native Size for accurate pixel selection

### Saving the Crop

Press **Ctrl+S** to save. The cropped region replaces the original file (respects the Confirm Deletions setting).

### Disabling Crop

Select the currently active crop mode again from the Crop menu to toggle it off.

---

## Image Enhancement

### Opening the Enhancement Dialog

**From the main window:**
- **Tools > Image Enhance...** -- opens for selected thumbnails (or all, with confirmation)
- **Right-click thumbnail > Enhance...** -- opens for selected thumbnails

**From the preview window:**
- **Enhance** menu -- select a specific operation to open the dialog pre-configured

### Available Operations

| Operation | Description | Parameters |
|-----------|-------------|------------|
| **Sharpen (Unsharp Mask)** | GIMP-style sharpening using Gaussian blur + weighted addition | Amount (0.1-5.0), Radius (0.5-10.0), Threshold (0-255) |
| **Denoise** | Non-Local Means denoising preserving detail | Strength (1-30), Color Strength (1-30) |
| **Brightness / Contrast** | Linear brightness offset and contrast scaling | Brightness (-100 to +100), Contrast (0.1-3.0) |
| **Auto Levels (CLAHE)** | Adaptive histogram equalization on the L channel | Clip Limit (1.0-10.0), Grid Size (4-16) |
| **Color Saturation** | HSV saturation adjustment | Factor (0.0-3.0, where 0=grayscale, 1=unchanged) |
| **Resize** | Resize by dimensions or percentage with Lanczos resampling | Width, Height, Maintain Aspect Ratio, Scale % |

### Using the Enhancement Dialog

1. Select an operation from the **Operation** dropdown
2. Adjust parameters using the sliders and spinboxes in the left panel
3. In single-image mode, the **split-view preview** updates automatically:
   - Left side shows the **Original** (cyan label)
   - Right side shows the **Enhanced** result (green label)
   - **Drag the centre divider** left/right to compare different areas
4. Choose a **Save Mode**:
   - **Overwrite original** -- replaces the source file
   - **Save as copy** -- creates a new file with a configurable suffix (default: `_enhanced`)
5. Click **Apply** (or **Apply to All** in batch mode)

### Batch Enhancement

When multiple images are selected (or no selection with confirmation to process all):
- The dialog opens in batch mode showing the file count
- No preview is available; adjust parameters then click **Apply to All**
- A progress dialog tracks processing

---

## AI Model Enhancement

AI Model Enhancement uses `.pth` and `.safetensors` model files via the [Spandrel](https://github.com/chaiNNer-org/spandrel) library. This supports 50+ architectures including ESRGAN, SwinIR, HAT, SPAN, SCUNet, GFPGAN, CodeFormer, and more -- for upscaling, denoising, restoration, and other enhancement tasks.

### Prerequisites

```bash
pip install spandrel torch
```

### Getting Model Files

Download models from [OpenModelDB](https://openmodeldb.info/). Place `.pth` or `.safetensors` files in the `models/` directory within the project folder (or any directory of your choice).

### Using AI Models

1. Open the Enhancement dialog (Tools > Image Enhance, or right-click > Enhance)
2. Select **AI Model (Upscale / Denoise / Restore)** from the Operation dropdown
3. If spandrel is not installed, the left panel shows install instructions
4. In the **AI Model Settings** panel at the bottom:
   - **Model Directory** defaults to the project's `models/` folder; click **Browse** to change
   - **Model** dropdown lists all detected `.pth`/`.safetensors`/`.pt`/`.ckpt` files with sizes
   - Click **Refresh** after adding new model files
   - **Info** shows the detected architecture, scale factor, and channel count
   - **Device**: CPU or CUDA (GPU, if available)
   - **Tile Size**: Controls memory usage for large images (512x512 default; 0 = no tiling)
5. Click **Apply**

### Understanding Model Names

Model filenames typically encode their function:
- **1x-** prefix: Same-size processing (denoising, cleanup, restoration)
- **2x-** prefix: 2x upscaling
- **4x-** prefix: 4x upscaling
- Keywords like "Denoise", "Cleanup", "Sharp", "ReFocus" indicate the purpose

---

## Auto Captioning

### Opening the Caption Window

- **Tools > Auto Caption...** from the main menu
- **Right-click thumbnail > Auto Caption...**
- The window opens maximised with the current image list

### Caption Window Layout

The window shows the current image with two text boxes:

- **Generated Caption** (top) -- AI-generated caption, editable
- **Saved Caption** (bottom) -- contents of the `.txt` file saved alongside the image

### Generating a Caption

1. Navigate to an image using **Left/Right** arrow keys
2. The backend generates a caption automatically (or click **Generate**)
3. Review and edit the generated text in the top box

### Caption Backends

Configure by right-clicking the **Caption** toolbar button in the main window:

| Backend | Type | Best For |
|---------|------|----------|
| **Florence2** | Local GPU/CPU | Detailed natural-language descriptions |
| **WD14** | Local GPU/CPU | Anime/illustration tag-style captions |
| **Ollama** | Server | Flexible prompting with any vision model |

**Florence2 options:**
- Model size: Base (~460MB) or Large (~1.5GB), with fine-tuned variants
- Detail level: Brief, Standard, or Detailed

**WD14 options:**
- Models: ViT v2, SwinV2 v2, ConvNeXtV2 v2, MOAT v2, ViT v3
- Threshold controls tag inclusion sensitivity (default 0.55)

**Ollama options:**
- Shows all models available on your Ollama server
- Requires Ollama running at the configured URL

### Saving Captions

- **Replace** (down arrow) -- copies the generated caption to the saved caption box
- Click **Save** or press **Ctrl+S** to write the saved caption to a `.txt` file alongside the image
- The `.txt` file has the same name as the image (e.g. `photo.jpg` -> `photo.txt`)

### Appending Captions

Click **Append** (+) to add the generated caption text after the existing saved caption, separated by a space.

---

## Batch Caption Generation

### Quick Batch Captioning

1. In the main window, optionally select specific thumbnails
2. **Left-click** the Caption toolbar button (notepad icon)
3. If no thumbnails are selected, you're asked whether to caption all images in view
4. The application checks for existing `.txt` files and asks:
   - **Skip existing** -- only generate captions for images without `.txt` files
   - **Overwrite all** -- regenerate all captions
5. A progress dialog shows the current file and count
6. Results summary shows success/failure counts

### Configuring the Backend

**Right-click** the Caption toolbar button to select:
- Caption backend (Florence2, WD14, Ollama)
- Model and detail settings specific to each backend
- **Open Caption Window...** to launch the full caption interface

---

## Caption Blending with Ollama

Caption blending takes two captions (generated + saved) and uses an Ollama LLM to merge them intelligently.

### Prerequisites

- Ollama installed and running (`ollama serve`)
- At least one text model pulled (e.g. `ollama pull llama3`)
- A vision model is **not** required for blending (text-only models work)

### Blending Procedure

1. Open the **Auto Caption** window
2. Generate a caption using any backend
3. Ensure the saved caption box has existing text (from a previous generation or manual entry)
4. Optionally edit the **blend instructions** field to guide the merge
5. Select an Ollama model for blending from the dropdown
6. Click **Blend**
7. The blended result appears in the Generated Caption box
8. Use **Ctrl+Z** / **Ctrl+Y** to toggle between original and blended versions
9. Click **Replace** when satisfied, then save

---

## YOLO Labeling

### Opening the YOLO Window

- **Tools > YOLO Labelling...** from the main menu
- The window opens maximised with the current thumbnail image list

### Creating Labels

1. In the label panel, type a label name and click **Add** (or press Enter)
2. Labels are assigned colours automatically
3. Click a label to make it the active drawing class

### Drawing Bounding Boxes

1. Select a label from the panel
2. Click and drag on the image to draw a bounding box
3. Release to complete the annotation
4. The box appears with the label name and colour

### Editing Annotations

- **Click** an existing box to select it
- **Drag** to move the box
- **Drag handles** to resize
- Press **Delete** to remove the selected box

### Saving Annotations

Annotations are saved in YOLO format as `.txt` files alongside the source images:
- One line per bounding box: `class_id center_x center_y width height`
- All coordinates normalised to 0-1 range

### Output Options

- **Output Path**: Configure where label files are saved
- **Resolution**: Set the target resolution for label generation
- **Size Mode**: Choose between image-relative or fixed-size output

---

## Settings and Preferences

### Options Menu

| Setting | Description |
|---------|-------------|
| **Remember Settings** | When enabled, saves all settings on exit and restores on launch |
| **Show Filenames** | Display file names below thumbnails (mutually exclusive with Dimensions/Size) |
| **Show Dimensions** | Display image dimensions below thumbnails |
| **Show File Size** | Display file size below thumbnails |
| **Show Folder Counts** | Display number of images next to folder names in both trees |
| **Confirm Deletions** | Show a confirmation dialog before deleting files |
| **Face Detection Settings...** | Choose detection preset (Fast / Balanced / Thorough) |

### Persisted Settings

The following are automatically saved and restored between sessions:

- Window size and position
- Splitter pane proportions
- Source and destination root paths and selected folders
- Sort criteria and filter state
- Thumbnail zoom level and display options
- Preview window geometry and fit mode
- YOLO window settings (labels, output path, resolution)
- Caption backend and model selections
- Enhancement settings (last operation, parameters, model directory, save mode)
- Face detection preset
- All menu toggle states

---

## Keyboard Shortcuts Reference

### Main Window

| Shortcut | Action |
|----------|--------|
| **Delete** | Delete selected files (with undo support) |
| **Shift+Delete** | Permanently delete (no undo) |
| **Ctrl+Z** | Undo last operation (move, copy, or delete) |
| **Enter** | Open preview window for selected thumbnail |
| **Ctrl+Q** | Exit application |
| **F1** | Open built-in manual |
| **Ctrl+Scroll** | Zoom thumbnails (over grid) or folder tree font (over tree) |

### Preview Window

| Shortcut | Action |
|----------|--------|
| **Left / Right** | Navigate to previous / next image |
| **Escape** | Close preview window |
| **Ctrl+S** | Save cropped image |
| **Ctrl+R** | Rotate image 90 degrees clockwise |
| **Ctrl+Scroll** | Zoom in/out (when not in Fit to Window mode) |

### Auto Caption Window

| Shortcut | Action |
|----------|--------|
| **Ctrl+S** | Save caption to .txt file |
| **Ctrl+Z** | Undo (restore previous generated caption) |
| **Ctrl+Y** | Redo (restore blended caption) |
| **Left / Right** | Navigate between images |
| **Escape** | Close window |

### Mouse Controls

| Action | Control |
|--------|---------|
| Open preview | Double-click thumbnail |
| Move files | Drag thumbnails to destination tree |
| Copy files | Ctrl + drag thumbnails to destination tree |
| Context menu | Right-click thumbnail |
| Folder context menu | Right-click folder in tree |
| Zoom | Ctrl + scroll wheel (over thumbnails, trees, or preview) |

---

## Troubleshooting

### Application Won't Start

1. Ensure the virtual environment is activated:
   ```bash
   source ~/scripts/venv/bin/activate
   ```
2. Verify dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
3. Check Python version (requires 3.8+):
   ```bash
   python --version
   ```
4. Run with debug output for more information:
   ```bash
   python main.py --debug
   ```

### Images Not Loading

- Check that the folder contains files with supported extensions: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.avif`, `.tiff`, `.bmp`
- Verify file permissions allow reading
- Check the status bar for error messages

### Face Detection Not Working Well

- Switch to **Thorough** preset (Options > Face Detection Settings)
- Download DNN models for improved accuracy:
  ```bash
  cd models && python download_models.py
  ```
- Without DNN models, the application uses Haar Cascade (less accurate but always available)

### "findimagedupes not found"

Visual duplicate detection requires the `findimagedupes` command-line tool:
```bash
# Ubuntu/Debian
sudo apt install findimagedupes

# Fedora
sudo dnf install findimagedupes

# Arch
sudo pacman -S findimagedupes
```

Exact (MD5) duplicate detection works without this tool.

### Ollama Not Detected

1. Check if Ollama is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```
2. Start Ollama if not running:
   ```bash
   ollama serve
   ```
3. If using Docker/Open-WebUI, ensure the Ollama API port (11434) is accessible

### AI Model Enhancement Shows "Install Required"

Install the Spandrel library and PyTorch:
```bash
pip install spandrel torch
```
Then restart the application. The AI Model option will become fully functional.

### Slow Performance with Large Collections

- Reduce thumbnail size with **Ctrl+Scroll** (smaller thumbnails load faster)
- Switch face detection to **Fast** preset if filtering is slow
- Close preview windows when not in use
- AI model operations on CPU are slower; use CUDA if available

### ONNX CUDAExecutionProvider Warning (WD14)

If you see a warning about CUDAExecutionProvider not available:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```
This requires an NVIDIA GPU with CUDA. The warning is harmless on CPU-only systems.

### Debug Mode

Run with `--debug` flag or set the environment variable for detailed console output:
```bash
IMG_SORT_DEBUG=1 python main.py
```

---

*Visual Image Master 2026 -- Organise your images with ease.*

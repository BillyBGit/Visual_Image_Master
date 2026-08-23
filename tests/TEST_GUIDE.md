# TEST_GUIDE.md — img_sort

## Quick Reference

**Import style:** `from caption_generator import CaptionGenerator` (no package prefix)
**conftest provides:** sys.path (tests/unit/conftest.py), QApplication offscreen (tests/conftest.py)
**Qt binding:** PySide6 — always `from PySide6.QtWidgets import ...` (NOT PyQt6)
**Test runner:** `source ~/scripts/venv/bin/activate && python -m pytest tests/unit/ -v`

---

## What to Mock

### Always Mock — will hang or crash without mocking
- `torch`, `torchvision`, `transformers`, `timm` — lazy ML imports; patch as whole modules (see below)
- `onnxruntime` — lazy import in caption_generator.py
- `huggingface_hub` — lazy import; patches model downloads
- `spandrel` — lazy import in image_enhance.py
- Model loading methods: `CaptionGenerator.load_florence()`, `CaptionGenerator.load_wd14()`
- `CaptionGenerator._load_wd14_pytorch()`, `CaptionGenerator._load_wd14_onnx()`
- `cv2` (OpenCV) — in face_detection.py, object_detection.py, scene_recognition.py, quality_assessment.py

### Mock for isolation
- `subprocess.run` — used in image_sorter_operations.py for external commands
- `urllib.request`, `urllib.error` — lazy imports in caption_generator.py download methods
- `csv` — lazy import in caption_generator.py tag loading
- `get_caption_generator()` — singleton; patch in tests that don't need a real generator

### Never Mock
- Qt widgets (QLabel, QSplitter, QScrollArea, etc.) — instantiate them for real in offscreen mode
- PySide6 signals — work fine in offscreen mode
- Pure Python logic (dicts, lists, string manipulation)

---

## Lazy Imports — CRITICAL (Rule F)

These imports occur **inside function bodies**. Patch as whole modules, not individual attributes.

### caption_generator.py

| Function | Lazy Import | Patch Target |
|----------|-------------|--------------|
| `CaptionGenerator.__init__` | `torch`, `onnxruntime`, `huggingface_hub` | `caption_generator.torch`, etc. |
| `CaptionGenerator.load_florence` | `torch`, `transformers` | `caption_generator.torch`, `caption_generator.transformers` |
| `CaptionGenerator._load_wd14_pytorch` | `torch`, `timm`, `huggingface_hub` | `caption_generator.torch`, `caption_generator.timm` |
| `CaptionGenerator._load_wd14_onnx` | `onnxruntime`, `torch` | `caption_generator.onnxruntime` |
| `CaptionGenerator._generate_wd14_pytorch` | `torch` | `caption_generator.torch` |
| `CaptionGenerator._generate_wd14_onnx` | `onnxruntime` | `caption_generator.onnxruntime` |
| Download helpers | `urllib.request`, `urllib.error` | `caption_generator.urllib` |
| Tag loading | `csv` | `caption_generator.csv` |

**Correct pattern:**
```python
with patch("caption_generator.torch") as mock_torch:
    mock_tensor = MagicMock()
    mock_torch.from_numpy.return_value = mock_tensor
    mock_tensor.unsqueeze.return_value = mock_tensor
    mock_tensor.to.return_value = mock_tensor      # dtype=torch.float32 is also mocked
    mock_torch.float32 = MagicMock()               # prevent real dtype errors
    mock_torch.sigmoid.return_value = MagicMock()
    result = caption_gen._generate_wd14_pytorch(img)
```

### image_enhance.py

| Function | Lazy Import | Patch Target |
|----------|-------------|--------------|
| `check_spandrel_available` | `spandrel`, `torch` | `image_enhance.spandrel`, `image_enhance.torch` |
| `check_cuda_available` | `torch` | `image_enhance.torch` |
| `load_upscale_model` | `spandrel`, `torch` | `image_enhance.spandrel`, `image_enhance.torch` |
| `apply_upscale` | `torch` | `image_enhance.torch` |

### image_sorter_operations.py

| Function | Lazy Import | Patch Target |
|----------|-------------|--------------|
| Various file ops | `hashlib`, `subprocess`, `tempfile`, `shutil` | `image_sorter_operations.subprocess` etc. |

---

## Qt Widget Classes — __init__ Signatures

**Main Windows (QMainWindow):**
```python
CaptionWindow(image_list: list, current_index: int, parent=None)
ImageSorter()           # No parameters
YoloWindow()            # No parameters
ImagePreviewWindow()    # No parameters
```

**Dialogs (QDialog):**
```python
AutoCaptionDialog(image_paths: list, parent=None)
AdvancedFilterDialog(parent=None)
EnhanceDialog(parent=None)
FaceDetectionDialog(parent=None)
```

**Workers (QThread):**
```python
CaptionWorker(...)     # Read source for specific args before testing
BlendWorker(...)
```

**Rules for Qt widget testing:**
- NEVER pass `MagicMock()` as `parent` — PySide6 C++ rejects it with TypeError
- Use `parent=None` (default) in tests, or a real `QWidget()` if parent behavior is tested
- `QApplication` is created by `tests/conftest.py` — never create one in a test file
- Import from `PySide6`, not `PyQt6` — the source uses PySide6 throughout

---

## CaptionWindow Method Testing — CRITICAL PATTERN

**NEVER do this:**
```python
window = CaptionWindow([], 0)          # Creates real Qt widgets
window.parent_app = MagicMock()        # auto-attributes are MagicMock objects
window._load_settings()                # FAILS: setText(MagicMock) → TypeError
```

PySide6's C++ layer validates argument types at runtime. `QLineEdit.setText(MagicMock)` raises
`TypeError: setText called with wrong argument types`. `MagicMock()` auto-created attributes
(e.g. `parent_app.caption_florence_custom_path` when not explicitly set) are `MagicMock` objects,
not strings. Passing them to any real Qt widget method causes this error.

**ALWAYS patch all setup methods when creating a CaptionWindow for method testing:**
```python
@pytest.fixture
def caption_window_instance():
    with patch.object(CaptionWindow, '_setup_ui', lambda self: None), \
         patch.object(CaptionWindow, '_setup_menu', lambda self: None), \
         patch.object(CaptionWindow, '_setup_shortcuts', lambda self: None), \
         patch.object(CaptionWindow, '_load_settings', lambda self: None), \
         patch.object(CaptionWindow, '_check_backend_status', lambda self: None), \
         patch('caption_window.get_caption_generator') as mock_gen, \
         patch('caption_window.find_main_window', return_value=None):
        mock_gen.return_value = MagicMock()
        window = CaptionWindow([], 0)
    yield window
```

Then in each test, set up ONLY the specific widgets the method needs, with proper types:
```python
def test_load_settings_defaults(caption_window_instance):
    window = caption_window_instance
    # Replace specific widgets with MagicMock
    window.florence_custom_path = MagicMock()
    window.ollama_url = MagicMock()
    window.ollama_model = MagicMock()
    window.backend_combo = MagicMock()
    window.backend_combo.count.return_value = 3
    window.backend_combo.itemData.side_effect = ["florence2", "ollama", "wd14"]
    # Set parent_app attributes with REAL PYTHON TYPES (str/bool/int) — NOT MagicMock auto-attrs
    window.parent_app = MagicMock()
    window.parent_app.caption_backend = "florence2"          # str
    window.parent_app.caption_florence_custom_path = ""      # str
    window.parent_app.caption_ollama_url = "http://localhost:11434"  # str
    window.parent_app.caption_ollama_model = "llava"         # str
    window.parent_app.caption_wd14_threshold = 0.55          # float
    window.parent_app.caption_wd14_custom_paths = {}         # dict
    window.parent_app.caption_blend_instructions = ""        # str
    window._load_settings()
    window.ollama_url.setText.assert_called_with("http://localhost:11434")
```

**Rule H — Always set explicit typed values on MagicMock parent_app:**
Any `parent_app` attribute that `_load_settings()`, `_save_settings()`, `_check_backend_status()`,
or similar methods read and pass to Qt widget methods MUST be explicitly set to a proper Python type.
Never rely on MagicMock auto-attributes for values that flow into `setText()`, `setValue()`,
`setCurrentIndex()`, `setPlainText()`, etc.

---

## Availability Check Functions — Always Patch False

```python
with patch("caption_generator.CaptionGenerator.is_florence_available", return_value=(False, "mocked")):
    ...
with patch("caption_generator.CaptionGenerator.is_wd14_available", return_value=(False, "mocked")):
    ...
with patch("image_enhance.check_spandrel_available", return_value=False):
    ...
with patch("image_enhance.check_cuda_available", return_value=False):
    ...
```

---

## CaptionGenerator Singleton

`get_caption_generator()` returns a module-level singleton. Patch it in tests that don't
need a real generator:

```python
with patch("caption_window.get_caption_generator") as mock_gen:
    mock_gen.return_value = MagicMock()
    window = CaptionWindow([], 0)
```

---

## Known Pitfalls

1. **torch.float32 TypeError**: When `_generate_wd14_pytorch` calls `tensor.to(device, dtype=torch.float32)`, if only `device` is mocked but `torch.float32` is real, you get `TypeError`. Always patch `caption_generator.torch` as a whole module.

2. **MagicMock as Qt parent**: `CaptionWindow([], 0, MagicMock())` crashes at `super().__init__(parent)`. Use `None`.

3. **PyQt6 vs PySide6 mismatch**: The project uses PySide6. `tests/conftest.py` uses PyQt6 for QApplication (legacy). In test files, import from PySide6.

4. **load_florence / load_wd14 hang**: These methods block on actual model downloads. Always patch them before instantiating CaptionGenerator in tests.

5. **QGroupBox title lookup via findChild**: `window.findChild(QGroupBox, "Caption Backend")` fails if the title string doesn't match exactly. Read `_setup_ui` source to get exact strings before writing assertions.

6. **signals.receivers() API**: PySide6 does not have a `.receivers()` method on signals. To verify a signal is connected, trigger the signal and observe side effects instead.

---

## Module Cheat Sheet

| Module | Import | Key Classes | Primary Mock Target |
|--------|--------|-------------|---------------------|
| caption_generator | `from caption_generator import CaptionGenerator` | CaptionGenerator, CaptionWorker, BlendWorker | `caption_generator.torch`, `caption_generator.transformers` |
| caption_window | `from caption_window import CaptionWindow` | CaptionWindow | `caption_window.get_caption_generator`, `caption_window.find_main_window` |
| image_sorter | `from image_sorter import ImageSorter` | ImageSorter | `subprocess.run`, file I/O |
| image_enhance | `from image_enhance import ...` | functions | `image_enhance.torch`, `image_enhance.spandrel` |
| face_detection | `from face_detection import FaceDetector` | FaceDetector | `face_detection.cv2` |
| object_detection | `from object_detection import ObjectDetector` | ObjectDetector | `object_detection.cv2` |
| scene_recognition | `from scene_recognition import SceneRecognizer` | SceneRecognizer | `scene_recognition.cv2`, `scene_recognition.torch` |
| yolo_window | `from yolo_window import YoloWindow` | YoloWindow | `yolo_window.cv2` |
| utils | `from utils import find_main_window` | — | `caption_window.find_main_window` |

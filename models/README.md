# AI Models for Visual Image Master 2026

This directory contains pre-trained AI models for intelligent image analysis and filtering.

## Quick Setup

Run the download script to fetch all required model files:

```bash
python3 download_models.py
```

## Model Categories

### 1. Face Detection Models
Enhanced face detection using deep neural networks.

**Files:**
- `deploy.prototxt` - Model architecture (ResNet-10 SSD)
- `res10_300x300_ssd_iter_140000_fp16.caffemodel` - Model weights (~10 MB)

**Features:**
- Detects faces at various angles
- Works with partially occluded faces
- Handles multiple faces per image
- Better accuracy than traditional Haar Cascades

### 2. Object Detection Models (YOLO)
Detect 80 different object types in images (people, animals, vehicles, food, etc.).

**Files:**
- `yolov4-tiny.cfg` - YOLOv4-tiny architecture configuration
- `yolov4-tiny.weights` - Pre-trained weights (~24 MB)

**Features:**
- Filter images containing specific objects
- Categories: people, animals, vehicles, food, furniture, electronics, sports equipment
- CPU-optimized for fast inference
- Based on COCO dataset (80 object classes)

### 3. Scene Recognition Models (Places365)
Classify images by scene type (beach, forest, office, restaurant, etc.).

**Files:**
- `resnet18_places365.onnx` - ResNet18 model trained on Places365 (~45 MB)
- `places365_labels.txt` - Scene category labels (365 categories)

**Features:**
- Recognize indoor vs outdoor scenes
- Identify specific locations (beach, mountain, office, bedroom, etc.)
- Group images by scene type
- CPU-optimized ResNet18 architecture

## Total Download Size

Approximately **80 MB** total for all models.

## Performance

All models are optimized for CPU-only operation:
- **CPU-friendly**: No GPU required
- **Portable**: Works on any system with OpenCV
- **Reasonable speed**: Optimized for balance between speed and accuracy
- **Modular**: GPU support can be added in the future

## Fallback Behavior

If model files are missing:
- **Face Detection**: Falls back to Haar Cascade (faster, less accurate)
- **Object Detection**: Feature disabled until models are downloaded
- **Scene Recognition**: Feature disabled until models are downloaded
- **Quality Assessment**: Works without models (uses OpenCV metrics)

## Model Sources

- **Face Detection**: OpenCV official repository
- **YOLO**: Darknet/AlexeyAB repository (YOLOv4)
- **Places365**: MIT CSAIL Vision Group

## License

All models are freely available for both academic and commercial use:
- Face detection models: OpenCV license
- YOLO models: YOLO license (freely usable)
- Places365 models: MIT license

## Custom YOLO Models

You can use your own trained YOLO models for object detection! Supported formats:
- **PyTorch models**: `.pt`, `.pth`, `.safetensors` (requires `pip install ultralytics`)
- **Darknet models**: `.weights` (requires corresponding `.cfg` file)
- **ONNX models**: `.onnx`

### Using Custom Models

1. In the Advanced AI Filters dialog, expand the "Object Detection (YOLO)" section
2. Click "Browse..." under "Custom YOLO Model"
3. Select your custom model file
4. **Important**:
   - If you select object categories, filtering works as normal
   - If you leave all categories unchecked, the filter will match ANY detection from your custom model
   - This is perfect for custom-trained models that detect specific objects not in COCO dataset

### Custom Model Behavior

When using a custom model with no categories selected:
- **"Contains" mode**: Shows images where the model detected ANYTHING
- **"Does NOT contain" mode**: Shows images where the model detected NOTHING

This allows you to train a model for a specific object and use it to filter your image collection!

## Usage in Application

Once downloaded, these models enable:
1. Advanced face detection filtering
2. Filter images by detected objects ("Show me images with dogs")
3. Filter images by quality metrics (sharpness, brightness, etc.)
4. Filter images by scene type ("Show me beach photos")
5. **NEW**: Use custom-trained YOLO models for specialized object detection

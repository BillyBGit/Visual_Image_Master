#!/usr/bin/env python3
"""
Download pre-trained models for image analysis.
Downloads models for: face detection, object detection, and scene recognition.
"""

import urllib.request
import os

# Model file URLs organized by category
MODEL_FILES = {
    # Face detection (DNN-based)
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    # Object detection (YOLOv4-tiny - CPU friendly)
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
    # Scene recognition (Places365 - ResNet18)
    # Note: Using PyTorch model since ONNX no longer available
    "resnet18_places365.pth.tar": "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
    "IO_places365.txt": "https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt",
}


def download_model(filename, url):
    """Download a model file if it doesn't already exist."""
    if os.path.exists(filename):
        print(f"✓ {filename} already exists")
        return True

    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False


def main():
    """Download all required model files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("Visual Image Master 2026 - Model Downloader")
    print("=" * 60)
    print("\nDownloading AI models for image analysis:")
    print("  • Face detection (ResNet-10 SSD)")
    print("  • Object detection (YOLOv4-tiny)")
    print("  • Scene recognition (Places365 ResNet18)")
    print("\nNote: Total download size is approximately 35 MB")
    print("All models are CPU-optimized for portability.\n")

    success = True
    total_files = len(MODEL_FILES)
    completed = 0

    for filename, url in MODEL_FILES.items():
        completed += 1
        print(f"[{completed}/{total_files}] ", end="")
        if not download_model(filename, url):
            success = False

    print("\n" + "=" * 60)
    if success:
        print("✓ All models downloaded successfully!")
        print("\nEnabled features:")
        print("  • Advanced face detection")
        print("  • Filter images by objects (people, animals, vehicles, etc.)")
        print("  • Filter images by quality (sharpness, brightness, etc.)")
        print("  • Filter images by scene type (outdoor, indoor, beach, etc.)")
    else:
        print("✗ Some models failed to download.")
        print("The application will use fallback methods where available.")

    print("=" * 60)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

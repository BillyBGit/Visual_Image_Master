"""
Image enhancement engine for Visual Image Master 2026.

Provides stateless functions for image enhancement operations:
- Tier 1 (Built-in): Sharpen, Denoise, Brightness/Contrast, Auto Levels, Color Saturation, Resize
- Tier 2 (Optional): AI Upscaling via Spandrel + PyTorch

No Qt dependency - pure OpenCV, Pillow, and numpy.
"""

import os
import cv2
import numpy as np
from PIL import Image


# --- Operation metadata registry ---

OPERATIONS = {
    "sharpen": {
        "name": "Sharpen (Unsharp Mask)",
        "params": [
            {"key": "amount", "label": "Amount", "type": "float",
             "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1},
            {"key": "radius", "label": "Radius", "type": "float",
             "min": 0.5, "max": 10.0, "default": 2.0, "step": 0.5},
            {"key": "threshold", "label": "Threshold", "type": "int",
             "min": 0, "max": 255, "default": 0, "step": 1},
        ],
    },
    "denoise": {
        "name": "Denoise",
        "params": [
            {"key": "strength", "label": "Strength", "type": "int",
             "min": 1, "max": 30, "default": 10, "step": 1},
            {"key": "color_strength", "label": "Color Strength", "type": "int",
             "min": 1, "max": 30, "default": 10, "step": 1},
        ],
    },
    "brightness_contrast": {
        "name": "Brightness / Contrast",
        "params": [
            {"key": "brightness", "label": "Brightness", "type": "int",
             "min": -100, "max": 100, "default": 0, "step": 1},
            {"key": "contrast", "label": "Contrast", "type": "float",
             "min": 0.1, "max": 3.0, "default": 1.0, "step": 0.05},
        ],
    },
    "auto_levels": {
        "name": "Auto Levels (CLAHE)",
        "params": [
            {"key": "clip_limit", "label": "Clip Limit", "type": "float",
             "min": 1.0, "max": 10.0, "default": 2.0, "step": 0.5},
            {"key": "grid_size", "label": "Grid Size", "type": "int",
             "min": 4, "max": 16, "default": 8, "step": 1},
        ],
    },
    "saturation": {
        "name": "Color Saturation",
        "params": [
            {"key": "factor", "label": "Saturation", "type": "float",
             "min": 0.0, "max": 3.0, "default": 1.0, "step": 0.05},
        ],
    },
    "resize": {
        "name": "Resize",
        "params": [
            {"key": "width", "label": "Width", "type": "int",
             "min": 1, "max": 16384, "default": 1024, "step": 1},
            {"key": "height", "label": "Height", "type": "int",
             "min": 1, "max": 16384, "default": 1024, "step": 1},
            {"key": "maintain_aspect", "label": "Maintain Aspect Ratio", "type": "bool",
             "default": True},
            {"key": "scale_percent", "label": "Scale %", "type": "int",
             "min": 1, "max": 1000, "default": 100, "step": 1},
            {"key": "use_scale", "label": "Use Scale Mode", "type": "bool",
             "default": False},
        ],
    },
    "ai_upscale": {
        "name": "AI Model (Upscale / Denoise / Restore)",
        "params": [
            {"key": "model_path", "label": "Model File", "type": "path", "default": ""},
            {"key": "tile_size", "label": "Tile Size", "type": "int",
             "min": 0, "max": 1024, "default": 512, "step": 128},
            {"key": "device", "label": "Device", "type": "choice",
             "choices": ["cpu", "cuda"], "default": "cpu"},
        ],
    },
}


# --- Tier 1: Built-in enhancement functions ---

def sharpen(img_bgr, amount=1.0, radius=2.0, threshold=0):
    """
    GIMP-style unsharp mask sharpening.

    Args:
        img_bgr: numpy array (BGR, uint8)
        amount: float 0.1-5.0 (strength of sharpening)
        radius: float 0.5-10.0 (blur kernel radius)
        threshold: int 0-255 (min brightness diff to sharpen)
    Returns:
        numpy array (BGR, uint8)
    """
    ksize = max(3, int(radius * 2) | 1)  # Ensure odd and >= 3
    blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), radius)

    if threshold == 0:
        sharpened = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
    else:
        # Only sharpen where difference exceeds threshold
        diff = cv2.absdiff(img_bgr, blurred)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
        mask = (gray_diff > threshold).astype(np.uint8)
        mask_3ch = cv2.merge([mask, mask, mask]) if len(img_bgr.shape) == 3 else mask

        sharpened_full = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
        sharpened = np.where(mask_3ch > 0, sharpened_full, img_bgr)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def denoise(img_bgr, strength=10, color_strength=10):
    """
    Non-Local Means denoising.

    Args:
        img_bgr: numpy array (BGR, uint8)
        strength: int 1-30 (luminance denoising strength)
        color_strength: int 1-30 (color denoising strength)
    Returns:
        numpy array (BGR, uint8)
    """
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, strength, color_strength, 7, 21)


def adjust_brightness_contrast(img_bgr, brightness=0, contrast=1.0):
    """
    Linear brightness and contrast adjustment.

    Args:
        img_bgr: numpy array (BGR, uint8)
        brightness: int -100 to +100 (additive offset)
        contrast: float 0.1 to 3.0 (multiplicative scale)
    Returns:
        numpy array (BGR, uint8)
    """
    return cv2.convertScaleAbs(img_bgr, alpha=contrast, beta=brightness)


def auto_levels(img_bgr, clip_limit=2.0, grid_size=8):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) on L channel.

    Args:
        img_bgr: numpy array (BGR, uint8)
        clip_limit: float 1.0-10.0 (contrast limit)
        grid_size: int 4-16 (tile grid size)
    Returns:
        numpy array (BGR, uint8)
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l = clahe.apply(l)
    result = cv2.merge([l, a, b])
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def adjust_saturation(img_bgr, factor=1.0):
    """
    Adjust color saturation.

    Args:
        img_bgr: numpy array (BGR, uint8)
        factor: float 0.0-3.0 (0=grayscale, 1=unchanged, >1=vivid)
    Returns:
        numpy array (BGR, uint8)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def resize_image(pil_image, width=1024, height=1024, maintain_aspect=True,
                 scale_percent=100, use_scale=False):
    """
    Resize image using Lanczos resampling.

    Args:
        pil_image: PIL Image object
        width: int target width
        height: int target height
        maintain_aspect: bool - fit within dimensions preserving ratio
        scale_percent: int percentage to scale (1-1000)
        use_scale: bool - if True, use scale_percent instead of width/height
    Returns:
        PIL Image object
    """
    if use_scale:
        factor = scale_percent / 100.0
        new_w = max(1, int(pil_image.width * factor))
        new_h = max(1, int(pil_image.height * factor))
        return pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if maintain_aspect:
        img_copy = pil_image.copy()
        img_copy.thumbnail((width, height), Image.Resampling.LANCZOS)
        return img_copy
    else:
        return pil_image.resize((width, height), Image.Resampling.LANCZOS)


# --- Tier 2: AI Upscaling (optional) ---

def check_spandrel_available():
    """Check if spandrel and torch are available."""
    try:
        import spandrel  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def check_cuda_available():
    """Check if CUDA is available for torch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def scan_model_directory(dir_path):
    """
    Scan a directory for model files (.pth, .safetensors, .pt, .ckpt).

    Returns:
        list of dict: [{"name": "model.pth", "path": "/full/path/model.pth", "size_mb": 66.7}, ...]
    """
    if not dir_path or not os.path.isdir(dir_path):
        return []

    valid_exts = {'.pth', '.safetensors', '.pt', '.ckpt'}
    models = []

    for entry in sorted(os.listdir(dir_path)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in valid_exts:
            full_path = os.path.join(dir_path, entry)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            models.append({
                "name": entry,
                "path": full_path,
                "size_mb": round(size_mb, 1),
            })

    return models


def load_upscale_model(model_path, device='cpu'):
    """
    Load a super-resolution model using Spandrel.

    Args:
        model_path: str path to .pth or .safetensors file
        device: 'cpu' or 'cuda'
    Returns:
        Spandrel ModelDescriptor
    Raises:
        ImportError if spandrel/torch not available
        RuntimeError if model can't be loaded
    """
    import spandrel
    import torch

    model = spandrel.ModelLoader(device=torch.device(device)).load_from_file(model_path)
    model.eval()
    return model


def get_model_info(model_path, device='cpu'):
    """
    Get info about a model without running inference.

    Returns:
        dict: {"architecture": str, "scale": int, "input_channels": int, "output_channels": int}
        or None on error
    """
    try:
        import spandrel
        import torch

        model = spandrel.ModelLoader(device=torch.device(device)).load_from_file(model_path)

        info = {
            "architecture": model.architecture.name if hasattr(model, 'architecture') else "Unknown",
            "scale": model.scale if hasattr(model, 'scale') else 1,
            "input_channels": model.input_channels if hasattr(model, 'input_channels') else 3,
            "output_channels": model.output_channels if hasattr(model, 'output_channels') else 3,
        }

        # Clean up
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

        return info
    except Exception as e:
        debug = os.environ.get('IMG_SORT_DEBUG') == '1'
        if debug:
            print(f"[DEBUG] Error getting model info for {model_path}: {e}")
        return None


def _tile_process(tensor, model, tile_size=512, overlap=32):
    """
    Process image tensor in overlapping tiles to manage memory.

    Args:
        tensor: torch.Tensor shape (1, C, H, W)
        model: Spandrel ModelDescriptor
        tile_size: int size of each tile
        overlap: int overlap between tiles
    Returns:
        torch.Tensor shape (1, C, H*scale, W*scale)
    """
    import torch

    _, c, h, w = tensor.shape
    scale = model.scale if hasattr(model, 'scale') else 1

    # Output dimensions
    out_h = h * scale
    out_w = w * scale
    output = torch.zeros((1, c, out_h, out_w), device=tensor.device, dtype=tensor.dtype)
    weight = torch.zeros((1, 1, out_h, out_w), device=tensor.device, dtype=tensor.dtype)

    step = tile_size - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            # Clamp tile boundaries
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = tensor[:, :, y_start:y_end, x_start:x_end]

            with torch.no_grad():
                tile_out = model(tile)

            # Output coordinates
            oy_start = y_start * scale
            ox_start = x_start * scale
            oy_end = y_end * scale
            ox_end = x_end * scale

            output[:, :, oy_start:oy_end, ox_start:ox_end] += tile_out
            weight[:, :, oy_start:oy_end, ox_start:ox_end] += 1.0

    # Average overlapping regions
    output = output / weight.clamp(min=1.0)
    return output


def upscale_image(img_bgr, model, tile_size=512):
    """
    Run image through a loaded Spandrel model for upscaling/restoration.

    Args:
        img_bgr: numpy array (BGR, uint8)
        model: Spandrel ModelDescriptor
        tile_size: int tile size (0 = process whole image at once)
    Returns:
        numpy array (BGR, uint8)
    """
    import torch

    # BGR uint8 -> RGB float32 tensor [0,1] shape (1,C,H,W)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(model.device)

    with torch.no_grad():
        if tile_size > 0:
            output = _tile_process(tensor, model, tile_size)
        else:
            output = model(tensor)

    # Convert back to BGR uint8
    result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# --- Unified processing interface ---

def apply_enhancement(image_path, operation, params, output_path=None):
    """
    Apply an enhancement operation to an image file.

    Args:
        image_path: str path to source image
        operation: str key from OPERATIONS dict
        params: dict of parameter values
        output_path: str path to save result (None = overwrite original)
    Returns:
        tuple: (success: bool, message: str)
    """
    if output_path is None:
        output_path = image_path

    debug = os.environ.get('IMG_SORT_DEBUG') == '1'

    try:
        if operation == "resize":
            return _apply_resize(image_path, params, output_path)
        elif operation == "ai_upscale":
            return _apply_ai_upscale(image_path, params, output_path)
        else:
            return _apply_cv_operation(image_path, operation, params, output_path)
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        return False, str(e)


def _apply_cv_operation(image_path, operation, params, output_path):
    """Apply an OpenCV-based enhancement and save."""
    img = cv2.imread(image_path)
    if img is None:
        return False, f"Could not read image: {image_path}"

    if operation == "sharpen":
        result = sharpen(img, **{k: v for k, v in params.items()
                                 if k in ('amount', 'radius', 'threshold')})
    elif operation == "denoise":
        result = denoise(img, **{k: v for k, v in params.items()
                                 if k in ('strength', 'color_strength')})
    elif operation == "brightness_contrast":
        result = adjust_brightness_contrast(img, **{k: v for k, v in params.items()
                                                     if k in ('brightness', 'contrast')})
    elif operation == "auto_levels":
        result = auto_levels(img, **{k: v for k, v in params.items()
                                     if k in ('clip_limit', 'grid_size')})
    elif operation == "saturation":
        result = adjust_saturation(img, **{k: v for k, v in params.items()
                                           if k in ('factor',)})
    else:
        return False, f"Unknown operation: {operation}"

    # Determine save format from output path extension
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == '.png':
        cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif ext == '.webp':
        cv2.imwrite(output_path, result, [cv2.IMWRITE_WEBP_QUALITY, 95])
    else:
        cv2.imwrite(output_path, result)

    return True, f"Applied {OPERATIONS[operation]['name']}"


def _apply_resize(image_path, params, output_path):
    """Apply resize using PIL and save."""
    pil_img = Image.open(image_path)

    # Preserve EXIF if present
    exif_data = None
    try:
        exif_data = pil_img.info.get('exif')
    except Exception:
        pass

    result = resize_image(
        pil_img,
        width=params.get('width', 1024),
        height=params.get('height', 1024),
        maintain_aspect=params.get('maintain_aspect', True),
        scale_percent=params.get('scale_percent', 100),
        use_scale=params.get('use_scale', False),
    )

    # Save with format preservation
    ext = os.path.splitext(output_path)[1].lower()
    save_kwargs = {}
    if ext in ('.jpg', '.jpeg'):
        save_kwargs['quality'] = 95
        if result.mode == 'RGBA':
            result = result.convert('RGB')
    elif ext == '.png':
        pass  # PNG is lossless
    elif ext == '.webp':
        save_kwargs['quality'] = 95

    if exif_data:
        save_kwargs['exif'] = exif_data

    result.save(output_path, **save_kwargs)
    pil_img.close()

    return True, f"Resized to {result.width}x{result.height}"


def _apply_ai_upscale(image_path, params, output_path):
    """Apply AI upscaling using Spandrel model."""
    if not check_spandrel_available():
        return False, "AI Upscale requires 'spandrel' and 'torch' packages. Install with: pip install spandrel torch"

    model_path = params.get('model_path', '')
    if not model_path or not os.path.isfile(model_path):
        return False, f"Model file not found: {model_path}"

    device = params.get('device', 'cpu')
    tile_size = params.get('tile_size', 512)

    debug = os.environ.get('IMG_SORT_DEBUG') == '1'
    if debug:
        print(f"[DEBUG] AI Upscale: model={model_path}, device={device}, tile_size={tile_size}")

    import torch

    # Load model
    model = load_upscale_model(model_path, device)
    scale = model.scale if hasattr(model, 'scale') else 1

    # Load and process image
    img = cv2.imread(image_path)
    if img is None:
        return False, f"Could not read image: {image_path}"

    if debug:
        print(f"[DEBUG] Input: {img.shape[1]}x{img.shape[0]}, scale={scale}")

    result = upscale_image(img, model, tile_size)

    if debug:
        print(f"[DEBUG] Output: {result.shape[1]}x{result.shape[0]}")

    # Save
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == '.png':
        cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif ext == '.webp':
        cv2.imwrite(output_path, result, [cv2.IMWRITE_WEBP_QUALITY, 95])
    else:
        cv2.imwrite(output_path, result)

    # Clean up GPU memory
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return True, f"Upscaled {scale}x to {result.shape[1]}x{result.shape[0]}"

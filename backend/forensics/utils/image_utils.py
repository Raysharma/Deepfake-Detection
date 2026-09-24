import io
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Any

try:
    import torch
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Standard ImageNet normalization for deep CNN feature extractors
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """
    Safely loads raw image bytes into a PIL Image and converts to RGB mode.
    Guarantees 3-channel standard color space regardless of palette or alpha channels.
    """
    img = Image.open(io.BytesIO(file_bytes))
    return img.convert("RGB")

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """
    Converts PIL RGB Image to OpenCV BGR numpy array without loss of color channels.
    """
    rgb_arr = np.array(pil_img)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """
    Converts OpenCV BGR numpy array to PIL RGB Image.
    """
    rgb_arr = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_arr)

def get_grayscale_array(pil_img: Image.Image) -> np.ndarray:
    """
    Extracts high-fidelity single-channel grayscale 2D float/uint8 numpy array for 2D FFT analysis.
    Uses standard luminance weighting: Y = 0.299R + 0.587G + 0.114B
    """
    gray_pil = pil_img.convert("L")
    return np.array(gray_pil, dtype=np.uint8)

def prepare_model_tensor(pil_img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Any:
    """
    Prepares deep learning inference tensor without altering original image resolution.
    Transforms:
    1. Bilinear resize copy to target dimensions (224, 224).
    2. Convert to float32 normalized in [0.0, 1.0].
    3. Normalize by ImageNet mean and std.
    4. Add batch dimension -> Shape: (1, 3, H, W).
    """
    if HAS_TORCH:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        tensor = transform(pil_img).unsqueeze(0)
        return tensor
    else:
        # Fallback numpy tensor representation if torch is loading
        resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        # Normalize
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        # Transpose to (3, H, W) and add batch dim -> (1, 3, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)

def save_image_safely(image_obj: Any, destination_path: str, quality: int = 95) -> None:
    """
    Persists PIL Image or OpenCV ndarray to disk safely.
    """
    if isinstance(image_obj, Image.Image):
        image_obj.save(destination_path, "JPEG", quality=quality)
    elif isinstance(image_obj, np.ndarray):
        # Check if 1-channel or 3-channel
        if len(image_obj.shape) == 2:
            cv2.imwrite(destination_path, image_obj)
        else:
            # Assume BGR if passed as ndarray
            cv2.imwrite(destination_path, image_obj, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        raise ValueError(f"Unsupported image type for saving: {type(image_obj)}")

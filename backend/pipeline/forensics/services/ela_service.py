import io
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from typing import Dict, Any, Optional
import cv2

from backend.pipeline.forensics.utils.file_utils import get_artifact_paths
from backend.pipeline.forensics.utils.image_utils import save_image_safely

DEFAULT_QUALITY = 90
DEFAULT_HIGH_ERROR_THRESHOLD = 20.0  # Empirical threshold for high-error artifact pixels

def generate_ela_analysis(
    pil_image: Image.Image,
    prefix: str,
    quality: int = DEFAULT_QUALITY,
    high_error_threshold: float = DEFAULT_HIGH_ERROR_THRESHOLD
) -> Dict[str, Any]:
    """
    Performs real Error Level Analysis (ELA) on an input PIL image:
    1. Recompresses the original RGB image to an in-memory JPEG stream at specified quality.
    2. Measures absolute per-pixel difference between original and recompressed representations.
    3. Computes descriptive error distribution statistics (mean, max, std, high-error ratio).
    4. Dynamically amplifies error differences to the full 0-255 dynamic range for human visual audit.
    5. Saves enhanced ELA visualization and returns structured forensic metrics.
    
    NOTE: ELA highlights variations in JPEG compression history and potential composite boundaries.
    It serves as a supporting forensic indicator, NOT definitive proof of deepfake manipulation.
    """
    try:
        # Guarantee RGB space
        orig_rgb = pil_image.convert("RGB")
        w, h = orig_rgb.size
        
        if w < 2 or h < 2:
            return {
                "available": False,
                "error": f"Image dimensions too small for ELA analysis ({w}x{h})."
            }

        # Step 1 & 2: Save to in-memory JPEG stream
        buffer = io.BytesIO()
        orig_rgb.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed_rgb = Image.open(buffer).convert("RGB")

        # Step 3: Compute absolute difference
        diff = ImageChops.difference(orig_rgb, recompressed_rgb)
        diff_arr = np.array(diff, dtype=np.float32)

        # Step 4: Calculate numerical statistics
        # Channel-wise max error per pixel gives magnitude of alteration
        pixel_max_channel = np.max(diff_arr, axis=2)
        mean_error = float(np.mean(diff_arr))
        max_error = float(np.max(diff_arr))
        std_error = float(np.std(diff_arr))
        
        # High error ratio: fraction of pixels exceeding configurable empirical threshold
        high_error_ratio = float(np.mean(pixel_max_channel > high_error_threshold))

        # Step 5: Amplify difference for visualization
        extrema = diff.getextrema()
        # Find maximum difference across all 3 channels
        max_channel_diff = max([ex[1] for ex in extrema]) if extrema else 0
        if max_channel_diff > 0:
            scale_factor = 255.0 / float(max_channel_diff)
        else:
            scale_factor = 1.0

        enhanced_diff = ImageEnhance.Brightness(diff).enhance(scale_factor)

        # Step 6: Save visualization
        abs_path, url_path = get_artifact_paths(prefix, "ela", ext=".jpg")
        save_image_safely(enhanced_diff, abs_path, quality=95)

        return {
            "available": True,
            "mean_error": round(mean_error, 4),
            "max_error": round(max_error, 4),
            "std_error": round(std_error, 4),
            "high_error_ratio": round(high_error_ratio, 4),
            "image_url": url_path,
            "quality": quality,
            "description": (
                f"ELA performed at JPEG quality {quality}. High-error pixel ratio: {high_error_ratio*100:.2f}%. "
                f"Elevated differences along localized edges may indicate digital splicing or resaving."
            ),
            "error": None
        }

    except Exception as e:
        return {
            "available": False,
            "mean_error": None,
            "max_error": None,
            "std_error": None,
            "high_error_ratio": None,
            "image_url": None,
            "quality": quality,
            "description": "Error Level Analysis could not be completed.",
            "error": f"ELA processing failed: {str(e)}"
        }

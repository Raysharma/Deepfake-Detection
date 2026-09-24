import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Optional

from backend.forensics.utils.file_utils import get_artifact_paths
from backend.forensics.utils.image_utils import save_image_safely, get_grayscale_array

def generate_fft_analysis(
    pil_image: Image.Image,
    prefix: str,
    radial_mask_ratio: float = 0.1667  # Radius = min(h, w) // 6 (~16.7%)
) -> Dict[str, Any]:
    """
    Performs 2D Fast Fourier Transform (FFT) frequency spectrum analysis:
    1. Converts image to high-fidelity single-channel grayscale representation.
    2. Computes discrete 2D FFT: np.fft.fft2().
    3. Shifts the zero-frequency DC component to the center: np.fft.fftshift().
    4. Computes logarithmic magnitude spectrum: log(1 + |F(u, v)|).
    5. Measures spectral distribution: mean magnitude, max magnitude, std dev.
    6. Measures high-to-low frequency energy ratio via radial spatial frequency masking.
       (Generative models/GANs frequently exhibit unnatural high-frequency energy spikes or periodic grid artifacts).
    7. Normalizes and color-maps spectrum for visual inspection.
    8. Persists visualization to outputs/forensic/{prefix}_fft.jpg.

    NOTE: FFT spectral anomalies provide evidence of synthetic upsampling and frequency domain artifacts.
    It is a supporting forensic signal to be interpreted alongside other evidence, NOT standalone proof of deepfakes.
    """
    try:
        # Convert to 2D numpy grayscale
        gray = get_grayscale_array(pil_image)
        h, w = gray.shape

        if h < 4 or w < 4:
            return {
                "available": False,
                "error": f"Image dimensions too small for 2D FFT ({w}x{h})."
            }

        # Step 1: 2D Fast Fourier Transform
        f = np.fft.fft2(gray.astype(np.float64))

        # Step 2: Shift DC zero-frequency component to center
        fshift = np.fft.fftshift(f)

        # Step 3: Magnitude spectrum with logarithmic scaling
        magnitude = np.abs(fshift)
        log_spectrum = np.log1p(magnitude)

        # Step 4: Numerical statistics
        mean_magnitude = float(np.mean(log_spectrum))
        max_magnitude = float(np.max(log_spectrum))
        std_magnitude = float(np.std(log_spectrum))

        # Step 5: High-frequency energy ratio vs low-frequency energy
        cy, cx = h // 2, w // 2
        radius = max(1, int(min(h, w) * radial_mask_ratio))

        y, x = np.ogrid[:h, :w]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        low_freq_mask = dist_sq <= (radius ** 2)
        high_freq_mask = ~low_freq_mask

        low_freq_energy = float(np.mean(log_spectrum[low_freq_mask]))
        high_freq_energy = float(np.mean(log_spectrum[high_freq_mask])) if np.any(high_freq_mask) else 0.0

        ratio = float(high_freq_energy / (low_freq_energy + 1e-7))

        # Step 6: Normalize logarithmic spectrum for visualization
        spec_min = np.min(log_spectrum)
        spec_max = np.max(log_spectrum)
        if spec_max > spec_min:
            norm_spectrum = ((log_spectrum - spec_min) / (spec_max - spec_min) * 255.0).astype(np.uint8)
        else:
            norm_spectrum = np.zeros_like(log_spectrum, dtype=np.uint8)

        # Apply colormap (INFERNO) for high-contrast frequency distribution
        color_spectrum = cv2.applyColorMap(norm_spectrum, cv2.COLORMAP_INFERNO)

        # Step 7: Persist visualization
        abs_path, url_path = get_artifact_paths(prefix, "fft", ext=".jpg")
        save_image_safely(color_spectrum, abs_path, quality=95)

        return {
            "available": True,
            "mean_magnitude": round(mean_magnitude, 4),
            "max_magnitude": round(max_magnitude, 4),
            "std_magnitude": round(std_magnitude, 4),
            "high_frequency_ratio": round(ratio, 4),
            "image_url": url_path,
            "description": (
                f"2D FFT Magnitude Spectrum computed. Mean magnitude: {mean_magnitude:.2f}, "
                f"High/Low frequency ratio: {ratio:.3f}. Analyzes high-frequency attenuation and unnatural periodic grid spikes."
            ),
            "error": None
        }

    except Exception as e:
        return {
            "available": False,
            "mean_magnitude": None,
            "max_magnitude": None,
            "std_magnitude": None,
            "high_frequency_ratio": None,
            "image_url": None,
            "description": "2D Fourier Transform frequency analysis could not be completed.",
            "error": f"FFT processing failed: {str(e)}"
        }

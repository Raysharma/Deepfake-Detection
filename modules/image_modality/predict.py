"""
modules/image_modality/predict.py
=================================
Inference Entrypoint for Person 1 (Image & Spatial AI)
Conforms to common/contracts.py ImageModalityOutput schema.
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.contracts import ImageModalityOutput
from modules.image_modality.spatial_detector import analyze_video_frames


def predict(video_path: str) -> dict:
    """
    Standardized inference function for Person 1 (Image & Spatial AI).
    Analyzes video frames for JPEG compression inconsistency (ELA),
    spectral grid artifacts (2D FFT), and generates Grad-CAM overlays.

    Args:
        video_path (str): Path to target video file.

    Returns:
        dict: Standardized ImageModalityOutput dictionary.
    """
    if not os.path.exists(video_path):
        return ImageModalityOutput(
            is_fake=False,
            fake_probability=0.0,
            spatial_score=0.0,
            frames_analyzed=0,
            face_detected=False,
            heatmap_path=None,
            detected_artifacts=["file_not_found"],
            error_message=f"Video file not found at {video_path}"
        ).to_dict()

    raw_result = analyze_video_frames(video_path, max_frames=30)

    output = ImageModalityOutput(
        modality="image_spatial",
        is_fake=raw_result.get("is_fake", False),
        fake_probability=raw_result.get("fake_probability", 0.0),
        spatial_score=raw_result.get("spatial_score", 0.0),
        frames_analyzed=raw_result.get("frames_analyzed", 0),
        face_detected=raw_result.get("face_detected", True),
        heatmap_path=raw_result.get("heatmap_path", None),
        detected_artifacts=raw_result.get("detected_artifacts", []),
        error_message=raw_result.get("error_message", None)
    )

    return output.to_dict()


if __name__ == "__main__":
    test_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/real_speaker.mp4"
    print(f"[*] Testing Image Modality on: {test_path}")
    res = predict(test_path)
    import pprint
    pprint.pprint(res)

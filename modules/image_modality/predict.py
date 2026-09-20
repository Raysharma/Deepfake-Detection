"""
modules/image_modality/predict.py
=================================
Inference Entrypoint for Person 1 (Image & Spatial AI)
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.contracts import ImageModalityOutput


def predict(video_path: str) -> dict:
    """
    Analyzes spatial facial frames of the video for blending artifacts and warping.
    
    Args:
        video_path (str): Path to the target .mp4 video file.
        
    Returns:
        dict: Standardized ImageModalityOutput dictionary.
    """
    if not os.path.exists(video_path):
        return ImageModalityOutput(
            is_fake=False,
            error_message=f"Video file not found at {video_path}"
        ).to_dict()

    # NOTE: Person 1 will plug their trained EfficientNet model & Grad-CAM logic here.
    # Current stub provides baseline return contract.
    output = ImageModalityOutput(
        is_fake=False,
        fake_probability=0.15,
        spatial_score=0.15,
        frames_analyzed=30,
        face_detected=True,
        heatmap_path="temp_uploads/sample_face_gradcam.jpg",
        detected_artifacts=["natural_illumination_consistent"]
    )
    return output.to_dict()


if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "sample.mp4"
    print("Testing Image Modality on:", test_path)
    print(predict(test_path))

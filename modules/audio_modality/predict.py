"""
modules/audio_modality/predict.py
=================================
Inference Entrypoint for Person 2 (Audio & Acoustic AI)
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.contracts import AudioModalityOutput


def predict(video_path: str) -> dict:
    """
    Analyzes acoustic features of the audio track for voice cloning anomalies.
    
    Args:
        video_path (str): Path to the target .mp4 video file.
        
    Returns:
        dict: Standardized AudioModalityOutput dictionary.
    """
    if not os.path.exists(video_path):
        return AudioModalityOutput(
            is_fake=False,
            error_message=f"Video file not found at {video_path}"
        ).to_dict()

    # NOTE: Person 2 will plug their Librosa Mel-Spectrogram + CNN model here.
    # Current stub provides baseline return contract.
    output = AudioModalityOutput(
        is_fake=False,
        fake_probability=0.20,
        frequency_score=0.20,
        audio_duration_sec=5.2,
        spectrogram_heatmap_path="temp_uploads/sample_audio_gradcam.jpg",
        detected_anomalies=["natural_formant_structure"]
    )
    return output.to_dict()


if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "sample.mp4"
    print("Testing Audio Modality on:", test_path)
    print(predict(test_path))

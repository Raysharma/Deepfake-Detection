import os
import torch
from huggingface_hub import hf_hub_download

from common.contracts import AudioModalityOutput
from .audio_extractor import extract_audio
from .spectrogram import audio_to_mel
from .model import create_model
from .gradcam import save_gradcam


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

HF_REPO_ID = "Somya09/deepfake-audio-cnn"
MODEL_FILENAME = "audio_cnn_best.pth"


_AUDIO_MODEL = None

def load_model():
    global _AUDIO_MODEL
    if _AUDIO_MODEL is not None:
        return _AUDIO_MODEL

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME
    )
    model = create_model().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _AUDIO_MODEL = model
    return _AUDIO_MODEL


def predict_audio(audio_path: str) -> dict:

    if not os.path.exists(audio_path):
        return AudioModalityOutput(
            is_fake=False,
            error_message=f"Audio file not found: {audio_path}"
        ).to_dict()

    try:
        # 1. Convert audio to Mel-spectrogram
        mel = audio_to_mel(audio_path)

        # 2. Prepare tensor
        input_tensor = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # 3. Load trained CNN from Hugging Face
        model = load_model()

        # 4. Prediction
        with torch.no_grad():
            logits = model(input_tensor)
            fake_probability = torch.sigmoid(logits).item()

        is_fake = fake_probability >= 0.5

        # 5. Generate Grad-CAM
        heatmap_path = os.path.join(
            "temp_uploads",
            "audio_gradcam_test.jpg"
        )

        save_gradcam(
            model,
            input_tensor,
            mel,
            heatmap_path
        )

        # 6. Detected anomaly
        anomalies = []

        if is_fake:
            anomalies.append(
                "synthetic_or_manipulated_acoustic_pattern"
            )
        else:
            anomalies.append(
                "natural_acoustic_pattern"
            )

        # 7. Return standard project contract
        return AudioModalityOutput(
            is_fake=is_fake,
            fake_probability=float(fake_probability),
            frequency_score=float(fake_probability),
            audio_duration_sec=4.0,
            spectrogram_heatmap_path=heatmap_path,
            detected_anomalies=anomalies
        ).to_dict()

    except Exception as e:

        return AudioModalityOutput(
            is_fake=False,
            error_message=str(e)
        ).to_dict()


def predict(video_path: str) -> dict:

    if not os.path.exists(video_path):
        return AudioModalityOutput(
            is_fake=False,
            error_message=f"Video file not found: {video_path}"
        ).to_dict()

    try:
        # 1. Extract audio from video
        audio_path = extract_audio(video_path)

        # 2. Use audio prediction pipeline
        return predict_audio(audio_path)

    except Exception as e:

        return AudioModalityOutput(
            is_fake=False,
            error_message=str(e)
        ).to_dict()


if __name__ == "__main__":

    # Test using the downloaded ASVspoof audio
    test_audio = (
        r"D:\deepfake_detection\LA\ASVspoof2019_LA_dev"
        r"\flac\LA_D_1000265.flac"
    )

    print(predict_audio(test_audio))
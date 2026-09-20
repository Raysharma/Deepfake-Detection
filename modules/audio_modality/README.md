# 🎙️ Person 2: Audio & Acoustic AI Engineer

## 🎯 Your Responsibilities
1. Extract audio track (WAV at 16kHz mono) from uploaded video files.
2. Use **Librosa** to generate 2D Mel-Spectrogram representations.
3. Train/fine-tune an Audio CNN (**ResNet50** or custom 2D CNN) on spectrogram images to classify real vs synthetic/cloned speech (e.g. ElevenLabs, Tacotron).
4. Implement **Grad-CAM** on the spectrogram to highlight unnatural frequency anomalies (e.g. robotic cutoffs).
5. Ensure your entrypoint function in `predict.py` takes `video_path: str` and returns a dictionary conforming to `common/contracts.py`.

## 📁 File Layout
```
audio_modality/
├── README.md             # This file
├── predict.py            # Primary inference entrypoint for the backend
├── audio_extractor.py    # Extracts WAV audio track from video
├── spectrogram.py        # Librosa Mel-Spectrogram transform
├── model.py              # Spectrogram CNN definition
└── gradcam.py            # Spectrogram Grad-CAM generator
```

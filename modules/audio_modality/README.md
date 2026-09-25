# 🎙️ Audio Modality — Acoustic Deepfake Detection

## Overview

The Audio Modality is responsible for detecting synthetic or manipulated
speech/audio content from uploaded video files.

The module extracts the audio track from a video, converts it into a
Mel-Spectrogram representation, processes the spectrogram using a custom
2D Audio CNN, and returns a standardized prediction for the backend.

The trained model is hosted separately on Hugging Face and is automatically
downloaded and cached during inference.

---

## Responsibilities

This module handles:

1. Audio extraction from uploaded video files.
2. Conversion of audio into 16 kHz mono WAV format.
3. Mel-Spectrogram generation using Librosa.
4. Audio CNN training on the ASVspoof 2019 LA dataset.
5. Binary classification of real/bona fide vs spoof audio.
6. Grad-CAM visualization for spectrogram-based explanations.
7. Standardized inference output for backend integration.
8. Loading the trained model from Hugging Face.

---

## Architecture

```text
                         Uploaded Video
                              │
                              ▼
                    ┌──────────────────┐
                    │ Audio Extraction │
                    │      FFmpeg      │
                    │   16kHz / Mono   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Mel-Spectrogram │
                    │      Librosa     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     AudioCNN     │
                    │   Custom 2D CNN  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Fake Probability│
                    │       0 - 1      │
                    └────────┬─────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
           ┌──────────────┐      ┌────────────────┐
           │   Grad-CAM   │      │  Standardized  │
           │ Spectrogram  │      │ Audio Output   │
           │   Heatmap    │      │    Contract    │
           └──────────────┘      └────────┬───────┘
                                         │
                                         ▼
                                  Backend Aggregator
```

---

## File Structure

```text
modules/
└── audio_modality/
    │
    ├── README.md
    │
    ├── predict.py
    │   └── Main inference entrypoint for backend integration
    │
    ├── audio_extractor.py
    │   └── Extracts audio from video using FFmpeg
    │
    ├── spectrogram.py
    │   └── Generates normalized Mel-Spectrograms using Librosa
    │
    ├── dataset.py
    │   └── ASVspoof protocol parsing and PyTorch Dataset classes
    │
    ├── precompute.py
    │   └── Precomputes Mel-Spectrogram features for training
    │
    ├── model.py
    │   └── Defines the custom 2D Audio CNN architecture
    │
    ├── train.py
    │   └── Model training and validation pipeline
    │
    └── gradcam.py
        └── Generates Grad-CAM explanations for spectrograms
```

---

## Training Artifacts

```text
audio_modality/
│
├── cache/
│   ├── train_mels.npy
│   ├── train_labels.npy
│   ├── dev_mels.npy
│   └── dev_labels.npy
│
└── checkpoints/
    └── audio_cnn_best.pth
```

---

## Model Storage

```text
GitHub
└── Source code + README

Hugging Face
└── Somya09/deepfake-audio-cnn
    └── audio_cnn_best.pth
```
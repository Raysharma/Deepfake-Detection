# 👄 Person 3: Multimodal & Lip-Sync AI Engineer (Your Work)

> **Role**: Multimodal & Sync ML Engineer (The Novelty Role)  
> **Core Objective**: Determine if spoken audio phonemes match visual lip movements (Lip-Sync Mismatch Detection).

---

## 💡 Why This Module Matters (The Deepfake Novelty)
Many state-of-the-art deepfakes can fool spatial CNNs because blending boundaries have become very smooth. However, **audio-to-lip synchronization** is extraordinarily difficult for generative models to fake:
- Deepfakes often generate lips that move slightly before or after the sound is uttered.
- In voice-cloned video dubbing, the speaker’s mouth says words that do not match the audio phonemes (e.g. bilabial plosives like "P", "B", "M" require lips to close, but dubbing models fail to synchronize this).
- Our **SyncNet dual-stream architecture** computes the temporal distance between the visual mouth movements and acoustic phonemes to expose this mismatch.

---

## 📁 Directory Architecture
```
video_sync_modality/
├── README.md                     # Documentation and mathematical rationale
├── predict.py                    # Contract-compliant inference entrypoint
├── inference.py                  # End-to-end video sync evaluation engine
├── run_test.py                   # Local standalone test runner
├── data_preprocessing/
│   ├── __init__.py
│   ├── lip_extractor.py          # MediaPipe / OpenCV mouth landmark ROI extraction
│   └── audio_extractor.py        # 16kHz audio separation & Mel-spectrogram chunker
└── models/
    ├── __init__.py
    └── syncnet.py                # Dual-stream PyTorch SyncNet architecture
```

---

## 🔬 How the SyncNet Pipeline Works

1. **Audio Stream**:
   - The audio track is extracted from the video and converted to a 16kHz mono signal.
   - For every 0.2-second window (corresponding to 5 consecutive video frames at 25 fps), an acoustic feature chunk (13-dim MFCC / Mel-spectrogram) is computed.

2. **Visual Stream**:
   - MediaPipe detects the face and isolates lip landmarks.
   - The mouth region is cropped, converted to grayscale, resized to $96 \times 96$, and grouped into 5-frame sequences.

3. **Dual-Stream Feature Projection**:
   - Both the visual frames and audio chunks are passed through their respective convolutional sub-networks, projecting both into a common **1024-dimensional embedding space**.

4. **Cross-Modal Distance & Offset**:
   - The Euclidean Distance $D = \| v - a \|_2$ or Cosine Similarity is evaluated across a sliding temporal window of $\pm 15$ frames:
     - **Authentic Video**: A sharp minimum distance appears near **offset = 0 frames**.
     - **Deepfake / Dubbed Video**: The minimum distance is high across all offsets, or the offset is unnaturally shifted.

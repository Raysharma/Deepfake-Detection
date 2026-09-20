# 🏛️ System Architecture & Multimodal Pipeline Workflow

```
                             [ Uploaded Media (.mp4) ]
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
  [ Person 1: Image ]             [ Person 2: Audio ]             [ Person 3: Lip-Sync ]
MediaPipe / OpenCV Frame Crop    Extract Audio & Mel-Spectrogram   MediaPipe Lip Landmark Tracking
        │                                │                                │
        ▼                                ▼                                ▼
  [ Spatial CNN ]                 [ Audio CNN ]                      [ SyncNet ]
(EfficientNet / Xception)            (ResNet50)                 (Dual-Stream Alignment)
        │                                │                                │
  ┌─────┴─────┐                    ┌─────┴─────┐                    ┌─────┴─────┐
  ▼           ▼                    ▼           ▼                    ▼           ▼
[Grad-CAM] [Spatial Score]     [Grad-CAM] [Frequency Score]    [Offset Frames] [Sync Score]
(Face Map)                     (Audio Map)
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         │
                                         ▼
                             [ Person 4: Backend API ]
                       FastAPI + Celery Task Orchestrator
                                         │
                                         ▼
                            [ Final JSON Aggregation ]
                             Weight-averaged scoring
                                         │
                                         ▼
                           [ MongoDB Persistent Store ]
                     Reports, Heatmaps & Confidence Timeline
```

---

## ⚖️ Multimodal Decision Aggregation (Weighted Heuristic)

Person 4's backend combines the 3 independent streams into a final decision score:

$$\text{Overall Fake Score} = (w_1 \times \text{Spatial Fake}) + (w_2 \times \text{Audio Fake}) + (w_3 \times \text{Sync Mismatch})$$

* Default Baseline Weights:
  - **Spatial Artifacts ($w_1$)**: 0.35
  - **Audio Spectral Anomalies ($w_2$)**: 0.35
  - **Lip-Sync Alignment ($w_3$)**: 0.30

If the aggregated score $> 0.50$, the system classifies the media as **DEEPFAKE**, and provides the breakdown showing exactly which modalities triggered the flag.

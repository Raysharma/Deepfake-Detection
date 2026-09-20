# 🛡️ Multimodal Explainable Deepfake Detection System

A multimodal deepfake detection system evaluating media across three complementary dimensions:
1. **Image (Spatial Artifacts)**: Face boundary warping & blending artifacts via Vision CNN (EfficientNet/XceptionNet) + Grad-CAM heatmaps.
2. **Audio (Acoustic Anomalies)**: Cloned speech & synthetic frequency anomalies via Mel-Spectrogram CNN + Grad-CAM.
3. **Video (Audio-Visual Lip-Sync Mismatch)**: Cross-modal synchronization distance & phoneme-lip alignment via **SyncNet**.

---

## 🏛️ Project Directory Structure

```
Deepfake-Detection/
├── docs/                                # Project architecture & integration contract
│   ├── ARCHITECTURE.md                  # Detailed dataflow diagrams
│   └── INTEGRATION_CONTRACT.md          # Input/output contracts for Persons 1, 2, 3, 4
│
├── common/                              # Shared schemas and dataclasses
│   ├── __init__.py
│   └── contracts.py                     # Standardized Python dictionary contracts
│
├── modules/
│   ├── image_modality/                  # Person 1 (Spatial AI - EfficientNet + Grad-CAM)
│   │   ├── README.md                    # Instructions & deliverables
│   │   └── predict.py                   # Standard entrypoint: predict(video_path) -> dict
│   │
│   ├── audio_modality/                  # Person 2 (Acoustic AI - Librosa + ResNet + Grad-CAM)
│   │   ├── README.md                    # Instructions & deliverables
│   │   └── predict.py                   # Standard entrypoint: predict(video_path) -> dict
│   │
│   ├── video_sync_modality/             # Person 3 (Multimodal Lip-Sync / SyncNet - YOUR WORK)
│   │   ├── README.md                    # SyncNet technical design & math
│   │   ├── predict.py                   # Standard entrypoint: predict(video_path) -> dict
│   │   ├── inference.py                 # Core alignment & offset scanner
│   │   ├── run_test.py                  # Standalone local test runner
│   │   ├── data_preprocessing/          # MediaPipe lip landmark isolation & audio extraction
│   │   └── models/                      # Dual-stream PyTorch SyncNet definition
│   │
│   └── backend_service/                 # Person 4 (FastAPI + Celery + Redis + MongoDB)
│       ├── README.md                    # Backend architecture instructions
│       └── orchestrator_example.py      # Pipeline coordinator importing the 3 ML modules
│
├── sample_data/                         # Local test media (e.g. demo_mouth.mp4)
├── requirements.txt                     # Unified project dependencies
├── .gitignore                           # Excludes heavy weights, datasets, temp caches
└── README.md                            # Project overview
```

---

## 👥 Division of Work (4-Member Team)

| Role | Domain | Tech Stack | Deliverable |
| :--- | :--- | :--- | :--- |
| **Person 1** | Image & Spatial AI | PyTorch, OpenCV, MediaPipe, EfficientNet, Grad-CAM | Face crops, spatial fake score, face heatmap |
| **Person 2** | Audio & Acoustic AI | PyTorch, Librosa, ResNet50, Grad-CAM | Mel-spectrograms, synthetic voice score, audio heatmap |
| **Person 3** | Multimodal & Lip-Sync AI | PyTorch, MediaPipe, OpenCV, SyncNet | 5-frame mouth sequences, temporal offset, sync score |
| **Person 4** | Backend & Infrastructure | FastAPI, Celery, Redis, MongoDB | Asynchronous API, DB schemas, pipeline orchestrator |

---

## 🚀 Independent Testing (Zero Merge Conflicts)

Each engineer works strictly in their own subfolder under `modules/` and tests locally:

### To test the Lip-Sync module (Person 3):
```powershell
python modules/video_sync_modality/run_test.py
```

### To test the Backend Orchestrator (Person 4):
```powershell
python modules/backend_service/orchestrator_example.py
```

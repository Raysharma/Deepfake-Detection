# 🤝 Integration Contract & Team Architecture Guide

> **Project**: Multimodal Explainable Deepfake Detection System  
> **Team Size**: 4 Engineers  
> **Architecture Principle**: Modular Monorepo with Contract-First Isolation  

---

## 📌 Executive Summary

To prevent code conflicts and allow all 4 team members to develop independently and in parallel:
1. **Isolated Folders**: Each engineer owns one dedicated folder under `modules/`. Nobody edits another teammate's folder without communication.
2. **Standardized Contract**: Every ML module (Image, Audio, Video-Sync) implements one exact function signature:
   ```python
   def predict(video_path: str) -> dict:
   ```
3. **Orchestrator Role**: Person 4 (Backend) imports `predict()` from each of the 3 modules into their Celery worker pipeline.

---

## 👥 Division of Work & Responsibilities

| Role | Engineer | Directory | Core Deliverable |
| :--- | :--- | :--- | :--- |
| **Person 1** | Image & Spatial AI Engineer | `modules/image_modality/` | EfficientNet/XceptionNet face classifier + Grad-CAM heatmap generator |
| **Person 2** | Audio & Acoustic AI Engineer | `modules/audio_modality/` | Librosa Mel-Spectrogram extractor + Audio CNN + Spectrogram Grad-CAM |
| **Person 3** | Multimodal & Lip-Sync AI Engineer | `modules/video_sync_modality/` | MediaPipe lip ROI isolation + SyncNet audio-visual alignment scorer |
| **Person 4** | Backend & Infrastructure Engineer | `modules/backend_service/` | FastAPI REST API + Celery background worker + Redis + MongoDB store |

---

## 📥 The Input Contract

Every ML prediction script **must** accept a local video path as a string:

```python
video_path = "/path/to/uploaded_video.mp4"
```

Person 4's backend is responsible for:
1. Receiving the uploaded media via FastAPI (`/upload` or `/analyze`).
2. Saving it to a temporary staging folder (`temp_uploads/<task_id>.mp4`).
3. Passing that absolute file path to Persons 1, 2, and 3 simultaneously.

---

## 📤 The Output Contract

Every ML module's `predict(video_path: str)` function **must** return a Python dictionary conforming to the schemas in `common/contracts.py`.

### 1. Person 1 (Image & Spatial) Output:
```python
{
    "modality": "image_spatial",
    "is_fake": bool,               # True if manipulated face detected
    "fake_probability": float,     # 0.0 to 1.0 (e.g. 0.87)
    "spatial_score": float,        # Raw model logit/score
    "frames_analyzed": int,        # e.g. 30
    "face_detected": bool,         # True/False
    "heatmap_path": str,           # Saved Grad-CAM JPEG path
    "detected_artifacts": list,    # e.g. ["chin_blending_warp", "eye_boundary_noise"]
    "error_message": None          # String if failed, else None
}
```

### 2. Person 2 (Audio & Acoustic) Output:
```python
{
    "modality": "audio_acoustic",
    "is_fake": bool,               # True if synthetic/cloned voice
    "fake_probability": float,     # 0.0 to 1.0
    "frequency_score": float,      # Spectrogram classifier score
    "audio_duration_sec": float,   # e.g. 4.8
    "spectrogram_heatmap_path": str, # Saved Grad-CAM spectrogram path
    "detected_anomalies": list,    # e.g. ["robotic_spectral_flatness", "synthetic_cutoff"]
    "error_message": None
}
```

### 3. Person 3 (Multimodal & Lip-Sync) Output:
```python
{
    "modality": "video_lip_sync",
    "is_fake": bool,               # True if speech and lip movement mismatch
    "fake_probability": float,     # 0.0 to 1.0
    "sync_confidence": float,      # Higher = higher synchronization match
    "offset_frames": int,          # Temporal delay (0 = perfect sync, +/- frames)
    "min_distance": float,         # Feature embedding distance (L2 or Cosine)
    "frames_evaluated": int,       # Number of mouth frame sequences
    "lip_landmarks_found": bool,   # True/False
    "detected_anomalies": list,    # e.g. ["lip_sync_mismatch", "delayed_phoneme_correlation"]
    "error_message": None
}
```

---

## 🔀 Git Branching Strategy (Zero Conflicts)

To make sure team members don't step on each other's work:
1. Keep the `main` branch protected.
2. Every engineer creates their own branch:
   - `git checkout -b feature/image-spatial` (Person 1)
   - `git checkout -b feature/audio-acoustic` (Person 2)
   - `git checkout -b feature/video-sync` (Person 3)
   - `git checkout -b feature/backend-infra` (Person 4)
3. **Commit Rule**: Only modify files within your assigned folder (`modules/<your_folder>/`).
4. **Weights & Datasets Rule**: **DO NOT** commit large `.pt`, `.h5`, or `.pth` model weights or video datasets to Git! They must be in `.gitignore`. Share model checkpoints using a shared Google Drive / OneDrive folder or Hugging Face.

# 🖼️ Person 1: Image & Spatial AI Engineer

## 🎯 Your Responsibilities
1. Extract video frames using OpenCV at a designated sampling rate.
2. Detect and crop the face boundary using **MediaPipe Face Detection**.
3. Implement / fine-tune a Vision CNN (**EfficientNet-B0** or **XceptionNet**) on cropped face images.
4. Implement **Grad-CAM** on the final convolutional layer to generate face heatmap overlays showing manipulated regions (e.g. blurring around chin, eyes).
5. Ensure your entrypoint function in `predict.py` takes `video_path: str` and returns a dictionary conforming to `common/contracts.py`.

## 📁 File Layout
```
image_modality/
├── README.md             # This file
├── predict.py            # Primary inference entrypoint for the backend
├── frame_extractor.py    # OpenCV frame sampling logic
├── face_cropper.py       # MediaPipe face bounding & alignment
├── model.py              # EfficientNet / XceptionNet definition
└── gradcam.py            # Grad-CAM heatmap visualization
```

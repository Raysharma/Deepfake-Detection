import os
import sys
import cv2

# Add project root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.video_sync_modality.data_preprocessing.lip_extractor import extract_lip_rois_from_video

def main():
    video_path = "sample_data/real_speaker.mp4"
    
    if not os.path.exists(video_path):
        print(f"[!] Please place a video with a speaking face at: {video_path}")
        return

    print(f"[*] Analyzing video: {video_path}")
    # Extract mouth frames (up to 30 frames)
    lip_frames, landmarks_found = extract_lip_rois_from_video(video_path, max_frames=30)

    print(f"[✓] Total mouth frames extracted: {len(lip_frames)}")
    print(f"[✓] Facial landmarks tracked by MediaPipe: {landmarks_found}")

    if len(lip_frames) > 0:
        sample_frame = lip_frames[0] # First frame
        print(f"[✓] Shape of single mouth frame: {sample_frame.shape} (Height, Width)")
        print(f"[✓] Pixel value range: min={sample_frame.min():.2f}, max={sample_frame.max():.2f}")

        # Save an image to disk so you can open and verify it!
        output_image_path = "sample_mouth_preview.jpg"
        # Convert from 0.0-1.0 float back to 0-255 uint8
        preview_img = (sample_frame * 255).astype("uint8")
        cv2.imwrite(output_image_path, preview_img)
        print(f"\n🎉 SUCCESS: Open '{output_image_path}' in VS Code to see your cropped mouth!")

if __name__ == "__main__":
    main()

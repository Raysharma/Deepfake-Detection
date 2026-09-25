import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.video_sync_modality.predict import predict

def print_result_card(title: str, res: dict):
    print("=" * 65)
    print(f" [CLIP] {title}")
    print("=" * 65)
    verdict = "[!] DEEPFAKE (LIP-SYNC MISMATCH)" if res['is_fake'] else "[OK] AUTHENTIC (LIP-SYNC MATCH)"
    print(f" Verdict            : {verdict}")
    print(f" Fake Probability   : {res['fake_probability'] * 100:.1f}%")
    print(f" Sync Confidence    : {res['sync_confidence'] * 100:.1f}%")
    print(f" Best Temporal Offset: {res['offset_frames']} frames")
    print(f" Min Feature Distance: {res['min_distance']:.4f}")
    print(f" Landmark Tracking  : {'Success (MediaPipe)' if res['lip_landmarks_found'] else 'Fallback'}")
    print(f" Detected Signals   : {', '.join(res['detected_anomalies'])}")
    print("=" * 65 + "\n")

def main():
    real_video = "sample_data/real_speaker.mp4"
    fake_video = "sample_data/fake_desync.mp4"

    print("\n[+] EVALUATING MULTIMODAL LIP-SYNC INFERENCE PIPELINE\n")

    # 1. Run on Authentic Video
    if os.path.exists(real_video):
        print(f"[*] Processing Authentic Video: {real_video} ...")
        real_result = predict(real_video)
        print_result_card("TEST 1: AUTHENTIC HUMAN SPEAKER", real_result)

    # 2. Run on Deepfake Video
    if os.path.exists(fake_video):
        print(f"[*] Processing Manipulated Video: {fake_video} ...")
        fake_result = predict(fake_video)
        print_result_card("TEST 2: DEEPFAKE / LIP-SYNC MISMATCH", fake_result)

if __name__ == "__main__":
    main()

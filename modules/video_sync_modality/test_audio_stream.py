import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.video_sync_modality.data_preprocessing.audio_extractor import extract_wav_from_video, compute_audio_mfcc_chunks

def main():
    video_path = "sample_data/real_speaker.mp4"
    print(f"[*] Extracting audio from: {video_path}")

    # 1. Extract raw 16kHz mono audio wave
    audio_data, sample_rate = extract_wav_from_video(video_path, sample_rate=16000)
    duration_sec = len(audio_data) / sample_rate

    print(f"[✓] Audio Sample Rate: {sample_rate} Hz")
    print(f"[✓] Total Audio Samples: {len(audio_data):,}")
    print(f"[✓] Audio Duration: {duration_sec:.2f} seconds")

    # 2. Compute 13-dim MFCC acoustic chunks
    mfcc_chunks = compute_audio_mfcc_chunks(audio_data, sample_rate=sample_rate, fps=25.0)
    print(f"[✓] Total Acoustic Hops/Frames: {mfcc_chunks.shape[0]}")
    print(f"[✓] Shape of single acoustic frame: {mfcc_chunks.shape[1]} frequency bands")
    
    print("\n🎉 SUCCESS: Audio extraction and acoustic chunking are fully functional!")

if __name__ == "__main__":
    main()

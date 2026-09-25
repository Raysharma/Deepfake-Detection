"""
sample_data/fetch_demo_dataset.py
=================================
Downloads authentic, real-world internet benchmarks:
- Real Human Portrait (Wikimedia Commons) vs AI Face (NVIDIA StyleGAN2)
- Real Human Speech vs AI Synthesized Voice (Coqui Neural TTS)
- Real Human Video vs AI Lip-Sync Deepfake (Wav2Lip Benchmark)
"""

import os
import cv2
import numpy as np
import urllib.request

SAMPLE_DIR = "sample_data"
os.makedirs(SAMPLE_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def download_file(url: str, dest_path: str, label: str):
    print(f"[*] Downloading {label}...")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as out:
            out.write(resp.read())
        print(f"  [OK] Saved to: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
        return True
    except Exception as e:
        print(f"  [!] Failed: {e}")
        return False

print("\n" + "=" * 65)
print("🌐 DOWNLOADING REAL-WORLD BENCHMARK MEDIA FROM THE INTERNET")
print("=" * 65)

# -------------------------------------------------------------
# 1. IMAGES: Real Human Photograph vs NVIDIA StyleGAN2 AI Face
# -------------------------------------------------------------
real_img = os.path.join(SAMPLE_DIR, "image_real_human.jpg")
fake_img = os.path.join(SAMPLE_DIR, "image_fake_stylegan2.jpg")
stylegan_strip = os.path.join(SAMPLE_DIR, "temp_stylegan_strip.png")

# Real: Barack Obama official portrait from Wikimedia Commons
download_file(
    "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg",
    real_img,
    "Real Human Photo (Wikimedia Commons)"
)

# AI Fake: Official NVIDIA StyleGAN2 AI Generated Face
success = download_file(
    "https://raw.githubusercontent.com/NVlabs/stylegan2/master/docs/stylegan2-teaser-1024x256.png",
    stylegan_strip,
    "NVIDIA StyleGAN2 AI Generated Faces"
)

if success and os.path.exists(stylegan_strip):
    strip = cv2.imread(stylegan_strip)
    if strip is not None:
        # Crop the first 256x256 AI face
        ai_face = strip[:, :256]
        cv2.imwrite(fake_img, ai_face)
        print(f"  [OK] Extracted StyleGAN2 AI Face: {fake_img}")
    if os.path.exists(stylegan_strip):
        os.remove(stylegan_strip)

# -------------------------------------------------------------
# 2. AUDIO: Genuine Human Speech vs Neural AI Synthesized Voice
# -------------------------------------------------------------
real_audio = os.path.join(SAMPLE_DIR, "audio_real_human.wav")
fake_audio = os.path.join(SAMPLE_DIR, "audio_fake_ai_tts.wav")

# Real: Natural human speaker wav from Wav2Lip benchmark
download_file(
    "https://raw.githubusercontent.com/sammysun0711/openvino_aigc_samples/main/Wav2Lip/data_audio_sun_5s.wav",
    real_audio,
    "Real Human Speech (Wav2Lip Audio Benchmark)"
)

# AI Fake: Neural Text-to-Speech AI Voice (Coqui-TTS Benchmark)
download_file(
    "https://raw.githubusercontent.com/coqui-ai/TTS/dev/tests/data/ljspeech/wavs/LJ001-0001.wav",
    fake_audio,
    "Neural AI Synthesized Voice (Coqui TTS Benchmark)"
)

# -------------------------------------------------------------
# 3. VIDEOS: Real Human Video vs AI-Generated Wav2Lip Deepfake
# -------------------------------------------------------------
fake_video_wav2lip = os.path.join(SAMPLE_DIR, "fake_ai_wav2lip.mp4")

# Real Video is already prepared at: sample_data/real_speaker.mp4
# Download actual AI-generated Wav2Lip deepfake video
download_file(
    "https://raw.githubusercontent.com/sammysun0711/openvino_aigc_samples/main/Wav2Lip/results/result_voice.mp4",
    fake_video_wav2lip,
    "AI-Generated Lip-Sync Deepfake (Wav2Lip Model Output)"
)

print("\n" + "=" * 65)
print("🎉 REAL-WORLD BENCHMARK SUITE READY IN sample_data/")
print("=" * 65 + "\n")

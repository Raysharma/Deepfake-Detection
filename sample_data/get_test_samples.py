import os
import urllib.request
import subprocess
import imageio_ffmpeg

os.makedirs("sample_data", exist_ok=True)

real_video_path = "sample_data/real_speaker.mp4"
fake_video_path = "sample_data/fake_desync.mp4"
temp_video_raw = "sample_data/temp_raw_video.mp4"
temp_audio_raw = "sample_data/temp_raw_audio.wav"

video_url = "https://raw.githubusercontent.com/sammysun0711/openvino_aigc_samples/main/Wav2Lip/data_video_sun_5s.mp4"
audio_url = "https://raw.githubusercontent.com/sammysun0711/openvino_aigc_samples/main/Wav2Lip/data_audio_sun_5s.wav"

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

print("[*] Downloading genuine talking sample video & authentic speech audio...")
req_v = urllib.request.Request(video_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req_v) as resp, open(temp_video_raw, 'wb') as f:
    f.write(resp.read())

req_a = urllib.request.Request(audio_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req_a) as resp, open(temp_audio_raw, 'wb') as f:
    f.write(resp.read())

print("[OK] Raw video and audio downloaded successfully.")

# 1. Produce authentic synchronized video
print("[*] Assembling authentic synchronized clip (real_speaker.mp4)...")
cmd_real = [
    ffmpeg_exe, "-y",
    "-i", temp_video_raw,
    "-i", temp_audio_raw,
    "-c:v", "copy",
    "-c:a", "aac",
    "-shortest",
    real_video_path
]
subprocess.run(cmd_real, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(f"[OK] Created AUTHENTIC video with audio: {real_video_path}")

# 2. Produce manipulated / desynchronized video (delay audio by 0.6 seconds / 15 frames)
print("[*] Assembling desynchronized lip-sync mismatch clip (fake_desync.mp4)...")
cmd_fake = [
    ffmpeg_exe, "-y",
    "-i", temp_video_raw,
    "-itsoffset", "0.6",
    "-i", temp_audio_raw,
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c:v", "copy",
    "-c:a", "aac",
    "-shortest",
    fake_video_path
]
subprocess.run(cmd_fake, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(f"[OK] Created DEEPFAKE / DESYNCHRONIZED video: {fake_video_path}")

# Cleanup temporary raw downloads
for tmp in [temp_video_raw, temp_audio_raw]:
    if os.path.exists(tmp):
        os.remove(tmp)

print("\n[SUCCESS] Both REAL and FAKE test videos are now fully assembled with real speech audio!")

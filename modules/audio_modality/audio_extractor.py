import os
import subprocess
import imageio_ffmpeg


SAMPLE_RATE = 16000


def extract_audio(video_path: str, output_dir: str = "temp_uploads") -> str:
    """
    Extract audio from a video and convert it to:
    - WAV
    - 16 kHz
    - mono

    Returns the generated WAV file path.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_audio.wav")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        output_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg audio extraction failed:\n" + result.stderr
        )

    if not os.path.exists(output_path):
        raise RuntimeError("Audio file was not created.")

    return output_path


if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else "sample_data/demo_mouth.mp4"

    path = extract_audio(video)
    print("Audio extracted successfully:")
    print(path)
import numpy as np
import librosa


SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
DURATION = 4


def load_audio(audio_path: str):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    target_length = SAMPLE_RATE * DURATION

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    return y, SAMPLE_RATE


def audio_to_mel(audio_path: str):
    y, sr = load_audio(audio_path)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=8000
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    return mel_db.astype(np.float32)
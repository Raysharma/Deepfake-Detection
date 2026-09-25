import os
import numpy as np

from .dataset import load_protocol
from .spectrogram import audio_to_mel


BASE = r"D:\deepfake_detection\LA"
CACHE_DIR = r"D:\deepfake_detection\Deepfake-Detection\modules\audio_modality\cache"


def create_cache(protocol_path, audio_dir, name):
    os.makedirs(CACHE_DIR, exist_ok=True)

    audio_files, labels = load_protocol(
        protocol_path,
        audio_dir
    )

    print(f"{name}: {len(audio_files)} files")

    first_mel = audio_to_mel(audio_files[0])

    shape = (
        len(audio_files),
        first_mel.shape[0],
        first_mel.shape[1]
    )

    cache_path = os.path.join(
        CACHE_DIR,
        f"{name}_mels.npy"
    )

    labels_path = os.path.join(
        CACHE_DIR,
        f"{name}_labels.npy"
    )

    cache = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=shape
    )

    cache[0] = first_mel.astype(np.float16)

    for i in range(1, len(audio_files)):
        mel = audio_to_mel(audio_files[i])
        cache[i] = mel.astype(np.float16)

        if (i + 1) % 100 == 0:
            print(f"{name}: {i + 1}/{len(audio_files)}")

    cache.flush()

    np.save(
        labels_path,
        np.array(labels, dtype=np.int64)
    )

    print(f"{name} cache completed.")


def main():
    create_cache(
        os.path.join(
            BASE,
            "ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.train.trn.txt"
        ),
        os.path.join(
            BASE,
            "ASVspoof2019_LA_train",
            "flac"
        ),
        "train"
    )

    create_cache(
        os.path.join(
            BASE,
            "ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.dev.trl.txt"
        ),
        os.path.join(
            BASE,
            "ASVspoof2019_LA_dev",
            "flac"
        ),
        "dev"
    )


if __name__ == "__main__":
    main()
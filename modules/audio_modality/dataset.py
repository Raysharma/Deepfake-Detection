import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .spectrogram import audio_to_mel


class ASVspoofDataset(Dataset):
    def __init__(self, audio_files, labels):
        if len(audio_files) != len(labels):
            raise ValueError("audio_files and labels must have same length")

        self.audio_files = audio_files
        self.labels = labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        mel = audio_to_mel(self.audio_files[index])

        mel_tensor = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0)

        label_tensor = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return mel_tensor, label_tensor


class CachedMelDataset(Dataset):
    def __init__(self, cache_path, labels_path):
        self.mels = np.load(cache_path, mmap_mode="r")
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mel = torch.from_numpy(
            self.mels[index].astype(np.float32)
        ).unsqueeze(0)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return mel, label


def load_protocol(protocol_path, audio_dir):
    audio_files = []
    labels = []

    with open(protocol_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            file_id = parts[1]
            label_name = parts[-1].lower()

            audio_path = os.path.join(
                audio_dir,
                file_id + ".flac"
            )

            if not os.path.exists(audio_path):
                continue

            if label_name == "bonafide":
                label = 0
            elif label_name == "spoof":
                label = 1
            else:
                continue

            audio_files.append(audio_path)
            labels.append(label)

    return audio_files, labels
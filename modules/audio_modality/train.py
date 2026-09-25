import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import CachedMelDataset
from .model import create_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-4


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = r"D:\deepfake_detection\Deepfake-Detection\modules\audio_modality\cache"
CHECKPOINT_DIR = r"D:\deepfake_detection\Deepfake-Detection\modules\audio_modality\checkpoints"


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (mel, labels) in enumerate(loader, start=1):
        mel = mel.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(mel)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mel.size(0)

        predictions = (torch.sigmoid(logits) >= 0.5).float()

        correct += (predictions == labels).sum().item()
        total += labels.numel()

        if batch_idx % 20 == 0:
            print(
                f"Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f}",
                flush=True
            )

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for mel, labels in loader:
        mel = mel.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        logits = model(mel)
        loss = criterion(logits, labels)

        total_loss += loss.item() * mel.size(0)

        predictions = (torch.sigmoid(logits) >= 0.5).float()

        correct += (predictions == labels).sum().item()
        total += labels.numel()

    return total_loss / total, correct / total


def main():

    print("Training device:", DEVICE)

    train_mels = os.path.join(
        CACHE_DIR,
        "train_mels.npy"
    )

    train_labels = os.path.join(
        CACHE_DIR,
        "train_labels.npy"
    )

    dev_mels = os.path.join(
        CACHE_DIR,
        "dev_mels.npy"
    )

    dev_labels = os.path.join(
        CACHE_DIR,
        "dev_labels.npy"
    )

    required_files = [
        train_mels,
        train_labels,
        dev_mels,
        dev_labels
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required cache file not found: {path}"
            )

    train_dataset = CachedMelDataset(
        train_mels,
        train_labels
    )

    dev_dataset = CachedMelDataset(
        dev_mels,
        dev_labels
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(dev_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = create_model().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = validate(
            model,
            dev_loader,
            criterion
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "audio_cnn_best.pth"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "epoch": epoch + 1,
                },
                checkpoint_path
            )

            print(
                "Best model saved:",
                checkpoint_path
            )


if __name__ == "__main__":
    main()
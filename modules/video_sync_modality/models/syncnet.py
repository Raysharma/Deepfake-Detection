"""
modules/video_sync_modality/models/syncnet.py
=============================================
SyncNet Dual-Stream PyTorch Architecture
----------------------------------------
Compares temporal sequences of 5 mouth frames against acoustic feature windows.
Projects both into a shared 1024-dimensional latent embedding space.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualStreamNet(nn.Module):
    """
    Visual sub-network processing 5 consecutive mouth frames.
    Input shape: (batch_size, 5, 96, 96)
    Output shape: (batch_size, 1024)
    """
    def __init__(self):
        super(VisualStreamNet, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(5, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1), # 48x48

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1), # 24x24

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1), # 12x12

            # Layer 4
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1), # 6x6
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        emb = self.fc(feat)
        # L2 Normalization
        return F.normalize(emb, p=2, dim=1)


class AudioStreamNet(nn.Module):
    """
    Audio sub-network processing 2D MFCC / spectrogram acoustic chunks.
    Input shape: (batch_size, 1, 13, 20)
    Output shape: (batch_size, 1024)
    """
    def __init__(self):
        super(AudioStreamNet, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), # 13x10

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # 6x5

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)) # 3x3
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        emb = self.fc(feat)
        # L2 Normalization
        return F.normalize(emb, p=2, dim=1)


class SyncNet(nn.Module):
    """
    Full Dual-Stream SyncNet Model.
    Computes Euclidean distance between visual mouth embedding and audio embedding.
    """
    def __init__(self):
        super(SyncNet, self).__init__()
        self.visual_stream = VisualStreamNet()
        self.audio_stream = AudioStreamNet()

    def forward(self, visual_input: torch.Tensor, audio_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_emb = self.visual_stream(visual_input)
        a_emb = self.audio_stream(audio_input)
        return v_emb, a_emb

    def compute_distance(self, visual_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        """Computes Euclidean Distance (L2 norm) between embeddings."""
        v_emb, a_emb = self.forward(visual_input, audio_input)
        return F.pairwise_distance(v_emb, a_emb)

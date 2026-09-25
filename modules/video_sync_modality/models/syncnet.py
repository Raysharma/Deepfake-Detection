"""
modules/video_sync_modality/models/syncnet.py
=============================================
Official SyncNet Dual-Stream PyTorch Architecture (Chung & Zisserman)
---------------------------------------------------------------------
Loads official pretrained weights trained on 100,000+ real talking videos
to compute true cross-modal audio-visual synchronization embeddings.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "syncnet_v2.model")


WEIGHTS_URL = "https://huggingface.co/chunyu-li/LatentSync/resolve/main/auxiliary/syncnet_v2.model"


class SyncNet(nn.Module):
    """
    Official SyncNet Dual-Stream Convolutional Architecture.
    - netcnnaud / netfcaud: Audio stream for 2D MFCC acoustic slices
    - netcnnlip / netfclip: Visual stream for 3D mouth frame sequences
    """
    def __init__(self, pretrained: bool = True):
        super(SyncNet, self).__init__()

        # 1. Audio Convolutional Sub-network
        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2), padding=1),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
        )

        # 2. Visual Lip 3D Convolutional Sub-network
        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
        )

        if pretrained:
            if not os.path.exists(WEIGHTS_PATH):
                os.makedirs(WEIGHTS_DIR, exist_ok=True)
                print("[*] Pretrained SyncNet weights not found locally. Auto-downloading...")
                try:
                    import urllib.request
                    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
                    print("[OK] Downloaded official SyncNet weights.")
                except Exception as e:
                    print(f"[!] Warning: Could not auto-download weights ({e}).")

            if os.path.exists(WEIGHTS_PATH):
                self.load_pretrained_weights(WEIGHTS_PATH)

    def load_pretrained_weights(self, path: str):
        """Loads state_dict from official pretrained syncnet_v2.model."""
        try:
            sd = torch.load(path, map_location="cpu")
            self.load_state_dict(sd)
            print(f"[OK] Successfully loaded official SyncNet pretrained weights from {path}")
        except Exception as e:
            print(f"[!] Warning: Could not load pretrained weights ({e}). Using initialized weights.")

    def forward_visual(self, visual_input: torch.Tensor) -> torch.Tensor:
        """
        Visual input shape: (batch_size, 3, 5, H, W)
        Output shape: (batch_size, 1024)
        """
        feat = self.netcnnlip(visual_input)
        feat = feat.view(feat.size(0), -1)
        emb = self.netfclip(feat)
        return F.normalize(emb, p=2, dim=1)

    def forward_audio(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Audio input shape: (batch_size, 1, 13, 20)
        Output shape: (batch_size, 1024)
        """
        feat = self.netcnnaud(audio_input)
        feat = feat.view(feat.size(0), -1)
        emb = self.netfcaud(feat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, visual_input: torch.Tensor, audio_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_emb = self.forward_visual(visual_input)
        a_emb = self.forward_audio(audio_input)
        return v_emb, a_emb

    def compute_distance(self, visual_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        """Computes Euclidean Distance (L2 norm) between visual and acoustic embeddings."""
        v_emb, a_emb = self.forward(visual_input, audio_input)
        return F.pairwise_distance(v_emb, a_emb)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Lip Encoder (3D CNN → 2D CNN)
# ----------------------------
class LipEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            # (B, 3, 96, 96)
            nn.Conv2d(3, 32, 5, stride=2, padding=2),  # 48x48
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 24x24
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 12x12
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 6x6
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1)),  # (256,1,1)
        )

        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        # x: (B, 3, 96, 96)
        feat = self.net(x)
        feat = feat.view(x.size(0), -1)
        feat = self.fc(feat)
        feat = F.normalize(feat, dim=-1)
        return feat



# ----------------------------
# Audio Encoder (Mel Encoder)
# ----------------------------
class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            # input: (B, 1, 80, T)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=(1,2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=(1,2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1)),  # → (128, 1, 1)
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, mel):
        # mel: (B, 80, T)
        mel = mel.unsqueeze(1)  # → (B,1,80,T)
        feat = self.net(mel)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)
        feat = F.normalize(feat, dim=-1)
        return feat



# ----------------------------
# SyncNet: Lip ↔ Audio Matching
# ----------------------------
class SyncNet(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.lip_encoder = LipEncoder(embed_dim)
        self.audio_encoder = AudioEncoder(embed_dim)

    def forward(self, lips, mels):
        """
        lips : (B, 3, 96, 96)
        mels : (B, 80, T)
        """
        lip_feat = self.lip_encoder(lips)
        mel_feat = self.audio_encoder(mels)

        return lip_feat, mel_feat

    def similarity(self, lips, mels):
        lip_feat, mel_feat = self.forward(lips, mels)
        return torch.sum(lip_feat * mel_feat, dim=-1)

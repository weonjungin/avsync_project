
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class VideoEncoder(nn.Module):
    """입술 ROI 시퀀스를 인코딩하는 비디오 인코더.

    Input:
        video: (B, T, C, H, W)
            B: batch size
            T: time steps (프레임 개수)
            C: 채널 수 (1 또는 3)
            H, W: 이미지 크기 (예: 96 x 96)

    Output:
        emb: (B, D)  # L2-normalized embedding
    """

    def __init__(self, emb_dim: int = 256, in_channels: int = 1):
        super().__init__()

        # ResNet18 backbone 정의 (ImageNet weight 없이)
        self.backbone = resnet18(weights=None)

        # 입력 채널이 1인 경우 첫 conv를 수정
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        # 마지막 FC 제거해서 512-dim feature만 받도록 설정
        self.backbone.fc = nn.Identity()

        # Audio encoder와 차원을 맞추기 위한 projection head
        self.proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, C, H, W)
        B, T, C, H, W = video.shape

        # (B*T, C, H, W)로 reshape 해서 frame 단위로 ResNet 적용
        x = video.view(B * T, C, H, W)
        feat = self.backbone(x)           # (B*T, 512)

        # 다시 time 차원을 복원
        feat = feat.view(B, T, -1)        # (B, T, 512)

        # 간단한 temporal aggregation: 평균 pooling
        feat = feat.mean(dim=1)           # (B, 512)

        # projection + 정규화
        emb = self.proj(feat)             # (B, D)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


class AudioEncoder(nn.Module):
    """mel-spectrogram을 인코딩하는 오디오 인코더.

    Input:
        audio: (B, 1, N_MELS, T_MEL)

    Output:
        emb: (B, D)  # L2-normalized embedding
    """

    def __init__(self, emb_dim: int = 256, in_channels: int = 1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # /2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # /4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # /8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 필요하면 여기서 한 번 더 pooling 추가 가능
        )

        # feature map 전체를 하나의 벡터로 줄이기 위한 global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Sequential(
            nn.Linear(256, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, 1, N_MELS, T_MEL)
        feat = self.conv(audio)           # (B, 256, H', W')
        feat = self.global_pool(feat)     # (B, 256, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, 256)

        emb = self.proj(feat)             # (B, D)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


class AVSyncEncoder(nn.Module):
    """Video + Audio 인코더를 함께 감싸는 wrapper 모듈.

    사용 예시:
        model = AVSyncEncoder(emb_dim=256)
        v_emb, a_emb = model(video_batch, audio_batch)
        sim = AVSyncEncoder.cosine_sim(v_emb, a_emb)
    """

    def __init__(
        self,
        emb_dim: int = 256,
        video_in_channels: int = 1,
        audio_in_channels: int = 1,
    ):
        super().__init__()
        self.video_encoder = VideoEncoder(
            emb_dim=emb_dim, in_channels=video_in_channels
        )
        self.audio_encoder = AudioEncoder(
            emb_dim=emb_dim, in_channels=audio_in_channels
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
    ):
        """video와 audio를 각각 인코딩해서 embedding 쌍을 반환한다.

        Args:
            video: (B, T, C, H, W)
            audio: (B, 1, N_MELS, T_MEL)

        Returns:
            v_emb: (B, D)
            a_emb: (B, D)
        """
        v_emb = self.video_encoder(video)
        a_emb = self.audio_encoder(audio)
        return v_emb, a_emb

    @staticmethod
    def cosine_sim(v_emb: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """L2-normalized embedding 사이의 cosine similarity.

        Args:
            v_emb: (B, D)
            a_emb: (B, D)

        Returns:
            sim: (B,)  # 각 샘플별 cosine similarity
        """
        return (v_emb * a_emb).sum(dim=-1)

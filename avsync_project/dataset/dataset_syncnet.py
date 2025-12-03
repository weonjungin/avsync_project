import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random


class DatasetSyncNet(Dataset):
    def __init__(self, root_dir, num_frames=5, mel_len=16):
        """
        root_dir: data/train
        num_frames: lip 프레임 개수
        mel_len: 모델에 넣을 mel spectrogram 길이 (T)
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.mel_len = mel_len

        self.samples = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    def __len__(self):
        return len(self.samples)

    def load_lips(self, lips_dir):
        frames = sorted(glob.glob(os.path.join(lips_dir, "*.png")))

        if len(frames) < self.num_frames:
            frames = frames * (self.num_frames // len(frames) + 1)

        frames = frames[:self.num_frames]

        imgs = []
        for f in frames:
            img = Image.open(f).convert("RGB")
            img = img.resize((96, 96))
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
            imgs.append(img)

        return torch.stack(imgs)  # (T, 3, 96, 96)

    def load_mel(self, mel_path):
        mel = np.load(mel_path)  # (80, T)
        mel = torch.from_numpy(mel).float()

        total_len = mel.size(1)

        # mel 길이가 원하는 mel_len보다 짧으면 패딩
        if total_len < self.mel_len:
            pad = self.mel_len - total_len
            mel = torch.cat([mel, torch.zeros(80, pad)], dim=1)
            return mel[:, :self.mel_len]

        # 길이가 길면 crop
        start = random.randint(0, total_len - self.mel_len)
        mel = mel[:, start:start + self.mel_len]

        return mel  # (80, mel_len)

    def __getitem__(self, idx):

        sample_dir = self.samples[idx]
        lips_dir = os.path.join(sample_dir, "lips")
        mel_path = os.path.join(sample_dir, "mel.npy")

        lips_all = self.load_lips(lips_dir)   # (T, 3, 96, 96)
        mel = self.load_mel(mel_path)         # (80, mel_len)

        # 랜덤 lip 프레임 선택
        frame_idx = random.randint(0, lips_all.shape[0] - 1)
        lips = lips_all[frame_idx]            # (3, 96, 96)

        return lips, mel, sample_dir

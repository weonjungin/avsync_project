import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random


class DatasetSyncNet(Dataset):
    def __init__(self, root_dir, num_frames=5):
        """
        root_dir: data/train
        num_frames: lip 프레임 개수 (영상에서 load할 frame 수)
        """
        self.root_dir = root_dir
        self.num_frames = num_frames

        # 각 샘플 폴더 경로들
        self.samples = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    def __len__(self):
        return len(self.samples)

    # -----------------------------
    # Lips Loader (frames → tensor)
    # -----------------------------
    def load_lips(self, lips_dir):
        frames = sorted(glob.glob(os.path.join(lips_dir, "*.png")))

        # 프레임 부족하면 반복해서 채우기
        if len(frames) < self.num_frames:
            frames = frames * (self.num_frames // len(frames) + 1)

        frames = frames[:self.num_frames]

        imgs = []
        for f in frames:
            img = Image.open(f).convert("RGB")
            img = img.resize((96, 96))

            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
            imgs.append(img)

        imgs = torch.stack(imgs)  # (T, 3, 96, 96)
        return imgs

    # -----------------------------
    # Mel Loader
    # -----------------------------
    def load_mel(self, mel_path):
        mel = np.load(mel_path)
        mel = torch.from_numpy(mel).float()  # (80, T)
        return mel

    # -----------------------------
    # Return a single sample
    # -----------------------------
    def __getitem__(self, idx):

        sample_dir = self.samples[idx]

        # 경로 설정
        lips_dir = os.path.join(sample_dir, "lips")
        mel_path = os.path.join(sample_dir, "mel.npy")

        # Load data
        lips_all = self.load_lips(lips_dir)        # shape = (T, 3, 96, 96)
        mel = self.load_mel(mel_path)              # shape = (80, T)

        # -----------------------------
        # 🔥 핵심 수정: 랜덤 1프레임 선택
        # -----------------------------
        frame_idx = random.randint(0, lips_all.shape[0] - 1)
        lips = lips_all[frame_idx]                # shape = (3, 96, 96)

        return lips, mel, sample_dir

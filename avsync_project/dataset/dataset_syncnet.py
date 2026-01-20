import os
import glob
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class DatasetSyncNet(Dataset):
    """
    SyncNet-style dataset:
    - Offline-preprocessed mouth ROI frames (roi/*.png or lips/*.png)
    - mel.npy (80, Tmel)
    - meta.json (recommended): fps, sr, hop_length

    Key behavior:
    - Sample a random start video frame t0
    - Take lip window: [t0, t0+Twin)
    - Take mel window aligned to the SAME time span
    """

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 5,        # Twin
        mel_len: int | None = None, # if None -> derived from (fps,sr,hop)
        img_size: int = 96,
        fps_default: float = 25.0,
        sr_default: int = 16000,
        hop_default: int = 160,
        recursive: bool = True,
        strict: bool = True,
    ):
        """
        Args:
            root_dir: dataset root. Can contain nested speaker/clip dirs.
            num_frames: Twin (video frames per sample).
            mel_len: # mel frames per sample. If None, derive from meta.
            img_size: ROI frame size (img_size x img_size).
            fps_default/sr_default/hop_default: fallback if meta.json missing.
            recursive: if True, find samples by locating mel.npy recursively.
            strict: if True, raise on invalid samples (recommended for QC scan).
        """
        self.root_dir = root_dir
        self.num_frames = int(num_frames)
        self.mel_len = None if mel_len is None else int(mel_len)
        self.img_size = int(img_size)

        self.fps_default = float(fps_default)
        self.sr_default = int(sr_default)
        self.hop_default = int(hop_default)

        self.recursive = bool(recursive)
        self.strict = bool(strict)

        self.samples = self._build_samples()

    # -------------------------
    # Index building
    # -------------------------
    def _build_samples(self):
        if self.recursive:
            # Robust: treat any directory containing mel.npy as a sample_dir
            mel_paths = glob.glob(os.path.join(self.root_dir, "**", "mel.npy"), recursive=True)
            sample_dirs = sorted({os.path.dirname(p) for p in mel_paths})
            return sample_dirs

        # Fallback: 1-depth only (old behavior)
        return sorted([
            os.path.join(self.root_dir, d)
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

    def __len__(self):
        return len(self.samples)

    # -------------------------
    # Meta + file discovery
    # -------------------------
    def _load_meta(self, sample_dir):
        meta_path = os.path.join(sample_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            fps = float(meta.get("fps", self.fps_default))
            sr = int(meta.get("sr", self.sr_default))
            hop = int(meta.get("hop_length", self.hop_default))
            return fps, sr, hop, meta
        else:
            return self.fps_default, self.sr_default, self.hop_default, {}

    def _get_frames_list(self, sample_dir):
        # preferred
        roi_dir = os.path.join(sample_dir, "roi")
        if os.path.isdir(roi_dir):
            frames = sorted(glob.glob(os.path.join(roi_dir, "*.png")))
            if frames:
                return frames

        # backward compat
        lips_dir = os.path.join(sample_dir, "lips")
        frames = sorted(glob.glob(os.path.join(lips_dir, "*.png")))
        return frames

    # -------------------------
    # Loading windows
    # -------------------------
    def _load_lip_window(self, frames, t0):
        """
        Returns: (Twin, 1, img_size, img_size)
        """
        Twin = self.num_frames
        T = len(frames)
        if T < Twin:
            # For sync training, repeating frames can inject fake motion/timing.
            # Better to skip such samples (preprocess should enforce min frames).
            msg = f"Too few ROI frames: T={T} < Twin={Twin}"
            if self.strict:
                raise ValueError(msg)
            # non-strict fallback: pad by repeating last frame (least bad)
            idxs = list(range(T)) + [T - 1] * (Twin - T)
        else:
            idxs = list(range(t0, t0 + Twin))

        imgs = []
        for k in idxs:
            img = Image.open(frames[k]).convert("L")  # grayscale
            if img.size != (self.img_size, self.img_size):
                img = img.resize((self.img_size, self.img_size))
            arr = np.array(img, dtype=np.float32) / 255.0   # (H,W)
            ten = torch.from_numpy(arr).unsqueeze(0)        # (1,H,W)
            imgs.append(ten)

        return torch.stack(imgs, dim=0)

    def _mel_per_vframe(self, fps, sr, hop):
        # mel frames per second = sr/hop
        # video frames per second = fps
        return (sr / hop) / fps

    def _load_mel_window(self, mel_path, t0, fps, sr, hop):
        """
        Returns: (80, mel_len)
        """
        mel = np.load(mel_path)  # (80, Tmel)
        mel = torch.from_numpy(mel).float()
        Tmel = mel.size(1)

        m_per_v = self._mel_per_vframe(fps, sr, hop)

        # Determine mel_len
        if self.mel_len is None:
            mel_len = int(round(self.num_frames * m_per_v))
            mel_len = max(mel_len, 1)
        else:
            mel_len = int(self.mel_len)

        # Map video frame t0 -> mel index
        mel_start = int(round(t0 * m_per_v))
        mel_end = mel_start + mel_len

        # Pad if needed
        if mel_end > Tmel:
            pad = mel_end - Tmel
            mel = torch.cat([mel, torch.zeros(80, pad)], dim=1)

        return mel[:, mel_start:mel_start + mel_len]

    # -------------------------
    # Main
    # -------------------------
    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        frames = self._get_frames_list(sample_dir)
        if len(frames) == 0:
            raise FileNotFoundError(f"missing roi/lips frames in {sample_dir}")

        mel_path = os.path.join(sample_dir, "mel.npy")
        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"missing mel.npy in {sample_dir}")

        fps, sr, hop, meta = self._load_meta(sample_dir)

        Twin = self.num_frames
        T = len(frames)

        # Choose t0 so that lip window doesn't wrap
        if T >= Twin:
            t0 = random.randint(0, T - Twin)
        else:
            t0 = 0  # will be handled by strict/non-strict in _load_lip_window

        # -------------------------
        # Positive (aligned)
        # -------------------------
        lips = self._load_lip_window(frames, t0)                   # (Twin, 1, S, S)
        mel_pos = self._load_mel_window(mel_path, t0, fps, sr, hop) # (80, mel_len)

        # -------------------------
        # Negative (offset shift)
        # -------------------------
        # 예: ±5 프레임 범위에서 0 제외
        max_off = getattr(self, "neg_max_offset", 5)  # 없으면 5로
        offs = [o for o in range(-max_off, max_off + 1) if o != 0]
        neg_off = random.choice(offs)

        # t0 기준으로 mel만 misalign 시키는 게 SyncNet 정석
        # (lip은 그대로 두고 audio를 민다)
        t0_neg = t0 + neg_off

        # mel window가 깨지지 않도록 클램프
        # (t0_neg가 너무 앞/뒤로 가면 _load_mel_window에서 에러 날 수 있음)
        if T >= Twin:
            t0_neg = max(0, min(t0_neg, T - Twin))
        else:
            t0_neg = 0

        mel_neg = self._load_mel_window(mel_path, t0_neg, fps, sr, hop)  # (80, mel_len)

        return lips, mel_pos, mel_neg, neg_off, sample_dir


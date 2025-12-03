import os
import glob
import torch
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from avsync_project.models.syncnet_model import SyncNet


# ---------------------------------------
# Load lips frames
# ---------------------------------------
def load_lips(lips_dir, num_frames=5):
    frame_paths = sorted(glob.glob(os.path.join(lips_dir, "*.png")))
    if len(frame_paths) == 0:
        raise ValueError(f"No lips frames found in {lips_dir}")

    # 부족하면 반복해서 채움
    if len(frame_paths) < num_frames:
        frame_paths = frame_paths * (num_frames // len(frame_paths) + 1)
    frame_paths = frame_paths[:num_frames]

    imgs = []
    for f in frame_paths:
        img = Image.open(f).convert("RGB")
        img = img.resize((96, 96))
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.permute(2, 0, 1)  # (3, 96, 96)
        imgs.append(img)

    return torch.stack(imgs)  # (num_frames, 3, 96, 96)


# ---------------------------------------
# Load mel (80, T)
# ---------------------------------------
def load_mel(mel_path):
    mel = np.load(mel_path)
    mel = torch.from_numpy(mel).float()
    return mel


# ---------------------------------------
# Compute Sync Score
# ---------------------------------------
def compute_sync_score(model, lips, mel, device):

    model.eval()

    # lips shape: (frames, 3, 96, 96)
    # mel shape : (80, T)

    # 1) mean pooling → (1, 3, 96, 96)
    lips = lips.mean(dim=0, keepdim=True)

    # 2) mel → (1, 80, T)
    mel = mel.unsqueeze(0)

    # 3) 여기 추가!!  🔥🔥🔥 중요
    lips = lips.to(device)
    mel = mel.to(device)

    with torch.no_grad():
        v_emb, a_emb = model(lips, mel)

    cos = torch.nn.CosineSimilarity(dim=1)
    score = cos(v_emb, a_emb).item()

    return score



# ---------------------------------------
# Main
# ---------------------------------------
def main():
    # 선택할 샘플 경로
    sample_dir = "data/train/AvWWVOgaMlk_90000"
    ckpt_path = "logs/checkpoints/syncnet_ckpt.pth"

    lips_dir = os.path.join(sample_dir, "lips")
    mel_path = os.path.join(sample_dir, "mel.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # Load model
    model = SyncNet(embed_dim=256).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")

    # Load data
    lips = load_lips(lips_dir)
    mel = load_mel(mel_path)

    # Compute score
    score = compute_sync_score(model, lips, mel, device)
    print("\n====================================")
    print(f" Sync Score for sample: {score:.4f}")
    print("====================================\n")


if __name__ == "__main__":
    main()

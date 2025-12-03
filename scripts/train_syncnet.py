import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# dataset & model import
from avsync_project.dataset.dataset_syncnet import DatasetSyncNet
from avsync_project.models.syncnet_model import SyncNet
from avsync_project.utils.utils import set_seed

from avsync_project.loss.contrastive_loss import ContrastiveSyncLoss  # ← 방금 만든 loss 파일


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for lips, mel, _ in dataloader:
        lips = lips.to(device)      # (B, T, 3, 96, 96)
        mel = mel.to(device)        # (B, 80, Tm)

        # -------------------------
        # 1) Positive pair
        # -------------------------
        v_pos, a_pos = model(lips, mel)

        # -------------------------
        # 2) Negative pair 생성
        # -------------------------
        B = lips.size(0)
        perm = torch.randperm(B)
        mel_neg = mel[perm]         # 오디오만 섞기

        v_neg, a_neg = model(lips, mel_neg)

        # -------------------------
        # 3) Loss 계산
        # -------------------------
        loss = criterion(v_pos, a_pos, v_neg, a_neg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():

    # -------------------------
    # Config
    # -------------------------
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data_root = "data/train"
    batch_size = 4
    lr = 1e-4
    num_epochs = 10

    save_dir = "logs/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "syncnet_ckpt.pth")

    # -------------------------
    # Dataset
    # -------------------------
    dataset = DatasetSyncNet(data_root)

    valid_samples = []
    for i in range(len(dataset)):
        try:
            _ = dataset[i]   # 한 번 로딩해보고
            valid_samples.append(dataset.samples[i])
        except Exception as e:
            print("⚠ Skip:", dataset.samples[i], "| Reason:", e)

    dataset.samples = valid_samples
    print(f"Valid dataset size: {len(dataset.samples)} samples")

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    print(f"Dataset size: {len(dataset)} samples")

    # -------------------------
    # Model / Loss / Optimizer
    # -------------------------
    model = SyncNet(embed_dim=256).to(device)
    criterion = ContrastiveSyncLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # Train Loop
    # -------------------------
    for epoch in range(1, num_epochs + 1):
        avg_loss = train(model, dataloader, optimizer, criterion, device)
        print(f"[Epoch {epoch}/{num_epochs}] Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), ckpt_path)
        print(f"✔ Saved: {ckpt_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()

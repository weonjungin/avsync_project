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

from avsync_project.loss.contrastive_loss import ContrastiveSyncLoss 


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for lips, mel, _ in dataloader:
        lips = lips.to(device)      # (B, ...)
        mel = mel.to(device)        # (B, 80, Tm)

        B = lips.size(0)
        if B < 2:
            # 배치가 1이면 in-batch negative 불가 → 스킵 or 기존 방식으로 대체
            continue

        # -------------------------
        # 1) Positive pair
        # -------------------------
        v_pos, a_pos = model(lips, mel)

        # -------------------------
        # 2) In-batch negatives (배치 내 다른 mel 전부)
        # -------------------------
        neg_losses = []
        for shift in range(1, B):  # 1..B-1
            mel_neg = mel.roll(shifts=shift, dims=0)  # (B, 80, Tm)
            v_neg, a_neg = model(lips, mel_neg)
            neg_losses.append(criterion(v_pos, a_pos, v_neg, a_neg))

        loss = torch.stack(neg_losses).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(dataloader))



def main():

    # -------------------------
    # Config
    # -------------------------
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data_root = "data/avspeech_1000_me25"
    batch_size = 8
    lr = 5e-5
    num_epochs = 80

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

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        avg_loss = train(model, dataloader, optimizer, criterion, device)
        print(f"[Epoch {epoch}/{num_epochs}] Loss: {avg_loss:.4f}")

        if device == "cuda":
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"[GPU] peak allocated: {peak:.2f} GB")

        torch.save(model.state_dict(), ckpt_path)
        print(f"✔ Saved: {ckpt_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()

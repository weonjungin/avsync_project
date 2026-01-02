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
from avsync_project.loss.info_nce_loss import InfoNCESyncLoss
from avsync_project.loss.hard_infonce_loss import HardInfoNCESyncLoss

import argparse
from pathlib import Path
import yaml


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for lips, mel, _ in dataloader:
        lips = lips.to(device)
        mel = mel.to(device)

        B = lips.size(0)
        if B < 2:
            continue

        v_emb, a_emb = model(lips, mel)
        loss = criterion(v_emb, a_emb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))

"""
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

    return total_loss / max(1, len(dataloader)) """""

def _load_yaml(path: str):
    p = Path(path).expanduser()

    if not p.is_absolute():
        proj_root = Path(__file__).resolve().parents[1]
        p = (proj_root / p).resolve()
    with open(p, "r") as f:
        return yaml.safe_load(f)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp.yaml")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--mel_len", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--ckpt_name", type=str, default=None)
    
    args = parser.parse_args()
    
    cfg = _load_yaml(args.config) if args.config else {}

    seed = cfg.get("common", {}).get("seed", 42)

    data_root = (
        args.root
        or cfg.get("dataset", {}).get("root", "data/avspeech_1000_me25"))

    batch_size = int(
        args.batch_size
        or cfg.get("train", {}).get("batch_size", 8))

    lr = float(
        args.lr
        or cfg.get("train", {}).get("lr", 5e-5))

    num_epochs = int(
        args.epochs
        or cfg.get("train", {}).get("epochs", 80))

    num_frames = int(
        args.num_frames
        or cfg.get("common", {}).get("num_frames", 5))

    mel_len = int(
        args.mel_len
        or cfg.get("common", {}).get("mel_len", 16))

    save_dir = (
        args.save_dir
        or cfg.get("train", {}).get("save_dir", "logs/checkpoints"))

    ckpt_name = (
        args.ckpt_name
        or cfg.get("train", {}).get("ckpt_name", "syncnet_ckpt.pth"))



    # dataset
    if "dataset" in cfg:
        if hasattr(args, "root"):
            args.root = cfg["dataset"].get("root", args.root)

    # common
    if "common" in cfg:
        for k in ["seed", "num_frames", "mel_len"]:
            if hasattr(args, k) and k in cfg["common"]:
                setattr(args, k, cfg["common"][k])

    # eval
    if "eval" in cfg:
        e = cfg["eval"]

        for k in ["neg_k", "do_pair", "do_rank", "do_offset"]:
            if hasattr(args, k) and k in e:
                setattr(args, k, e[k])

        if "rank_ks" in e:
            args.rank_ks = ",".join(str(x) for x in e["rank_ks"])

        if "offsets" in e:
            args.offsets = ",".join(str(x) for x in e["offsets"])

        if "save_csv" in e:
            args.save_csv = e["save_csv"]

        if "save_offset_csv" in e:
            args.save_offset_csv = e["save_offset_csv"]

    # -------------------------
    # Config
    # -------------------------
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, ckpt_name)


    # -------------------------
    # Dataset
    # -------------------------
    dataset = DatasetSyncNet(data_root, num_frames=num_frames, mel_len=mel_len)


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
    
    # criterion = ContrastiveSyncLoss(margin=0.3)
    # criterion = InfoNCESyncLoss(temperature=0.07)
    temperature = float(cfg.get("train", {}).get("temperature", 0.07))
    hard_k = int(cfg.get("train", {}).get("hard_k", 5))
    criterion = HardInfoNCESyncLoss(temperature=temperature, hard_k=hard_k)
   
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

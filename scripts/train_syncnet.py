import os
import sys
import argparse
from pathlib import Path
import yaml
import re
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from avsync_project.dataset.dataset_syncnet import DatasetSyncNet
from avsync_project.models.syncnet_model import SyncNet
from avsync_project.utils.utils import set_seed

from avsync_project.loss.contrastive_loss import ContrastiveSyncLoss


def _load_yaml(path: str):
    p = Path(path).expanduser()
    if not p.is_absolute():
        proj_root = Path(__file__).resolve().parents[1]
        p = (proj_root / p).resolve()
    with open(p, "r") as f:
        return yaml.safe_load(f)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# -------------------------
# Pairwise InfoNCE (pos vs neg 2-way)
# -------------------------
class PairInfoNCELoss(torch.nn.Module):
    """
    각 샘플별로 (v, a_pos) vs (v, a_neg) 두 개 중 pos가 더 크도록 하는 2-way InfoNCE.
    logits = [sim(v,a_pos), sim(v,a_neg)] / T
    label = 0 (pos)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, a_pos, a_neg):
        v = F.normalize(v, dim=1)
        a_pos = F.normalize(a_pos, dim=1)
        a_neg = F.normalize(a_neg, dim=1)

        sim_pos = (v * a_pos).sum(dim=1, keepdim=True)  # (B,1)
        sim_neg = (v * a_neg).sum(dim=1, keepdim=True)  # (B,1)
        logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature  # (B,2)

        labels = torch.zeros(v.size(0), dtype=torch.long, device=v.device)  # pos=0
        return F.cross_entropy(logits, labels)


def build_pair_criterion(cfg):
    loss_name = cfg.get("train", {}).get("loss", "contrastive")  # "contrastive" | "pair_infonce"

    if loss_name == "pair_infonce":
        temperature = float(cfg.get("train", {}).get("temperature", 0.07))
        return PairInfoNCELoss(temperature=temperature)

    margin = float(cfg.get("train", {}).get("margin", 0.3))
    return ContrastiveSyncLoss(margin=margin)


# -------------------------
# Speaker split (GRID: s1_processed ~ s34_processed)
# -------------------------
def extract_speaker_id(sample_dir: str) -> str:
    """
    sample_dir 예:
      data/s1_processed/xxxxxx
      /media/.../s10_processed/xxxxxx
    -> return: "s1", "s10"
    """
    parts = sample_dir.replace("\\", "/").split("/")
    for p in parts:
        m = re.match(r"^(s\d+)_processed$", p)  # s10_processed
        if m:
            return m.group(1)  # s10
        m = re.match(r"^(s\d+)$", p)  # 혹시 s10 폴더가 있으면
        if m:
            return m.group(1)
    raise ValueError(f"Cannot parse speaker id from: {sample_dir}")


def split_by_speaker_count(samples, seed=42, n_train=28, n_val=3):
    """
    speaker-disjoint 고정 count split.
    - train speakers: n_train
    - val speakers: n_val
    - test speakers: remaining
    """
    spk_to_samples = {}
    for s in samples:
        spk = extract_speaker_id(s)
        spk_to_samples.setdefault(spk, []).append(s)

    speakers = sorted(spk_to_samples.keys())  # ["s1","s2",...]
    rng = random.Random(seed)
    rng.shuffle(speakers)

    if len(speakers) < (n_train + n_val + 1):
        raise ValueError(f"Not enough speakers: got {len(speakers)}, need >= {n_train + n_val + 1}")

    train_spk = speakers[:n_train]
    val_spk = speakers[n_train:n_train + n_val]
    test_spk = speakers[n_train + n_val:]

    def gather(spk_list):
        out = []
        for spk in spk_list:
            out.extend(spk_to_samples[spk])
        return out

    return gather(train_spk), gather(val_spk), gather(test_spk), (train_spk, val_spk, test_spk)


# -------------------------
# Train / Eval
# -------------------------
def run_one_epoch(model, dataloader, optimizer, criterion, device, *, train: bool, grad_clip=None, print_shapes_once=False):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_steps = 0

    for step, batch in enumerate(dataloader):
        # Dataset returns:
        # lips, mel_pos, mel_neg, neg_off, sample_dir
        lips, mel_pos, mel_neg, neg_off, sample_dir = batch

        lips = lips.to(device, non_blocking=True)
        mel_pos = mel_pos.to(device, non_blocking=True)
        mel_neg = mel_neg.to(device, non_blocking=True)

        B = lips.size(0)
        if B < 2:
            continue

        if print_shapes_once and step == 0:
            neg_preview = neg_off[:4].tolist() if torch.is_tensor(neg_off) else neg_off
            print(
                "[Sanity]",
                "lips:", tuple(lips.shape),
                "mel_pos:", tuple(mel_pos.shape),
                "mel_neg:", tuple(mel_neg.shape),
                "neg_off:", neg_preview
            )

        with torch.set_grad_enabled(train):
            # Positive pair (aligned)
            v_pos, a_pos = model(lips, mel_pos)
            # Negative pair (offset-shifted audio)
            v_neg, a_neg = model(lips, mel_neg)

            # Loss
            if isinstance(criterion, PairInfoNCELoss):
                loss = criterion(v_pos, a_pos, a_neg)
            else:
                loss = criterion(v_pos, a_pos, v_neg, a_neg)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += float(loss.item())
        n_steps += 1

    return total_loss / max(1, n_steps)


def save_checkpoint(path, model, optimizer, epoch, cfg, best_val=None):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
        "best_val": best_val,
    }
    torch.save(ckpt, path)


def load_model_only(path, model, device):
    ckpt = torch.load(path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        # fallback: state_dict만 저장된 경우
        model.load_state_dict(ckpt)
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp.yaml")

    # overrides
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--mel_len", type=int, default=None)

    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--ckpt_name", type=str, default=None)

    # training utils
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--qc_scan", action="store_true")
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--print_shapes", action="store_true")

    # split fixed counts
    parser.add_argument("--train_spk", type=int, default=28)
    parser.add_argument("--val_spk", type=int, default=3)

    args = parser.parse_args()
    cfg = _load_yaml(args.config) if args.config else {}

    seed = int(cfg.get("common", {}).get("seed", 42))
    set_seed(seed)

    data_root = args.root or cfg.get("dataset", {}).get("root", "data/grid_processed")
    batch_size = int(args.batch_size or cfg.get("train", {}).get("batch_size", 8))
    lr = float(args.lr or cfg.get("train", {}).get("lr", 5e-5))
    num_epochs = int(args.epochs or cfg.get("train", {}).get("epochs", 80))
    num_frames = int(args.num_frames or cfg.get("common", {}).get("num_frames", 5))
    mel_len = int(args.mel_len or cfg.get("common", {}).get("mel_len", 16))

    save_dir = args.save_dir or cfg.get("train", {}).get("save_dir", "logs/checkpoints")
    ckpt_name = args.ckpt_name or cfg.get("train", {}).get("ckpt_name", "syncnet_ckpt.pth")
    ckpt_path = os.path.join(save_dir, ckpt_name)

    best_ckpt_path = os.path.join(save_dir, "best_" + ckpt_name)

    num_workers = int(args.num_workers if args.num_workers is not None else cfg.get("train", {}).get("num_workers", 4))
    grad_clip = args.grad_clip if args.grad_clip is not None else cfg.get("train", {}).get("grad_clip", None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Dataset (full)
    # -------------------------
    dataset = DatasetSyncNet(data_root, num_frames=num_frames, mel_len=mel_len)

    if args.qc_scan:
        valid_samples = []
        for i in range(len(dataset)):
            try:
                _ = dataset[i]
                valid_samples.append(dataset.samples[i])
            except Exception as e:
                print("⚠ Skip:", dataset.samples[i], "| Reason:", e)
        dataset.samples = valid_samples
        print(f"[QC] Valid dataset size: {len(dataset.samples)} samples")

    print(f"[All] clips: {len(dataset)}")

    # -------------------------
    # Speaker-disjoint split (fixed count: 28/3/rest)
    # -------------------------
    all_samples = list(dataset.samples)
    train_samples, val_samples, test_samples, (train_spk, val_spk, test_spk) = split_by_speaker_count(
        all_samples, seed=seed, n_train=args.train_spk, n_val=args.val_spk
    )

    print(f"[Split] speakers: train={len(train_spk)} val={len(val_spk)} test={len(test_spk)}")
    print(f"[Split] clips:    train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"[Split] train speakers: {train_spk}")
    print(f"[Split] val speakers:   {val_spk}")
    print(f"[Split] test speakers:  {test_spk}")

    train_dataset = deepcopy(dataset)
    val_dataset = deepcopy(dataset)

    train_dataset.samples = train_samples
    val_dataset.samples = val_samples

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(not args.no_shuffle),
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )

    # val/test는 shuffle False
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )


    # -------------------------
    # Model / Loss / Optimizer
    # -------------------------
    model = SyncNet(embed_dim=int(cfg.get("model", {}).get("embed_dim", 256))).to(device)
    criterion = build_pair_criterion(cfg)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # Train Loop (with val + best)
    # -------------------------
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loss = run_one_epoch(
            model, train_loader, optimizer, criterion, device,
            train=True,
            grad_clip=grad_clip,
            print_shapes_once=args.print_shapes
        )

        val_loss = run_one_epoch(
            model, val_loader, optimizer, criterion, device,
            train=False,
            grad_clip=None,
            print_shapes_once=False
        )

        print(f"[Epoch {epoch}/{num_epochs}] train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        if device == "cuda":
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"[GPU] peak allocated: {peak:.2f} GB")

        # always save last
        save_checkpoint(ckpt_path, model, optimizer, epoch, cfg, best_val=best_val)
        print(f"✔ Saved (last): {ckpt_path}")

        # save best on val
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, cfg, best_val=best_val)
            print(f"⭐ Saved (best): {best_ckpt_path}  (best_val={best_val:.4f})")

    print("Training complete!")



if __name__ == "__main__":
    main()

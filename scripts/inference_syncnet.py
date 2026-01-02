import os
import glob
import argparse
import random

import torch
import numpy as np
from PIL import Image
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from avsync_project.models.syncnet_model import SyncNet

from pathlib import Path
import yaml

# ---------------------------------------
# Load lips frames
# ---------------------------------------
def load_lips(lips_dir: str, num_frames: int = 5) -> torch.Tensor:
    frame_paths = sorted(glob.glob(os.path.join(lips_dir, "*.png")))
    if len(frame_paths) == 0:
        raise ValueError(f"No lips frames found in {lips_dir}")

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

    return torch.stack(imgs)  # (T, 3, 96, 96)


# ---------------------------------------
# Load mel (80, T) + crop/pad to mel_len
# ---------------------------------------
def load_mel(mel_path: str, mel_len: int = 16) -> torch.Tensor:
    mel = np.load(mel_path)  # (80, T)
    mel = torch.from_numpy(mel).float()

    if mel.ndim != 2 or mel.shape[0] != 80:
        raise ValueError(f"Bad mel shape: {tuple(mel.shape)} in {mel_path}")

    T = mel.size(1)
    if T < mel_len:
        pad = mel_len - T
        mel = torch.cat([mel, torch.zeros(80, pad)], dim=1)
        mel = mel[:, :mel_len]
    elif T > mel_len:
        start = (T - mel_len) // 2  # deterministic center crop
        mel = mel[:, start:start + mel_len]

    return mel  # (80, mel_len)


@torch.no_grad()
def compute_score(model: SyncNet, lips: torch.Tensor, mel: torch.Tensor, device: str) -> float:
    model.eval()
    # lips: (T,3,96,96) -> (1,T,3,96,96)
    lips = lips.unsqueeze(0)
    mel = mel.unsqueeze(0)  # (1,80,T)

    lips = lips.to(device)
    mel = mel.to(device)

    v_emb, a_emb = model(lips, mel)
    return float(torch.nn.functional.cosine_similarity(v_emb, a_emb, dim=1).item())


def is_valid_sample(sample_dir: str) -> bool:
    lips_dir = os.path.join(sample_dir, "lips")
    mel_path = os.path.join(sample_dir, "mel.npy")
    if not os.path.isdir(lips_dir):
        return False
    if not os.path.isfile(mel_path):
        return False
    if len(glob.glob(os.path.join(lips_dir, "*.png"))) == 0:
        return False
    return True


def pick_negative_sample(all_samples, exclude_sd: str) -> str:
    # pick a different sample dir
    if len(all_samples) <= 1:
        raise ValueError("Need at least 2 valid samples for negative sampling.")
    while True:
        cand = random.choice(all_samples)
        if cand != exclude_sd:
            return cand
        
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
    parser.add_argument("--root", default="data/train", help="sample root dir")
    parser.add_argument("--ckpt", default="logs/checkpoints/syncnet_ckpt.pth", help="checkpoint path")
    parser.add_argument("--sample", default="", help="run only one sample dir (optional)")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--mel_len", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--neg_k", type=int, default=5, help="number of negative mels per sample (mean over K)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for negative sampling")
    parser.add_argument("--save_csv", default="logs/inference_scores_with_neg.csv")
    args = parser.parse_args()

    # -------------------------
    # Apply YAML config
    # -------------------------
    cfg = _load_yaml(args.config) if args.config else {}

    # dataset.root
    if "dataset" in cfg and "root" in cfg["dataset"]:
        args.root = cfg["dataset"]["root"]

    # paths.ckpt
    if "paths" in cfg and "ckpt" in cfg["paths"]:
        args.ckpt = cfg["paths"]["ckpt"]

    # common
    if "common" in cfg:
        c = cfg["common"]
        if "seed" in c:
            args.seed = c["seed"]
        if "num_frames" in c:
            args.num_frames = c["num_frames"]
        if "mel_len" in c:
            args.mel_len = c["mel_len"]

    # inference
    if "inference" in cfg:
        inf = cfg["inference"]
        for k in ["max_samples", "neg_k", "save_csv", "sample"]:
            if k in inf:
                setattr(args, k, inf[k])


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # Load model
    model = SyncNet(embed_dim=256).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.ckpt}")

    # Build sample list
    if args.sample:
        samples = [args.sample]
    else:
        samples = sorted([
            os.path.join(args.root, d) for d in os.listdir(args.root)
            if os.path.isdir(os.path.join(args.root, d))
        ])
        if args.max_samples and args.max_samples > 0:
            samples = samples[:args.max_samples]

    # Pre-filter valid samples for negative sampling pool
    valid_pool = [sd for sd in samples if is_valid_sample(sd)] if args.sample else [
        sd for sd in sorted([
            os.path.join(args.root, d) for d in os.listdir(args.root)
            if os.path.isdir(os.path.join(args.root, d))
        ]) if is_valid_sample(sd)
    ]

    results = []
    ok, skip = 0, 0

    for sd in samples:
        try:
            if not is_valid_sample(sd):
                raise ValueError("missing lips/*.png or mel.npy")

            lips_dir = os.path.join(sd, "lips")
            mel_path = os.path.join(sd, "mel.npy")

            lips = load_lips(lips_dir, num_frames=args.num_frames)
            mel_pos = load_mel(mel_path, mel_len=args.mel_len)

            pos = compute_score(model, lips, mel_pos, device)

            # negatives: same lips, different mel(s)
            neg_scores = []
            for _ in range(max(1, args.neg_k)):
                neg_sd = pick_negative_sample(valid_pool, exclude_sd=sd)
                neg_mel_path = os.path.join(neg_sd, "mel.npy")
                mel_neg = load_mel(neg_mel_path, mel_len=args.mel_len)
                neg_scores.append(compute_score(model, lips, mel_neg, device))

            neg_mean = float(np.mean(neg_scores))
            margin = float(pos - neg_mean)

            results.append({
                "sample_dir": sd,
                "pos": pos,
                "neg_mean": neg_mean,
                "margin": margin,
                "neg_k": int(max(1, args.neg_k)),
                "neg_scores": ",".join([f"{v:.6f}" for v in neg_scores]),
            })

            if args.sample:
                print("\n====================================")
                print(f" POS  : {pos:.4f}")
                print(f" NEG  : {neg_mean:.4f} (mean over K={max(1, args.neg_k)})")
                print(f" MARG : {margin:.4f} (pos - neg)")
                print("====================================\n")

            ok += 1

        except Exception as e:
            print(f"⚠ Skip: {sd} | Reason: {e}")
            skip += 1

    print(f"\nDone. OK={ok} SKIP={skip}")

    # Save csv
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    df = pd.DataFrame(results)

    # 아무것도 처리된 게 없으면 여기서 종료
    if df.empty:
        print("No inference results to summarize (all samples skipped or failed).")
        return

    # margin 컬럼이 없으면 만들어주기 (안전장치)
    if "margin" not in df.columns:
        if ("pos" in df.columns) and ("neg_mean" in df.columns):
            df["margin"] = df["pos"] - df["neg_mean"]
        else:
            print("Results exist but missing required columns for margin:", df.columns.tolist())
            return

    df = df.sort_values("margin", ascending=False)

    df.to_csv(args.save_csv, index=False)
    print(f"Saved: {args.save_csv}")

    # Print top/bottom by margin
    if len(df) > 0:
        print("\nTop 5 by margin:")
        print(df[["sample_dir", "pos", "neg_mean", "margin"]].head(5).to_string(index=False))
        print("\nBottom 5 by margin:")
        print(df[["sample_dir", "pos", "neg_mean", "margin"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
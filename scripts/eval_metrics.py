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
# Load mel (80, T) + crop/pad to mel_len (center crop)
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


# ---------------------------------------
# Load full mel (80, T) without cropping
# ---------------------------------------
def load_mel_full(mel_path: str) -> np.ndarray:
    mel = np.load(mel_path)
    if mel.ndim != 2 or mel.shape[0] != 80:
        raise ValueError(f"Bad mel shape: {tuple(mel.shape)} in {mel_path}")
    return mel.astype(np.float32)  # (80, T)


def crop_pad_mel_np(mel: np.ndarray, mel_len: int = 16) -> np.ndarray:
    """Take first mel_len frames after any shift. Deterministic."""
    T = mel.shape[1]
    if T < mel_len:
        pad = mel_len - T
        mel = np.concatenate([mel, np.zeros((80, pad), dtype=mel.dtype)], axis=1)
        return mel[:, :mel_len]
    if T > mel_len:
        return mel[:, :mel_len]
    return mel


def shift_mel_np(mel: np.ndarray, shift: int) -> np.ndarray:
    """
    mel: (80, T)
    shift > 0: move right (pad zeros at beginning)  -> audio delayed
    shift < 0: move left  (pad zeros at end)        -> audio advanced
    """
    if shift == 0:
        return mel
    out = np.zeros_like(mel)
    T = mel.shape[1]
    if shift > 0:
        s = shift
        if s < T:
            out[:, s:] = mel[:, :T - s]
    else:
        s = -shift
        if s < T:
            out[:, :T - s] = mel[:, s:]
    return out


@torch.no_grad()
def compute_score(model: SyncNet, lips: torch.Tensor, mel: torch.Tensor, device: str) -> float:
    model.eval()
    lips = lips.unsqueeze(0).to(device)  # (1,T,3,96,96)
    mel = mel.unsqueeze(0).to(device)    # (1,80,mel_len)
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
    if len(all_samples) <= 1:
        raise ValueError("Need at least 2 valid samples for negative sampling.")
    while True:
        cand = random.choice(all_samples)
        if cand != exclude_sd:
            return cand


# ---------------------------
# Metrics helpers
# ---------------------------
def build_pair_dataset(df: pd.DataFrame):
    """From per-sample pos & neg_scores(csv string), build pair-level arrays."""
    scores, labels = [], []
    for _, row in df.iterrows():
        pos = float(row["pos"])
        scores.append(pos); labels.append(1)
        neg_str = str(row.get("neg_scores", "") or "")
        if neg_str.strip():
            for x in neg_str.split(","):
                x = x.strip()
                if x:
                    scores.append(float(x))
                    labels.append(0)
    return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int64)


def best_threshold(scores: np.ndarray, labels: np.ndarray):
    cand = np.unique(scores)
    best_tau, best_acc = None, -1.0
    for tau in cand:
        pred = (scores > tau).astype(np.int64)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_tau = float(tau)
    return best_tau, best_acc


def pair_accuracy(scores: np.ndarray, labels: np.ndarray, tau: float) -> float:
    pred = (scores > tau).astype(np.int64)
    return float((pred == labels).mean())


def ranking_metrics(df: pd.DataFrame, ks=(1, 5)):
    pos_scores = df["pos"].astype(float).to_numpy()
    # parse neg lists
    neg_lists = []
    for s in df["neg_scores"].fillna("").astype(str).to_list():
        if s.strip() == "":
            neg_lists.append([])
        else:
            neg_lists.append([float(x) for x in s.split(",") if x.strip()])

    max_m = max((len(x) for x in neg_lists), default=0)
    neg_arr = np.full((len(neg_lists), max_m), -np.inf, dtype=np.float32)
    for i, row in enumerate(neg_lists):
        neg_arr[i, :len(row)] = row

    rank = 1 + (neg_arr > pos_scores[:, None]).sum(axis=1)  # 1 is best
    out = {f"recall@{k}": float((rank <= k).mean()) for k in ks}
    out["mean_rank"] = float(rank.mean())
    return out


def offset_identification_accuracy(model: SyncNet, sample_dirs, device: str, num_frames: int, mel_len: int, offsets):
    """
    For each sample:
      - load lips
      - load full mel
      - for each offset, shift full mel, crop/pad to mel_len, score
      - predict offset = argmax score
      - success if predicted == 0
    """
    offsets = list(offsets)
    ok, hit0 = 0, 0
    per_sample = []  # optional debug

    for sd in sample_dirs:
        try:
            if not is_valid_sample(sd):
                continue
            lips = load_lips(os.path.join(sd, "lips"), num_frames=num_frames)
            mel_full = load_mel_full(os.path.join(sd, "mel.npy"))  # (80, T)

            scores = []
            for off in offsets:
                mel_shifted = shift_mel_np(mel_full, off)
                mel_clip = crop_pad_mel_np(mel_shifted, mel_len=mel_len)
                mel_t = torch.from_numpy(mel_clip).float()
                scores.append(compute_score(model, lips, mel_t, device))

            pred_off = offsets[int(np.argmax(scores))]
            ok += 1
            hit0 += int(pred_off == 0)

            per_sample.append({
                "sample_dir": sd,
                "pred_offset": pred_off,
                "scores": scores
            })

        except Exception:
            continue

    acc = (hit0 / ok) if ok > 0 else 0.0
    return acc, ok, per_sample

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

    # data params
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--mel_len", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--neg_k", type=int, default=5, help="number of negative mels per sample (mean over K)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for negative sampling")

    # metrics options
    parser.add_argument("--do_pair", action="store_true", help="compute Pair classification accuracy")
    parser.add_argument("--do_rank", action="store_true", help="compute Ranking metrics (Recall@K)")
    parser.add_argument("--do_offset", action="store_true", help="compute Offset identification accuracy (shift sweep)")

    parser.add_argument("--tau", type=float, default=None, help="pair threshold; if None, tune on current set")
    parser.add_argument("--rank_ks", default="1,5", help="comma-separated ks for Recall@K")
    parser.add_argument("--offsets", default="-5,-4,-3,-2,-1,0,1,2,3,4,5", help="comma-separated integer offsets")

    parser.add_argument("--save_csv", default="logs/inference_scores_with_neg.csv", help="save per-sample pos/neg scores")
    parser.add_argument("--save_offset_csv", default="", help="optional: save per-sample sweep scores to csv")
    args = parser.parse_args()

    # -------------------------
    # Apply YAML config
    # -------------------------
    cfg = _load_yaml(args.config) if args.config else {}

    # dataset.root -> args.root
    if "dataset" in cfg and "root" in cfg["dataset"]:
        args.root = cfg["dataset"]["root"]

    # paths.ckpt -> args.ckpt (있으면)
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

    # eval
    if "eval" in cfg:
        e = cfg["eval"]
        for k in ["neg_k", "do_pair", "do_rank", "do_offset", "save_csv", "save_offset_csv"]:
            if k in e:
                setattr(args, k, e[k])

        # YAML에서 list로 받는 걸 기존 코드(split(","))와 호환되게 문자열로 변환
        if "rank_ks" in e:
            args.rank_ks = ",".join(str(x) for x in e["rank_ks"])
        if "offsets" in e:
            args.offsets = ",".join(str(x) for x in e["offsets"])


    # default: all metrics
    if not (args.do_pair or args.do_rank or args.do_offset):
        args.do_pair = args.do_rank = args.do_offset = True

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

    # Pool for negative sampling (valid only)
    valid_pool = [
        sd for sd in sorted([
            os.path.join(args.root, d) for d in os.listdir(args.root)
            if os.path.isdir(os.path.join(args.root, d))
        ]) if is_valid_sample(sd)
    ]

    # ---------------------------------------
    # Step 1) Per-sample inference (pos + K neg)
    #   -> needed for (A) and (C)
    # ---------------------------------------
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
            ok += 1

        except Exception as e:
            print(f"⚠ Skip: {sd} | Reason: {e}")
            skip += 1

    print(f"\nPer-sample scoring done. OK={ok} SKIP={skip}")

    df = pd.DataFrame(results)
    if df.empty:
        print("No valid samples processed. Exiting.")
        return

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    df = df.sort_values("margin", ascending=False)
    df.to_csv(args.save_csv, index=False)
    print(f"Saved per-sample scores: {args.save_csv}")

    # ---------------------------------------
    # Step 2) (A) Pair classification accuracy
    # ---------------------------------------
    if args.do_pair:
        pair_scores, pair_labels = build_pair_dataset(df)
        if args.tau is None:
            tau, tuned_acc = best_threshold(pair_scores, pair_labels)
        else:
            tau, tuned_acc = args.tau, None
        acc = pair_accuracy(pair_scores, pair_labels, tau)
        msg = f"[A] Pair-Cls Accuracy: {acc:.4f} (tau={tau:.6f})"
        if tuned_acc is not None:
            msg += f"  [tuned on same set: {tuned_acc:.4f}]"
        print(msg)

    # ---------------------------------------
    # Step 3) (C) Ranking metrics
    # ---------------------------------------
    if args.do_rank:
        ks = [int(x) for x in args.rank_ks.split(",")]
        rm = ranking_metrics(df, ks=ks)
        print("[C] Ranking:", ", ".join([f"{k}={v:.4f}" for k, v in rm.items()]))

    # ---------------------------------------
    # Step 4) (B) Offset identification accuracy
    # ---------------------------------------
    if args.do_offset:
        offsets = [int(x) for x in args.offsets.split(",")]
        acc_off, n_used, per_sample = offset_identification_accuracy(
            model=model,
            sample_dirs=samples,
            device=device,
            num_frames=args.num_frames,
            mel_len=args.mel_len,
            offsets=offsets
        )
        print(f"[B] Offset-Id Accuracy: {acc_off:.4f} (N={n_used}, gt_offset=0)")

        if args.save_offset_csv:
            rows = []
            for r in per_sample:
                row = {"sample_dir": r["sample_dir"], "pred_offset": r["pred_offset"]}
                for i, off in enumerate(offsets):
                    row[f"s_{off}"] = float(r["scores"][i])
                rows.append(row)
            odf = pd.DataFrame(rows)
            os.makedirs(os.path.dirname(args.save_offset_csv), exist_ok=True)
            odf.to_csv(args.save_offset_csv, index=False)
            print(f"Saved sweep scores: {args.save_offset_csv}")


if __name__ == "__main__":
    main()
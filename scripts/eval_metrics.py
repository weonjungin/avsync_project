import os
import glob
import argparse
import random
import re
from pathlib import Path
import yaml

import torch
import numpy as np
from PIL import Image
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from avsync_project.models.syncnet_model import SyncNet


# ---------------------------------------
# Utils: YAML
# ---------------------------------------
def _load_yaml(path: str):
    p = Path(path).expanduser()
    if not p.is_absolute():
        proj_root = Path(__file__).resolve().parents[1]
        p = (proj_root / p).resolve()
    with open(p, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------
# Speaker split (GRID: s1_processed ~ s34_processed)
# ---------------------------------------
def extract_speaker_id(sample_dir: str) -> str:
    parts = sample_dir.replace("\\", "/").split("/")
    for p in parts:
        m = re.match(r"^(s\d+)_processed$", p)  # s10_processed
        if m:
            return m.group(1)  # s10
        m = re.match(r"^(s\d+)$", p)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot parse speaker id from: {sample_dir}")


def split_by_speaker_count(samples, seed=42, n_train=28, n_val=3):
    spk_to_samples = {}
    for s in samples:
        spk = extract_speaker_id(s)
        spk_to_samples.setdefault(spk, []).append(s)

    speakers = sorted(spk_to_samples.keys())
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


# ---------------------------------------
# Collect samples recursively (GRID 구조 대응)
# sample_dir 내부에 lips/ 와 mel.npy가 있어야 유효
# ---------------------------------------
def collect_sample_dirs(root: str):
    root = os.path.abspath(root)
    mel_paths = glob.glob(os.path.join(root, "**", "mel.npy"), recursive=True)

    sample_dirs = []
    for mp in mel_paths:
        sd = os.path.dirname(mp)
        lips_dir = os.path.join(sd, "lips")
        if os.path.isdir(lips_dir) and len(glob.glob(os.path.join(lips_dir, "*.png"))) > 0:
            sample_dirs.append(sd)

    sample_dirs = sorted(set(sample_dirs))
    return sample_dirs


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


# ---------------------------------------
# Load lips frames (match training: grayscale 1-channel)
# returns (T, 1, 96, 96)
# ---------------------------------------
def load_lips(lips_dir: str, num_frames: int = 5, size: int = 96) -> torch.Tensor:
    frame_paths = sorted(glob.glob(os.path.join(lips_dir, "*.png")))
    if len(frame_paths) == 0:
        raise ValueError(f"No lips frames found in {lips_dir}")

    if len(frame_paths) < num_frames:
        frame_paths = frame_paths * (num_frames // len(frame_paths) + 1)
    frame_paths = frame_paths[:num_frames]

    imgs = []
    for f in frame_paths:
        img = Image.open(f).convert("L")        # grayscale
        img = img.resize((size, size))
        arr = np.array(img).astype(np.float32) / 255.0  # (H,W)
        t = torch.from_numpy(arr).unsqueeze(0)          # (1,H,W)
        imgs.append(t)

    return torch.stack(imgs, dim=0)  # (T, 1, 96, 96)


# ---------------------------------------
# mel helpers
# ---------------------------------------
def load_mel_full_np(mel_path: str) -> np.ndarray:
    mel = np.load(mel_path)
    if mel.ndim != 2 or mel.shape[0] != 80:
        raise ValueError(f"Bad mel shape: {tuple(mel.shape)} in {mel_path}")
    return mel.astype(np.float32)  # (80, T)


def crop_pad_mel_np(mel: np.ndarray, mel_len: int = 16) -> np.ndarray:
    """
    Deterministic: take FIRST mel_len frames (after any shift).
    Stable + simple.
    """
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
    shift is in "mel frames index"
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


def mel_per_vframe(fps: float, sr: int, hop: int) -> float:
    return (sr / float(hop)) / float(fps)


def load_meta_defaults(sample_dir: str):
    """
    meta.json이 있으면 fps/sr/hop 읽고,
    없으면 (fps=25, sr=16000, hop=160) 가정.
    """
    meta_path = os.path.join(sample_dir, "meta.json")
    fps, sr, hop = 25.0, 16000, 160

    if os.path.isfile(meta_path):
        try:
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
            fps = float(meta.get("fps", fps))
            sr = int(meta.get("sr", sr))
            hop = int(meta.get("hop_length", meta.get("hop", hop)))
        except Exception:
            pass
    return fps, sr, hop


def mel_clip_at_offset_frames(sample_dir: str, mel_full: np.ndarray, mel_len: int, off_frames: int) -> np.ndarray:
    """
    off_frames: video frame offset (e.g., -5..+5)
    convert to mel index shift using fps/sr/hop then shift mel and take first mel_len frames.
    """
    fps, sr, hop = load_meta_defaults(sample_dir)
    m_per_v = mel_per_vframe(fps, sr, hop)
    shift_mel = int(round(off_frames * m_per_v))  # mel-index shift
    mel_shifted = shift_mel_np(mel_full, shift_mel)
    mel_clip = crop_pad_mel_np(mel_shifted, mel_len=mel_len)
    return mel_clip


# ---------------------------------------
# Model score
# ---------------------------------------
@torch.no_grad()
def compute_score(model: SyncNet, lips: torch.Tensor, mel: torch.Tensor, device: str) -> float:
    """
    lips: (T,1,96,96)
    mel:  (80,mel_len)
    """
    model.eval()
    lips = lips.unsqueeze(0).to(device)  # (1,T,1,96,96)
    mel = mel.unsqueeze(0).to(device)    # (1,80,mel_len)
    v_emb, a_emb = model(lips, mel)
    return float(torch.nn.functional.cosine_similarity(v_emb, a_emb, dim=1).item())


# ---------------------------------------
# AUC (no sklearn): Mann–Whitney U / rank statistic
# AUC = P(score_pos > score_neg) + 0.5 * P(equal)
# ---------------------------------------
def auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    if pos_scores.size == 0 or neg_scores.size == 0:
        return float("nan")

    all_scores = np.concatenate([pos_scores, neg_scores], axis=0)
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)], axis=0)

    # ranks with tie handling: average rank
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1, dtype=np.float64)

    # tie correction: average ranks for equal values
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j

    # sum ranks for positives
    R_pos = ranks[labels == 1].sum()
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()

    # U statistic for positives
    U_pos = R_pos - n_pos * (n_pos + 1) / 2.0
    auc = U_pos / (n_pos * n_neg)
    return float(auc)


def safe_mean(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def safe_std(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return float(arr.std(ddof=1)) if arr.size >= 2 else float("nan")


def main():
    parser = argparse.ArgumentParser()

    # config / paths
    parser.add_argument("--config", type=str, default="configs/exp.yaml")
    parser.add_argument("--root", default=None, help="GRID processed root (contains s*_processed)")
    parser.add_argument("--ckpt", default=None, help="checkpoint path")

    # IMPORTANT: evaluate TEST only (speaker-disjoint split 28/3/3)
    parser.add_argument("--train_spk", type=int, default=28)
    parser.add_argument("--val_spk", type=int, default=3)

    # data params
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--mel_len", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all test samples")
    parser.add_argument("--seed", type=int, default=42)

    # evaluation controls
    parser.add_argument("--offsets", default="-5,-4,-3,-2,-1,0,1,2,3,4,5",
                        help="comma-separated FRAME offsets for offset curve & offset-id acc")
    parser.add_argument("--save_offset_csv", default="logs/offset_sweep_test.csv",
                        help="save per-sample offset sweep scores (for plotting)")
    parser.add_argument("--save_summary_csv", default="logs/eval_summary_test.csv",
                        help="save aggregated summary metrics")

    args = parser.parse_args()

    # -------------------------
    # Apply YAML config
    # -------------------------
    cfg = _load_yaml(args.config) if args.config else {}

    if args.root is None:
        args.root = cfg.get("dataset", {}).get("root", "data/grid_processed")

    if args.ckpt is None:
        args.ckpt = cfg.get("paths", {}).get("ckpt", "logs/checkpoints/best_syncnet_ckpt.pth")

    c = cfg.get("common", {})
    args.seed = int(c.get("seed", args.seed))
    args.num_frames = int(c.get("num_frames", args.num_frames))
    args.mel_len = int(c.get("mel_len", args.mel_len))

    # offsets can be overridden by cfg.eval.offsets (frame offsets)
    e = cfg.get("eval", {})
    if "offsets" in e:
        args.offsets = ",".join(str(x) for x in e["offsets"])

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # -------------------------
    # Load model
    # -------------------------
    model = SyncNet(embed_dim=int(cfg.get("model", {}).get("embed_dim", 256))).to(device)

    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.ckpt}")

    # -------------------------
    # Collect all valid samples (recursive)
    # -------------------------
    all_samples = [sd for sd in collect_sample_dirs(args.root) if is_valid_sample(sd)]
    if len(all_samples) == 0:
        print(f"No valid samples found under: {args.root}")
        return

    # speaker-disjoint split (28/3/3), then evaluate TEST ONLY
    train_s, val_s, test_s, (train_spk, val_spk, test_spk) = split_by_speaker_count(
        all_samples, seed=args.seed, n_train=args.train_spk, n_val=args.val_spk
    )
    samples = test_s

    print(f"[Split] speakers: train={len(train_spk)} val={len(val_spk)} test={len(test_spk)}")
    print(f"[EvalSplit=TEST] N={len(samples)}")
    print(f"[Test speakers]: {test_spk}")

    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]
        print(f"[Eval] max_samples applied -> N={len(samples)}")

    offsets_f = [int(x) for x in args.offsets.split(",") if x.strip()]
    if 0 not in offsets_f:
        raise ValueError("offsets must include 0 (for gt_offset=0).")

    offsets_f_sorted = offsets_f[:]  # preserve given order for columns
    zero_idx = offsets_f_sorted.index(0)

    # ---------------------------------------
    # Evaluate per-sample sweep on TEST
    #   -> enables:
    #   1) Offset Curve (mean curve)
    #   2) Offset Margin (per-sample + mean/std)
    #   3) Offset Identification Accuracy
    #   4) AUC (pos=score@0, neg=scores@others)
    # ---------------------------------------
    rows = []
    ok, skip = 0, 0

    all_pos = []
    all_neg = []
    margins = []
    hit0 = 0  # for offset-id acc

    for sd in samples:
        try:
            lips = load_lips(os.path.join(sd, "lips"), num_frames=args.num_frames)
            mel_full = load_mel_full_np(os.path.join(sd, "mel.npy"))

            # score sweep across offsets
            sweep_scores = []
            for off_f in offsets_f_sorted:
                mel_np = mel_clip_at_offset_frames(sd, mel_full, args.mel_len, off_frames=off_f)
                mel_t = torch.from_numpy(mel_np).float()
                sweep_scores.append(compute_score(model, lips, mel_t, device))

            sweep_scores = np.asarray(sweep_scores, dtype=np.float64)

            # pos/neg for margin + AUC
            pos = float(sweep_scores[zero_idx])
            neg_scores = np.delete(sweep_scores, zero_idx)
            neg_mean = float(neg_scores.mean()) if neg_scores.size else float("nan")
            margin = float(pos - neg_mean)

            # offset-id: argmax offset
            pred_idx = int(np.argmax(sweep_scores))
            pred_off = int(offsets_f_sorted[pred_idx])
            hit0 += int(pred_off == 0)

            # accumulate for global metrics
            all_pos.append(pos)
            all_neg.extend(list(neg_scores))
            margins.append(margin)

            # per-sample row for csv
            row = {
                "sample_dir": sd,
                "pred_offset": pred_off,
                "pos_s0": pos,
                "neg_mean": neg_mean,
                "margin": margin,
            }
            for i, off_f in enumerate(offsets_f_sorted):
                row[f"s_{off_f}"] = float(sweep_scores[i])
            rows.append(row)

            ok += 1
        except Exception as ex:
            print(f"⚠ Skip: {sd} | Reason: {ex}")
            skip += 1

    print(f"\nOffset sweep done on TEST. OK={ok} SKIP={skip}")

    if ok == 0:
        print("No valid samples processed. Exiting.")
        return

    df = pd.DataFrame(rows)

    # ---------------------------------------
    # 1) Offset Curve (mean +/- std over TEST)
    # ---------------------------------------
    curve_mean = []
    curve_std = []
    for off_f in offsets_f_sorted:
        vals = df[f"s_{off_f}"].to_numpy(dtype=np.float64)
        curve_mean.append(safe_mean(vals))
        curve_std.append(safe_std(vals))

    # ---------------------------------------
    # 2) Offset Margin summary
    # ---------------------------------------
    margins_np = np.asarray(margins, dtype=np.float64)
    margin_mean = safe_mean(margins_np)
    margin_std = safe_std(margins_np)

    # ---------------------------------------
    # 3) Offset Identification Accuracy
    # ---------------------------------------
    offset_id_acc = hit0 / float(ok)

    # ---------------------------------------
    # 4) AUC (offset-based ROC)
    #   pos scores: score@0 per sample (N=ok)
    #   neg scores: all non-zero offset scores pooled (N=ok*(len(offsets)-1))
    # ---------------------------------------
    auc = auc_from_scores(np.asarray(all_pos, dtype=np.float64), np.asarray(all_neg, dtype=np.float64))

    # ---------------------------------------
    # Save CSVs
    # ---------------------------------------
    os.makedirs(os.path.dirname(args.save_offset_csv), exist_ok=True)
    df.to_csv(args.save_offset_csv, index=False)
    print(f"Saved per-sample offset sweep: {args.save_offset_csv}")

    summary = {
        "split": "test",
        "n_samples": ok,
        "offsets_frames": [int(x) for x in offsets_f_sorted],
        "offset_curve_mean": curve_mean,
        "offset_curve_std": curve_std,
        "margin_mean": margin_mean,
        "margin_std": margin_std,
        "offset_id_acc": float(offset_id_acc),
        "auc": float(auc),
        "ckpt": args.ckpt,
        "root": args.root,
        "num_frames": int(args.num_frames),
        "mel_len": int(args.mel_len),
        "seed": int(args.seed),
        "train_speakers": train_spk,
        "val_speakers": val_spk,
        "test_speakers": test_spk,
    }

    # summary csv (one row, human-readable columns)
    out_row = {
        "split": "test",
        "N": ok,
        "margin_mean": margin_mean,
        "margin_std": margin_std,
        "offset_id_acc": float(offset_id_acc),
        "auc": float(auc),
        "offsets": args.offsets,
        "ckpt": args.ckpt,
    }
    for i, off_f in enumerate(offsets_f_sorted):
        out_row[f"curve_mean_s{off_f}"] = curve_mean[i]
        out_row[f"curve_std_s{off_f}"] = curve_std[i]

    os.makedirs(os.path.dirname(args.save_summary_csv), exist_ok=True)
    pd.DataFrame([out_row]).to_csv(args.save_summary_csv, index=False)
    print(f"Saved summary: {args.save_summary_csv}")

    # ---------------------------------------
    # Print summary (console)
    # ---------------------------------------
    print("\n=== TEST EVAL SUMMARY ===")
    print(f"- Split: TEST (speaker-disjoint 28/3/3)")
    print(f"- N: {ok}")
    print(f"- Offsets (frames): {offsets_f_sorted}")

    print("\n[1] Offset Curve (mean ± std)")
    for off_f, m, s in zip(offsets_f_sorted, curve_mean, curve_std):
        if np.isnan(s):
            print(f"  off {off_f:+d}: mean={m:.4f}")
        else:
            print(f"  off {off_f:+d}: mean={m:.4f} ± {s:.4f}")

    print("\n[2] Offset Margin")
    if np.isnan(margin_std):
        print(f"  mean={margin_mean:.4f}")
    else:
        print(f"  mean={margin_mean:.4f} ± {margin_std:.4f}")

    print("\n[3] Offset Identification Accuracy")
    print(f"  acc={offset_id_acc:.4f}  (baseline≈1/{len(offsets_f_sorted)}={1.0/len(offsets_f_sorted):.4f})")

    print("\n[4] AUC (offset-based ROC)")
    print(f"  AUC={auc:.4f}")


if __name__ == "__main__":
    main()

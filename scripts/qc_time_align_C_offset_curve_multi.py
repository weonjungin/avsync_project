import os, json, argparse, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def zpad(i, w=5):
    return str(i).zfill(w)

def cosine(a, b, eps=1e-8):
    a = a.astype(np.float32); b = b.astype(np.float32)
    a = a - a.mean(); b = b - b.mean()
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def load_roi_sequence(roi_dir, t0, T):
    frames = []
    for fi in range(t0, t0 + T):
        fp = os.path.join(roi_dir, f"{zpad(fi)}.png")
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        img = np.array(Image.open(fp).convert("L"), dtype=np.float32) / 255.0
        frames.append(img)
    return np.stack(frames, axis=0)  # (T,H,W)

def lip_motion_sequence(frames):
    T = frames.shape[0]
    mot = np.zeros((T,), dtype=np.float32)
    for i in range(1, T):
        mot[i] = np.mean(np.abs(frames[i] - frames[i-1]))
    return mot

def audio_energy_per_video_frame(mel, fps, sr, hop, t0, T, offset_frames=0):
    mel_fps = sr / hop
    n_mels, Tm = mel.shape
    seq = np.zeros((T,), dtype=np.float32)
    shift_sec = offset_frames / fps

    for i, fi in enumerate(range(t0, t0 + T)):
        s = (fi / fps) + shift_sec
        e = ((fi + 1) / fps) + shift_sec
        m0 = int(round(s * mel_fps))
        m1 = int(round(e * mel_fps))
        m0 = max(0, min(m0, Tm))
        m1 = max(0, min(m1, Tm))
        if m1 <= m0:
            m1 = min(Tm, m0 + 1)
        seq[i] = float(np.mean(mel[:, m0:m1]))
    return seq

def list_samples(processed_root, speaker=None):
    # processed_root: /media/HDD/jiweon/GRID/processed_syncnet
    if speaker is None:
        speakers = [d for d in os.listdir(processed_root) if d.startswith("s")]
    else:
        speakers = [speaker]
    out = []
    for sp in speakers:
        sp_dir = os.path.join(processed_root, sp)
        if not os.path.isdir(sp_dir):
            continue
        for clip in os.listdir(sp_dir):
            sd = os.path.join(sp_dir, clip)
            if os.path.isfile(os.path.join(sd, "meta.json")) and \
               os.path.isfile(os.path.join(sd, "mel.npy")) and \
               os.path.isdir(os.path.join(sd, "roi")):
                out.append(sd)
    return sorted(out)

def choose_t0(meta, T, mode="mid"):
    n_frames = int(meta.get("num_frames_saved", meta.get("frame_count_reported", 0)))
    if n_frames <= T:
        return 0
    if mode == "start":
        return 0
    if mode == "mid":
        return max(0, (n_frames - T)//2)
    if mode == "random":
        return random.randint(0, n_frames - T)
    raise ValueError("mode must be start|mid|random")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_root", type=str, required=True,
                    help="예: /media/HDD/jiweon/GRID/processed_syncnet")
    ap.add_argument("--speaker", type=str, default=None, help="예: s9 (없으면 전체)")
    ap.add_argument("--N", type=int, default=20, help="샘플 개수")
    ap.add_argument("--T", type=int, default=20, help="video window length (frames)")
    ap.add_argument("--t0_mode", type=str, default="mid", help="start|mid|random")
    ap.add_argument("--max_off", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="qc_out_multi")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    all_samples = list_samples(args.processed_root, speaker=args.speaker)
    if len(all_samples) == 0:
        raise RuntimeError("No valid samples found. Check processed_root path.")
    chosen = random.sample(all_samples, k=min(args.N, len(all_samples)))

    offsets = list(range(-args.max_off, args.max_off + 1))
    S = len(chosen)
    curves = np.zeros((S, len(offsets)), dtype=np.float32)

    per_sample_rows = []

    for si, sample_dir in enumerate(chosen):
        meta = json.load(open(os.path.join(sample_dir, "meta.json")))
        fps = float(meta["fps"])
        sr  = int(meta["sr"])
        hop = int(meta["hop_length"])
        roi_dir = os.path.join(sample_dir, meta.get("roi_dir", "roi"))

        mel = np.load(os.path.join(sample_dir, "mel.npy"))
        if mel.ndim != 2:
            continue

        t0 = choose_t0(meta, args.T, mode=args.t0_mode)

        roi_frames = load_roi_sequence(roi_dir, t0, args.T)
        lip_seq = lip_motion_sequence(roi_frames)

        scores = []
        for oi, off in enumerate(offsets):
            aud_seq = audio_energy_per_video_frame(mel, fps, sr, hop, t0, args.T, offset_frames=off)
            scores.append(cosine(lip_seq, aud_seq))
            curves[si, oi] = scores[-1]

        best_i = int(np.argmax(scores))
        best_off = offsets[best_i]
        best_score = scores[best_i]
        per_sample_rows.append((sample_dir, t0, best_off, best_score))

        # per-sample plot 저장(원하면)
        fig_path = os.path.join(args.out_dir, f"offset_curve_{si:03d}.png")
        plt.figure()
        plt.plot(offsets, scores, marker="o")
        plt.axvline(0, linestyle="--")
        plt.title(f"sample {si} best={best_off} (score={best_score:.3f})")
        plt.xlabel("audio offset (video frames)")
        plt.ylabel("cosine(lip_motion, mel_energy)")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

    # 평균/표준편차
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)

    # summary plot
    sum_png = os.path.join(args.out_dir, "offset_curve_mean.png")
    plt.figure()
    plt.plot(offsets, mean_curve, marker="o")
    plt.fill_between(offsets, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    plt.axvline(0, linestyle="--")
    plt.title(f"Mean offset curve over {S} samples (T={args.T}, t0_mode={args.t0_mode})")
    plt.xlabel("audio offset (video frames)")
    plt.ylabel("cosine(lip_motion, mel_energy)")
    plt.tight_layout()
    plt.savefig(sum_png, dpi=200)
    plt.close()

    # csv summary
    csv_path = os.path.join(args.out_dir, "per_sample_best.csv")
    with open(csv_path, "w") as f:
        f.write("sample_dir,t0,best_offset,best_score\n")
        for r in per_sample_rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f}\n")

    print("==== C-multi done ====")
    print(f"processed_root: {args.processed_root}")
    print(f"speaker: {args.speaker} | N={len(chosen)} | T={args.T} | t0_mode={args.t0_mode} | max_off=±{args.max_off}")
    print("saved:")
    print(" -", sum_png)
    print(" -", csv_path)
    print(" - per-sample plots:", args.out_dir)

if __name__ == "__main__":
    main()

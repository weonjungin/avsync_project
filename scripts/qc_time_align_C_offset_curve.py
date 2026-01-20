# scripts/qc_time_align_C_offset_curve.py
import os, json, argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def zpad(i, w=5):
    return str(i).zfill(w)

def cosine(a, b, eps=1e-8):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def load_roi_sequence(roi_dir, t0, T):
    frames = []
    for fi in range(t0, t0 + T):
        fp = os.path.join(roi_dir, f"{zpad(fi)}.png")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing ROI: {fp}")
        img = np.array(Image.open(fp).convert("L"), dtype=np.float32) / 255.0
        frames.append(img)
    return np.stack(frames, axis=0)  # (T, H, W)

def lip_motion_sequence(frames):
    # 프레임 간 절대차의 평균 -> 길이 T (첫 프레임은 0)
    T = frames.shape[0]
    mot = np.zeros((T,), dtype=np.float32)
    for i in range(1, T):
        mot[i] = np.mean(np.abs(frames[i] - frames[i-1]))
    return mot

def audio_energy_per_video_frame(mel, fps, sr, hop, t0, T, offset_frames=0):
    """
    mel: (n_mels, Tm)
    각 비디오 프레임 구간 [t/fps, (t+1)/fps) 에 해당하는 mel 인덱스를 평균내어
    길이 T의 에너지 시퀀스를 만든다.
    offset_frames: 오디오를 (비디오 대비) 얼마나 이동시켜 볼지 (비디오 프레임 단위)
    """
    mel_fps = sr / hop
    n_mels, Tm = mel.shape
    seq = np.zeros((T,), dtype=np.float32)

    shift_sec = offset_frames / fps

    for i, fi in enumerate(range(t0, t0 + T)):
        s = (fi / fps) + shift_sec
        e = ((fi + 1) / fps) + shift_sec

        m0 = int(round(s * mel_fps))
        m1 = int(round(e * mel_fps))

        # 안전장치
        m0 = max(0, min(m0, Tm))
        m1 = max(0, min(m1, Tm))
        if m1 <= m0:
            # 너무 짧으면 1프레임이라도
            m1 = min(Tm, m0 + 1)

        patch = mel[:, m0:m1]  # (n_mels, span)
        seq[i] = float(np.mean(patch))  # 전체 mel 평균 에너지
    return seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_dir", type=str, required=True)
    ap.add_argument("--t0", type=int, default=15)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--max_off", type=int, default=10, help="offset 범위(비디오 프레임 단위), 예: 10이면 -10..+10")
    ap.add_argument("--out_dir", type=str, default="qc_out")
    args = ap.parse_args()

    sample_dir = args.sample_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(sample_dir, "meta.json")))
    fps = float(meta["fps"])
    sr  = int(meta["sr"])
    hop = int(meta["hop_length"])
    roi_dir = os.path.join(sample_dir, meta.get("roi_dir", "roi"))

    mel = np.load(os.path.join(sample_dir, "mel.npy"))
    assert mel.ndim == 2 and mel.shape[0] == 80, f"Unexpected mel shape: {mel.shape}"

    # lip motion sequence
    roi_frames = load_roi_sequence(roi_dir, args.t0, args.T)     # (T,H,W)
    lip_seq = lip_motion_sequence(roi_frames)                    # (T,)

    offsets = list(range(-args.max_off, args.max_off + 1))
    scores = []

    for off in offsets:
        aud_seq = audio_energy_per_video_frame(mel, fps, sr, hop, args.t0, args.T, offset_frames=off)
        s = cosine(lip_seq, aud_seq)  # 또는 corr처럼 사용
        scores.append(s)

    # best offset
    best_i = int(np.argmax(scores))
    best_off = offsets[best_i]
    best_score = scores[best_i]

    # save csv
    csv_path = os.path.join(out_dir, "offset_curve.csv")
    with open(csv_path, "w") as f:
        f.write("offset_frames,score\n")
        for o, s in zip(offsets, scores):
            f.write(f"{o},{s}\n")

    # plot
    plt.figure()
    plt.plot(offsets, scores, marker="o")
    plt.axvline(0, linestyle="--")
    plt.title(f"Offset curve (lip-motion vs mel-energy) | best={best_off} (score={best_score:.4f})")
    plt.xlabel("audio offset (video frames)")
    plt.ylabel("cosine(lip_motion, mel_energy)")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "offset_curve.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("==== C) Offset curve outputs ====")
    print("sample_dir:", sample_dir)
    print(f"video window: t0={args.t0}, T={args.T} @ {fps}fps")
    print(f"max_off: ±{args.max_off} frames")
    print(f"best_offset: {best_off} frames, best_score={best_score:.4f}")
    print("saved:")
    print(" -", csv_path)
    print(" -", fig_path)

if __name__ == "__main__":
    main()

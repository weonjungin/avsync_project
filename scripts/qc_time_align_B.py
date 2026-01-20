# scripts/qc_time_align_B.py
import os, json, argparse
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import matplotlib.pyplot as plt

def zpad(i, w=5):  # 00030 같은 파일명
    return str(i).zfill(w)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_dir", type=str, required=True)
    ap.add_argument("--t0", type=int, default=30)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="qc_out")
    ap.add_argument("--gif_fps", type=int, default=10)  # 보기용
    args = ap.parse_args()

    sample_dir = args.sample_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(sample_dir, "meta.json")))
    fps = float(meta["fps"])
    sr  = int(meta["sr"])
    hop = int(meta["hop_length"])
    mel_fps = sr / hop

    mel = np.load(os.path.join(sample_dir, "mel.npy"))  # (80, Tm)
    assert mel.ndim == 2, f"Unexpected mel shape: {mel.shape}"
    n_mels, Tm = mel.shape

    roi_dir = os.path.join(sample_dir, meta.get("roi_dir", "roi"))

    # ---- compute matched mel range for the video window [t0, t0+T) ----
    t0 = args.t0
    T  = args.T
    v_sec0 = t0 / fps
    v_sec1 = (t0 + T) / fps

    m0 = int(round(v_sec0 * mel_fps))
    m1 = int(round(v_sec1 * mel_fps))
    m0 = max(0, min(m0, Tm))
    m1 = max(0, min(m1, Tm))

    # ---- 1) Make lip GIF ----
    frames = []
    for fi in range(t0, t0 + T):
        fp = os.path.join(roi_dir, f"{zpad(fi)}.png")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"ROI frame missing: {fp}")
        img = Image.open(fp).convert("L")  # grayscale
        frames.append(np.array(img))

    gif_path = os.path.join(out_dir, "lip_window.gif")
    imageio.mimsave(gif_path, frames, fps=args.gif_fps)

    # ---- 2) Make mel image with highlight ----
    mel_img_path = os.path.join(out_dir, "mel_with_window.png")
    plt.figure()
    plt.imshow(mel, aspect="auto", origin="lower")
    plt.axvspan(m0, m1, alpha=0.25)  # highlight same-time window
    plt.title(
        f"Mel (n_mels={n_mels}) | video [{t0},{t0+T}) @ {fps}fps -> mel [{m0},{m1}) @ {mel_fps:.1f}fps"
    )
    plt.xlabel("mel frame index")
    plt.ylabel("mel bin")
    plt.tight_layout()
    plt.savefig(mel_img_path, dpi=200)
    plt.close()

    # ---- 3) (Optional) Summary image: first ROI + mel highlight in one figure ----
    summary_path = os.path.join(out_dir, "qc_summary.png")
    plt.figure(figsize=(10, 4))
    # left: first ROI frame
    plt.subplot(1, 2, 1)
    plt.imshow(frames[0], cmap="gray")
    plt.title(f"ROI frame {t0} (window {T} frames)")
    plt.axis("off")

    # right: mel
    plt.subplot(1, 2, 2)
    plt.imshow(mel, aspect="auto", origin="lower")
    plt.axvspan(m0, m1, alpha=0.25)
    plt.title("Mel + same-time window")
    plt.xlabel("mel frame index")
    plt.ylabel("mel bin")

    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()

    # --- 4) Combined GIF: ROI + moving mel cursor ---
    combo_frames = []

    for i, fi in enumerate(range(t0, t0 + T)):
        cur_sec = fi / fps
        cur_mel = int(round(cur_sec * mel_fps))
        cur_mel = max(0, min(cur_mel, Tm - 1))

        fig = plt.figure(figsize=(8, 3))

        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(frames[i], cmap="gray")
        ax1.set_title(f"ROI frame {fi}")
        ax1.axis("off")

        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(mel, aspect="auto", origin="lower")
        ax2.axvspan(m0, m1, alpha=0.15)          # window도 같이 표시(선택)
        ax2.axvline(cur_mel, linewidth=2)        # 색 지정 안 해도 됨
        ax2.set_title(f"Mel @ t={cur_sec:.2f}s (mel {cur_mel})")
        ax2.set_xlabel("mel frame index")
        ax2.set_ylabel("mel bin")

        plt.tight_layout()

        # ---- Matplotlib canvas -> numpy (robust across versions) ----
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())   # (H, W, 4)
        img = buf[:, :, :3].copy()                   # (H, W, 3) RGB
        combo_frames.append(img)

        plt.close(fig)

    combo_gif_path = os.path.join(out_dir, "lip_mel_sync.gif")
    imageio.mimsave(combo_gif_path, combo_frames, fps=args.gif_fps)
    print(" -", combo_gif_path)


    # ---- Print log (for 교수님 보고용) ----
    print("==== B) QC Outputs ====")
    print("sample_dir:", sample_dir)
    print(f"video: frame [{t0},{t0+T}) -> time [{v_sec0:.3f},{v_sec1:.3f}) sec (dur={(v_sec1-v_sec0):.3f}s)")
    print(f"mel  : idx   [{m0},{m1}) -> time [{m0/mel_fps:.3f},{m1/mel_fps:.3f}) sec (dur={((m1-m0)/mel_fps):.3f}s)")
    print("saved:")
    print(" -", gif_path)
    print(" -", mel_img_path)
    print(" -", summary_path)
    print(" -", combo_gif_path)

if __name__ == "__main__":
    main()

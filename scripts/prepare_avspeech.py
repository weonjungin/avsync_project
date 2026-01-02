import os
import argparse
import subprocess
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from insightface.app import FaceAnalysis
import librosa
import shutil
from pathlib import Path
import yaml


# ---------------------------------------
# Run command
# ---------------------------------------
def run_cmd(cmd):
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)


# ---------------------------------------
# Download YouTube + clip extraction
# ---------------------------------------
def download_and_clip(youtube_id, start, end, out_dir):
    raw_path = os.path.join(out_dir, "raw.mp4")
    clip_path = os.path.join(out_dir, "clip.mp4")
    wav_path = os.path.join(out_dir, "audio.wav")

    url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Step 1: Download mp4 (raw)
    run_cmd(
        f'yt-dlp -f "bv*+ba/b" --remux-video mp4 "{url}" '
        f'-o "{raw_path}" --quiet'
    )

    # Step 2: Extract video only (for ROI extraction) - start~end clip
    run_cmd(
        f'ffmpeg -y -i "{raw_path}" -ss {start} -to {end} '
        f'-c:v libx264 -preset fast -pix_fmt yuv420p '
        f'-an "{clip_path}" -loglevel quiet'
    )

    # Step 3: Extract audio to wav (torchaudio/mp4 불안정 회피용)
    run_cmd(
        f'ffmpeg -y -i "{raw_path}" -ss {start} -to {end} '
        f'-vn -ac 1 -ar 16000 -c:a pcm_s16le "{wav_path}" -loglevel quiet'
    )

    return raw_path, clip_path, wav_path, url


# ---------------------------------------
# Extract lips ROI using insightface (improved)
# - ratio-based margin (fix far speaker over-crop)
# - square crop (consistent scale)
# - temporal smoothing (EMA on cx,cy,side)
# ---------------------------------------
def extract_lips(
    detector,
    clip_path,
    out_dir,
    size=96,
    # margin as ratio of mouth bbox
    margin_x_ratio=0.35,
    margin_y_ratio=0.55,
    # square crop scale (after margin)
    square_scale=1.00,
    # temporal smoothing
    smooth_alpha=0.4,
    # clamp minimal side to avoid too tiny crops
    min_side_px=20
):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(clip_path)

    idx = 0
    n_frames = 0
    n_no_face = 0
    n_multi_face = 0
    n_saved = 0
    prev_gray = None
    motion_vals = []
    smooth_state = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        n_frames += 1
        faces = detector.get(frame)

        if len(faces) == 0:
            n_no_face += 1
            idx += 1
            continue

        if len(faces) >= 2:
            n_multi_face += 1

        face = faces[0]
        lm = face.landmark_3d_68
        if lm is None or lm.shape[0] != 68:
            idx += 1
            continue

        # mouth landmarks (48~67)
        mouth = lm[48:68, :2]
        x1 = float(np.min(mouth[:, 0]))
        y1 = float(np.min(mouth[:, 1]))
        x2 = float(np.max(mouth[:, 0]))
        y2 = float(np.max(mouth[:, 1]))

        # bbox size
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        # ratio-based margin (fix: far speaker gets too much margin with fixed px)
        mx = 0.20 * w
        
        top = 0.08 * h
        bot = 0.50 * h 

        x1m = x1 - mx
        y1m = y1 - top
        x2m = x2 + mx
        y2m = y2 + bot

        # square crop around center (more consistent)
        cx = (x1m + x2m) / 2.0
        cy = (y1m + y2m) / 2.0 + 0.02 * h

        side = max(x2m - x1m, y2m - y1m) * 0.90
        side = max(float(min_side_px), side)

        # temporal smoothing (EMA on cx,cy,side) to reduce jitter
        if smooth_state is None:
            cx_s, cy_s, side_s = cx, cy, side
        else:
            pcx, pcy, pside = smooth_state
            a = float(smooth_alpha)
            cx_s = a * cx + (1 - a) * pcx
            cy_s = a * cy + (1 - a) * pcy
            side_s = a * side + (1 - a) * pside

        smooth_state = (cx_s, cy_s, side_s)

        # back to bbox
        half = side_s / 2.0
        x1c = int(round(cx_s - half))
        y1c = int(round(cy_s - half))
        x2c = int(round(cx_s + half))
        y2c = int(round(cy_s + half))

        # clamp to image boundary
        H, W = frame.shape[:2]
        x1c = max(0, x1c)
        y1c = max(0, y1c)
        x2c = min(W, x2c)
        y2c = min(H, y2c)

        lip_roi = frame[y1c:y2c, x1c:x2c]
        if lip_roi.size > 0 and (x2c - x1c) > 1 and (y2c - y1c) > 1:
            lip_roi = cv2.resize(lip_roi, (size, size), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                motion_vals.append(float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))))
            prev_gray = gray

            cv2.imwrite(os.path.join(out_dir, f"frame_{idx:04d}.png"), lip_roi)
            n_saved += 1

        idx += 1

    cap.release()

    stats = {
        "n_frames": n_frames,
        "n_no_face": n_no_face,
        "n_multi_face": n_multi_face,
        "n_saved": n_saved,
        "no_face_ratio": (n_no_face / n_frames) if n_frames > 0 else 1.0,
        "multi_face_ratio": (n_multi_face / n_frames) if n_frames > 0 else 0.0,
        "motion_energy": float(np.mean(motion_vals)) if len(motion_vals) > 0 else 0.0,
        
        # 기록해두면 디버깅 편함
        "params": {
            "margin_x_ratio": margin_x_ratio,
            "margin_y_ratio": margin_y_ratio,
            "square_scale": square_scale,
            "smooth_alpha": smooth_alpha,
            "min_side_px": min_side_px,
            "out_size": size,
        }
    }
    return stats



# ---------------------------------------
# Extract mel spectrogram (from wav)
# ---------------------------------------
def extract_mel(wav_path, out_path):
    wav, sr = librosa.load(wav_path, sr=16000)

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )

    mel_db = librosa.power_to_db(mel).astype("float32")
    np.save(out_path, mel_db)


# ---------------------------------------
# Success criteria + cleanup 
# ---------------------------------------
def is_success(vid_dir, min_lip_frames=25, min_mel_frames=10):
    clip_path = os.path.join(vid_dir, "clip.mp4")
    mel_path = os.path.join(vid_dir, "mel.npy")
    lips_dir = os.path.join(vid_dir, "lips")

    if not (os.path.isfile(clip_path) and os.path.getsize(clip_path) > 0):
        return False
    if not os.path.isfile(mel_path):
        return False

    try:
        mel = np.load(mel_path)
        if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] < min_mel_frames:
            return False
    except Exception:
        return False

    if not os.path.isdir(lips_dir):
        return False
    lip_frames = [
        f for f in os.listdir(lips_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if len(lip_frames) < min_lip_frames:
        return False

    return True


def cleanup_keep_clip_only(vid_dir):
    for fname in ["raw.mp4", "audio.wav"]:
        p = os.path.join(vid_dir, fname)
        if os.path.isfile(p):
            os.remove(p)


def write_meta(vid_dir, meta: dict):
    meta_path = os.path.join(vid_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------
# Process one AVSpeech row
# ---------------------------------------
def process_one_row(row, detector, out_root, min_lip_frames=25, max_multi_face_ratio=0.2, max_no_face_ratio=0.5):
    youtube_id = str(row["youtube_id"])
    start = float(row["start"])
    end = float(row["end"])

    vid_dir = os.path.join(out_root, f"{youtube_id}_{int(start*1000)}")

    try:
        os.makedirs(out_root, exist_ok=True)
        os.makedirs(vid_dir, exist_ok=True)

        # Step 1: Download & cut (raw/clip/wav)
        raw_path, clip_path, wav_path, url = download_and_clip(youtube_id, start, end, vid_dir)

        # meta 저장
        meta = {
            "youtube_id": youtube_id,
            "url": url,
            "start": start,
            "end": end,
            "clip_path": "clip.mp4",
            "mel_path": "mel.npy",
            "lips_dir": "lips",
            "sr": 16000
        }
        write_meta(vid_dir, meta)

        # Step 2: ROI extraction + stats
        lips_dir = os.path.join(vid_dir, "lips")
        lip_stats = extract_lips(detector, clip_path, lips_dir)  
        meta["lip_stats"] = lip_stats
        write_meta(vid_dir, meta)

        # DROP rule: multi-face / no-face / low-motion
        if lip_stats.get("multi_face_ratio", 0.0) >= max_multi_face_ratio:
            raise RuntimeError(f"DROP multi_face_ratio={lip_stats['multi_face_ratio']:.3f}")

        if lip_stats.get("no_face_ratio", 0.0) >= max_no_face_ratio:
            raise RuntimeError(f"DROP no_face_ratio={lip_stats['no_face_ratio']:.3f}")

        if lip_stats.get("n_saved", 0) < 2:
            raise RuntimeError(f"DROP too_few_lip_frames n_saved={lip_stats.get('n_saved',0)}")

        if lip_stats.get("motion_energy", 0.0) < 2.5:
            raise RuntimeError(f"DROP low motion_energy={lip_stats.get('motion_energy',0.0):.2f}")


        # Step 3: mel extraction (from wav)
        mel_path = os.path.join(vid_dir, "mel.npy")
        extract_mel(wav_path, mel_path)

        # Step 4: 성공 샘플만 raw/audio 삭제
        if is_success(vid_dir, min_lip_frames=min_lip_frames):
            cleanup_keep_clip_only(vid_dir)
            print(f"✔ SUCCESS (cleaned raw/audio): {youtube_id}")
        else:
            shutil.rmtree(vid_dir, ignore_errors=True)
            print(f"⚠ DONE but NOT success criteria (kept raw/audio): {youtube_id}")
            return

    except Exception as e:
        if os.path.isdir(vid_dir):
            shutil.rmtree(vid_dir, ignore_errors=True)
        print(f"✘ FAILED {youtube_id}: {e}")


# ---------------------------------------
# Main
# ---------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min_lip_frames", type=int, default=25)

    def _load_yaml(path: str):
        p = Path(path).expanduser()

        if not p.is_absolute():
            proj_root = Path(__file__).resolve().parents[1]
            p = (proj_root / p).resolve()
        with open(p, "r") as f:
            return yaml.safe_load(f)

    args = parser.parse_args()

    # YAML config
    if args.config:
        cfg = _load_yaml(args.config)

        if args.csv is None and "dataset" in cfg and "csv" in cfg["dataset"]:
            args.csv = cfg["dataset"]["csv"]

        prep = cfg.get("prepare", {})

        if args.out is None and "out" in prep:
            args.out = prep["out"]

        if args.limit is None and "limit" in prep:
            args.limit = prep["limit"]

        if "min_lip_frames" in prep and args.min_lip_frames == 25:
            args.min_lip_frames = prep["min_lip_frames"]

    # 필수값 체크
    if args.csv is None:
        raise ValueError("csv가 필요합니다. --csv 또는 --config의 dataset.csv를 지정하세요.")
    if args.out is None:
        raise ValueError("out이 필요합니다. --out 또는 --config의 prepare.out을 지정하세요.")

    df = pd.read_csv(args.csv)

    # AVSpeech format columns
    df.columns = ["youtube_id", "start", "end", "x1", "y1"]

    if args.limit:
        df = df.iloc[: args.limit]

    os.makedirs(args.out, exist_ok=True)

    # Initialize insightface
    detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=0)

    # Process each row
    for _, row in tqdm(df.iterrows(), total=len(df)):
        process_one_row(row, detector, args.out, min_lip_frames=args.min_lip_frames)


if __name__ == "__main__":
    main()

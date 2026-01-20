import os
import argparse
import subprocess
import json
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
import librosa
import shutil
from pathlib import Path
import yaml
import glob


# ---------------------------------------
# Run command
# ---------------------------------------
def run_cmd(cmd):
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)


# ---------------------------------------
# YAML loader (project-relative 지원)
# ---------------------------------------
def load_yaml(path: str):
    p = Path(path).expanduser()
    if not p.is_absolute():
        proj_root = Path(__file__).resolve().parents[1]
        p = (proj_root / p).resolve()
    with open(p, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------
# Write meta.json
# ---------------------------------------
def write_meta(sample_dir, meta: dict):
    meta_path = os.path.join(sample_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------
# Extract audio wav (16k mono) from video
# ---------------------------------------
def extract_audio_wav(video_path, wav_path, sr=16000):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    run_cmd(
        f'ffmpeg -y -i "{video_path}" -vn -ac 1 -ar {sr} -c:a pcm_s16le "{wav_path}" -loglevel error'
    )


# ---------------------------------------
# Extract mel spectrogram (from wav)
# ---------------------------------------
def extract_mel(wav_path, out_path, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    wav, _ = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel).astype("float32")
    np.save(out_path, mel_db)
    return mel_db.shape  # (80, Tmel)


# ---------------------------------------
# Extract lips ROI using insightface
# - square crop
# - temporal smoothing (EMA)
# - grayscale 저장 + size x size
# ---------------------------------------
def extract_lips(
    detector,
    video_path,
    roi_out_dir,
    size=96,
    smooth_alpha=0.4,
    min_side_px=20,
):
    os.makedirs(roi_out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

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

        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        # ratio-based margins (tunable later)
        mx = 0.20 * w
        top = 0.08 * h
        bot = 0.50 * h

        x1m = x1 - mx
        y1m = y1 - top
        x2m = x2 + mx
        y2m = y2 + bot

        # square crop around center
        cx = (x1m + x2m) / 2.0
        cy = (y1m + y2m) / 2.0 + 0.02 * h

        side = max(x2m - x1m, y2m - y1m) * 0.90
        side = max(float(min_side_px), side)

        # temporal smoothing (EMA on cx,cy,side)
        if smooth_state is None:
            cx_s, cy_s, side_s = cx, cy, side
        else:
            pcx, pcy, pside = smooth_state
            a = float(smooth_alpha)
            cx_s = a * cx + (1 - a) * pcx
            cy_s = a * cy + (1 - a) * pcy
            side_s = a * side + (1 - a) * pside

        smooth_state = (cx_s, cy_s, side_s)

        # bbox coords
        half = side_s / 2.0
        x1c = int(round(cx_s - half))
        y1c = int(round(cy_s - half))
        x2c = int(round(cx_s + half))
        y2c = int(round(cy_s + half))

        # clamp
        H, W = frame.shape[:2]
        x1c = max(0, x1c)
        y1c = max(0, y1c)
        x2c = min(W, x2c)
        y2c = min(H, y2c)

        roi = frame[y1c:y2c, x1c:x2c]
        if roi.size > 0 and (x2c - x1c) > 1 and (y2c - y1c) > 1:
            roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                m = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
                motion_vals.append(m)
            prev_gray = gray

            # ✅ 저장은 grayscale로
            cv2.imwrite(os.path.join(roi_out_dir, f"{idx:05d}.png"), gray)
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
        "params": {
            "smooth_alpha": float(smooth_alpha),
            "min_side_px": int(min_side_px),
            "out_size": int(size),
            "gray": True
        }
    }
    return stats


# ---------------------------------------
# Success criteria
# ---------------------------------------
def is_success(sample_dir, min_roi_frames=5, min_mel_frames=10):
    mel_path = os.path.join(sample_dir, "mel.npy")
    roi_dir = os.path.join(sample_dir, "roi")

    if not os.path.isfile(mel_path):
        return False
    if not os.path.isdir(roi_dir):
        return False

    try:
        mel = np.load(mel_path)
        if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] < min_mel_frames:
            return False
    except Exception:
        return False

    frames = [f for f in os.listdir(roi_dir) if f.lower().endswith(".png")]
    if len(frames) < min_roi_frames:
        return False

    return True


# ---------------------------------------
# Process one GRID mpg
# ---------------------------------------
def process_one_mpg(
    mpg_path,
    detector,
    out_root,
    roi_size=96,
    smooth_alpha=0.4,
    min_side_px=20,
    sr=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    min_roi_frames=5,
    keep_wav=False,
):
    mpg_path = str(mpg_path)

    parent = os.path.basename(os.path.dirname(mpg_path))          # s1_processed
    speaker = parent.replace("_processed", "")
    clip_id = os.path.splitext(os.path.basename(mpg_path))[0]     # bbaf2n

    sample_dir = os.path.join(out_root, speaker, clip_id)
    roi_dir = os.path.join(sample_dir, "roi")
    wav_path = os.path.join(sample_dir, "audio.wav")
    mel_path = os.path.join(sample_dir, "mel.npy")

    # -------------------------
    # RESUME: 이미 완료된 샘플(meta.json + 성공조건 충족)은 스킵
    # -------------------------
    meta_path = os.path.join(sample_dir, "meta.json")
    if os.path.isfile(meta_path) and is_success(sample_dir, min_roi_frames=min_roi_frames, min_mel_frames=10):
        return True

    try:
        os.makedirs(sample_dir, exist_ok=True)

        cap = cv2.VideoCapture(mpg_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        meta = {
            "dataset": "GRID",
            "orig_path": mpg_path,
            "speaker": speaker,
            "clip_id": clip_id,
            "fps": float(fps) if fps and fps > 0 else 25.0,
            "sr": int(sr),
            "hop_length": int(hop_length),
            "n_fft": int(n_fft),
            "n_mels": int(n_mels),
            "roi_dir": "roi",
            "mel_path": "mel.npy",
            "out_size": int(roi_size),
            "frame_count_reported": int(frame_count),
        }
        write_meta(sample_dir, meta)

        lip_stats = extract_lips(
            detector,
            mpg_path,
            roi_dir,
            size=roi_size,
            smooth_alpha=smooth_alpha,
            min_side_px=min_side_px,
        )
        meta["lip_stats"] = lip_stats
        meta["num_frames_saved"] = int(lip_stats.get("n_saved", 0))
        write_meta(sample_dir, meta)

        # audio + mel
        extract_audio_wav(mpg_path, wav_path, sr=sr)
        mel_shape = extract_mel(
            wav_path,
            mel_path,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        meta["mel_shape"] = [int(mel_shape[0]), int(mel_shape[1])]
        meta["mel_frames"] = int(mel_shape[1])
        write_meta(sample_dir, meta)

        if not is_success(sample_dir, min_roi_frames=min_roi_frames, min_mel_frames=10):
            raise RuntimeError("NOT success criteria")

        # wav 삭제는 옵션
        if (not keep_wav) and os.path.isfile(wav_path):
            os.remove(wav_path)

        return True

    except Exception as e:
        shutil.rmtree(sample_dir, ignore_errors=True)
        print(f"✘ FAILED {mpg_path}: {e}")
        return False


# ---------------------------------------
# Main
# ---------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    prep = cfg.get("prepare", {})
    ds = cfg.get("dataset", {})

    mode = prep.get("mode", "grid")  # 기본 grid
    out_root = prep.get("out", None)
    grid_root = prep.get("grid_root", None)

    if mode != "grid":
        raise ValueError(f"This script version supports only prepare.mode=grid (got {mode})")

    if out_root is None:
        raise ValueError("YAML prepare.out 이 필요합니다.")
    if grid_root is None:
        raise ValueError("YAML prepare.grid_root 이 필요합니다. (GRID raw data root)")

    limit = prep.get("limit", None)
    min_roi_frames = int(prep.get("min_lip_frames", 5))  # 기존 키 그대로 사용

    # ROI params (minimal)
    roi_size = int(prep.get("roi_size", 96))
    smooth_alpha = float(prep.get("smooth_alpha", 0.4))
    min_side_px = int(prep.get("min_side_px", 20))

    # Mel params (minimal)
    sr = int(prep.get("sr", 16000))
    hop_length = int(prep.get("hop_length", 256))
    n_fft = int(prep.get("n_fft", 1024))
    n_mels = int(prep.get("n_mels", 80))

    keep_wav = bool(prep.get("keep_wav", False))

    os.makedirs(out_root, exist_ok=True)

    mpg_files = sorted(glob.glob(os.path.join(grid_root, "**", "*.mpg"), recursive=True))
    if limit:
        mpg_files = mpg_files[: int(limit)]

    print(f"[GRID] Found mpg files: {len(mpg_files)}")
    print(f"[GRID] out_root: {out_root}")
    print(f"[GRID] roi_size={roi_size}, smooth_alpha={smooth_alpha}, sr={sr}, hop={hop_length}")

    detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=0)

    ok = 0
    for mpg_path in tqdm(mpg_files):
        if process_one_mpg(
            mpg_path,
            detector,
            out_root=out_root,
            roi_size=roi_size,
            smooth_alpha=smooth_alpha,
            min_side_px=min_side_px,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            min_roi_frames=min_roi_frames,
            keep_wav=keep_wav,
        ):
            ok += 1

    print(f"Done. SUCCESS={ok}/{len(mpg_files)}")


if __name__ == "__main__":
    main()

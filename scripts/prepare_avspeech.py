import os
import argparse
import subprocess
import cv2
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from insightface.app import FaceAnalysis


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

    # Step 1: Download mp4
    run_cmd(
        f'yt-dlp -f "bv*+ba/b" --remux-video mp4 "{url}" '
        f'-o "{raw_path}" --quiet'
    )

    # Step 2: Extract video only (for ROI extraction)
    run_cmd(
        f'ffmpeg -y -i "{raw_path}" -ss {start} -to {end} '
        f'-c:v libx264 -preset fast -pix_fmt yuv420p '
        f'-an "{clip_path}" -loglevel quiet'
    )

    # Step 3: Extract audio to wav (because torchaudio cannot load mp4 reliably)
    run_cmd(
        f'ffmpeg -y -i "{raw_path}" -ss {start} -to {end} '
        f'-vn -ac 1 -ar 16000 -c:a pcm_s16le "{wav_path}" -loglevel quiet'
    )

    return clip_path, wav_path


# ---------------------------------------
# Extract lips ROI using insightface
# ---------------------------------------
def extract_lips(detector, clip_path, out_dir, size=96):
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(clip_path)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.get(frame)
        if len(faces) == 0:
            idx += 1
            continue

        face = faces[0]

        # -----------------------------
        # 68 landmark 버전 (강력 안정적)
        # -----------------------------
        lm = face.landmark_3d_68  # (68, 3)

        if lm is None or lm.shape[0] != 68:
            idx += 1
            continue

        # 입술: 48~67
        mouth = lm[48:68, :2]  # x,y만 사용

        x1 = int(np.min(mouth[:, 0]))
        y1 = int(np.min(mouth[:, 1]))
        x2 = int(np.max(mouth[:, 0]))
        y2 = int(np.max(mouth[:, 1]))

        # margin
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)

        lip_roi = frame[y1:y2, x1:x2]

        if lip_roi.size > 0:
            lip_roi = cv2.resize(lip_roi, (size, size))
            cv2.imwrite(os.path.join(out_dir, f"frame_{idx:04d}.png"), lip_roi)

        idx += 1

    cap.release()





# ---------------------------------------
# Extract mel spectrogram
# ---------------------------------------
import librosa

def extract_mel(clip_path, out_path):
    # librosa로 읽기
    wav, sr = librosa.load(clip_path, sr=16000)

    # 멜 스펙트로그램 계산
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
# Process one AVSpeech row
# ---------------------------------------
def process_one_row(row, detector, out_root):
    youtube_id = str(row["youtube_id"])
    start = float(row["start"])
    end = float(row["end"])

    vid_dir = os.path.join(out_root, f"{youtube_id}_{int(start*1000)}")
    os.makedirs(vid_dir, exist_ok=True)

    try:
        # Step 1: Download & cut
        clip_path, wav_path = download_and_clip(youtube_id, start, end, vid_dir)

        # Step 2: ROI extraction
        lips_dir = os.path.join(vid_dir, "lips")
        extract_lips(detector, clip_path, lips_dir)

        # Step 3: mel extraction
        mel_path = os.path.join(vid_dir, "mel.npy")
        extract_mel(wav_path, mel_path)

        print(f"✔ SUCCESS: {youtube_id}")

    except Exception as e:
        print(f"✘ FAILED {youtube_id}: {e}")


# ---------------------------------------
# Main
# ---------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

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
        process_one_row(row, detector, args.out)


if __name__ == "__main__":
    main()

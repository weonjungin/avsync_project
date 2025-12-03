
"""AVSyncEncoder가 잘 동작하는지 랜덤 텐서로 테스트하는 스크립트.

VSCode 터미널에서:

    python demo_forward.py

를 실행했을 때, 모델이 에러 없이 forward만 잘 돌아가면
구조 설계는 정상적으로 된 것이다.
"""

import torch
from models.avsync_encoder import AVSyncEncoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 하이퍼파라미터 예시
    B = 4          # batch size
    T = 8          # video frame 개수
    C = 1          # video channel (grayscale 기준)
    H = W = 96     # 입술 ROI 크기
    N_MELS = 80    # mel 차원
    T_MEL = 32     # mel time step

    # 랜덤 입력 생성 (실제 프로젝트에서는 AVSpeech 전처리 결과가 들어올 예정)
    video = torch.randn(B, T, C, H, W, device=device)
    audio = torch.randn(B, 1, N_MELS, T_MEL, device=device)

    model = AVSyncEncoder(emb_dim=256,
                          video_in_channels=C,
                          audio_in_channels=1).to(device)

    v_emb, a_emb = model(video, audio)
    sim = AVSyncEncoder.cosine_sim(v_emb, a_emb)

    print("Video embedding shape:", v_emb.shape)  # (B, 256)
    print("Audio embedding shape:", a_emb.shape)  # (B, 256)
    print("Cosine similarity shape:", sim.shape)  # (B,)
    print("Cosine similarity values:", sim)


if __name__ == "__main__":
    main()

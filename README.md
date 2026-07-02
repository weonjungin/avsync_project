# avsync_project

Audio-visual 동기화 연구의 출발점. AVSpeech 데이터셋 기반으로 SyncNet을 학습해 오디오-입술 동기화 점수를 계산하는 모델을 개발했습니다.

## 배경 및 결과

초기에는 대규모 화자 다양성을 가진 [AVSpeech](https://looking-to-listen.github.io/avspeech/) 데이터셋으로 SyncNet을 학습하려 했으나, 유튜브 다운로드 정책 변경으로 데이터 수급이 중단되며 프로젝트가 중단되었습니다.

이후 GRID / HDTF 데이터셋으로 전환해 연구를 이어갔고, 그 결과물이 [lip-sync-score](https://github.com/weonjungin/lip-sync-score)입니다. 거기서 개발한 SyncNetTemporal(SyncLT)은 최종적으로 [ADLip2](https://github.com/weonjungin/ADLip2)의 립싱크 손실 함수로 사용되었습니다.

    avsync_project (AVSpeech, 중단)
          │  데이터 수급 문제로 GRID/HDTF 전환
          ▼
    lip-sync-score (SyncNetTemporal/SyncLT 개발)
          │  손실 함수로 결합
          ▼
    ADLip2 (최종 립싱크 생성 모델)

## 레포 구조

    avsync_project/
    ├── avsync_project/           # 핵심 패키지
    │   ├── models/
    │   │   ├── syncnet_model.py      # SyncNet 모델 정의
    │   │   ├── avsync_encoder.py     # 오디오/영상 인코더
    │   │   └── demo_forward.py       # forward pass 데모
    │   ├── dataset/
    │   │   └── dataset_syncnet.py    # AVSpeech 데이터로더
    │   ├── loss/
    │   │   ├── contrastive_loss.py
    │   │   ├── info_nce_loss.py
    │   │   └── hard_infonce_loss.py  # hard negative 샘플링 InfoNCE
    │   └── utils/
    │
    ├── scripts/
    │   ├── prepare_avspeech.py       # AVSpeech 원본 → 전처리
    │   ├── train_syncnet.py          # SyncNet 학습
    │   ├── inference_syncnet.py      # 추론
    │   ├── eval_metrics.py           # 정량 평가
    │   ├── qc_visual_check.py        # 프레임 단위 시각 QC
    │   └── qc_offset_curve.py        # 오디오-영상 offset curve QC
    │
    ├── configs/
    │   └── exp.yaml
    │
    └── logs/checkpoints/
        └── syncnet_ckpt.pth

## 환경 설정

    conda create -n avsync2 python=3.10
    conda activate avsync2
    conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia

    pip install numpy==1.26.4 pandas opencv-python librosa yt-dlp onnxruntime insightface
    pip install retina-face
    pip install git+https://github.com/serengil/retinaface.git

## 실행 방법

### 1. 데이터 준비

AVSpeech CSV를 `data/raw/avspeech_train.csv`, `data/raw/avspeech_test.csv`에 저장한 뒤 전처리합니다.

    python scripts/prepare_avspeech.py --config configs/exp.yaml

### 2. 학습

    python scripts/train_syncnet.py --config configs/exp.yaml

### 3. 평가

    python scripts/eval_metrics.py --config configs/exp.yaml --split test
    python scripts/eval_metrics.py --config configs/exp.yaml --split all

### 4. 추론

    python scripts/inference_syncnet.py --config configs/exp.yaml

### 5. QC (시간축 정렬 확인)

    python scripts/qc_visual_check.py --sample_dir <샘플 폴더> --t0 30 --T 5
    python scripts/qc_offset_curve.py --sample_dir <샘플 폴더>

## 관련 프로젝트

- [lip-sync-score](https://github.com/weonjungin/lip-sync-score): 이 프로젝트를 이어받아 GRID/HDTF 기반으로 개발한 SyncNetTemporal(SyncLT)
- [ADLip2](https://github.com/weonjungin/ADLip2): SyncLT를 손실 함수로 사용한 최종 립싱크 생성 모델

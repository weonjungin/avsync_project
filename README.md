# Setup & Usage

## 1. 데이터 준비

```bash
# data 폴더 생성
mkdir data

# AVSpeech CSV 다운로드 후 아래 위치에 저장
# data/avspeech_train.csv
# data/avspeech_test.csv
```

## 2. AVSpeech 데이터 전처리

```bash
python scripts/prepare_avspeech.py --csv data/avspeech_train.csv --out data/train --limit 50
```

## 3. SyncNet 학습

```bash
python scripts/train_syncnet.py
```

## 4. SyncNet 추론

```bash
python scripts/inference_syncnet.py
```

---

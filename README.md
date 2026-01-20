# Setup & Usage

## 1. 데이터 준비

```bash
# data 폴더 생성
mkdir data

# AVSpeech CSV 다운로드 후 아래 위치에 저장
# data/raw/avspeech_train.csv
# data/raw/avspeech_test.csv
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

## Create environment

```bash
conda create -n avsync2 python=3.10
conda activate avsync2
```

## Install PyTorch (CUDA 11.8)

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
```

## Install dependencies

```bash
pip install numpy==1.26.4
pip install retina-face
pip install pandas
pip install git+https://github.com/serengil/retinaface.git
pip install insightface
pip install onnxruntime
pip install opencv-python
pip install yt-dlp
pip install librosa
```

---
cd ~/projects/avsync_project
conda activate syncnet

# 전처리
python scripts/prepare_avspeech.py --config configs/exp.yaml

# 학습
python scripts/train_syncnet.py --config configs/exp.yaml

# 평가
python scripts/eval_metrics.py --config configs/exp.yaml : test split만 평가 
python scripts/eval_metrics.py --config configs/exp.yaml --split test
python scripts/eval_metrics.py --split all


# 추론
python scripts/inference_syncnet.py --config configs/exp.yaml

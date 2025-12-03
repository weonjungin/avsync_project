data폴더 생성 후 
avspeech_train.csv, avspeech_test.csv 다운 후 넣기
----
python scripts/prepare_avspeech.py --csv data/avspeech_train.csv --out data/train --limit 50
python scripts/train_syncnet.py
python scripts/inference_syncnet.py
----
conda create -n avsync2 python=3.10
conda activate avsync2
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
pip install numpy==1.26.4
pip install retina-face
pip install pandas
pip install git+https://github.com/serengil/retinaface.git
pip install insightface
pip install onnxruntime
pip install opencv-python
pip install insightface onnxruntime opencv-python
pip install yt-dlp
pip install librosa

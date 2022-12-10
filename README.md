# AutoVideoEdit (Frames)

## Features
Clip searching: 
- Search by frame description
- (Dev) Search by audio - matching text
- (Dev) Search by audio - descriptiom
- (Dev) Search by OCR
- (Dev) Search by proper noun + proper noun tagging

Audio-Video Syncing:
- TBD

## Install
Create and activate conda enviorment with python 3.7:
```
conda create -n AutoVideoEdit python=3.7
conda activate AutoVideoEdit
```

Install CLIP:
```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Install other dependancies:
```
pip install numpy opencv-python scenedetect pytesseract
```


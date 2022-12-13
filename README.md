# OpenAutoVideoEdit 

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
Clone and enter the repo:
```
git clone https://github.com/GaneshPimpale/AutoVideoEdit
cd AutoVideoEdit
```

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

## Useage
AVE can be used like a library (look at [edit.py] for an example).

First,
``` python
from AutoVideoEdit import AVE
vid1 = AVE()
```

where ```vid1``` is a video edit instance. Multiple video editing istances can be created, however each instance will use lots of memory so be cautious.

Then,
```python
vid1.addVideo('[Path to video]')
vid1.addVideo('[Path to a different video]')
# Add as many videos as needed

vid1.newQuery('[Ask something here]')
vid1.newQuery('[Ask something else here]')
vid1.newQuery('[Ask another thing here]')
vid1.newQuery('[Ask something here]')
# Add as many queries as needed

# To create a spliced together video:
vid1.compile_vid()
```

Note, the order in which the ```addVideo()``` function is called does **not** matter, however, the order in which ```newQuery()``` is called **DOES MATTER** and will define the sequne in which the videos are spliced togther.

Finally, calling ```.compile_vid()``` is the final function to call to compile the video.

NOTE: If the application exits with no code (windows) or killed by the system, you have run out memory. 

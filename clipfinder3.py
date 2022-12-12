import numpy as np
from tqdm import tqdm
import concurrent
import csv

# CLIP dependancies
import torch
import clip
from PIL import Image

import cv2
from scenedetect import detect, ContentDetector
import pytesseract

# CLIP init
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
cashe, text = {}, {}
print("device:" + device)

# Main Functions
def getFrames(filename): 
    """ return list of frames and FPS from filename
    :param filename: string of filepath
    :return frames: list of images
    :return fps: integer of frames per second
    """
    vid_cap = cv2.VideoCapture(filename)
    success, image = vid_cap.read()
    frames = []
    while success:
        frames.append(image)
        success, image = vid_cap.read()
    
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    return frames, fps

def getFrames_downscale(filename): 
    """ return list of frames from filename, downscaled to 450 x 450
    :param filename: string of filepath
    :return frames: list of images
    """
    vid_cap = cv2.VideoCapture(filename)
    success, image = vid_cap.read()
    frames = []
    while success:
        image = cv2.resize(image, (450, 450))
        frames.append(image)
        success, image = vid_cap.read()
    return frames

def showFrame(frames, index):
    """ Display a frame from its frame list and index
    :param frames: list containing frame
    :param index: index of frame in list
    """
    image = Image.fromarray(frames[index])
    image.show()

def encodeAndCalculate(frame, encoded_prompt):
    """ Encode and calculate the eucclidian dist for a single frame
    :param frame: referance to frame image
    :param encoded_prompt: string
    :return dist: Euc dist between frame and prompt
    """
    image = Image.fromarray(frame)
    image = preprocess(image).unsqueeze(0).to(device)
    tensor_b = encoded_prompt

    with torch.no_grad():
        tensor_a = model.encode_image(image)
        #NOTE: Currently using euclidean dist here:
        dist = torch._euclidean_dist(tensor_a, tensor_b)[0][0].tolist()

    return dist

def encodeAndCalculate_list(frames, encoded_prompt):
    """ Multiprocess encodeAndCalculate with a list of frames and display a progress bar
    :param frames: 
    :param encoded_prompt: 
    :return: list of distances
    """
    with concurrent.futures.ProcessPoolExecutor() as Executor:
        results = []
        with tqdm(total=len(frames)) as pbar:
            for frame in frames:
                results.append(Executor.submit(encodeAndCalculate, frame, encoded_prompt))
    return [r.result() for r in results]

def returnTopClips(QUERY, tupleList, num_clips = 3, data = False, name="data"):
    fullList = []
    iteratorLen = 5
    text_features = model.encode_text(clip.tokenize([QUERY]). to(device))

    for i in tqdm(range(len(myFrame))):
        if i % iteratorLen == 0:
            frame = myFrameLowRes[i]
            image = Image.fromarray(frame)
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                if i not in cashe:
                    image_features = model.encode_image(image)
                    cashe[i] = image_features
                else:
                    image_features = cashe[i]
                tensor = torch.cosine_similarity(image_features, text_features, dim=1)
                fullList.append((tensor.tolist())[0]*100)
        else:
            fullList.append(-1)
    
    if data:
        csvFormat = [[i] for i in fullList]

        with open(name + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csvFormat)

    topClips, iterator, fullListMax = [], 0, max(fullList)
    
    while iterator < num_clips and fullListMax > 0:
        maxIndex = fullList.index(fullListMax)
        clipRange = getClipRange(maxIndex, tupleList, len(myFrame), fps)
        topClips.append(clipRange)
        fullList[clipRange[0]:clipRange[1]+1] = [0] * (clipRange[1] - clipRange[0])
        iterator += 1
        fullListMax = max(fullList)
    
    return topClips


def getClipRange(index, tupleList, frames, fps):
    """
    :param index:
    :param tupleList:
    :param frames: 
    :param fps:
    :return: 
    """
    for i in range(len(tupleList)):
        if index >= tupleList[i][0] and index <= tupleList[i][1]:
            return tupleList[i]
        else: 
            start = index - round(fps)
            if start < 0:
                start = 0
            end = index + (2*round(fps))
            if end > frames - 1:
                end = frames -1
            return (start, end)
    

def spliceVideo(topClipIndex):
    """
    :param topClipIndex: 
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('topclips.mp4', fourcc, 30, (myFrame[0].shape[1], myFrame[0].shape[0]))
    for i in range(len(topClipIndex)):
        for j in range(topClipIndex[i][0], topClipIndex[i][1]):
            out.write(myFrame[j])
    out.release()


# Extra features
def ocr(frame):
    """ Extract the text displayed in a frame
    :param frame: referance to image frame
    :return text: string containing text extracted from frame 
    """
    text = pytesseract.image_to_string(frame)
    return text

# Performace metrics
def euc_dist(a, b):
    """ Measure the euclidean distance between two tensors
    :param a: Tensor a
    :param b: Tensor b
    :return: Euclidean distance between two tensors
    """
    return torch.dist(a, b)

def cos_sim(a, b):
    """ Return the cosine similarity between two tesnors
    :param a: Tensor a
    :param b: Tensor b
    :return: consine similarity between two tensors
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Run script
VIDEO = './test_vids/baby_penguin.mp4'
QUERY = 'small white penguin'
scene_list = detect(VIDEO, ContentDetector())
tupleList = [ (scene_list[i][0].get_frames(), scene_list[i+1][0].get_frames()) for i in range(len(scene_list) - 1) ]
myFrame, fps = getFrames(VIDEO)
myFrameLowRes = getFrames_downscale(VIDEO)

topClipIndex = returnTopClips(QUERY, tupleList, 3)
print(topClipIndex)
spliceVideo(topClipIndex)
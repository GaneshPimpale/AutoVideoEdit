import cv2
from PIL import Image
import torch
import clip
from tqdm import tqdm
import csv
from scenedetect import detect, ContentDetector
import numpy as np
import pytesseract
import framesToEmbeddings as fte
import concurrent

VIDEO_PATH = "./test_vids/random_cat.mp4" #TODO: change file to desired test
PROMPT = "cat in bathroom"
scene_list = detect(VIDEO_PATH, ContentDetector())

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
cos = torch.nn.CosineSimilarity(dim=0)
cashe = {}

def getFrames(filename):
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    frames = []
    while success:
        frames.append(image)
        success,image = vidcap.read()
    return frames

def getFramesLowRes(filename):
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    frames = []
    while success:
        image = cv2.resize(image, (450, 450))
        frames.append(image)
        success,image = vidcap.read()
    return frames

def showFrame(frameList, index):
    image = Image.fromarray(frameList[index])
    image.show()

myFrame = getFrames(VIDEO_PATH)
fullList = []
iteratorLen = 5
for i in tqdm(range(len(myFrame))):
    if i % iteratorLen != 0:
        continue
    frame = myFrame[i]
    image = Image.fromarray(frame)
    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize([PROMPT]).to(device)

    with torch.no_grad():
        if i not in cashe:
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            cashe[i] = image_features
        else:
            image_features = cashe[i]
        logits_per_image, logits_per_text = model(image, text)
        ##probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        fullList.append(logits_per_image.tolist()[0][0])
        

        
maxIndex = fullList.index(max(fullList)) * iteratorLen
image = Image.fromarray(myFrame[maxIndex])
image.show()

csvFormat = [[i] for i in fullList]

with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csvFormat)
    
## input a frame, perform OCR, and return the text if any exist
def OCR(frame):
    text = pytesseract.image_to_string(frame)
    return text

## get the euclidean distance between two tensors
def euc(a, b):
    return torch.dist(a, b)


def encodeAndCalculate(frame, encodedprompt):
    image = Image.fromarray(frame)
    image = preprocess(image).unsqueeze(0).to(device)
    b = encodedprompt

    with torch.no_grad():
        a = model.encode_image(image)
        dist = torch._euclidean_dist(a, b)[0][0].tolist()
    
    return dist

## multiprocess encodeAndCalculate with a list of frames
## start the processes, and show a progress bar as they finish
## return a list of the results
def encodeAndCalculateList(frames, encodedprompt):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = []
        with tqdm(total=len(frames)) as pbar:
            for frame in frames:
                results.append(executor.submit(encodeAndCalculate, frame, encodedprompt))
                pbar.update(1)
    return [r.result() for r in results]

## NOTES:
##logits_per_image, logits_per_text = model(image, clip.tokenize([PROMPT]).to(device))
##fullList.append(logits_per_image.tolist()[0][0])
##fullList.append(torch._euclidean_dist(image_features, text_features)[0][0].tolist())

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
import threading
from multiprocess import Pool, Process

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
cashe, text = {}, {}

def getFrames(filename):
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    frames = []
    while success:
        frames.append(image)
        success,image = vidcap.read()
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    return frames, fps

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

threadingEmbeddings = []
def threadRipperFast(myFrameLowRes, taskID):
    print("starting thread", taskID)
    cashe = {}
    iteratorLen = 5
    for i in range(len(myFrameLowRes)):
        if i % iteratorLen == 0:
            frame = myFrameLowRes[i]
            image = Image.fromarray(frame)
            image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                cashe[i] = image_features
    cashe['ID'] = taskID
    threadingEmbeddings.append(cashe)
    return

def preComputeFast(numThreads):
    ## start threads and save the outputs in a list
    shardSize = len(myFrameLowRes) // numThreads
    myThreads, threadingEmbeddings = [], []
    for i in range(numThreads):
        start = i * shardSize
        end = (i+1) * shardSize
        if i == numThreads - 1:
            end = len(myFrameLowRes)
            
        def dictCutter(dict, start, end):
            newDict = {}
            indexVal = 0
            for i in range(start, end):
                newDict[indexVal] = dict[i]
                indexVal += 1
            return newDict
        
        t = threading.Thread(target=threadRipperFast, args=(dictCutter(myFrameLowRes, start, end), i))
        myThreads.append(t)
        t.start()
        
    for t in myThreads:
        t.join()
    
    temp = {}
    for i in range(len(threadingEmbeddings)):
        ID = threadingEmbeddings[i]['ID']
        threadingEmbeddings[i].pop('ID')
        temp[ID] = threadingEmbeddings[i]
    
    index = 0
    thecashe = {}
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            thecashe[index] = temp[i][j]
            index += 1
    
    return thecashe
    
def preComputeSlow(fps):
    iteratorLen = 5
    last_text, img_text = "", ""
    for i in tqdm(range(len(myFrame))):
        if i % iteratorLen == 0 and i not in cashe:
            frame = myFrameLowRes[i]
            image = Image.fromarray(frame)
            image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                cashe[i] = image_features
                
        if i % round(fps) == 0 and i not in cashe:
            img_text = OCR(myFrameLowRes[i])
                    
            if img_text != "" and img_text not in text:
                text[img_text] = [i]
            elif img_text in text and img_text != last_text:
                text[img_text] = text[img_text] + [i]
                
            last_text = img_text
    
def returnTopClips(PROMPT, tupleList, num_clips = 3, data = False, name = "data"):
    fullList = []
    iteratorLen = 5
    text_features = model.encode_text(clip.tokenize([PROMPT]).to(device))
    last_text, img_text = "", ""

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
                fullList.append(torch.cosine_similarity(image_features, text_features, dim=1).tolist()[0] * 100)
        else:
            fullList.append(-1)
            
    if data:
        csvFormat = [[i] for i in fullList]

        with open(name + ".csv", "w", newline="") as f:
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
    for i in range(len(tupleList)):
        if index >= tupleList[i][0] and index <= tupleList[i][1]:
            return tupleList[i]
        else:
            start = index - round(fps)
            if start < 0:
                start = 0
            end = index + (2 * round(fps))
            if end > frames - 1:
                end = frames - 1
            return (start, end)
        
def spliceVizVideo(topClipIndex):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('topclips.mp4', fourcc, 30, (myFrame[0].shape[1], myFrame[0].shape[0]))
    for i in range(len(topClipIndex)):
        for j in range(topClipIndex[i][0], topClipIndex[i][1]):
            out.write(myFrame[j])
    out.release()

VIDEO_PATH = "./test_vids/random_cat.mp4"
PROMPT = "a calendar"

scene_list = detect(VIDEO_PATH, ContentDetector())
tupleList = [ (scene_list[i][0].get_frames(), scene_list[i+1][0].get_frames()) for i in range(len(scene_list) - 1) ]
myFrame, fps = getFrames(VIDEO_PATH)
myFrameLowRes = getFramesLowRes(VIDEO_PATH)
preComputeSlow(fps)

topClipIndex = returnTopClips(PROMPT, tupleList, 3)
spliceVizVideo(topClipIndex)

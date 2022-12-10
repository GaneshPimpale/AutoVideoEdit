import cv2
from PIL import Image
import torch
import clip
from tqdm import tqdm
import csv
from scenedetect import detect, ContentDetector
import numpy as np
import pytesseract

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def encodeAndCalculate(frame, encodedprompt):
    image = Image.fromarray(frame)
    image = preprocess(image).unsqueeze(0).to(device)
    b = encodedprompt

    ##with torch.no_grad():
    a = model.encode_image(image)
    dist = torch._euclidean_dist(a, b)[0][0].tolist()
    
    return dist
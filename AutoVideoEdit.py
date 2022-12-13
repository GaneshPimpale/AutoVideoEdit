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
print("device: " + device)


class AVE:
    def __init__(self):
        # Global variables
        self.VIDEOS = []
        self.VID_DATA = [] #NOTE: Indexes match with self.VIDEOS
        self.QUERIES = []

    def compile_vid(self):
        print("Compiling video...")

        # Preprocess each video
        for VIDEO in self.VIDEOS:
            print("Processing: " + VIDEO)
            scene_list = detect(VIDEO, ContentDetector())
            tupleList = [ (scene_list[i][0].get_frames(), scene_list[i+1][0].get_frames()) for i in range(len(scene_list) - 1) ]
            myFrame, fps = self.getFrames(VIDEO)
            myFrameLowRes = self.getFrames_downscale(VIDEO)
            vid_process = [tupleList, myFrame, myFrameLowRes, fps]
            self.VID_DATA.append(vid_process)
        
        # Process each query
        clip_pkgs = [] # [clip, vid_index]
        for QUERY in self.QUERIES:
            print("Processing: " + QUERY)
            for vid_data in self.VID_DATA:
                self.myFrame = vid_data[1]
                self.myFrameLowRes = vid_data[2]
                self.fps = vid_data[3]
                clips, score = self.returnTopClips(QUERY, vid_data[0], 1)
                for clip in clips:
                    clip_pkg = [clip, self.VID_DATA.index(vid_data), score]
                    clip_pkgs.append(clip_pkg)

        print(clip_pkgs)

        # Choose the clips that scored higher:
        clip_pkgs_splice = []
        i = 0
        while i + len(self.VIDEOS) <= len(clip_pkgs):
            pkgs = clip_pkgs[i:i+len(self.VIDEOS)]
            maxScore = 0
            maxScoreIndex = -1
            for j in range(len(pkgs)):
                score = pkgs[j][2]
                if score > maxScore:
                    maxScore = score
                    maxScoreIndex = i+j
            clip_pkgs_splice.append(clip_pkgs[maxScoreIndex])
            i += len(self.VIDEOS)
        

        self.spliceVideo(clip_pkgs_splice)

    def addVideo(self, fileName):
        """ Add filepath of video to list of videos
        :param fileName: path to video file
        :return index: index of video in self.VIDEOS
        """
        self.VIDEOS.append(fileName)
        index = len(self.VIDEOS)
        print("Added vid file: " + fileName + " ;index: " + str(index))
        return index

    def newQuery(self, query):
        """ Add query string to list of queries
        :param query: string containing user provided query
        :return index: index of new query in self.QUERIES
        """
        self.QUERIES.append(query)
        index = len(self.QUERIES)
        print("New query: " + query + " ;index: " + str(index))
        return index

    def getVideos(self):
        """ Display all video file paths
        :return: all video file paths
        """
        print(self.VIDEOS)
        return self.VIDEOS

    def getQueries(self):
        """ Display all queries
        :return: list of all queries
        """
        print(self.QUERIES)
        return self.QUERIES

    def getFrames(self, filename): 
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

    def getFrames_downscale(self, filename): 
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

    def showFrame(self, frames, index):
        """ Display a frame from its frame list and index
        :param frames: list containing frame
        :param index: index of frame in list
        """
        image = Image.fromarray(frames[index])
        image.show()

    def encodeAndCalculate(self, frame, encoded_prompt):
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

    def encodeAndCalculate_list(self, frames, encoded_prompt):
        """ Multiprocess encodeAndCalculate with a list of frames and display a progress bar
        :param frames: 
        :param encoded_prompt: 
        :return: list of distances
        """
        with concurrent.futures.ProcessPoolExecutor() as Executor:
            results = []
            with tqdm(total=len(frames)) as pbar:
                for frame in frames:
                    results.append(Executor.submit(self.encodeAndCalculate, frame, encoded_prompt))
        return [r.result() for r in results]

    def getClipRange(self, index, tupleList, frames, fps):
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

    def returnTopClips(self, QUERY, tupleList, num_clips = 3, data = True, name="data"):
        fullList = []
        iteratorLen = 5
        text_features = model.encode_text(clip.tokenize([QUERY]). to(device))

        for i in tqdm(range(len(self.myFrame))):
            if i % iteratorLen == 0:
                frame = self.myFrameLowRes[i]
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
            clipRange = self.getClipRange(maxIndex, tupleList, len(self.myFrame), self.fps)
            topClips.append(clipRange)
            fullList[clipRange[0]:clipRange[1]+1] = [0] * (clipRange[1] - clipRange[0])
            iterator += 1
            fullListMax = max(fullList)
        
        return topClips, fullListMax

    def spliceVideo(self, clip_pkgs):
        """
        :param topClipIndex: 
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('topclips.mp4', fourcc, 30, (self.myFrame[0].shape[1], self.myFrame[0].shape[0]))
        for clip_pkg in clip_pkgs:
            clip = clip_pkg[0]
            vid_data = self.VID_DATA[clip_pkg[1]]
            for j in range(clip[0], clip[1]):
                myFrame = vid_data[1]
                out.write(myFrame[j])
        out.release()

    # Extra features
    def ocr(self, frame):
        """ Extract the text displayed in a frame
        :param frame: referance to image frame
        :return text: string containing text extracted from frame 
        """
        text = pytesseract.image_to_string(frame)
        return text

    # Performace metrics
    def euc_dist(self, a, b):
        """ Measure the euclidean distance between two tensors
        :param a: Tensor a
        :param b: Tensor b
        :return: Euclidean distance between two tensors
        """
        return torch.dist(a, b)

    def cos_sim(self, a, b):
        """ Return the cosine similarity between two tesnors
        :param a: Tensor a
        :param b: Tensor b
        :return: consine similarity between two tensors
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == '__main__':
    print('Running Frammes.py internal debug...')



from groundingdino.util.inference import load_model, Model

import cv2
import os
from urllib import request
import re


class RoadSignExtractor:
    __GDinoModelUrl = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    __GDinoConfigUrl = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/refs/tags/v0.1.0-alpha2/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    __WeightDir = os.path.join(os.getcwd(), "Weights")
    __Model = None
    __MinimumSignSize = (44, 44)

    def __init__(self, Device="cpu", SignsPath="Signs"):
        self.__SignsPath = SignsPath
        self.Device = Device

        if not os.path.exists(self.__SignsPath):
            os.makedirs(self.__SignsPath)

        self.__DownloadModel()
        self.__FileIndex = self.__GetMaxIndex() + 1

    def setMinimumSignSize(self, width, height):
        self.__MinimumSignSize = (width, height)

    def __GetMaxIndex(self):
        def extract_number(f):
            s = re.findall("(\d+).jpg", f)
            return (int(s[0]) if s else -1, f)

        fileList = os.listdir(self.__SignsPath)

        if len(fileList) == 0:
            return -1

        maxFile = max(fileList, key=extract_number)
        if maxFile and maxFile.endswith(".jpg"):
            maxFile = os.path.splitext(maxFile)[0]
            return int(maxFile.split("_", 1)[1])
        return -1

    def __DownloadModel(self):
        for url in [self.__GDinoModelUrl, self.__GDinoConfigUrl]:
            fileName = url.rsplit("/", 1)[-1]
            completeFileName = os.path.join(self.__WeightDir, fileName)

            if not os.path.exists(completeFileName):
                request.urlretrieve(url, completeFileName)

    def __LoadModel(self):
        if self.__Model == None:
            configPath = os.path.join(
                self.__WeightDir, self.__GDinoConfigUrl.rsplit("/", 1)[-1]
            )
            checkPoint = os.path.join(
                self.__WeightDir, self.__GDinoModelUrl.rsplit("/", 1)[-1]
            )
            self.__Model = Model(
                model_config_path=configPath,
                model_checkpoint_path=checkPoint,
                device=self.Device,
            )

    def __ExtractSignsFromFrame(self, frame):
        self.__LoadModel()
        detections, _ = self.__Model.predict_with_caption(
            image=frame,
            caption="All road signs",
            box_threshold=0.40,
            text_threshold=0.35,
        )

        for detected in detections:
            bbox = detected[0].astype(int)
            # Sign size should be greater than __MinimumSignSize pixels
            if (bbox[3] - bbox[1] < self.__MinimumSignSize[0]) or (
                bbox[2] - bbox[0] < self.__MinimumSignSize[1]
            ):
                continue

            sign = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            signName = "Sign_%08d.jpg" % (self.__FileIndex)
            cv2.imwrite(os.path.join(self.__SignsPath, signName), sign)
            self.__FileIndex += 1

    def extractFromImage(self, ImagePath):
        frame = cv2.imread(ImagePath)
        self.__ExtractSignsFromFrame(frame)

    def extractFromVideo(self, VideoFile):
        videoCapture = cv2.VideoCapture(VideoFile)
        videoFPS = videoCapture.get(cv2.CAP_PROP_FPS)
        frameCount = 0
        imagesPerSeconds = 1
        read = True
        while read:
            read, image = videoCapture.read()

            if not read:
                print("Error reading video frames from file " + VideoFile)
                break

            if frameCount % int(videoFPS / imagesPerSeconds) == 0:
                self.__ExtractSignsFromFrame(image)
            frameCount += 1

        videoCapture.release()

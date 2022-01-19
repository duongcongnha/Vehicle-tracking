import numpy as np
import cv2
import dlib

class FrontalFaceModel:
    """
    
    """

    def __init__(self):
        self.model = None
        self.faces = None

    def process(self, img):
        faces = None
        return faces


class HogModel(FrontalFaceModel):

    def __init__(self):
        super().__init__()
        self.model = dlib.get_frontal_face_detector()

    def process(self, img):
        self.faces = self.model(img, 1)
        return (self.faces, len(self.faces))


class SsdModel(FrontalFaceModel):

    def __init__(self, frame_height:int, frame_width:int):
        super().__init__()
        self.frame_height = frame_height
        self.frame_width = frame_width

        modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "models/deploy.prototxt"
        self.model = cv2.dnn.readNetFromCaffe(configFile, modelFile)


    def process(self, img):
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
        self.model.setInput(blob)
        self.faces = self.model.forward()
        return (self.faces, self.faces.shape[2])


    
        
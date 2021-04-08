# Licensed under the EUPL.

from mtcnn import MTCNN
import cv2
from FacedetectionInterface import FacedetectionInterface

class MyMTCNN(FacedetectionInterface):
    def __init__(self):
        super().__init__()
        
        self.detector = MTCNN()
        
    def get_amount_faces_from_img(self, cv2_img) -> int:
        result = self.detector.detect_faces(cv2_img)
        return(len(result))
    
    def __str__(self):
        return "mtcnn"
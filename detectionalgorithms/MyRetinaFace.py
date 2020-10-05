# Licensed under the EUPL.

from face_detection import RetinaFace
from FacedetectionInterface import FacedetectionInterface


class MyRetinaFace(FacedetectionInterface):
    def __init__(self):
        super().__init__()
        
        self.detector = RetinaFace()
        
    def get_amount_faces_from_img(self, cv2_img) -> int:
        result = self.detector(cv2_img)
        amount_detected_faces = 0
        for r in result:
            if r[2] > 0.95:
                amount_detected_faces += 1
        return(amount_detected_faces)
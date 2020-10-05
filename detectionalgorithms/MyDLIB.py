# Licensed under the EUPL.

from FacedetectionInterface import FacedetectionInterface
import face_recognition

class MyDLIB(FacedetectionInterface):
    def __init__(self):
        super().__init__()
        
    def get_amount_faces_from_img(self, cv2_img) -> int:
        face_locations = face_recognition.face_locations(cv2_img)
        return(len(face_locations))
# Licensed under the EUPL.

import time
import cv2

class FacedetectionInterface:
    def __init__(self):
        self.duration_each_img = []
    
    def get_amount_faces_from_path(self, path):
        start = time.time()
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        amount_faces = self.get_amount_faces_from_img(img)
        self.duration_each_img.append(time.time()-start)
        return amount_faces
        
    def get_amount_faces_from_img(self, cv2_img) -> int:
        pass
    
    def has_one_face(self, path):
        return self.get_amount_faces_from_path(path) == 1
    
    def get_avg_computationtime_per_image(self):
        return "{} s".format(sum(self.duration_each_img) / len(self.duration_each_img))

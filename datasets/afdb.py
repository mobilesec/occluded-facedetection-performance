# Licensed under the EUPL.

from FacedatasetInterface import FacedatasetInterface
import os

class AFDB(FacedatasetInterface):
    def __init__(self, path):
        self.path=path
        self.img_paths = []
        self.get_paths()
        
    def get_paths(self):
        for person_id in os.listdir(self.path):
            person_imgs_path = os.path.join(self.path, person_id)
            for person_img_path in os.listdir(person_imgs_path):
                self.img_paths.append(os.path.join(person_imgs_path, person_img_path))                
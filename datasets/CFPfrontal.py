# Licensed under the EUPL.

from FacedatasetInterface import FacedatasetInterface
import os

class CFPFrontal(FacedatasetInterface):
    def __init__(self, path):
        self.path=path # Path to folder which contains the 'Data' folder of the CFP dataset.
        self.img_paths = []
        self.get_paths()
        
    def get_paths(self):
        person_id_path = os.path.join(self.path, "Data/Images")
        for person_id in os.listdir(person_id_path):
            person_imgs_path = os.path.join(person_id_path, person_id+"/frontal")
            for person_img_path in os.listdir(person_imgs_path):
                self.img_paths.append(os.path.join(person_imgs_path, person_img_path))
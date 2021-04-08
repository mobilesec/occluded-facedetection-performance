# Licensed under the EUPL.

from FacedatasetInterface import FacedatasetInterface
import os

class ImageFolder(FacedatasetInterface):
    def __init__(self, path):
        self.path=path
        self.img_paths = []
        self.get_paths()
        
    def get_paths(self):
        for img_file in os.listdir(self.path):
            self.img_paths.append(os.path.join(self.path, img_file))
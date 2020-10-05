# Licensed under the EUPL.

from tqdm import tqdm
from shutil import copyfile
import shutil
import ntpath
import os
import time

class FaceAnalysis:
    def __init__(self, algorithm, dataset):
        self.algorithm = algorithm
        self.dataset = dataset
        self.errors = dict()
        self.amount_correct = 0
        
    def calcAcc(self, copy_file_to_downloads_folder=False, path_to_wrong_detections="~/Downloads/wrong/", path_to_correct_detections="~/Downloads/correct/"):
        if copy_file_to_downloads_folder:
            if os.path.isdir(path_to_wrong_detections):
                shutil.rmtree(path_to_wrong_detections) 
            if os.path.isdir(path_to_correct_detections):
                shutil.rmtree(path_to_correct_detections) 
            os.makedirs(path_to_wrong_detections)
            os.makedirs(path_to_correct_detections)
        imgs = self.dataset.get_paths_to_imgs()
        start = time.time()
        for img in tqdm(imgs):
            if self.algorithm.has_one_face(img):
                self.amount_correct += 1
                if copy_file_to_downloads_folder:
                    copyfile(img, os.path.join(path_to_correct_detections,img.replace("/","_")))
            else:
                amount_faces_detected = str(self.algorithm.get_amount_faces_from_path(img))
                if amount_faces_detected in self.errors:
                    self.errors[amount_faces_detected] += 1
                else:
                    self.errors[amount_faces_detected] = 1
                
                if copy_file_to_downloads_folder:
                    copyfile(img, os.path.join(path_to_wrong_detections,amount_faces_detected+"_"+img.replace("/","_")))
        self.elapsed_time = time.time()-start
    
    def __str__(self):
        ret = "{} images successfully detected.\n".format(self.amount_correct)
        for error in self.errors:
            ret += "In {} images {} face(s) have been found\n".format(self.errors[error], error)
        ret += "Computation time: {}s".format(self.elapsed_time)
        return ret
        
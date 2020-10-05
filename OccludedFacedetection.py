# Licensed under the EUPL.

from facerec.faceextractor import FaceExtractor
from FaceAnalysis import FaceAnalysis

from detectionalgorithms.MyMTCNN import MyMTCNN
from detectionalgorithms.MyRetinaFace import MyRetinaFace
from detectionalgorithms.MyDLIB import MyDLIB

from datasets.CFPfrontal import CFPFrontal
from datasets.CFPprofile import CFPProfile
from datasets.afdb import AFDB


from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image


extractor = FaceExtractor()
mt = MyMTCNN()
rf = MyRetinaFace()
dl = MyDLIB()

path="~/Downloads/" # Path which contains 'cfp-dataset'



def print_results(name):
    dataset = CFPFrontal("~/Downloads/" + name)

    print("=== {} ===".format(name))
    analysis = FaceAnalysis(mt,dataset)
    analysis.calcAcc(copy_file_to_downloads_folder=False)
    print("= MTCNN =")
    print(analysis)

    analysis = FaceAnalysis(rf,dataset)
    analysis.calcAcc(copy_file_to_downloads_folder=False)
    print("= RETINAFACE =")
    print(analysis)

    analysis = FaceAnalysis(dl,dataset)
    analysis.calcAcc(copy_file_to_downloads_folder=False)
    print("= DLIB =")
    print(analysis)


name = "cfp-no-eyes-perc"
original_path=os.path.join(path,"cfp-dataset/Data/Images")
new_path=os.path.join(path,name+"/Data/Images")
for person_nr in tqdm(os.listdir(original_path)):
    frontal_path = os.path.join(original_path, person_nr, "frontal")
    Path(os.path.join(new_path, person_nr, "frontal")).mkdir(parents=True, exist_ok=True)
    for img_nr in os.listdir(frontal_path):
        img_path = os.path.join(frontal_path, img_nr)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = extractor.get_landmarks_from_path(img_path)
        if len(landmarks) == 1:
            left_eye = landmarks[0]["keypoints"]["left_eye"][1]
            right_eye = landmarks[0]["keypoints"]["right_eye"][1]

            min_eye = min(left_eye,right_eye)
            max_eye = max(left_eye,right_eye)
            img[int(min_eye-img.shape[0]/10):int(max_eye+img.shape[0]/10), :, :] = 0
            img_to_print = Image.fromarray(img.astype(np.uint8))
            new_path_img = os.path.join(new_path, person_nr, "frontal", img_nr)
            img_to_print.save(new_path_img)
        else:
            pass
            #print("{} has {} faces".format(img_path, len(landmarks)))
            
print_results(name)


name = "cfp-no-eyes-fixed"
original_path=os.path.join(path,"cfp-dataset/Data/Images")
new_path=os.path.join(path,name+"/Data/Images")
for person_nr in tqdm(os.listdir(original_path)):
    frontal_path = os.path.join(original_path, person_nr, "frontal")
    Path(os.path.join(new_path, person_nr, "frontal")).mkdir(parents=True, exist_ok=True)
    for img_nr in os.listdir(frontal_path):
        img_path = os.path.join(frontal_path, img_nr)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = extractor.get_landmarks_from_path(img_path)
        if len(landmarks) == 1:
            left_eye = landmarks[0]["keypoints"]["left_eye"][1]
            right_eye = landmarks[0]["keypoints"]["right_eye"][1]

            min_eye = min(left_eye,right_eye)
            max_eye = max(left_eye,right_eye)
            img[int(min_eye-5):int(max_eye+5), :, :] = 0


            img_to_print = Image.fromarray(img.astype(np.uint8))
            new_path_img = os.path.join(new_path, person_nr, "frontal", img_nr)
            img_to_print.save(new_path_img)
        else:
            pass
            #print("{} has {} faces".format(img_path, len(landmarks)))

print_results(name)


name = "cfp-no-eyes-rectangle"
original_path=os.path.join(path,"cfp-dataset/Data/Images")
new_path=os.path.join(path,name+"/Data/Images")
for person_nr in tqdm(os.listdir(original_path)):
    frontal_path = os.path.join(original_path, person_nr, "frontal")
    Path(os.path.join(new_path, person_nr, "frontal")).mkdir(parents=True, exist_ok=True)
    for img_nr in os.listdir(frontal_path):
        img_path = os.path.join(frontal_path, img_nr)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = extractor.get_landmarks_from_path(img_path)
        if len(landmarks) == 1:
            left=landmarks[0]["keypoints"]["left_eye"]
            right=landmarks[0]["keypoints"]["right_eye"]

            width=int(abs(right[0]-left[0])/10)
            img[left[1]-width:left[1]+width, left[0]-width*3:left[0]+width*3, :] = 0
            img[right[1]-width:right[1]+width, right[0]-width*3:right[0]+width*3, :] = 0


            img_to_print = Image.fromarray(img.astype(np.uint8))
            new_path_img = os.path.join(new_path, person_nr, "frontal", img_nr)
            img_to_print.save(new_path_img)
        else:
            pass
            #print("{} has {} faces".format(img_path, len(landmarks)))

print_results(name)


name = "cfp-top-half-only"
original_path=os.path.join(path,"cfp-dataset/Data/Images")
new_path=os.path.join(path,name+"/Data/Images")
for person_nr in tqdm(os.listdir(original_path)):
    frontal_path = os.path.join(original_path, person_nr, "frontal")
    Path(os.path.join(new_path, person_nr, "frontal")).mkdir(parents=True, exist_ok=True)
    for img_nr in os.listdir(frontal_path):
        img_path = os.path.join(frontal_path, img_nr)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = extractor.get_landmarks_from_path(img_path)
        if len(landmarks) == 1:
            landmarks = landmarks[0]
            nose = landmarks["keypoints"]["nose"]
            img[int(nose[1]+nose[1]/20):, :, :] = 0


            img_to_print = Image.fromarray(img.astype(np.uint8))
            new_path_img = os.path.join(new_path, person_nr, "frontal", img_nr)
            img_to_print.save(new_path_img)
        else:
            pass
            #print("{} has {} faces".format(img_path, len(landmarks)))

print_results(name)


name = "cfp-top-half-only-mask"
original_path=os.path.join(path,"cfp-dataset/Data/Images")
new_path=os.path.join(path,name+"/Data/Images")
for person_nr in tqdm(os.listdir(original_path)):
    frontal_path = os.path.join(original_path, person_nr, "frontal")
    Path(os.path.join(new_path, person_nr, "frontal")).mkdir(parents=True, exist_ok=True)
    for img_nr in os.listdir(frontal_path):
        img_path = os.path.join(frontal_path, img_nr)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = extractor.get_landmarks_from_path(img_path)
        if len(landmarks) == 1:
            left_eye = landmarks[0]["keypoints"]["left_eye"][1]
            right_eye = landmarks[0]["keypoints"]["right_eye"][1]
            min_eye = max(left_eye,right_eye)
            nose = landmarks[0]["keypoints"]["nose"][1]
            middle = int((nose+min_eye)/2)
            img[middle:, :, :] = 0


            img_to_print = Image.fromarray(img.astype(np.uint8))
            new_path_img = os.path.join(new_path, person_nr, "frontal", img_nr)
            img_to_print.save(new_path_img)
        else:
            pass
            #print("{} has {} faces".format(img_path, len(landmarks)))

print_results(name)



name = "cfp-no-top"
original_path=os.path.join(path,"cfp-dataset/Data/Images")
new_path=os.path.join(path,name+"/Data/Images")
for person_nr in tqdm(os.listdir(original_path)):
    frontal_path = os.path.join(original_path, person_nr, "frontal")
    Path(os.path.join(new_path, person_nr, "frontal")).mkdir(parents=True, exist_ok=True)
    for img_nr in os.listdir(frontal_path):
        img_path = os.path.join(frontal_path, img_nr)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = extractor.get_landmarks_from_path(img_path)
        if len(landmarks) == 1:
            keypoints = landmarks[0]["keypoints"]
            offset = 25
            eye = min(keypoints["left_eye"][1]-offset, keypoints["right_eye"][1]-offset)
            img[:int(eye), :, :] = 0


            img_to_print = Image.fromarray(img.astype(np.uint8))
            new_path_img = os.path.join(new_path, person_nr, "frontal", img_nr)
            img_to_print.save(new_path_img)
        else:
            pass
            #print("{} has {} faces".format(img_path, len(landmarks)))

print_results(name)


import random

for i in range(10):
    name = "cfp-random-black-bars-{}".format(i)
    original_path=os.path.join(path,"cfp-dataset/Data/Images")
    new_path=os.path.join(path,name+"/Data/Images")
    for person_nr in tqdm(os.listdir(original_path)):
        frontal_path = os.path.join(original_path, person_nr, "frontal")
        Path(os.path.join(new_path, person_nr, "frontal")).mkdir(parents=True, exist_ok=True)
        for img_nr in os.listdir(frontal_path):
            img_path = os.path.join(frontal_path, img_nr)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            landmarks = extractor.get_landmarks_from_path(img_path)
            if len(landmarks) == 1:
                height = int(img.shape[0]/10)
                start_y = random.randint(0,int(img.shape[0]-height))
                img[start_y-height:start_y+height, :, :] = 0


                img_to_print = Image.fromarray(img.astype(np.uint8))
                new_path_img = os.path.join(new_path, person_nr, "frontal", img_nr)
                img_to_print.save(new_path_img)
            else:
                pass
                #print("{} has {} faces".format(img_path, len(landmarks)))

    print_results(name)



name = "cfp-top-half-only-more"
original_path=os.path.join(path,"cfp-dataset/Data/Images")
new_path=os.path.join(path,name+"/Data/Images")
for person_nr in tqdm(os.listdir(original_path)):
    frontal_path = os.path.join(original_path, person_nr, "frontal")
    Path(os.path.join(new_path, person_nr, "frontal")).mkdir(parents=True, exist_ok=True)
    for img_nr in os.listdir(frontal_path):
        img_path = os.path.join(frontal_path, img_nr)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = extractor.get_landmarks_from_path(img_path)
        if len(landmarks) == 1:
            landmarks = landmarks[0]
            nose = landmarks["keypoints"]["nose"]
            img[int(nose[1]+nose[1]/10):, :, :] = 0
            
            
            img_to_print = Image.fromarray(img.astype(np.uint8))
            new_path_img = os.path.join(new_path, person_nr, "frontal", img_nr)
            img_to_print.save(new_path_img)
        else:
            pass
            #print("{} has {} faces".format(img_path, len(landmarks)))
            
print_results(name)



name = "cfp-vert-bar-larger"
original_path=os.path.join(path,"cfp-dataset/Data/Images")
new_path=os.path.join(path,name+"/Data/Images")
for person_nr in tqdm(os.listdir(original_path)):
    frontal_path = os.path.join(original_path, person_nr, "frontal")
    Path(os.path.join(new_path, person_nr, "frontal")).mkdir(parents=True, exist_ok=True)
    for img_nr in os.listdir(frontal_path):
        img_path = os.path.join(frontal_path, img_nr)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = extractor.get_landmarks_from_path(img_path)
        if len(landmarks) == 1:
            landmarks = landmarks[0]
            nose = landmarks["keypoints"]["nose"]
            img[:, int(nose[0]+nose[0]/10):, :] = 0


            img_to_print = Image.fromarray(img.astype(np.uint8))
            new_path_img = os.path.join(new_path, person_nr, "frontal", img_nr)
            img_to_print.save(new_path_img)
        else:
            pass
            #print("{} has {} faces".format(img_path, len(landmarks)))

print_results(name)



name = "cfp-vert-bar-larger-small"
original_path=os.path.join(path,"cfp-dataset/Data/Images")
new_path=os.path.join(path,name+"/Data/Images")
for person_nr in tqdm(os.listdir(original_path)):
    frontal_path = os.path.join(original_path, person_nr, "frontal")
    Path(os.path.join(new_path, person_nr, "frontal")).mkdir(parents=True, exist_ok=True)
    for img_nr in os.listdir(frontal_path):
        img_path = os.path.join(frontal_path, img_nr)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = extractor.get_landmarks_from_path(img_path)
        if len(landmarks) == 1:
            landmarks = landmarks[0]
            nose = landmarks["keypoints"]["nose"]
            img[:, int(nose[0]+nose[0]/5):, :] = 0


            img_to_print = Image.fromarray(img.astype(np.uint8))
            new_path_img = os.path.join(new_path, person_nr, "frontal", img_nr)
            img_to_print.save(new_path_img)
        else:
            pass
            #print("{} has {} faces".format(img_path, len(landmarks)))

print_results(name)



name = "cfp-vert-bars"
original_path=os.path.join(path,"cfp-dataset/Data/Images")
new_path=os.path.join(path,name+"/Data/Images")
for person_nr in tqdm(os.listdir(original_path)):
    frontal_path = os.path.join(original_path, person_nr, "frontal")
    Path(os.path.join(new_path, person_nr, "frontal")).mkdir(parents=True, exist_ok=True)
    for img_nr in os.listdir(frontal_path):
        img_path = os.path.join(frontal_path, img_nr)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = extractor.get_landmarks_from_path(img_path)
        if len(landmarks) == 1:
            img_path = os.path.join(frontal_path, img_nr)
            img = cv2.imread(img_path)
            img[:, int(img.shape[0]*2/3):, :] = 0


            img_to_print = Image.fromarray(img.astype(np.uint8))
            new_path_img = os.path.join(new_path, person_nr, "frontal", img_nr)
            img_to_print.save(new_path_img)
        else:
            pass
            #print("{} has {} faces".format(img_path, len(landmarks)))

print_results(name)

# Licensed under the EUPL

from exceptions.NoFaceFoundError import NoFaceFoundError
from lib.align_trans import get_reference_facial_points, warp_and_crop_face

from mtcnn import MTCNN
import cv2
import numpy as np
import errno

class FaceExtractor:
    def __init__(self):
        self.reference = get_reference_facial_points(default_square=True)
        self.detector = MTCNN()

    def get_face_from_image(self, img):
        if img is None:
            raise TypeError()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        landmarks = self.detector.detect_faces(img)
        warped_faces = []
        if landmarks:
            for landmark in landmarks:
                kp = landmark["keypoints"]
                facial5points = [kp['left_eye'], kp['right_eye'], kp['nose'], kp["mouth_left"], kp["mouth_right"]]
                warped_face = warp_and_crop_face(np.array(img), facial5points, self.reference, crop_size=(112, 112))
                warped_faces.append(warped_face)
            return warped_faces
        else:
            raise NoFaceFoundError()

    def get_face_from_path(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img_path)
        return self.get_face_from_image(img)

    def get_landmarks_from_image(self, img):
        return self.detector.detect_faces(img)

    def get_landmarks_from_path(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.detector.detect_faces(img)

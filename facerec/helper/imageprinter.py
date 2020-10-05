# Licensed under the EUPL

import math
from PIL import Image, ImageOps

class ImagePrinter:
    def __init__(self, cols=2, size=200):
        self.imgs = []
        self.cols = cols
        self.size = size

    def add_img_real(self, img):
        thumb = ImageOps.fit(img, (self.size, self.size), Image.ANTIALIAS)
        self.imgs.append(thumb)

    def add_img(self, path):
        im = Image.open(path)
        self.add_img_real(im)
        
    def get_img(self):
        rows = math.ceil(len(self.imgs) / self.cols)
        new_im = Image.new('RGB', (self.cols * self.size, rows * self.size))
        counter = 0
        for j in range(0, rows * self.size, self.size):
            for i in range(0, self.cols * self.size, self.size):
                if counter < len(self.imgs):
                    new_im.paste(self.imgs[counter], (i, j))
                counter += 1
        return new_im

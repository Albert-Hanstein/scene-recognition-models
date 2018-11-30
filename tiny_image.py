'''
Take address as parameter.
cd into this address,
    - Take each image
        - Crop it into 16x16
        - Flatten it into a 1D array
    - Vstack 1D arrays
    - Normalise this stack
    - Return the stack
'''

import os
import cv2
import numpy as np
from sklearn import preprocessing

def stack_images(path):
    os.chdir(path)
    image_array = np.ones(256,)
    stack = np.ones(256,)
    # Use the fact that there are always 99 images per folder
    # to our advantage by using range of 100
    for image in range(100):
        img = cv2.imread(str(image) + '.jpg', cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (16, 16))
        resized_img = np.array(resized_img, dtype=float)
        image_array = resized_img.ravel()
        stack = np.vstack([stack, image_array])
    stack = stack[1:]
    os.chdir('../')
    return stack;

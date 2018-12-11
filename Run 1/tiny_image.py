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
from pathlib import Path
# from sklearn import preprocessing

def stack_images(path, num):
    os.chdir(path)
    image_array = np.ones(256,)
    stack = np.ones(256,)
    # Use the fact that there are always 99 images per folder
    # to our advantage by using range of 100
    for image in range(num):
        file = Path(str(image)+'.jpg')
        if file.is_file():
            img = cv2.imread(str(image) + '.jpg', cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, (16, 16))
            resized_img = np.array(resized_img, dtype=float)
            image_array = resized_img.ravel()
            stack = np.vstack([stack, image_array])
    stack = stack[1:]
    os.chdir('../')
    return stack;

def convert(num):
    if num == 0: return 'tallbuilding'
    elif num == 1: return 'suburb'
    elif num == 2: return 'insidecity'
    elif num == 3: return 'highway'
    elif num == 4: return 'bedroom'
    elif num == 5: return 'opencountry'
    elif num == 6: return 'livingroom'
    elif num == 7: return 'store'
    elif num == 8: return 'industrial'
    elif num == 9: return 'kitchen'
    elif num == 10: return 'office'
    elif num == 11: return 'coast'
    elif num == 12: return 'street'
    elif num == 13: return 'mountain'
    elif num == 14: return 'forest'
    else: return 'This is not a category.';

import os
# import cv2
import numpy as np
from sklearn import preprocessing
from tiny_image import stack_images

'''
img = cv2.imread('../training/bedroom/1.jpg', cv2.IMREAD_GRAYSCALE)
resized_img = cv2.resize(img, (16, 16))
resized_img = np.array(resized_img, dtype=float)
flattened = resized_img.ravel()
flattened2 = np.vstack([flattened, flattened])
# Only do this once all the images in the folder are saved
normalised = preprocessing.normalize(flattened2, norm='l1')
'''

folder = [x[0] for x in os.walk('../training/')]
folder = folder[1:]
for path in folder:
    categ_column = np.tile(folder.index(path), (100, 1)) # 100 because that's the number of images in each category
    flat_imgs = stack_images(path)
    norm_flat_imgs = preprocessing.normalize(flat_imgs, norm='l1')
    categ_dataset = np.hstack([norm_flat_imgs, categ_column])
    

# Create vertical np.array with index indicating category
# hstack that to the stack of images
# This will be the data fo rone category
# Do a bigger vstack to stack the categories

'''
test_image = cv2.imread('0.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

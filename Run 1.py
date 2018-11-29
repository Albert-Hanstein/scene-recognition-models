import cv2
import numpy as np
from sklearn import preprocessing

img = cv2.imread('../training/bedroom/1.jpg', cv2.IMREAD_GRAYSCALE)
resized_img = cv2.resize(img, (16, 16))
resized_img = np.array(resized_img, dtype=float)
flattened = resized_img.ravel()
flattened2 = np.vstack([flattened, flattened])
normalised = preprocessing.normalize(flattened2, norm='l1')
print(sum(normalised[1]))

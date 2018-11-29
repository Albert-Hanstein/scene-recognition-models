import cv2
import numpy as np

img = cv2.imread('../training/bedroom/0.jpg', cv2.IMREAD_GRAYSCALE)
resized_img = cv2.resize(img, (16, 16))
resized_img = np.array(resized_img, dtype=float)
flattened = resized_img.ravel()
print(flattened)
"""
cv2.imshow('image',resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

import os
import cv2

def sample(path):
    os.chdir(path)
    test_image = cv2.imread('0.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return;

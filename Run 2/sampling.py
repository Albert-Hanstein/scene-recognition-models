import os
import cv2

def sample(path):
    os.chdir(path)
    test_image = cv2.imread('0.jpg', cv2.IMREAD_GRAYSCALE)
    height, width = test_image.shape
    x_start_sample = 0
    y_start_sample = 0
    num_of_samples = 0
    while y_start_sample <= (height - 8):
        while x_start_sample <= (width - 8):
            num_of_samples += 1
            x_start_sample += 4
        x_start_sample = 0
        y_start_sample += 4
    print("Number of samples: " + str(num_of_samples)) # 3185
    # crop_img = test_image[0:100, 0:100]
    # cv2.imshow('crop1', crop_img)
    # cv2.imshow('orig', test_image)
    # cv2.imshow('crop2', test_image[150:200, 0:100])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return;

import os
import cv2
import numpy as np
from sklearn import preprocessing

def sample(image):
    sample_size = 8
    # print('Image: ' + str(image))
    test_image = cv2.imread(str(image) + '.jpg', cv2.IMREAD_GRAYSCALE)
    height, width = test_image.shape
    print('Height: ' + str(height) + ' Width: ' + str(width))
    x_start_sample = 0
    y_start_sample = 0
    num_of_samples = 0
    sample_stack = np.ones(sample_size**2,)
    while y_start_sample <= (height - sample_size):
        while x_start_sample <= (width - sample_size):
            sample = test_image[int(y_start_sample):int((y_start_sample+sample_size)), int(x_start_sample):int((x_start_sample+sample_size))]
            num_of_samples += 1

            flat_sample = np.array(sample, dtype=float)
            flat_sample = flat_sample.ravel()
            sample_stack = np.vstack([sample_stack, flat_sample])

            x_start_sample += (sample_size/2)
        x_start_sample = 0
        y_start_sample += (sample_size/2)
    sample_stack = sample_stack[1:,:]
    sample_stack = preprocessing.normalize(sample_stack, norm='l1') # mean=0 && sum==1
    print("Number of samples: " + str(num_of_samples)) # 3185
    print(sample_stack.shape) # expecting (XX, 64)

    # For debugging/experimenting purposes, use the code below to view image sections
    '''
    crop_img = test_image[0:100, 0:100]
    cv2.imshow('crop1', crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return sample_stack;

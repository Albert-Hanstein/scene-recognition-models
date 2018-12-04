import os
import cv2
import numpy as np
from sampling import sample
from quantisation import stack_training_dataset

path = '../training/bedroom/'
sample_stack = sample(path)

'''
TODO: Sample_stack 80% of all images in training dataset
- Go through all folders
- Sample + stack up 80% of all images
'''

path = '../training/'
stack_training_dataset(path) # expecting 15

import os
import cv2
import numpy as np
from joblib import dump, load
from sampling import sample
from quantisation import stack_training_dataset

# path = '../training/bedroom/'
# sample_stack = sample(0)

'''
TODO: Sample_stack 80% of all images in training dataset
- Go through all folders
- Sample + stack up 80% of all images
'''

path = '../training/'
dump(stack_training_dataset(path), 'stacked_training_set_before_kmc.joblib') # expecting 15

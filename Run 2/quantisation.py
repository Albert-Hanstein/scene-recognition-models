import os
import numpy as np
from sampling import sample

def stack_training_dataset(path):
    num_of_images = 80
    sample_size = 8

    folder = [x[0] for x in os.walk(path)]
    folder = folder[1:]
    stack = np.ones(sample_size**2,)
    print("Number of categories: " + str(len(folder)))
    for category in folder:
        os.chdir(category)
        print(category)
        for img in range(num_of_images):
            stack = np.vstack([stack, sample(img)])
        print(stack.shape)
        os.chdir('../')
    print('Shape of stack after piling up training set samples:')
    print(stack.shape)
    return stack;

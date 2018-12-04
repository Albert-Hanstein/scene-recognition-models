import os

def stack_training_dataset(path):
    folder = [x[0] for x in os.walk(path)]
    folder = folder[1:]
    print("Number of categories: " + str(len(folder)))
    return;

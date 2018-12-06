import os
import cv2
import numpy as np
#from sklearn.cluster import KMeans
from joblib import dump, load
# from sampling import sample
from quantisation import stack_training_dataset, k_means

def main():
    # To sample the training images and stack up the samples
    # path = '../training/'
    # dump(stack_training_dataset(path), 'stacked_training_set_before_kmc.joblib') # expecting 15

    # K-Means Clustering on this stack of samples
    stack_for_kmeans = load('../training/stacked_training_set_before_kmc.joblib')
    stack_for_kmeans = stack_for_kmeans[1:,:]
    np.random.shuffle(stack_for_kmeans)
    # kmeans = k_means(stack_for_kmeans[:10000,:])
    kmeans = load('kmeans_model.joblib')
    print(kmeans.predict(stack_for_kmeans[10000:10002,:]))

    # Test out the kmeans model


    return;

if __name__ == "__main__":
    main()

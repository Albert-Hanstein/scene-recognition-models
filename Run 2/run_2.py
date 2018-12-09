import os
import cv2
import numpy as np
#from sklearn.cluster import KMeans
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from BoVW_functions import *

def main():
    '''
    # To sample the training images and stack up the samples
    # path = '../training/'
    # dump(stack_training_dataset(path), 'stacked_training_set_before_kmc.joblib') # expecting 15

    # K-Means Clustering on this stack of samples
    #stack_for_kmeans = load('../training/stacked_training_set_before_kmc.joblib')
    #stack_for_kmeans = stack_for_kmeans[1:,:]
    #np.random.shuffle(stack_for_kmeans)
    # kmeans = k_means(stack_for_kmeans[:10000,:])
    '''
    # Just load a saved model to go faster
    kmeans = load('kmeans_model.joblib')
    '''
    # Test out the kmeans model
    image_path = '../training/Office/50'
    test_stack = sample(image_path)
    #image_path = '../training/bedroom/1'
    #test_stack = np.vstack([test_stack, sample(image_path)])
    prediction = kmeans.predict(test_stack)
    print(prediction)
    print(len(prediction))
    # Make a 1D histogram of this prediction array
    freq = np.bincount(prediction)
    print(len(freq))
    print(freq)
    '''
    # histogram_stack = one_d_histogram('../training/', kmeans)
    histogram_stack = load('../training/histogram_stack_no_labels.joblib')
    label_col_train = labels('train')

    # One-vs-Rest Logistic Regression
    clf = LogisticRegression(penalty='l1',tol=1e-3,random_state=0, max_iter=100, solver='liblinear', multi_class='ovr').fit(histogram_stack, label_col_train)
    print(clf.predict(histogram_stack[75:85,:]))
    print(clf.score(histogram_stack, label_col_train))

    hist_stack_test = one_d_histogram('../training/', kmeans, 'test')
    label_col_test = labels('test')
    predict_on_test = clf.predict(hist_stack_test)
    accuracy = 100 * np.sum(predict_on_test==label_col_test)/len(predict_on_test)
    print('Accuracy on test set: ' + str(accuracy) + '%')

    return;

if __name__ == "__main__":
    main()

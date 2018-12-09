import os
import cv2
import numpy as np
from joblib import dump, load
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from BoVW_functions import *

def main():

    # Sample the training images and stack up the samples
    path = '../training/'
    stack_for_kmeans = stack_training_dataset(path)
    #dump(stack_training_dataset(path), 'stacked_training_set_before_kmc.joblib') # expecting 15

    # K-Means Clustering on this stack of samples
    # stack_for_kmeans = load('../training/stacked_training_set_before_kmc.joblib')
    stack_for_kmeans = stack_for_kmeans[1:,:]
    np.random.shuffle(stack_for_kmeans)
    kmeans = k_means(stack_for_kmeans[:50000,:])
    # kmeans = load('kmeans_model.joblib')

    histogram_stack = one_d_histogram('../training/', kmeans, 'train')
    #histogram_stack = load('../training/histogram_stack_no_labels.joblib')
    label_col_train = labels('train')

    # Shuffle features and labels the same way without combining them first
    p = np.random.permutation(len(histogram_stack))
    histogram_stack = histogram_stack[p]
    label_col_train = label_col_train[p]

    # One-vs-Rest Logistic Regression
    clf = LogisticRegression(penalty='l1',tol=1e-3,random_state=0, solver='liblinear', multi_class='ovr').fit(histogram_stack, label_col_train)
    predict_on_train = clf.predict(histogram_stack)
    accuracy_on_train = 100 * np.sum(predict_on_train==label_col_train)/len(predict_on_train)
    print('Accuracy on training set: ' + str(accuracy_on_train) + '%')

    hist_stack_test = one_d_histogram('../training/', kmeans, 'test')
    # hist_stack_test = load('../training/histogram_stack_test.joblib')
    label_col_test = labels('test')
    predict_on_test = clf.predict(hist_stack_test)
    accuracy_on_test = 100 * np.sum(predict_on_test==label_col_test)/len(predict_on_test)
    print('Accuracy on test set: ' + str(accuracy_on_test) + '%')

    # Confusion matrix for in-depth analysis of model behaviour
    cm = metrics.confusion_matrix(label_col_test, predict_on_test)
    display_matrix(cm)

    return;

if __name__ == "__main__":
    main()

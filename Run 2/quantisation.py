import os
import numpy as np
from joblib import dump, load
from sklearn.cluster import KMeans
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

def k_means(data_points):
    print(data_points.shape)
    kmeans = KMeans(n_clusters=500, random_state=0, n_jobs=1).fit(data_points)
    dump(kmeans, 'kmeans_model.joblib')
    print('Clustering done')
    return kmeans;

def one_d_histogram(path, clustering_model):
    num_of_images = 10 # Set this back to 80 when making dataset
    folder = [x[0] for x in os.walk(path)]
    folder = folder[1:]
    histogram_stack = np.ones(500,)
    for category in folder:
        os.chdir(category)
        for img in range(num_of_images):
            samples = sample(img)
            prediction = clustering_model.predict(samples)
            freq = np.bincount(prediction)
            while(len(freq) < 500):
                freq = np.append(freq, 0)
            print('Freq shape: ' + str(freq.shape))
            histogram_stack = np.vstack([histogram_stack, freq])
            print('Histogram stack shape: ' + str(histogram_stack.shape))
        os.chdir('../')
    histogram_stack = histogram_stack[1:,:]
    dump(histogram_stack, 'histogram_stack.joblib')
    return histogram_stack;

def labels():
    samples_per_categ = 80
    num_of_categs = 15
    label_col = np.zeros(samples_per_categ)
    for categ in range(num_of_categs-1):
        label_col = np.hstack([label_col, np.full(samples_per_categ, categ+1)])
    return label_col;

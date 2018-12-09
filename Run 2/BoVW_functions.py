import os
import cv2
import numpy as np
from joblib import dump, load
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sampling import sample

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

def one_d_histogram(path, clustering_model, mode):
    if mode == 'train':
        num_of_images = 80
    elif mode == 'test':
        num_of_images = 20
    folder = [x[0] for x in os.walk(path)]
    folder = folder[1:]
    histogram_stack = np.ones(500,)
    for category in folder:
        os.chdir(category)
        for img in range(num_of_images):
            if mode == 'test':
                img = img + 80
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
    if mode == 'train':
        dump(histogram_stack, 'histogram_stack_train.joblib')
    elif mode == 'test':
        dump(histogram_stack, 'histogram_stack_test.joblib')
    return histogram_stack;

def labels(mode):
    num_of_categs = 15
    if mode == 'train':
        samples_per_categ = 80
    elif mode == 'test':
        samples_per_categ = 20
    label_col = np.zeros(samples_per_categ)
    for categ in range(num_of_categs-1):
        label_col = np.hstack([label_col, np.full(samples_per_categ, categ+1)])
    return label_col;

def display_matrix(confusion_matrix):
    plt.figure(figsize=(9,9))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size = 15)
    plt.colorbar()
    tick_marks = np.arange(15)
    plt.xticks(tick_marks, ["Tall Building", "Suburb", "Inside City", "Highway", "Bedroom", "Open Country", "Living Room", "Store", "Industrial", "Kitchen", "Office", "Coast", "Street", "Mountain", "Forest"], rotation=90, size = 10)
    plt.yticks(tick_marks, ["Tall Building", "Suburb", "Inside City", "Highway", "Bedroom", "Open Country", "Living Room", "Store", "Industrial", "Kitchen", "Office", "Coast", "Street", "Mountain", "Forest"], size = 10)
    plt.tight_layout()
    plt.ylabel('Actual label', size = 15)
    plt.xlabel('Predicted label', size = 15)
    width, height = confusion_matrix.shape

    for x in range(width):
     for y in range(height):
      plt.annotate(str(confusion_matrix[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')
    plt.show()
    return;

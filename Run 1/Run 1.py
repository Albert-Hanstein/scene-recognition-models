import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tiny_image import *

# Preparing numpy matrix dataset
folder = [x[0] for x in os.walk('../training/')]
folder = folder[1:]
whole_dataset = np.zeros(257)
for path in folder:
    categ_column = np.tile(folder.index(path), (100, 1)) # 100 because that's the number of images in each category
    flat_imgs = stack_images(path, 100)
    norm_flat_imgs = preprocessing.normalize(flat_imgs, norm='l1') # mean=0 && sum==1
    categ_dataset = np.hstack([norm_flat_imgs, categ_column])
    whole_dataset = np.vstack([whole_dataset, categ_dataset])
whole_dataset = whole_dataset[1:, :] # remove the row of zeroes
np.random.shuffle(whole_dataset)

# Preparing the processed data for input
features = whole_dataset[:,:-1]
labels = whole_dataset[:, -1]
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20)

scores = []
for i in range(15):
    # Train the model
    classifier = KNeighborsClassifier(n_neighbors=(i+1)) # K = 5
    classifier.fit(features_train, labels_train)
    # Predict
    labels_pred = classifier.predict(features_test)
    # Evaluate
    accuracy = sum(labels_test == labels_pred)/len(labels_test)
    print('K: ' + str(i+1) + ' Accuracy: ' + str(accuracy))
    scores.append(accuracy)

path = '../testing/'
os.chdir(path)
num_of_inference_img = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
text_list = []
flat_img_inference = stack_images(path, num_of_inference_img+3) # No. of images in testing/ folder
norm_flat_img_inference = preprocessing.normalize(flat_img_inference, norm='l1')
inference_pred = classifier.predict(norm_flat_img_inference)
inference_pred.astype(int)
index = 0
for pred in inference_pred:
    pred_text = str(index) + '.jpg ' + convert(pred) + '\n'
    text_list.append(pred_text)
    index = index + 1
text_file = open('/home/hans/Documents/Year 4/Computer Vision/Coursework 3/scene-recognition-models/Run 1/run1.txt', 'w')
for i in range(len(text_list)):
    text_file.write(text_list[i])
text_file.close()

k_values_tried = list(range(1, 16))
plt.plot(k_values_tried, scores, 'ro')
plt.xlabel('Values of K tried')
plt.ylabel('Accuracy')
plt.title('Accuracy for different values of K')
plt.show()

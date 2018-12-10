import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tiny_image import stack_images

# Preparing numpy matrix dataset
folder = [x[0] for x in os.walk('../training/')]
folder = folder[1:]
whole_dataset = np.zeros(257)
for path in folder:
    categ_column = np.tile(folder.index(path), (100, 1)) # 100 because that's the number of images in each category
    flat_imgs = stack_images(path)
    norm_flat_imgs = preprocessing.normalize(flat_imgs, norm='l1') # mean=0 && sum==1
    categ_dataset = np.hstack([norm_flat_imgs, categ_column])
    whole_dataset = np.vstack([whole_dataset, categ_dataset])
whole_dataset = whole_dataset[1:, :] # remove the row of zeroes
np.random.shuffle(whole_dataset)

# Data preprocessing
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
    print('Accuracy: ' + str(accuracy))
    # print(confusion_matrix(labels_test, labels_pred))
    # print(classification_report(labels_test, labels_pred))
    scores.append(accuracy)

k_values_tried = list(range(1, 16))
plt.plot(k_values_tried, scores, 'ro')
plt.xlabel('Values of K tried')
plt.ylabel('Accuracy')
plt.title('Accuracy for different values of K')
plt.show()

'''
test_image = cv2.imread('0.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from livelossplot import PlotLossesKeras
import numpy as np
import cv2
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import helper as hlp
# from IPython.display import display
# from PIL import Image
from helper import get_cm_string

# Keras VGG
# from keras.applications import vgg16
# classifier = vgg16.VGG16(weights='imagenet')


# Disable gpu - put this at the very top before keras import


image_dim = (224, 224)

train_dir = 'C:/Users/geoff/PycharmProjects/scene-recognition-models/training'

keras.optimizers.Adam(lr=0.0025, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)


train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   validation_split=0.10,
                                   featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   fill_mode='reflect',
                                   zoom_range=[0.5, 1.5],
                                   rotation_range=10
                                   )
test_datagen = ImageDataGenerator(rescale=1./255)

img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
im = np.reshape(cv2.resize(img, (224, 224)), (224, 224, 1, 1))

training_data_for_fitting = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_dim,
    batch_size=1000,
    color_mode='grayscale',
    class_mode='sparse',
    subset='training'
)

training_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_dim,
    batch_size=64,
    color_mode='grayscale',
    class_mode='sparse',
    subset='training'
)
train_datagen.fit(training_data_for_fitting[0][0])

validation_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_dim,
    batch_size=64,
    color_mode='grayscale',
    class_mode='sparse',
    subset='validation'
)

#classifier architecture

classifier = Sequential()

classifier.add(Conv2D(48, kernel_size=(11, 11), strides=(4, 4), input_shape=(image_dim[0], image_dim[1], 1), use_bias=False))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))

classifier.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

classifier.add(Conv2D(128, kernel_size=(5, 5), use_bias=False))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))

classifier.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

classifier.add(Conv2D(192, kernel_size=(3, 3), use_bias=False))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))

classifier.add(Conv2D(192, kernel_size=(3, 3), use_bias=False))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))

classifier.add(Conv2D(128, kernel_size=(3, 3), use_bias=False))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))

classifier.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=2048, use_bias=False))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))
classifier.add(Dropout(0.25))

classifier.add(Dense(units=1024, use_bias=False))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))

classifier.add(Dropout(0.25))

classifier.add(Dense(units=15, use_bias=False))
classifier.add(BatchNormalization())
classifier.add(Activation("softmax"))


classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


#training sequence
classifier.fit_generator(
    training_data,
    steps_per_epoch=training_data.samples / training_data.batch_size,
    epochs=200,
    validation_data=validation_data,
    validation_steps=validation_data.samples / validation_data.batch_size,
    callbacks=[PlotLossesKeras()]
    )


validation_data.reset()
predictions = classifier.predict_generator(validation_data, verbose=1, steps=validation_data.samples / validation_data.batch_size)

predicted_class_indices = np.argmax(predictions, axis=1)
ground_truth_class_indices = validation_data.classes

cm = confusion_matrix(ground_truth_class_indices, predicted_class_indices)
#hlp.display_matrix(cm)
cm_string = get_cm_string(cm, labels=validation_data.class_indices.keys())
print(cm_string)

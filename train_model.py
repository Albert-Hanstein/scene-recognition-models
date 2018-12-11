
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from livelossplot import PlotLossesKeras
import numpy as np
from sklearn.metrics import confusion_matrix
# from IPython.display import display
# from PIL import Image
# import helper as hlp
from helper import get_cm_string

image_dim = (64, 64)

train_dir = '/home/geoffrey893/PycharmProjects/scene-recognition-models/training'

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

training_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_dim,
    batch_size=10,
    color_mode='grayscale',
    class_mode='sparse',
    subset='training'
)

validation_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_dim,
    batch_size=10,
    color_mode='grayscale',
    class_mode='sparse',
    subset='validation'
)

# test_set = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(200, 200),
#     batch_size=10,
#     color_mode='grayscale',
#     class_mode='sparse'
# )

classifier = Sequential()

classifier.add(Conv2D(32, kernel_size=(5, 5), input_shape=(image_dim[0], image_dim[1], 1), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=64, activation='relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(output_dim=32, activation='relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(output_dim=15, activation='softmax'))

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classifier.fit_generator(
    training_gen,
    steps_per_epoch=training_gen.samples/training_gen.batch_size,
    epochs=300,
    validation_data=validation_gen,
    validation_steps=validation_gen.samples/validation_gen.batch_size,
    callbacks=[PlotLossesKeras()]
    )


validation_gen.reset()
predictions = classifier.predict_generator(validation_gen, verbose=1, steps=validation_gen.samples/validation_gen.batch_size)

predicted_class_indices = np.argmax(predictions, axis=1)
ground_truth_class_indices = validation_gen.classes

cm = confusion_matrix(ground_truth_class_indices, predicted_class_indices)
cm_string = get_cm_string(cm, labels=validation_gen.class_indices.keys())
print(cm_string)

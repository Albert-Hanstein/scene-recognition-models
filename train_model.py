import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image
import helper as hlp


train_dir = '/home/geoffrey893/PycharmProjects/scene-recognition-models/training'
test_dir = '/home/geoffrey893/PycharmProjects/scene-recognition-models/testing'

keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=100,
    color_mode='grayscale',
    class_mode='sparse'
)
test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(200, 200),
    batch_size=10,
    color_mode='grayscale',
    class_mode='sparse'
)

classifier = Sequential()

classifier.add(Convolution2D(16, 3, 3, input_shape=(200, 200, 1), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(8, 5, 5, input_shape=(100, 100, 1), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(8, 5, 5, input_shape=(50, 50, 1), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=64, activation='relu'))

classifier.add(Dense(output_dim=16, activation='relu'))

classifier.add(Dense(output_dim=1, activation='relu'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit_generator(
    training_set,
    steps_per_epoch=500,
    epochs=100,
    validation_data=test_set,
    validation_steps=800)




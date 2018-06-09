
# coding: utf-8
# Model Building and Weight

# import the required libraries
import numpy as np
import pandas as pd


# import the keras libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D,Dropout
from keras.layers import Flatten
from keras.layers import Dense
from PIL import Image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

# model archeticture desgin

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation ='relu'))
classifier.add(Dropout(0.1))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(64, 3, 3, input_shape = (150, 150, 3), activation ='relu'))
classifier.add(Dropout(0.1))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(128, 3, 3, input_shape = (150, 150, 3), activation ='relu'))
classifier.add(Dropout(0.1))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# the model so far outputs 3D feature maps (height, width, features)

# Flatten out the 3D feature into 1D feature vectors

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim = 6, activation = 'softmax'))

# Put the output dimenssion as the number of class variables
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


# this is a generator that will read pictures found in
# subfolers of './data/train_61326', and indefinitely generate
# batches of augmented image data
training_set = train_datagen.flow_from_directory(
        './data/train_61326',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = train_datagen.flow_from_directory(
        './data/validation_61326',  
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='categorical')  # since we use Categorical_crossentropy loss, we need categorical labels

# Fit the model and generate the classifier
classifier.fit_generator(training_set,
                         steps_per_epoch=16,
                         epochs=10,
                         validation_data=validation_generator,
                         validation_steps=5)


# serialize weights to HDF5 and save them for future references
classifier.save_weights("modelConv2d.h5")
classifier.save('model.h5')
print("Saved model to disk")


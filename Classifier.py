import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np


class Clasifier:
    def __init__(self):
        classifier = Sequential()

        # Step 1 - Convolution
        classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

        # Step 2 - Pooling
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Adding a second convolutional layer
        classifier.add(Convolution2D(32, 3, 3, activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Step 3 - Flattening
        classifier.add(Flatten())

        # Step 4 - Full connection
        classifier.add(Dense(output_dim=128, activation='relu'))
        classifier.add(Dense(output_dim=1, activation='sigmoid'))

        # Compiling the CNN
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

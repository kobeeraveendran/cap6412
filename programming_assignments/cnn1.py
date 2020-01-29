import tensorflow as tf
from tensorflow import keras

from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.models import Sequential
from keras.optimizers import Adam
from utils import load_data

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

import numpy as np
import time

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

if __name__ == "__main__":

    # load train and test set
    train, test = load_data()

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2)))
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2)))
    model.add(ZeroPadding2D())

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2)))
    model.add(ZeroPadding2D())

    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(200, activation = 'softmax'))

    model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    start = time.time()
    model.fit_generator(train, epochs = 20, verbose = 2)
    training_time = time.time() - start

    print('\n\nTotal training time taken (mins:sec): {0}:{1:.2f}\n\n'.format(int(training_time//60), training_time % 60))

    preds = model.predict(test, verbose = 1)

    pred_class_indices = np.argmax(preds, axis = 1)

    print(pred_class_indices)
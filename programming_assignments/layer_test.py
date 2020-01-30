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

def train_test(model, train, test_set, test_labels, label_map):

    model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    start = time.time()
    model.fit_generator(train, epochs = 20, verbose = 1)
    training_time = time.time() - start

    print('\n\nTotal training time taken (mins:sec): {0}:{1:.2f}\n\n'.format(int(training_time // 60), training_time % 60))

    preds = model.predict(test_set, verbose = 1)

    print('Predicted class indices:\n')
    pred_class_indices = np.argmax(preds, axis = 1)
    print(pred_class_indices)

    print('Num predictions: ', len(pred_class_indices))

    test_acc = 0
    num_test = len(pred_class_indices)

    for i in range(num_test):
        if label_map[pred_class_indices[i]] == test_labels[i]:
            test_acc += 1

    print('\nTest set accuracy: {0:.2f}'.format(test_acc / num_test * 100))

if __name__ == "__main__":

    # load train and test set
    train, test, test_labels = load_data()

    label_map = {v:k for k, v in train.class_indices.items()}

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

    train_test(model, train, test, test_labels, label_map)

    '''
    model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    start = time.time()
    model.fit_generator(train, epochs = 20, verbose = 1)
    training_time = time.time() - start

    print('\n\nTotal training time taken (mins:sec): {0}:{1:.2f}\n\n'.format(int(training_time//60), training_time % 60))

    preds = model.predict(test, verbose = 1)

    pred_class_indices = np.argmax(preds, axis = 1)

    print(pred_class_indices)
    '''
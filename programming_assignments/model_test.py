# Kobee Raveendran
# University of Central Florida
# CAP6412 - Programming Assignment 1

import tensorflow as tf
from tensorflow import keras

from keras.layers import Input, Dense, Flatten, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.metrics import accuracy

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.utils import plot_model

from keras import backend as K

from utils import load_data_limited

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

import numpy as np
import time

import argparse
import os
import sys

# TODO: add more documentation for functions, clarity

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

def train_model(model, train_x, train_y, batch_size = 64):

    model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    start = time.time()
    model.fit(train_x, train_y, epochs = 15, verbose = 1, batch_size = batch_size, validation_split = 0.1)
    training_time = time.time() - start

    print('\n\nTotal training time taken (mins:sec): {0}:{1:.2f}\n\n'.format(int(training_time // 60), training_time % 60))

    return model

def predict(model, test_set, test_labels):

    print('\n\nPredicting on test set...')
    preds = model.predict(test_set, verbose = 1)

    print('\nPredicted Class Indices: \n')
    pred_class_indices = np.argmax(preds, axis = 1)
    print(pred_class_indices)

    num_test = len(pred_class_indices)

    print('Num predictions: ', num_test)

    test_acc = accuracy(np.argmax(test_labels, axis = 1), pred_class_indices)

    print('\nTest set accuracy: {0:.2f}%'.format(np.sum(test_acc) / num_test * 100))

# ~21% test set accuracy
def model_3conv(train_x, train_y, test_x, test_y):

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(200, activation = 'softmax'))

    print(model.summary())
        
    print('\n\n3 Conv layer model')
    trained_model = train_model(model, train_x, train_y, batch_size = 256)
    print('\n3 Conv Layer model predictions:')
    predict(trained_model, test_x, test_y)

    #plot_model(model, to_file = 'figs/3conv_model.png')

# ~31% test set accuracy
def model_6conv(train_x, train_y, test_x, test_y):

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(256, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(256, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(200, activation = 'softmax'))

    print(model.summary())

    print('\n\n6 Conv layer model')
    trained_model = train_model(model, train_x, train_y, batch_size = 256)
    print('\n6 Conv Layer model predictions:')
    predict(trained_model, test_x, test_y)

    #plot_model(model, to_file = 'figs/6conv_model.png')


def model_9conv(train_x, train_y, test_x, test_y):

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(256, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(512, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(1024, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(512, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(256, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D())

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(200, activation = 'softmax'))

    print(model.summary())

    print('\n\n9 Conv layer model')
    trained_model = train_model(model, train_x, train_y, batch_size = 64)
    print('\n9 Conv Layer model predictions:')
    predict(trained_model, test_x, test_y)

    #plot_model(model, to_file = 'figs/9conv_model.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, default = 'conv3', 
                        help = "Which model to train. Options are 'conv3', 'conv6', 'conv9' or 'finetuned'")
    parser.add_argument('--samples_per_class', type = int, default = 500, 
                        help = "Number of training samples to gather from each class; Default = 500")

    args = parser.parse_args()

    samples_per_class = args.samples_per_class
    model_choice = args.model

    if model_choice not in ['conv3', 'conv6', 'conv9', 'finetuned']:
        print('Invalid model choice. Exiting...')
        sys.exit()

    print('\nConfig:')
    print('Samples per class: {}\nModel: {}'.format(samples_per_class, model_choice))

    # load train and test set
    print('\n\nLoading training and test sets...\n\n')

    if model_choice != 'finetuned':
        train_x, train_y, test_x, test_y = load_data_limited(num_samples = samples_per_class, finetune = False)
    else:
        train_x, train_y, test_x, test_y = load_data_limited(finetune = True)
    
    train_y = keras.utils.to_categorical(train_y)
    test_y = keras.utils.to_categorical(test_y)

    print('\n\n\nTrain_y shape: \n', train_y.shape)
    print('Test_y shape: \n', test_y.shape)


    os.makedirs('figs', exist_ok = True)

    if model_choice == 'conv3':


        model_3conv(train_x, train_y, test_x, test_y)

    elif model_choice == 'conv6':

        model_6conv(train_x, train_y, test_x, test_y)

    elif model_choice == 'conv9':

        model_9conv(train_x, train_y, test_x, test_y)

    else:
        # finetuned model - test set accuracy on SVHN: ~94%
        # based on the sample finetuning code provided by keras at
        # https://keras.io/applications/
        base_model = VGG16(weights = 'imagenet', include_top = False, input_tensor = Input(shape = (128, 128, 3)))

        print(base_model.summary())

        new_top = base_model.output
        new_top = GlobalAveragePooling2D()(new_top)
        new_top = Dense(512, activation = 'relu')(new_top)
        new_top = Dropout(0.5)(new_top)
        new_top = Dense(10, activation = 'softmax')(new_top)

        model = Model(inputs = base_model.input, outputs = new_top)

        # freeze vgg16 layers to train new output layers
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer = SGD(learning_rate = 1e-4, momentum = 0.9), 
                      loss = 'categorical_crossentropy', 
                      metrics = ['accuracy'])

        print('Training new output layers...')

        start = time.time()
        model.fit(train_x, train_y, batch_size = 64, epochs = 5, validation_split = 0.2)
        new_output_train_duration = time.time() - start

        print('Training time for re-initializing output: {0}:{1:.2f}'.format(int(new_output_train_duration // 60), new_output_train_duration % 60))

        for layer in base_model.layers[:15]:
            layer.trainable = True

        # train with both the new output layers and the early conv layers of VGG
        model.compile(optimizer = SGD(learning_rate = 1e-4, momentum = 0.9), 
                      loss = 'categorical_crossentropy', 
                      metrics = ['accuracy'])
        
        start2 = time.time()
        model.fit(train_x, train_y, batch_size = 64, epochs = 10, validation_split = 0.2)
        end = time.time()

        print(model.summary())

        print('Training time for output + retrained conv layers: {0}:{1:.2f}'.format(int((end - start2) // 60), (end - start2) % 60))

        print('Total training time for finetuned VGG16: {0}:{1:.2f}'.format(int((end - start) // 60), (end - start) % 60))

        predict(model, test_x, test_y)
        
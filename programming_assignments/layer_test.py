import tensorflow as tf
from tensorflow import keras

from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import accuracy

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras import backend as K

from utils import load_data, load_data_limited

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

import numpy as np
import time

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

def train_model(model, train_x, train_y):

    model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    start = time.time()
    model.fit(train_x, train_y, epochs = 15, verbose = 1, batch_size = 128)
    training_time = time.time() - start

    print('\n\nTotal training time taken (mins:sec): {0}:{1:.2f}\n\n'.format(int(training_time // 60), training_time % 60))

    return model

def predict(model, test_set, test_labels):

    preds = model.predict(test_set, verbose = 1)

    print('Predicted Class Indices: \n')
    pred_class_indices = np.argmax(preds, axis = 1)
    print(pred_class_indices)

    num_test = len(pred_class_indices)

    print('Num predictions: ', num_test)

    test_acc = accuracy(np.argmax(test_labels, axis = 1), pred_class_indices)

    print('\nTest set accuracy: {}'.format(np.sum(test_acc) / num_test))


def model_3conv():

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

    return model

def model_6conv():

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(256, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(256, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(200, activation = 'softmax'))

    return model

def model_12conv():

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(256, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(512, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(1024, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(2048, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(1024, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(512, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(256, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(strides = (2, 2), padding = 'same'))
    model.add(ZeroPadding2D())

    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(200, activation = 'softmax'))

    return model

if __name__ == "__main__":

    # load train and test set
    train_x, train_y, test_x, test_y = load_data_limited(500)

    train_y = keras.utils.to_categorical(train_y)
    test_y = keras.utils.to_categorical(test_y)

    #label_map = {v:k for k, v in train.class_indices.items()}

    # 3 convolutional blocks
    '''
    conv3 = model_3conv()
    trained_model = train_model(conv3, train_x, train_y)
    predict(trained_model, test_x, test_y)
    '''

    # 6 convolutional blocks
    '''
    conv6 = model_6conv()
    trained_model = train_model(conv6, train_x, train_y)
    predict(trained_model, test_x, test_y)
    '''

    # 12 convolutional blocks
    conv12 = model_12conv()
    trained_model = train_model(conv12, train_x, train_y)
    predict(trained_model, test_x, test_y)
    
    # InceptionV3 pretrained model fine-tuning

    '''
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor = Input(shape = (128, 128, 3)))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(200, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])

    # train the model on the new data for a few epochs
    model.fit(train_x, train_y, epochs = 5, verbose = 1, batch_size = 128)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit(train_x, train_y, epochs = 10, batch_size = 128, verbose = 1)

    predict(model, test_x, test_y)
    '''
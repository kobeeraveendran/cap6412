import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import os
import cv2

#from tf.compat.v1 import ConfigProto

def load_data():

    datagen = ImageDataGenerator(rescale = 1. / 255)

    # load training data with associated classes from directory structure
    train_gen = datagen.flow_from_directory('tiny-imagenet-200/train', target_size = (64, 64), batch_size = 128, class_mode = 'categorical')
    
    # load test images
    test_dir = os.path.join(os.getcwd(), 'tiny-imagenet-200/val/images/')
    test_set = []
    
    for sample in os.listdir(test_dir):
        img = cv2.imread(test_dir + sample)
        test_set.append(np.asarray(img))

    print(test_set[0])

    test_set = np.array(test_set)

    test_labels = []
    file = open('tiny-imagenet-200/val/val_annotations.txt', 'r')

    for line in file:
        img_label = line.split('\t')[1]
        test_labels.append(img_label)

    file.close()

    #test_gen = datagen.flow_from_directory('tiny-imagenet-200/val/images', target_size = (64, 64))

    return train_gen, test_set, test_labels

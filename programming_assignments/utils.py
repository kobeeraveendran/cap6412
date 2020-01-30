import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import os
import cv2
import re
import imageio
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#from tf.compat.v1 import ConfigProto

def load_data():

    datagen = ImageDataGenerator(rescale = 1. / 255)

    # load training data with associated classes from directory structure

    # for inception
    train_gen = datagen.flow_from_directory('tiny-imagenet-200/train', target_size = (299, 299), batch_size = 128, class_mode = 'categorical')
    
    # custom
    #train_gen = datagen.flow_from_directory('tiny-imagenet-200/train', target_size = (64, 64), batch_size = 128, class_mode = 'categorical')
    
    # load test images
    test_dir = os.path.join(os.getcwd(), 'tiny-imagenet-200/val/images/')
    test_set = []

    i = 0
    
    im_dir = sorted(os.listdir(test_dir), key = lambda f: int(re.sub(r'\D', '', f)))

    for sample in im_dir:
        
        # inception
        img = cv2.resize(cv2.imread(test_dir + sample), (299, 299))

        # custom
        #img = cv2.imread(test_dir + sample)
        test_set.append(np.asarray(img))

    #print(test_set[0])

    test_set = np.array(test_set)

    test_labels = []
    file = open('tiny-imagenet-200/val/val_annotations.txt', 'r')

    for line in file:
        img_label = line.split('\t')[1]
        test_labels.append(train_gen.class_indices[img_label])

    file.close()
    test_labels = np.array(test_labels)

    #test_gen = datagen.flow_from_directory('tiny-imagenet-200/val/images', target_size = (64, 64))

    return train_gen, test_set, test_labels

def load_data_limited(num_samples):

    # load training set

    train_x = np.zeros(shape = (200 * num_samples, 64, 64, 3), dtype = 'uint8')
    train_y = np.zeros(shape = (200 * num_samples), dtype = 'uint8')

    train_dir_path = 'tiny-imagenet-200/train/'

    label_to_class_index = {}

    sample_num = 0
    for i, subdir in enumerate(os.listdir(train_dir_path)):
        imgs_path = os.path.join(train_dir_path, subdir, 'images')
        label_to_class_index[subdir] = i

        for img in os.listdir(imgs_path):
            sample = cv2.imread(os.path.join(imgs_path, img))
            cvt_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
            sample = np.asarray(cvt_sample)

            train_x[sample_num] = sample
            train_y[sample_num] = i
            sample_num += 1

    # load test set (using val instead)

    test_x = np.zeros(shape = (200 * 50, 64, 64, 3), dtype = 'uint8')
    test_y = np.zeros(shape = (200 * 50), dtype = 'uint8')

    test_dir_path = 'tiny-imagenet-200/val/images'

    file = open('tiny-imagenet-200/val/val_annotations.txt', 'r')

    test_label_mapping = {}
    for line in file:
        sample_label = line.split('\t')
        file_name = sample_label[0]
        class_label = sample_label[1]
        test_label_mapping[file_name] = class_label

    file.close()

    sample_num = 0
    for i, img in enumerate(os.listdir(test_dir_path)):
        sample = cv2.imread(os.path.join(test_dir_path, img))
        cvt_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        sample = np.asarray(cvt_sample)

        test_x[sample_num] = sample
        test_y[sample_num] = int(label_to_class_index[test_label_mapping[img]])

        sample_num += 1

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)


    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    #train, test, test_labels = load_data()

    train_x, train_y, test_x, test_y = load_data_limited(num_samples = 500)

    print(np.shape(train_x))
    print(np.shape(train_y))
    print(np.shape(test_x))
    print(np.shape(test_y))
    print('\n\n')

    print('Label for sample 19: ', test_y[19])
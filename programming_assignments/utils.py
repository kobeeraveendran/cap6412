import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import os
import cv2
import re
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.io import loadmat
from skimage.transform import resize

#from tf.compat.v1 import ConfigProto

def load_data_limited(num_samples = 500, finetune = False):

    # load training set

    if finetune:
        training_data = loadmat('svhn/train_32x32.mat')
        test_data = loadmat('svhn/test_32x32.mat')

        train_x = np.zeros(shape = (73257, 128, 128, 3), dtype = 'uint8')

        train_x_small = np.transpose(training_data['X'], (3, 0, 1, 2))

        for i in range(len(train_x)):
            if i == 0:
                print(train_x[i].shape)
            train_x[i] = cv2.resize(train_x_small[i], (128, 128))
        
        train_y = training_data['y']

        #print('Unique classes (train): ', np.unique(train_y))
        
        train_y = np.array([x[0] for x in train_y])
        train_y[train_y == 10] = 0

        test_x = np.zeros(shape = (26032, 128, 128, 3), dtype = 'uint8')

        test_x_small = np.transpose(test_data['X'], (3, 0, 1, 2))

        for i in range(len(test_x)):
            test_x[i] = cv2.resize(test_x_small[i], (128, 128))

        test_y = test_data['y']
        test_y = np.array([x[0] for x in test_y])

        test_y[test_y == 10] = 0

        #print('Unique classes (test): ', np.unique(test_y))

    else:
        train_x = np.zeros(shape = (200 * num_samples, 64, 64, 3), dtype = 'uint8')

        train_y = np.zeros(shape = (200 * num_samples), dtype = 'uint8')

        train_dir_path = 'tiny-imagenet-200/train/'

        label_to_class_index = {}

        sample_num = 0
        for i, subdir in enumerate(os.listdir(train_dir_path)):
            imgs_path = os.path.join(train_dir_path, subdir, 'images')
            label_to_class_index[subdir] = i

            for img in os.listdir(imgs_path)[:num_samples]:
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
            
            #cvt_sample = cv2.resize(cvt_sample, (128, 128))

            sample = np.asarray(cvt_sample)

            test_x[sample_num] = sample
            test_y[sample_num] = int(label_to_class_index[test_label_mapping[img]])

            sample_num += 1

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)


    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    #train, test, test_labels = load_data()

    train_x, train_y, test_x, test_y = load_data_limited(num_samples = 500, finetune = False)

    print(np.shape(train_x))
    print(np.shape(train_y))
    print(np.shape(test_x))
    print(np.shape(test_y))
    print('\n\n')

    print(type(test_y))
    print(test_y[:5])

    print('Sample 0 image: ')

    plt.imshow(test_x[0, :, :, :])
    plt.show()

    print('Label for sample 0: ', test_y[0])
import keras
from keras.preprocessing.image import load_img
import os

if __name__ == '__main__':
    train = []
    test = []

    train_path = os.path.join(os.getcwd(), 'tiny-imagenet-200/train/')
    
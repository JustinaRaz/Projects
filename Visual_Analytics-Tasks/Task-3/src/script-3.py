import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

import functions


def main():
    args = functions.file_loader()
    data = args.data
    epochs_num = args.epochs

    main_folder_path = "in/{}/".format(data)

    labels_n = functions.get_labels(main_folder_path)
    print("Currently preprocessing the images...")
    prep_label_names = functions.get_preprocessed_imgs(labels_n, main_folder_path)
    print("Image preprocessing is done!")

    # Prepping images
    prepped_imgs = prep_label_names[0]
    full_labels = prep_label_names[2]


    print("Fitting a model... This could take a while.")
    hist = functions.fit_model(prepped_imgs, full_labels, epochs_num, labels_n)

    functions.plot_history(hist, epochs_num)

if __name__ == "__main__":
    main()

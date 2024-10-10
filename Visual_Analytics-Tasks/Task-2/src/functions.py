import os
import sys
import cv2
sys.path.append("..")
import pandas as pd
import numpy as np
import utils.classifier_utils as clf_util
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from joblib import dump, load
from tensorflow.keras.datasets import cifar10
import functions

def rename_labels(data):
    
    """
    A function that renames the labels.

    Input:
        - data - Cifar10 dataset, which contains data for labels, namely, numbers. Based on the information in the data documentation page, this code translates numbers into the actual labels - to what those numbers mean.

    Returns:
        - renamed_data - a dataset which contains strings as labels, not numbers. 
    """
    label_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'}

    # Converting numbers to words:
    label_array = np.vectorize(label_names.get)(data.flatten())

    # Reshaping the label array, to match the original one:
    renamed_data = label_array.reshape(data.shape)

    return renamed_data


# Function that performs preprocessing: 
def preprocessing(train, test):

    """
    A function that performs preprocessing on the images. It grayscales, normalizes, and reshapes the array.

    Input:
        - train - dataset which contains a list of array of numbers for training split.
        - test - dataset which contains a list of array of numbers for testing split.
    """
    grayscale_image_train = []
    grayscale_image_test = []

    for image in train:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_image_train.append(image)

    grayscale_image_train = np.array(grayscale_image_train)

    train_scaled = grayscale_image_train/255.0

    X_train_scaled = train_scaled.reshape(-1, 1024)

    for image in test:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_image_test.append(image)

    grayscale_image_test = np.array(grayscale_image_test)

    test_scaled = grayscale_image_test/255.0

    X_test_scaled = test_scaled.reshape(-1, 1024)

    preprocessed_arrays = (X_train_scaled, X_test_scaled)

    return preprocessed_arrays


def log_clf(X_train_scaled, y_train):

    """
    A function that creates a logistic regression classifier.
    """
    clf = LogisticRegression(tol=0.1, # tolerance for stopping criteria
                         solver='saga', # algorithm to use in the optimisation problem; 'saga' is for big data sets.
                         multi_class= 'multinomial', # better performance when dealing with larger number of classes.
                         random_state = 12).fit(X_train_scaled, y_train)
    return clf


def neural_clf(X_train_scaled, y_train):

    """
    A function that creates a neural network classifier.
    """
    clf = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (40,),
                           max_iter=1000,
                           random_state = 12).fit(X_train_scaled, y_train)
    return clf


def save_loss_curve(neural_clf):

    """
    A function which is designed to save a neural network classifier's loss curve.
    """
    plt.plot(neural_clf.loss_curve_)
    plt.title("Loss curve during training", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')
    plt.savefig("out/loss_curve.png") 
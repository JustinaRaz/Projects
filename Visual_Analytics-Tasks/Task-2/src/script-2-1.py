# Importing the modules

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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from joblib import dump, load
from tensorflow.keras.datasets import cifar10
import functions

def main():
    # Loading the data 
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Renaming the labels
    y_train = functions.rename_labels(y_train)
    y_test = functions.rename_labels(y_test)

    # Preprocessing
    preprocessing_data = functions.preprocessing(X_train, X_test)
    X_train_s = preprocessing_data[0]
    X_test_s = preprocessing_data[1]

    # Training the classifier on the actual data 
    classifier = functions.log_clf(X_train_s, y_train)

    # Predicting the labels
    y_pred = classifier.predict(X_test_s)

    # Calculating classification metrics
    clf_metrics = metrics.classification_report(y_test, y_pred, output_dict=True)

    # Saving the classification metrics
    report = pd.DataFrame(clf_metrics)
    save_path = 'out/log_reg_metrics.csv'
    report.to_csv(save_path, index=True)

if __name__ == "__main__":
    main()
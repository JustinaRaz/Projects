import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from joblib import dump, load
import functions
from codecarbon import EmissionsTracker


def main():
    tracker = EmissionsTracker(output_file = "emissions_assignment_2.csv", output_dir = "out/emissions/", project_name = "logistic_classification")
    tracker.start()

    tracker.start_task("load_data")
    # Index for saving logistic regression's vectorizer:
    ind = "logistic"
    # load the data:
    filename = os.path.join("in","fake_or_real_news.csv")
    data = pd.read_csv(filename, index_col=0)
    tracker.stop_task("load_data")

    tracker.start_task("split_logistic")
    splits = functions.get_splits(data)
    train_X = splits[0]
    test_X = splits[1]
    train_y = splits[2]
    test_y = splits[3]
    tracker.stop_task("split_logistic")

    tracker.start_task("get_features")
    features = functions.get_features(train_X, test_X, ind)
    train_X_features = features[0]
    test_X_features = features[1]
    tracker.stop_task("get_features")

    tracker.start_task("classification_logistic")
    functions.logistic_classification(train_X_features, train_y, test_X_features, test_y)
    tracker.stop_task("classification_logistic")

    tracker.stop()

if __name__ == "__main__":
    main()

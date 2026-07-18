import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from joblib import dump, load

def get_splits(data_frame):

    """
    This function is used to get the Train and Test splits for the data frame.

    Input:
        A data frame.

    Returns:
        Output is a tuple, which can be indexed to receive the following:
        [0] - data content for training;
        [1] - data content for testing;
        [2] - data labels for training;
        [3] - data labels for testing.
    """

    X = data_frame["text"] # data content - column in a data frame called "text".
    y = data_frame["label"] # data content's labels - column in a data frame called "label".

    # Train - test split:

    X_train, X_test, y_train, y_test = train_test_split(X,          # texts for the model
                                                        y,          # labels to clasify
                                                        test_size=0.2,   # create an 80/20 split
                                                        random_state= 12) # "seed" for reproducibility
    output = (X_train, X_test, y_train, y_test)

    return output


def get_features(training_content, testing_content, index):

    """
    A function that takes the text and tuns it into a numerical representations - vectorizes.

    Input:
        training_content - a variable that contains data content for training.
        testing_content - a variable that contains data content for testing.
        index - an index, which represents the type of classification ("logistic" or "neural")

    Returns:
        The function returns 2 variables in a tuple, and when indexed, these represent:
        [0] - a matrix of numbers to represent training dataset's features.
        [1] - a matrix of numbers to represent testing dataset's features.
    """

    # Vectorizing (turning the data into the numerical representations) and feature extraction:

    # creating a vectorized object:
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams (1 word and 2 word units)
                                lowercase =  True,       
                                max_df = 0.95,           # removing commong words
                                min_df = 0.05,           # removing rare words
                                max_features = 250)      # the maximum amount of features to keep

    vect_path = "out/models/tfidf_{}_vectorizer.joblib".format(index)
    dump(vectorizer, vect_path)

    # Fittint to the training data:
    X_train_features = vectorizer.fit_transform(training_content) # creates a matrix of numbers

    # Fitting to the test data:
    X_test_features = vectorizer.transform(testing_content) # creates a matrix of numbers

    output = (X_train_features, X_test_features)

    return output

def logistic_classification(training_features, training_labels, testing_features, testing_labels):

    """
    A function that trains a logistic classifier, saves the classifier and classification report.

    Input:
        training_features - a matrix which represents training dataset's features
        training_labels - data labels for training
        testing_features - a matrix which represents testing dataset's features
        testing_labels - data labels for testing
    Returns:
        The function returns 2 csv files in a folder out -> models.
    """

    logistic_classifier = LogisticRegression(random_state=12).fit(training_features, training_labels)

    #Saving the classifier
    class_path = "out/models/log_reg_classifier.joblib"
    dump(logistic_classifier, class_path)

    y_pred = logistic_classifier.predict(testing_features)
    logistic_classifier_metrics = metrics.classification_report(testing_labels, y_pred, output_dict=True)

    #Creating a report and saving it
    report = pd.DataFrame(logistic_classifier_metrics)
    save_path = "out/log_reg_report.csv"
    report.to_csv(save_path, index=True)    

def neural_classification(training_features, training_labels, testing_features, testing_labels):

    """
    A function that trains a neural classifier, saves the classifier and classification report.

    Input:
        training_features - a matrix which represents training dataset's features
        training_labels - data labels for training
        testing_features - a matrix which represents testing dataset's features
        testing_labels - data labels for testing
    Returns:
        The function returns 2 csv files in a folder out -> models.
    """

    neural_classifier = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (20,),
                           max_iter=1000,
                           random_state = 12).fit(training_features, training_labels)

    #Saving the classifier
    class_path = "out/models/neural_classifier.joblib"
    dump(neural_classifier, class_path)

    y_pred = neural_classifier.predict(testing_features)
    neural_classifier_metrics = metrics.classification_report(testing_labels, y_pred, output_dict=True)

    #Creating a report and saving it
    report = pd.DataFrame(neural_classifier_metrics)
    save_path = "out/neural_report.csv"
    report.to_csv(save_path, index=True) 
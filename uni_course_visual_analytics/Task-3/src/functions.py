import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

def file_loader():

    """
    A function that allows the user to enter specific information on what the function should look for.
    In this case, it requires the user to write the name of the dataset folder.

    Returns:
        args.data - will contain user's input for the dataset.
        args.epochs - number of epochs for the model.
    """

    parser = argparse.ArgumentParser(description = "Image classification")
    parser.add_argument("data", 
                    #required=True, 
                    help="Name of your dataset.")

    parser.add_argument("epochs", type=int,
                        #required=True, 
                        help="Number of epochs you want to run the model for.")

    args = parser.parse_args()
    return args

def plot_history(H, epochs):
    """
    A function to plot the loss and accuracy curve.

    Inputs: 
        H - a fit of the model;
        epochs - the number of epochs.
    Returns:
        The output is the image with two curves - loss and accuracy. The plot is saved in the folder called 'out'.
    """
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    
    plt.savefig(os.path.join("out", "curves.png"))
    plt.show()


def get_labels(path):
    """
    A function to get the labels (folder names) for images.

    Input:
        A path to the data folder, which contains multiple subfolders with images.
    
    Returns:
        A list of folder names, which are later treated as a label for an image (label of the image = the folder where the image is).
    """
    image_folders = sorted(os.listdir(path))
    labels = image_folders
    return labels

def get_preprocessed_imgs(labels, main_folder_path):

    """
    A function to preprocess images so that they could be used in a model.

    Input:
        labels - a list of labels for images.
        main_folder_path - a path to the data folder.

    Returns:
        the function returns a tupe, which can be indexed to extract values such as:
        [0] - a numerical representation - an array of pixels - for each image.
        [1] - 'grand' list of paths to all images in a data set.
        [2] - 'grand' label list; a list of labels to all 3000+ images.
    """

    image_labels = []
    path_to_images = []
    prepped_images_list = []

    for folder in labels:
        folder_dir = main_folder_path + folder + "/"
        image_names = sorted(os.listdir(folder_dir))
        for image in image_names:                       
            if image[-4:] == ".jpg":                                 # makes sure that only images are kept
                image_path = folder_dir + image                      # creates a path to an image
                image = load_img(image_path, target_size=(224, 224)) # loads the image of the right size for the VGG16 model
                arrayed_image = img_to_array(image)                  # turns the image into an array
                reshaped_image = arrayed_image.reshape((1, arrayed_image.shape[0], arrayed_image.shape[1], arrayed_image.shape[2]))
                prepped_image = preprocess_input(reshaped_image)
                prepped_images_list.append(prepped_image)                 # appends the pixels/array into the list       
                path_to_images.append(image_path)                    # keeps track of the path to the image           
                image_labels.append(folder)                          # keeps track of the image label              
    prepped_Images = np.stack(prepped_images_list, axis=0)
    prepped_images = np.squeeze(prepped_Images, axis=1)

    return (prepped_images, path_to_images, image_labels)



def fit_model(prepped_images, image_labels, eph, labels):
    """
    A function that fits a model and returns it, aditionally, saves the classification report and saves it.

    Input:
         - prepped_images - a variable that contains the arrays of pixels of images
         - image_labels - labels for images
         - eph - a number of epochs, specified by the user
         - labels - a list of unique labels.
    
    Returns:
        the function returns a tupe of 4 variables, which can be indexed and extracted:
        [0] - a subset of 80% of the original data set, which is used for model's training.
        [1] - a subset of 20% of the original data set, which is used for model's testing.
        [2] - respective labels for the training dataset.
        [3] - respective labels for the testing dataset.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(prepped_images, 
                                                        image_labels, 
                                                        random_state=12,
                                                        stratify=image_labels, # Ensures that image classes are proportionally similar in test and train datasets.
                                                        shuffle = True, # Shuffles the data before the split
                                                        test_size=0.2)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    model = VGG16(include_top=False, 
                pooling='avg',
                input_shape=(224, 224, 3))

    for layer in model.layers:
        layer.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu')(flat1)
    output = Dense(10, activation='softmax')(class1)

    model = Model(inputs=model.inputs, 
                outputs=output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    H = model.fit(X_train, y_train, 
                validation_split=0.2,
                batch_size=64,
                epochs= eph,
                verbose=1)
                
    predictions = model.predict(X_test, batch_size=64)
    class_rep = classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labels, 
                                output_dict=True)
    report = pd.DataFrame.from_dict(class_rep)
    report = report.transpose()
    report.to_csv('out/report.csv')

    return H
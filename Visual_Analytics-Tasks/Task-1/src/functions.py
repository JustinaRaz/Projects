import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.join(".."))
from utils.imutils import jimshow
from utils.imutils import jimshow_channel
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from numpy.linalg import norm
from tqdm import tqdm
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, 
                                                  img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors


def file_loader():

    """
    A function that allows the user to enter specific information on what the function should look for.

    Returns:
        User's inputs. 
        args.data - will contain user's input for the dataset
    """

    parser = argparse.ArgumentParser(description = "Finding an artist of user's topic")
    parser.add_argument("folder", 
                    #required=True, 
                    help="Name of your folder with flower images.")

    parser.add_argument("image_num", 
                    #required=True, 
                    help="A number from 0 to 1359, which represents the target image.")

    args = parser.parse_args()
    return args


def get_image(folder, users_pick):

    """
    A function that picks a target image from the data set.

    Input:
        - folder - a path to the folder with images of flowers.
        - users_pick - a number from the user, which represents an image that is picked.

    Returns:
        - A tuple, which the first value is a path to a randomly picked image, and the second value is a list with all directories to
        every image in the dataset folder.
    """

    image_names = sorted(os.listdir(folder))
    
    image_directories = []

    for image_name in image_names:
        image_dir = folder + image_name 
        image_directories.append(image_dir)
    
    random_image = image_directories[users_pick]

    return (random_image, image_directories)


def get_histogram(img):

    """
    A function that takes a loaded image and creates a normalized color histogram for that image.

    Input:
        - An image, loaded with OpenCV

    Returns:
        - A normalized color histogram for that image.
    """

    hist = cv2.calcHist([img], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
    hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)

    return hist


def hist_compare(image_directories, hist):

    """
    A function that takes target image's color histogram and compares it to the color histograms of other images.

    Input:
        - image_directories - a list of all image directories
        - hist - a color histogram of the target image

    Returns:
        - a list of tuples. The second value of a tuple shows the distance of the image (in terms of cv2.HISTCMP_CHISQR metric)
        to the user's picked image.
    """

    image_values = []

    for img in image_directories:
        
        filename = img

        # loading the image:
        img = cv2.imread(img)

        # Ectracting the color histogram
        img_hist = cv2.calcHist([img], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])

        # normalising it
        norm_img = cv2.normalize(img_hist, img_hist, 0, 1.0, cv2.NORM_MINMAX)

        # compare histogram of this looped image to the one of a picked random image
        distance = round(cv2.compareHist(hist, norm_img, cv2.HISTCMP_CHISQR), 2)

        image_values.append((filename, distance))

    return image_values


def pick_second(val):
    """
    A simple function that picks a second value, for instance, in a tuple of two values, and returns the second value.
    """
    return val[1]


def get_similar_images(values):

    """
    A function that produces a list of directories of the 5 images that are most close to the target image.

    Input:
        - values - a list of tuples, where the first value represents the filename/directory, and the second value represents
        the distance (cv2.HISTCMP_CHISQR metric) of the images to the target image.

    Returns:
        - A list of 5 directories - images closest to the target.
    """

    # Creating a data frame out of the list of tuples:
    data = pd.DataFrame(values, 
                        columns=["Directory", "Distance"])

    # Sort the "Distance" values:
    data_sorted = data.sort_values(by = "Distance")

    # Show only the first 5 images, closest to the picked image:
    closest_imgs = data_sorted[1:6]

    # Saving the resulting data frame with 5 closest images:
    outpath = os.path.join("out", "closest_imgs_opencv.csv")
    closest_imgs.to_csv(outpath, index = False)

    # using this function and sorting:
    sorted_image_values = sorted(values, key=pick_second)

    # extracting the first 5 directories to each one of the image:
    dirs_to_imgs = [x[0] for x in sorted_image_values]
    dirs_to_imgs = dirs_to_imgs[1:6]

    return dirs_to_imgs

def get_plots(directories_to_similar_images, target):

    """
    A function that creates plots and saves them into the folder out/similar_images_OpenCV.

    Input:
        - directories_to_similar_images - a list of directories to the 5 images that are similar to a target image.
        - target - a path to the target image.

    Returns:
        - 5 plots. Each plot contains two images, one of which is a target image, and the other - one of the closest images.
        Additional plot is saved to show a target image separately.
    """

    target = cv2.imread(target)
    cv2.imwrite('out/similar_images_OpenCV/target.jpg', target)

    height, width = target.shape[:2]

    s1 = cv2.imread(directories_to_similar_images[0])
    s2 = cv2.imread(directories_to_similar_images[1])
    s3 = cv2.imread(directories_to_similar_images[2])
    s4 = cv2.imread(directories_to_similar_images[3])
    s5 = cv2.imread(directories_to_similar_images[4])

    #Resizing:

    s1_resized = cv2.resize(s1, (width, height))
    s2_resized = cv2.resize(s2, (width, height))
    s3_resized = cv2.resize(s3, (width, height))
    s4_resized = cv2.resize(s4, (width, height))
    s5_resized = cv2.resize(s5, (width, height))

    pair_1 = np.hstack([target, s1_resized])
    cv2.imwrite('out/similar_images_OpenCV/pair_1.jpg', pair_1)

    pair_2 = np.hstack([target, s2_resized])
    cv2.imwrite('out/similar_images_OpenCV/pair_2.jpg', pair_2)

    pair_3 = np.hstack([target, s3_resized])
    cv2.imwrite('out/similar_images_OpenCV/pair_3.jpg', pair_3)

    pair_4 = np.hstack([target, s4_resized])
    cv2.imwrite('out/similar_images_OpenCV/pair_4.jpg', pair_4)

    pair_5 = np.hstack([target, s5_resized])
    cv2.imwrite('out/similar_images_OpenCV/pair_5.jpg', pair_5)


def extract_features(img_path, model):
    """
    Extract features from image data using pretrained model (e.g. VGG16)
    """

    input_shape = (224, 224, 3)
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img, verbose=False)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(features)
    return flattened_features



def get_similar_vgg16_images(filenames, model, users_num):

    """
    A function that saves the plots of closest image, found using VGG16 model. Images are saved to out/similar_images_VGG16 folder.
    Inputs:
        - filenames - a list of directories to all images.
        - model - a VGG16 model
        - users_num - a number which is provided by the user, that helps to find the target image.

    Returns:
        Two plots. One contains 5 similar images, the other - target image.
        It also returns a dataframe with metrics for each image.
    """

    target = cv2.imread(filenames[users_num])
    cv2.imwrite('out/similar_imgs_VGG16/target.jpg', target)

    feature_list = []
    
    for i in tqdm(range(len(filenames))): 
        feature_list.append(extract_features(filenames[i], model))
    
    neighbors = NearestNeighbors(n_neighbors=10, 
                             algorithm='brute', #good for smaller data sets.
                             metric='cosine').fit(feature_list)

    distances, indices = neighbors.kneighbors([feature_list[users_num]])

    idxs = []

    values = []
    
    for i in range(1,6):
        values.append((filenames[indices[0][i]], distances[0][i]))
        idxs.append(indices[0][i])
    
    data = pd.DataFrame(values, 
                        columns=["Directory", "Distance"])
    
    data_sorted = data.sort_values(by = "Distance")

    data_sorted.to_csv("out/similar_images_VGG16.csv", index = False)

    f, axarr = plt.subplots(1,5)

    axarr[0].imshow(mpimg.imread(filenames[idxs[0]]))
    axarr[1].imshow(mpimg.imread(filenames[idxs[1]]))
    axarr[2].imshow(mpimg.imread(filenames[idxs[2]]))
    axarr[3].imshow(mpimg.imread(filenames[idxs[3]]))
    axarr[4].imshow(mpimg.imread(filenames[idxs[4]]))

    plt.savefig("out/similar_images_VGG16/five_images.jpg") 
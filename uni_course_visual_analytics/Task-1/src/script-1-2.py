import os, sys
sys.path.append(os.path.join(".."))
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, 
                                                  img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors
import functions



def main():

    args = functions.file_loader()
    
    folder_name = args.folder
    num = int(args.image_num)
    
    directory = "in/{}/".format(folder_name)
    target_img_get = functions.get_image(directory, num)
    dirs = target_img_get[1]
    target_image = target_img_get[0]
    

    model = VGG16(weights='imagenet', # default
              include_top=False, # no classification layer
              pooling='avg',
              input_shape=(224, 224, 3)) 

    features = functions.extract_features(target_image, model)

    functions.get_similar_vgg16_images(dirs, model, num)
if __name__ == "__main__":
    main()
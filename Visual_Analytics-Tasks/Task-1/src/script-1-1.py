import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.join(".."))
from utils.imutils import jimshow
from utils.imutils import jimshow_channel
import matplotlib.pyplot as plt
import pandas as pd
import functions
import argparse


def main():

    args = functions.file_loader()
    folder_name = args.folder
    num = int(args.image_num)

    directory = "in/{}/".format(folder_name)
    target_img_get = functions.get_image(directory, num)
    target_img = cv2.imread(target_img_get[0])
    histogram = functions.get_histogram(target_img)

    dirs = target_img_get[1]

    img_values = functions.hist_compare(dirs, histogram)

    similar_imgs = functions.get_similar_images(img_values)
    
    functions.get_plots(similar_imgs, target_img_get[0])

if __name__ == "__main__":
    main()
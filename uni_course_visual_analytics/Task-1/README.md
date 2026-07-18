# Assignment 1: Building a simple image search algorithm

This repository contains the code for **Assignment 1** from *Visual Analytics* course at *Aarhus University*.

The task is to pick an image from the dataset and find five images that are most similar to the image that was "randomly" picked. This task is performed by:

- comparison of color histograms using **OpenCV** and computing **cv2.HISTCMP_CHISQR** metric;
- using feature extraction from the images with **VGG16** model.  

This document is structured as follows:

1. **Data and structure** - describes the data source and file structure.
2. **Reproduction instructions** - steps to replicate the analysis.
3. **Output summary** - key points from the outputs.
4. **Ways of improvement** - identifies potential limitations and suggests code enhancements.

## 1. Data and structure

*The 17 Category Flower Dataset* is used in the present assignment. The information about the data and its documentation can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/). Please download the dataset and store it in the folder called ```in```.

The directory structure is as follows:
```
assignment-1-VIS/
├── in/
│   └── flowers/
│       ├── image_0001.jpg
│       ├── ...
│       └── image_1360.jpg
├── out/
│   ├── similar_images_OpenCV/
│   │   ├── pair_1.jpg
│   │   ├── ...
│   │   ├── pair_5.jpg
│   │   └── target.jpg
│   ├── similar_images_VGG16/
│   │   ├── five_images.jpg
│   │   └── target.jpg
│   ├── closest_imgs_opencv.csv
│   └── similar_imgs_VGG16.csv
├── src/
│   ├── utils/
│   ├── functions.py
│   ├── script-1-1.py
│   └── script-1-2.py
├── README.md
├── requirements.txt
└── setup.sh
```
The **script-1-1.py** contains the main code for image comparison with OpenCV, **script-1-2.py** containst the main code for image comparison with VGG16 model. Meanwhile, **functions.py** contains functions utilized by both main scripts.

```Utils``` is a folder with functions, that are predefined by the teacher of the course and can be used for the analysis.

## 2. Reproduction instructions

In order to run both Python scripts smoothly, the following steps should be completed within the terminal:

1. Open the terminal and set the working directory to the folder:

    ```python
    cd your_path_to/assignment-1-VIS
    ```

2. Run the following command to install the required modules and set up the *virtual environment*:

    ```python
    bash setup.sh
    ```
3. Activate the virtual environment by running:

    ```python
    source ./env/bin/activate
    ```
    Before proceeding: if the folder of the dataset is zipped, please unzip by running:

    ```python
    unzip in/flowers.zip
    ```
    Make sure to adjust the name of the folder, if it is named differently.

4. Execute both Python scripts. Remember to specify the folder of your dataset, and a number from 0 to 1359, which represents the index of a target image. 

    Image comparison using *color histograms* with *OpenCV*:
    ```python
    python src/script-1-1.py flowers 16
    ```

    Image comparison using pretrained *VGG16* model:
    ```python
    python src/script-1-2.py flowers 16
    ```
    The code above was executed to produce the current output, which can be found in the folder ```out```. The dataset folder was named **flowers**. Again, if the dataset folder is called differently, enter the right name.

5. After running both scripts, deactivate the virtual environment:

    ```
    deactivate
    ```
    The output files will be stored and found in the folder called ```out```.

## 3. Output summary

Folder ```out/similar_images_OpenCV``` contains the result of the **script-1-1.py** Python script. The first 5 files display a pair of images for a comparison: a target image on the left, and one of the 5 closest images on the right. Additionally, the target image is saved separately. 

Folder ```out/similar_images_VGG16``` contains the result of the **script-1-2.py** Python script. This folder contains 2 files in total - one that displays a target image, and the other that displays 5 most similar images to the target one. 

Both scripts provide the user with a dataset, which contains the information about the directory of the closest images, and a respective metric of distance/similarity to the target image.

It is clear from the output that the use of VGG16 model provides way more accuracy in detecting the most similar images to the target one. It successfully detects the features of the flower category. The comparison of color histograms across the images using OpenCV does not provide as accurate results. It seems that maybe it can find a flower of the same color, however, it fails to find the right category of the flower.  


## 4. Ways of improvement

As with any programming task, there are always ways of how the code or the analysis can be improved. Here are a few points on how the code/analysis could be improved:

- Split the predefined functions into smaller ones, that perform only one task at a time. This would increase the amount of code of the main scripts, however, it would make it easier to figure out at what point the error comes up, if it happens.
- The output images of the script with OpenCV could have been saved into one file, with all similar images displayed at once. This would not only facilitate the comparison of the results, but also result in less storage taken by the files of the repository.

*Note: These scripts were run with 4 CPUs and took only a couple of minutes to produce the output.*
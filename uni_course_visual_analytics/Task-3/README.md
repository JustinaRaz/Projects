# Assignment 3: Document classification using pretrained image embeddings

This repository contains the code for **Assignment 3** from *Visual Analytics* course at *Aarhus University*.

The task is to develop one Python script to train a CNN (Convolutional Neural Network) classifier to predict the type of the document based on its visual features.

This document is structured as follows:

1. **Data and structure** - describes the data source and file structure.
2. **Reproduction instructions** - steps to replicate the analysis.
3. **Output summary** - key points from the outputs.
4. **Ways of improvement** - identifies potential limitations and suggests code enhancements.

## 1. Data and structure

*Tobacco3482* dataset is used in the present assignment. The information about the data can be found [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download). In short, the data contains 10 subfolders with different image data types, resulting in 3842 images in total. The dataset should be downloaded and stored in the folder ```in```.

The directory structure is as follows:

```
assignment-3-VIS/
├── in/
│   └── Tobacco3482/
│       ├── ADVE/
│       ├── ... /
│       └── Scientific/
├── out/
│   ├── curves.png
│   └── report.csv
├── src/
│   ├── functions.py
│   └── script-3.py
├── README.md
├── requirements.txt
└── setup.sh
```
The **script-3.py** contains the main code for document classification using CNN. Script **functions.py** contains shared functions utilized in the main code.


## 2. Reproduction instructions

In order to run the Python script smoothly, the following steps should be completed within the terminal:

1. Open the terminal and set the working directory to the folder:

    ```python
    cd your_path_to/assignment-3-VIS
    ```

2. Run the following command to install the required modules and set up the *virtual environment*:

    ```python
    bash setup.sh
    ```
3. Activate the virtual environment by running:

    ```python
    source ./env/bin/activate
    ```

    Before proceeding: if the dataset folder is zipped, make sure to unzip it by executing:

    ```python
    unzip in/the_name_of_your_data_folder.zip
    ```

4. Execute the Python script. Remember to adjust the folder name, which contains subfolders with document images, and the number of epochs you want the model to run for.
 
    ```python
    python src/script-3.py Tobacco3482 10
    ```
    By such specification, the model will take the data from ```Tobacco3482``` folder, and run for 10 epochs.

5. After running the scripts, deactivate the virtual environment:

    ```
    deactivate
    ```
    The output files will be stored and found in the folder called ```out```.

## 3. Output summary

The ```out``` folder contains two files:

- curves.png

    This plot contains two graphs: *a loss curve* and *an accuracy curve* over 10 epochs. It shows the performance of the CNN model during training for document classification.

    **Loss curve**:

    *Training loss* curve starts high, and drops evidently after the first epoch, which might indicate a significant learning. Then it stabilizes and shows a gradual decrease in loss, which might suggest that the model keeps improving.

    *Validation loss* curve does not show stability as the *training* one. Loss fluctuates from epoch to epoch, suggesting that the model needs some improvements.
    
    **Accuracy curve**:

    *Training accuracy* curve, just as *training loss* curve, indicates that the model effectively learns from the training dataset over 10 epochs.

    *Validation accuracy* curve, on the other hand, just as the *validation loss* curve, is unstable across the epochs. This might suggest that the model is not generalizing well on the validation data (drop in accuracy at 4th and 6th epochs).

- report.csv

    This file contains the classification report with detailed metrics for document classes that are predicted by the model.

    Some document classes, like *ADVE* and *Email* have high classification metrics in terms of *precision*, *recall* and *F-1 score*. However, document classes like *Scientific* and *Resume* have lower scores across all metrics. As document categories are balanced, it might be that these classes are harder to learn, and more data for all classes is needed. Based on these results, the model is not learning the document categories well.


## 4. Ways of improvement

As with any programming task, there are always ways of how the code or the analysis can be improved. Here are a few points on how the code/analysis could be improved:

- Adjust the model's architecture. The model's performance could be increased by adjusting the *learning rate*, the number of *epochs* (in this case, the result is with 10 epochs), or reducing the *batch size*. All these arguments for the model specification could also be let to the user to choose.

- More data in each document category could improve the model's performance. It would ensure a better training, as the model could learn more details about each category. Additionally, with more learning the model could generalize better on the validation dataset.


*Note: It is recommended to execute the script on larger number of CPUs.*
# Assignment 2: Classification benchmarks with Logistic Regression and Neural Networks

This repository contains the code for **Assignment 2** from *Visual Analytics* course at *Aarhus University*.

The task is to develop two Python scripts to classify an image dataset represented not by actual images, but by numpy arrays with corresponding labels. The assignment features classification using Logistic Regression and Neural Networks.

This document is structured as follows:

1. **Data and structure** - describes the data source and file structure.
2. **Reproduction instructions** - steps to replicate the analysis.
3. **Output summary** - key points from the outputs.
4. **Ways of improvement** - identifies potential limitations and suggests code enhancements.

## 1. Data and structure

*Cifar10* is used in the present assignment. The information about the data and its documentation can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html). There is no need to download the data, as it will be loaded directly within the Python scripts.

The directory structure is as follows:
```
assignment-2-VIS/
├── in/
├── out/
│   ├── log_reg_metrics.csv
│   ├── loss_curve.png
│   └── neural_net_metrics.csv
├── src/
│   ├── utils/
│   ├── functions.py
│   ├── script-2-1.py
│   └── script-2-2.py
├── README.md
├── requirements.txt
└── setup.sh
```
The **script-2-1.py** contains the main code for image classification with Logistic Regression classifier, **script-2-2.py** containst the main code for image classification with Neural Network classifier. Script **functions.py** contains shared functions utilized by both main scripts.

```Utils``` is a folder with functions predefined by the course instructor, and can be used for the analysis.

## 2. Reproduction instructions

In order to run both Python scripts smoothly, the following steps should be completed within the terminal:

1. Open the terminal and set the working directory to the folder:

    ```python
    cd your_path_to/assignment-2-VIS
    ```

2. Run the following command to install the required modules and set up the *virtual environment*:

    ```python
    bash setup.sh
    ```
3. Activate the virtual environment by running:

    ```python
    source ./env/bin/activate
    ```
4. Execute both Python scripts separately.
 
    Logistic Regression classifier:
    ```python
    python src/script-2-1.py
    ```
    Neural Network classifier:
    ```python
    python src/script-2-2.py
    ```

5. After running the scripts, deactivate the virtual environment:

    ```
    deactivate
    ```
    The output files will be stored and found in the folder called ```out```.

## 3. Output summary

The ```out``` folder contains three files:

- log_reg_metrics.csv

    The Logistic Regression classifier shows moderate performance, with better classification results for labels such as *automobile* and *truck* compared to *cat* or *bird*.

- loss_curve.png

    The loss curve from the Neural Network classification script shows rapid improvement in the initial ~50 iterations. The curve becomes flatter as it progresses, with minor oscillations indicating slight variations in loss. 
    
- neural_net_metrics.csv

    The Neural Network classifier outperforms the Logistic Regression classifier, with higher *F1-scores* across all categories. However, the overall accuracy is only 38%, indicating room for improvement.


## 4. Ways of improvement

As with any programming task, there are always ways of how the code or the analysis can be improved. Here are a few points on how the code/analysis could be improved:

- Both classifiers show room for improvement. Experimenting with different stopping criteria or exploring other algorithms might enhance performance. For the Neural Network, increasing the number of hidden layers could be beneficial.

- Flexibility in classifier specifications could be introduced using the *argparse* library to allow users to adjust parameters more dynamically.


*Note: The first script executes on 4 CPUs in just over a minute, whereas the second script requires significantly more time and computational resources.*
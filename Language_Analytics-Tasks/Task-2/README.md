# Assignment 2: Text classification benchmarks

This repository contains the code for **Assignment 2** from *Language Analytics* course at *Aarhus University*.

The task is to train two simple (binary) classification models on the text data.
More specifically:
- **script-2-1.py** script is designed to train a **logistic classifier**;
- **script-2-2.py** script is designed to train a **neural network**.

Both models are used to classify text into being *true* or *fake*.

This document is structured as follows:

1. **Data and structure** - describes the data source and file structure.
2. **Reproduction instructions** - steps to replicate the analysis.
3. **Output summary** - key points from the outputs.
4. **Ways of improvement** - identifies potential limitations and suggests code enhancements.
5. **CodeCarbon tracking** - environmental impact of running the code.

## 1. Data and structure

*The Fake News* dataset is used in the present assignment. The data can be downloaded [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). Make sure to download the dataset and store it in a folder called ```in``` under the name of *fake_or_real_news.csv*.

The overall structure of the folders should be as follows:

```
assignment-2-LANG/
├── in/
│   └── fake_or_real_news.csv
├── out/
│   ├── emissions/
│   │   ├── emissions_assignment_2.csv
│   │   ├── emissions-base-... .csv
|   │   └── emissions-base-... .csv
│   ├── models/
│   │   ├── log_reg_classifier.csv
│   │   ├── neural_classifier.csv
│   │   ├── tfidf_logistic_vectorizer.joblib
│   │   └── tfidf_neural_vectorizer.joblib
│   ├── log_reg_report.csv
│   └── neural_report.csv
├── src/
│   ├── functions.py
│   ├── script-2-1.py
│   └── script-2-2.py
├── README.md
├── requirements.txt
└── setup.sh
```
The **script-2-1.py** contains the main code for training a logistic classifier and **script-2-2.py** contains the code for training a neural network. The **functions.py** is a script which contains predefined functions which are used in latter scripts.

## 2. Reproduction instructions

In order to run the Python script smoothly, the following steps should be completed within the terminal:

1. Open the terminal and set the working directory to the folder:

    ```python
    cd your_path_to/assignment-2-LANG
    ```
2. Run the following command to install the required modules and set up the *virtual environment*:

    ```python
    bash setup.sh
    ```
3. Then, activate the virtual environment by running:

    ```python
    source ./env/bin/activate
    ```

4. Run both Python scripts:

    - The following code will execute the Python script for training a **logistic classifier** on the data:
 
        ```python
        python src/script-2-1.py
        ```

    - The following code will execute the Python script for training a **neural network** on the data:


        ```python
        python src/script-2-2.py
        ```

5. Finally, once the script has finished running, deactivate the virtual environment: 

    ```
    deactivate
    ```
    The output files will be stored and found in the folder called ```out```.

## 3. Output summary

The ```out``` folder will contain two scripts with logistic regression (*log_reg_report.csv*) and neural network's (*neural_report.csv*) classification reports. Subfolder ```models``` will contain both models and their vectorizers.

The classification reports of both models indicate approximately similar classification quality. In both cases, models seem to classify text as being *fake* or *true* pretty well, with *F1-score* resulting in 0.86 - 0.87 for both classification models.


## 4. Ways of improvement

As with any programming task, there are always ways of how the code or the analysis can be improved. Here are a few points on how the code/analysis could be improved:

- Provide the user with more flexibility, for example, allow the user to input their own choice for hidden layers for neural network classification model. Additionally, allowing the user to specify the name of the dataset could make the code usable on other datasets. 
- Create more (separate current functions into smaller ones) predefined functions in **functions.py**, which could potentially help in debugging the errors, once these arise.


## 5. CodeCarbon tracking

In this repository, the **CodeCarbon** was used to monitor environmental impact of the code.
For a more detailed analysis of these results, please see Assignment 5.

*Note 1: New emissions data (see ```out/emissions``` folder) is generated every time the scripts are run.*

*Note 2: Both scripts were run on 4 CPUs and took around 20-30 seconds to produce the output.*

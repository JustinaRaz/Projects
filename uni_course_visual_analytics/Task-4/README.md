# Assignment 4: Detecting faces in historical newspapers

This repository contains the code for **Assignment 4** from *Visual Analytics* course at *Aarhus University*.

The task is to develop a Python script that analyzes the prevalence of images of human faces in three newspapers over the last 200 years. The output should contain summary showing how the percentage of pages with human faces have changed per decade in each newspaper.

This document is structured as follows:

1. **Data and structure** - describes the data source and file structure.
2. **Reproduction instructions** - steps to replicate the analysis.
3. **Output summary** - key points from the outputs.
4. **Ways of improvement** - identifies potential limitations and suggests code enhancements.

## 1. Data and structure

*A corpus of historic Swiss newspapers (the Gazette de Lausanne (GDL, 1804-1991); the Impartial (IMP, 1881-2017) and the Journal de Genève (JDG, 1826-1994))* is used in the present assignment. The information about the data and its documentation can be found [here](https://zenodo.org/records/3706863). Please download the dataset and store it in the folder called ```in```.

The directory structure is as follows:
```
assignment-4-VIS/
├── in/
│   └── newspapers/
│       ├── GDL/
│       ├── IMP/
│       └── JDG/
├── out/
│   ├── plots_GDL.png
│   ├── plots_IMP.png
│   ├── plots_JDG.png
│   ├── summary_GDL.csv
│   ├── summary_IMP.csv
│   └── summary_JDG.csv
├── src/
│   ├── functions.py
│   └── script-4.py
├── README.md
├── requirements.txt
└── setup.sh
```
The **script-4.py** contains the main code which performs the assignment's task. Meanwhile, **functions.py** contains functions utilized by the main script.

## 2. Reproduction instructions

In order to run the Python script smoothly, the following steps should be completed within the terminal:

1. Open the terminal and set the working directory to the folder:

    ```python
    cd your_path_to/assignment-4-VIS
    ```

2. Run the following command to install the required modules and set up the *virtual environment*:

    ```python
    bash setup.sh
    ```
3. Activate the virtual environment by running:

    ```python
    source ./env/bin/activate
    ```

    Before proceeding: if the downloaded dataset is zipped, unzip it by:

    ```python
    unzip in/newspapers.zip
    ```
    Make sure to adjust the folder name (*newspapers*) if it is called differently.

4. Execute the Python script. Ensure to specify the name of the folder containing three subfolders with image files of each newspaper. For instance, if the folder is named "newspapers", execute the following code:
 
    ```python
    python src/script-4.py newspapers
    ```

5. After running the script, deactivate the virtual environment:

    ```
    deactivate
    ```
    The output files will be stored and found in the folder called ```out```.

## 3. Output summary

The ```out``` folder contains 6 files:

- **Plots of the results**

    Folder stores 3 plots for each newspaper, which contain 2 subplots with information of: *Variation in Percentage of Pages Featuring Human Faces*, and *Variation in Total Number of Human Faces* across decades.

    **Key points**:
    
    - Generally, the total number of human faces has increased across decades in all newspapers. Based on results, *the Impartial* newspaper contained the greatest amount of images with human faces.
    - Percentage-wise, the same trend holds, and the percentage of pages with human faces has increased from the earliest decades to the latest decades. However, the line shows way more "peaks" and "dips", which should be investigated a bit more in order to understand what they actually mean (see the section below).

- **Summaries of the results**

    Folder stores 3 datasets (*.csv*) for each paper, with a total number of faces per decade, and the percentage of pages containing human faces for that specific decade. These datasets are used to generate the plots, specified above.

## 4. Ways of improvement

As with any programming task, there are always ways of how the code or the analysis can be improved. Here is one way how the code/analysis could be improved:

- As mentioned above, plots that showcase the variation in percentage of pages featuring human faces require a bit more adjustments. These subplots do not provide accurate representation of how the prevalence of human faces in pages of the newspaper changes over time, as there might be huge differences in the amount of pages that the paper contained in each decade. Analyzing relative frequencies rather than the actual percentages of pages with human faces might provide a more comprehensive comparison across decades by accounting for differences in the number of pages.

*Note: The script requires significant amount of time to run, and requires more computational resources than usual.*
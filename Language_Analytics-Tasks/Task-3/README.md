# Assignment 3: Query expansion with word embeddings via ```gensim```

This repository contains the code for **Assignment 3** from *Language Analytics* course at *Aarhus University*.

The task of the present assignment is to analyze a corpus of song lyrics and evaluate how frequently a certain topic, for instance, "love", and similar words related to "love", appear in songs by various artists. Word embeddings are used to expand the query and include similar terms to the target word to later calculate the percentage of songs by a specific artist that contain these terms.

This document is structured as follows:

1. **Data and structure** - describes the data source and file structure.
2. **Reproduction instructions** - steps to replicate the analysis.
3. **Output summary** - key points from the outputs.
4. **Ways of improvement** - identifies potential limitations and suggests code enhancements.
5. **CodeCarbon tracking** - environmental impact of running the code.

## 1. Data and structure

*Spotify Million Song Dataset*  is used in the present assignment. The data can be downloaded [here](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs). Make sure to download the dataset and store it in a folder called ```in``` (see the directory tree below). The downloaded dataset can be renamed, and in this case, it is called *spotify.csv*.

The directory structure is as follows:
```
assignment-3-LANG/
├── in/
│   └── spotify.csv
├── out/
│   └── emissions/
│       ├── emissions_assignment_3.csv
│       └── emissions_base... .csv
├── src/
│   ├── functions.py
│   └── script-3.py
├── README.md
├── requirements.txt
└── setup.sh
```

The **script-3.py** contains the main code, which produces the output in the terminal. The **functions.py** script is a script which contains predefined functions, used in the main code.

## 2. Reproduction instructions

In order to run the Python script smoothly, the following steps should be completed within the terminal:

1. Open the terminal and set the working directory to the folder:

    ```python
    cd your_path_to/assignment-3-LANG
    ```
2. Run the following command to install the required modules and set up the *virtual environment*:

    ```python
    bash setup.sh
    ```
3. Then, activate the virtual environment by running:

    ```python
    source ./env/bin/activate
    ```

4. Run the Python script. Make sure to also specify the name of the *csv* file, the artist's name and a word:

    ```python
    python src/script-3.py spotify.csv ABBA love
    ```
    In this case, I am specifying that my dataset is called *spotify.csv*, I am interested in song lyrics by *ABBA*, and would like to find the percentage of ABBA's songs related to *love*.

5. Finally, once the script has finished running, deactivate the virtual environment: 

    ```
    deactivate
    ```

## 3. Output summary
Once both scripts have finished running, the main output can be seen in the terminal. Based on the inputs specified above, the output looks as below:

**76.99% of ABBA's songs contain words related to love.**


## 4. Ways of improvement

As with any programming task, there are always ways of how the code or the analysis can be improved. Here are some ways how the code/analysis could be improved:

- Create more (separate current functions into smaller ones) predefined functions in **functions.py**, which could potentially help in debugging the errors, once these arise. 
- These simplified functions would also allow to monitor environmental impact of the code in a more detailed way.


## 5. CodeCarbon tracking

In this repository, the **CodeCarbon** was used to monitor environmental impact of the code.
For a more detailed analysis of these results, please see Assignment 5.

*Note 1: New emissions data (see ```out/emissions``` folder)  is generated every time the script is run.*

*Note 2: The script was run on 4 CPUs and took around 25 seconds to produce the output.*

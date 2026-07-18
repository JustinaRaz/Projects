# Assignment 1: Extracting linguistic features using ```spaCy```

This repository contains the code for **Assignment 1** from *Language Analytics* course at *Aarhus University*.

The task is to extract the linguistic information from a corpus of multiple texts. In this case, the code has to extract the relative frequency (RF) of nouns, verbs, adjectives, adverbs, and a total number of unique personal names, locations or organizations from the given texts.

This document is structured as follows:

1. **Data and structure** - describes the data source and file structure.
2. **Reproduction instructions** - steps to replicate the analysis.
3. **Output summary** - key points from the outputs.
4. **Ways of improvement** - identifies potential limitations and suggests code enhancements.
5. **CodeCarbon tracking** - environmental impact of running the code.

## 1. Data and structure

*The Uppsala Student English Corpus (USE)* dataset is used in the present assignment. The information about the data and its documentation can be found [here](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457). Make sure to download the dataset, ensure that the folder is called *USEcorpus* and store it in a folder called ```in```.

The directory structure is/should be as follows:
```
assignment-1-LANG/
├── in/
│   ├── USEcorpus/
│       ├── USEcorpus/
│       │   ├── a1/
│       │   ├── .../
│       │   └── c1/
│       └── readme.md
├── out/
│   ├── emissions/
│   │   ├── emissions_assignment_1.csv
│   │   └── emissions-base-... .csv
│   ├── a1_counts.csv
│   ├── ...
│   └── c1_counts.csv
├── src/
│   ├── functions.py
│   └── script-1.py
├── README.md
├── requirements.txt
└── setup.sh
```
The **script-1.py** contains the main code, which produces the output. The **functions.py** script is a script which contains predefined functions, used in the main code.

## 2. Reproduction instructions

In order to run the Python script smoothly, the following steps should be completed within the terminal:

1. Open the terminal and set the working directory to the folder:

    ```python
    cd your_path_to/assignment-1-LANG
    ```
2. Run the following command to install the required modules and set up the *virtual environment*:

    ```python
    bash setup.sh
    ```
3. Then, activate the virtual environment by running:

    ```python
    source ./env/bin/activate
    ```
    Before proceeding: if the downloaded dataset is zipped, unzip it by executing:

    ```python
    unzip in/USEcorpus.zip
    ```
    Make sure that the structure of the directory specified above holds.

4. Run the Python script:
 
    ```python
    python src/script-1.py
    ```
5. Finally, once the script has finished running, deactivate the virtual environment: 

    ```
    deactivate
    ```
    The output files will be stored and found in the folder called ```out```.

## 3. Output summary
The ```out``` folder will contain multiple **csv** files, one for each data folder located in the folder ```in``` (```a1```, ```a2```, ..., ```c1```).

An example of the **csv** file:

Filename | RelFreq NOUN | RelFreq VERB | RelFreq ADJ | RelFreq ADV | Unique PER | Unique LOC | Unique ORG | 
--- | --- | --- | --- | --- | --- | --- | --- |
0100.a1.txt | 1526.39 | 1526.39 | 827.39 | 827.39 | 0 | 0 | 0 |
0101.a1.txt | 1181.7 | 1257.94 | 597.2 | 851.33 | 1 | 0 | 0 |
0102.a1.txt | 1506.68 | 1227.22 | 692.59 | 486.03 | 1 | 0 | 0 |
...

The columns represent:

-  *Filename* - text document for which the data was produced;
- *RelFreqNOUN* - the RF of nouns;
- *RelFreqVERB* - the RF of verbs;
- *RelFreqADJ* - the RF of adjectives;
- *RelFreqADV* - the RF of adverbs;
- *UniquePER* - the number of unique personal names in a given text;
- *UniqueLOC* - the number of unique locations in a given text;
- *UniqueORG* - the number of unique organizations in a given text.

## 4. Ways of improvement

As with any programming task, there are always ways of how the code or the analysis can be improved. Here are a few points on how the code/analysis could be improved:

- Use a larger English language model. This could enhance the accuracy of the results, however, it would require more time and resources to process the text.
- Provide the user with more flexibility, for example, where the input files are located and how the files are named. Moreover, the user could be allowed to set the size of the language model and choose which language model to load.
- The main code could be split into smaller parts, which would allow to monitor the environment impact of the subtasks.

## 5. CodeCarbon tracking

In this repository, the **CodeCarbon** was used to monitor environmental impact of the code.
For a more detailed analysis of these results, please see Assignment 5.

*Note 1: New emissions data (see ```out/emissions``` folder)  is generated every time the script is run.*

*Note 2: The script was run with 4 CPUs and took around 4 minutes to produce the output.*
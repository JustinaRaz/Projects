# Assignment 4: Emotion analysis with pretrained language models

This repository contains the code for **Assignment 4** from *Language Analytics* course at *Aarhus University*.

The task is to perform some computational text analysis. The code is designed to investigate the emotional profile of the television show called *Game of Thrones*. The code allows to explore this emotional profile, with plots showcasing how it has changed over the eight seasons of series. To do this, a pretrained emotional classifier ([loaded](https://huggingface.co/docs/transformers/v4.27.2/en/task_summary#natural-language-processing) via *HuggingFace*) is used for text classification.

This document is structured as follows:

1. **Data and structure** - describes the data source and file structure.
2. **Reproduction instructions** - steps to replicate the analysis.
3. **Output summary** - key points from the outputs.
4. **Ways of improvement** - identifies potential limitations and suggests code enhancements.
5. **CodeCarbon tracking** - environmental impact of running the code.

## 1. Data and structure

The script with the *Game of Thrones* data and its details can be found and downloaded [here](https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons?select=Game_of_Thrones_Script.csv). Make sure to download the dataset and store it in a folder called ```in``` (see the directory tree below). The downloaded dataset can be renamed. Here it is called *Game_of_Thrones_Script.csv*:

```
assignment-4-LANG/
├── in/
│   └── Game_of_Thrones_Script.csv
├── out/
│   └── emissions/
│   │   ├── emissions_assignment_4.csv
│   │   └── emissions_base... .csv
|   ├── emotion_scores_by_season.png
|   └── RF_of_emotion_across_seasons.png
├── src/
│   ├── functions.py
│   └── script-4.py
├── README.md
├── requirements.txt
└── setup.sh
```
The **script-4.py** contains the main code, which produces the output. The **functions.py** script is a script which contains predefined functions, used in the main code.

## 2. Reproduction instructions

In order to run the Python script smoothly, the following steps should be completed within the terminal:

1. Open the terminal and set the working directory to the folder:

    ```python
    cd your_path_to/assignment-4-LANG
    ```

2. Run the following command to install the required modules and set up the *virtual environment*:

    ```python
    bash setup.sh
    ```

3. Then, activate the virtual environment by running:

    ```python
    source ./env/bin/activate
    ```

4. Run the Python script. Make sure to also specify the name of your *csv* file:

    ```python
    python src/script-4.py Game_of_Thrones_Script.csv
    ```

5. Finally, once the script has finished running, deactivate the virtual environment: 

    ```
    deactivate
    ```

    The output files will be stored and found in the folder called ```out```.

## 3. Output summary

Once the scipt has finished running, there should appear 2 plots in a folder called ```out```.

**Plot 1**: ```emotion_scores_by_season.png```

This plot consists of 8 sublots, which represent the distribution of different type of emotions in each of the 8 seasons. 

Based on the plots, it is clear that the *Neutral* emotional profile is the most common across all seasons. The second most common emotional profile is *Anger*, which is followed by *Disgust* and *Surprise*. *Joy*, on the other hand, seems to be the emotional profile which is least prevelant across all seasons, surpassing *Fear* by a bit only in season 8.

**Plot 2**:  ```RF_of_emotion_across_seasons.png```

This plot consists of 7 sublots, which represent the relative frequency of different types of emotions in each of the 8 seasons.

Short description for each type of emotion:

1. *Anger*. Anger was least prevalent in season 3, and most in season 8.
2. *Disgust*. Disgust decreased with the time of the series - from season 3 to season 8 it got less prevalant.
3. *Fear*. Most of the fear was captured in season 7, least - season 8.
4. *Joy*. Most of the joy was captured in the middle of the series - season 4 and 5.
5. *Neutral*. Neutral emotion did not fluctuate too much throughout the series.
6. *Sadnesss*. Sadness was found to drop after season 1, and again, steadily increase from season 2 to 5, peaking at season 6. Season 7 was classified to have the smallest 'prevalence of sadness.
7. *Surprise*. The greatest prevalence of surprise was captured in season 1, the smallest - season 6.

## 4. Ways of improvement

As with any programming task, there are always ways of how the code or the analysis can be improved. In this assignment, predefined functions could have been split into smaller parts, which would allow a more specific CodeCarbon tracking.

## 5. CodeCarbon tracking

In this repository, the **CodeCarbon** was used to monitor environmental impact of the code.
For a more detailed analysis of these results, please see Assignment 5.

*Note 1: New emissions data (see ```out/emissions``` folder) is generated every time the script is run.*

*Note 2: The dataset for this assignment is very large, and therefore emotional classification of text takes a lot of computing power. Make sure to run this code with enough CPUs. It is recommended to run it with more than 4 CPUs. Otherwise, it might take around 1 hour to get the output.*
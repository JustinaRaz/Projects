import os
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import argparse


def get_label(data_frame):

    """
    A function to get the emotional profile label for each row in a data frame.

    Input: 
        - A data frame to work with, in this case, Game of Thrones script.
    
    Returns:
        - A data frame as the original one, with additional column with emotion label.
    """
    
    label = []

    classifier = pipeline("text-classification", 
                        model= "j-hartmann/emotion-english-distilroberta-base", 
                        return_all_scores = False)

    for sentence in data_frame["Sentence"]:
        output_dict = classifier(str(sentence))
        output_label = output_dict[0]["label"]
        label.append(output_label)

    # Adding labels to the data set:

    data_frame["Label"] = label

    return data_frame


def plot_emotions_per_season(data_frame):   

    """
    A function which plots and saves a figure, containing 8 subplots to represent the emotional profile in each season of the series.

    Input:
        - A data frame with classified emotional profile label. 

    Returns:
        - A plot of 8 subplots (in 2 columns and 4 rows), which is saved in folder -out-, representing a distribution of all emotion types in all 8 seasons.
    """

    seasons = data_frame['Season'].unique() # ['Season 1' 'Season 2' 'Season 3' 'Season 4' 'Season 5' 'Season 6', 'Season 7' 'Season 8']

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))

    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]

    # Initialize subplot index
    i = 0

    for season in seasons:
        # Initialize a new dictionary to keep track of counts:
        label_count = {}
        subset = data_frame[data_frame['Season'] == season]

        for label in subset["Label"]:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        # The following code extracts the list of keys (labels) and values (counts):

        emotions = list(label_count.keys())
        counts = list(label_count.values())

        df = pd.DataFrame({'Emotion': emotions, 'Frequency': counts})

        df_sorted = df.sort_values(by='Emotion')

        row, col = subplot_positions[i]
        
         # Plot the bar chart in the corresponding subplot
        axes[row, col].bar(df_sorted["Emotion"], df_sorted["Frequency"])
        axes[row, col].set_xlabel('Emotion', fontweight='bold')
        axes[row, col].set_ylabel('Frequency of emotion', fontweight='bold') 
        axes[row, col].set_title('Frequency of emotions in {}'.format(season), fontweight='bold')

        # Increment the subplot index
        i += 1

        axes[row, col].set_ylim(0, 2000)
        axes[row, col].grid(axis='y')

    # Prevents overlapping:
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig("out/emotion_scores_by_season.png")

    plt.show()


def plot_emotion_across_seasons(data_frame):

    """
    A function which plots and saves a figure, that represents the relative frequency of each type of emotion across all seasons of the series.

    Input: 
        - A data frame with the classified emotional profile label. 

    Returns:
        - A plot, which is saved in a folder -out-, and showcases the relative frequency of each emotion type across the whole series.
    """

    # Figure specifications 
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]

    # Observation count for each season, from 1 to 8.
    obs_per_season = []
    seasons = sorted(list(data_frame['Season'].unique()))

    for season_s in seasons:
        subset_s = data_frame[data_frame['Season'] == season_s]
        obs_num = len(subset_s)
        obs_per_season.append(obs_num)


    # Initialize subplot index
    i = 0

    labels = sorted(list(data_frame['Label'].unique()))


    for label in labels:
        label_count_across_seasons = [] # Count for S1, S2, S3... in the right order.
        for season in seasons:
            subset = data_frame[(data_frame['Label'] == label) & (data_frame['Season'] == season)]
            label_count = len(subset)
            label_count_across_seasons.append(label_count)

        df = pd.DataFrame({'Season': seasons, 'Label_count': label_count_across_seasons, 'Season_count': obs_per_season})
        

        df["RL"] = df["Label_count"]/df["Season_count"]
        
        row, col = subplot_positions[i]
        
         # Plot the bar chart in the corresponding subplot
        axes[row, col].bar(df["Season"], df["RL"])
        axes[row, col].set_xlabel('Season', fontweight='bold')
        axes[row, col].set_ylabel('Relative frequency of emotion', fontweight='bold') 
        axes[row, col].set_title(' "{}" across seasons'.format(label), fontweight='bold')

        # Increment the subplot index
        i += 1

        #axes[row, col].set_ylim(0, 2000)
        axes[row, col].grid(axis='y')

    # Deletes the last subplot:
    fig.delaxes(axes[3, 1])
    
    # Prevents overlapping:
    plt.tight_layout()
    plt.savefig("out/RF_of_emotion_across_seasons.png")
    plt.show()




def file_loader():

    """
    A function that allows the user to enter specific information on what the function should look for.
    In this script, the user is required to enter a dataset name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("data", 
                    #required=True, 
                    help="Name of your dataset.")
    args = parser.parse_args()
    return args
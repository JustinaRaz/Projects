import os
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import functions
from codecarbon import EmissionsTracker
import functions
import argparse


def main():
    tracker = EmissionsTracker(output_file = "emissions_assignment_4.csv", output_dir = "out/emissions/")
    tracker.start()

    tracker.start_task("splitting_args")
    args = functions.file_loader()
    data_file = args.data
    tracker.stop_task("splitting_args")

    tracker.start_task("reading_csv")
    df = pd.read_csv("in/{}".format(data_file))
    tracker.stop_task("reading_csv")

    tracker.start_task("get_label")
    print("Collecting the labels for the sentences...")
    data = functions.get_label(df)
    print("Done collecting labels!")
    tracker.stop_task("get_label")

    tracker.start_task("get_plot_per_season")
    print("Creating the first plot...")
    functions.plot_emotions_per_season(data)
    tracker.stop_task("get_plot_per_season")

    tracker.start_task("get_plot_across_season")
    print("Creating the second plot...")
    functions.plot_emotion_across_seasons(data)
    tracker.stop_task("get_plot_across_season")
    tracker.stop()
if __name__ == "__main__":
    main()

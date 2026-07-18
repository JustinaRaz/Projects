import pandas as pd
import os
import string
import gensim
import gensim.downloader
import gensim.downloader as api
import argparse
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
import functions


def main():
    tracker = EmissionsTracker(output_file = "emissions_assignment_3.csv", output_dir = "out/emissions/")
    tracker.start()
    print("Currently loading the word embedding model...")
    
    tracker.start_task("language_model_load")
    emb_model = api.load("glove-wiki-gigaword-50")
    print("Loading is done! ")
    tracker.stop_task("language_model_load")

    tracker.start_task("splitting_args")
    args = functions.file_loader()
    data_file = args.data
    artist_input = args.artist
    word_input = args.word
    tracker.stop_task("splitting_args")

    tracker.start_task("reading_csv")
    data = pd.read_csv("in/{}".format(data_file))
    tracker.stop_task("reading_csv")

    tracker.start_task("get_similar_words")
    print("Looking for words, similar to {}".format(word_input))
    sim_words = functions.get_words(word_input, emb_model)
    tracker.stop_task("get_similar_words")
    
    tracker.start_task("subsetting")
    artist_df = data[data['artist'] == artist_input]
    number_of_songs = len(artist_df)
    tracker.stop_task("subsetting")

    tracker.start_task("get_songs")
    print("Counting {}'s songs related to {}...".format(artist_input, word_input))
    dict = functions.get_count(artist_df, sim_words)
    value = functions.get_max(dict)
    tracker.stop_task("get_songs")

    tracker.start_task("save_output")
    answer = f"{round(value*100/number_of_songs, 2)}% of {artist_input}'s songs contain words related to {word_input}."
    #print(answer)
    tracker.stop_task("save_output")

    tracker.stop()
    print(answer)
if __name__ == "__main__":
    main()
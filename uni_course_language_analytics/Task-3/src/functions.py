import pandas as pd
import os
import string
import gensim
import gensim.downloader
import gensim.downloader as api
import argparse
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

# Find a list of words that are similar to the user's input:
def get_words(wd, model):
    """
    A function that finds 10 most similar words to the one chosen by the user.

    Input:
        wd - a single string - one word.
        model - a word embedding model which is used to find a list of similar words.

    Returns:
        The output is a list of words, similar to the user's word, and including user's word.
    """
    similar_words = model.most_similar(wd)
    words = []
    for pair in similar_words:
        words.append(pair[0])
    words.append(wd)
    return words

# Argparse:

def file_loader():

    """
    A function that allows the user to enter specific information on what the function should look for.
    It creates placements for the user's values, namely, the user can specify the dataset that should be used for the analysis,
    the name of the artist whose songs should be analyzed, and the word, which is used to find the songs related to it.

    Returns:
        User's inputs. 
        args.data - will contain user's input for the dataset;
        args.artist - will contain user's input with the name of the artist;
        args.word - will contain user's input with one word.
    """

    parser = argparse.ArgumentParser(description = "Finding an artist of user's topic")
    parser.add_argument("data", 
                    #required=True, 
                    help="Name of your dataset with spotify lyrics.")
    parser.add_argument("artist", 
                        #required=True, 
                        help="Artist name whose songs you want to look into.")
    parser.add_argument("word", 
                        #required=True,
                        help="Word which you want to look for in an artists songs.")

    args = parser.parse_args()
    return args

# Preprocessing of the lyrics:
def preprocess(text):

    """
    A function which is used to preprocess the song's lyrics.

    Input:
        One song's lyrics - a text.
    Returns:
        Returns a list of lowercased words from the lyrics. This list does not contain stop words.
    """

    word_list = []
    stop_words = {'for', 'a', 'of', 'the', 'and', 'to', 'in'} # list of stopwords - from gensim documentation
    tokens = text.lower().split() # splitting into words and lowercasing
    cleaned_tokens = [ ''.join(char for char in token if char not in string.punctuation) for token in tokens] #removing punctuation
    
    # remove stop words:
    for word in cleaned_tokens:
        if word not in stop_words:
            word_list.append(word)
    
    return word_list

# Getting the count of words per song:
def get_count(artist_data, similar_words):

    """
    A function that loops through specific artist's songs, and produces a dictionary of counts for 
    each word - the count of how many songs each word can be found in.

    Input:
        artist_data: a dataset which contains one specific artist's songs with lyrics.
        similar_words: a list of similar words, the output from 'preprocess()' function.
    Returns:
        a dictionary with counts for each word. This count represents how many songs have the respective words in its lyrics.
    """

    word_count = {}
    for text in artist_data["text"]:
        lyrics = preprocess(text)
        for word in similar_words:
            if word in lyrics and word not in word_count:
             word_count[word] = 1
            if word in lyrics and word in word_count:
             word_count[word] += 1
    return word_count

# Find the maximum value for words in songs according to user's input:
def get_max(dictionary):

    """
    A function that finds the number of songs related to the user's inputs.

    Input:
        a dictionary with similar words and their counts.

    Returns:
        the maximum number from the dictionary. This number represents the amount of songs, which are related to the user's word for specific artist.
    """
    values_list = list(dictionary.values())
    max_value = max(values_list)
    return max_value
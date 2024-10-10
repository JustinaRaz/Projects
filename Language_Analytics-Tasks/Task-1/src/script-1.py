import pandas as pd
import spacy
import os
import string
import numpy as np
import functions
from codecarbon import EmissionsTracker


def main():
    tracker = EmissionsTracker(output_file = "emissions_assignment_1.csv", output_dir = "out/emissions/")
    tracker.start()
    # Initialize spacy model: 

    tracker.start_task("model_load")
    nlp = spacy.load("en_core_web_md")
    tracker.stop_task("model_load")

    # Path to data; directories to each data folder: 
    tracker.start_task("get_directories_to_subfolders")
    data_path = "in/USEcorpus/USEcorpus/"
    dirs = sorted(os.listdir(data_path))

    # Create a list of paths to each folder: 

    subfolder_directories = []

    for directory in dirs:
        subfolder = data_path + directory
        subfolder_directories.append(subfolder)
        filenames = sorted(os.listdir(subfolder))
    tracker.stop_task("get_directories_to_subfolders")

    tracker.start_task("get_values")
    for directory in subfolder_directories:
        text_files = sorted(os.listdir(directory))
        data_values = []
        for file in text_files:
            path = directory + "/" + file
            filename = file
            with open(path, encoding = "latin-1") as f:
                text = f.read()

                # Using my function to clean the loaded text:
                c_text = functions.clean_text(text)
                
                doc = nlp(c_text)

                # count RelFreq for all words in that specific doc:
                counts = functions.count_freq(doc)

                # count Unique entities for all words in that specific doc:
                entities = functions.unique_ent(doc)
                
                # Appending a tuple of values to a list, which is later transformed into a data frame with respective columns:

                data_values.append((filename, counts[0], counts[1], counts[2], counts[3], entities[0], entities[1], entities[2]))

        data = pd.DataFrame(data_values, 
                        columns=["Filename", "RelFreq NOUN", "RelFreq VERB", "RelFreq ADJ", "RelFreq ADV", "Unique PER", "Unique LOC", "Unique ORG"])
        
        outpath = os.path.join("out", f"{directory[-2:]}_counts.csv")
        data.to_csv(outpath, index = False)
    tracker.stop_task("get_values")
    
    tracker.stop()

if __name__ == "__main__":
    main()


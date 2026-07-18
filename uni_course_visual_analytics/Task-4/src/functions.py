import os
from PIL import Image
import argparse

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

import pandas as pd
import matplotlib.pyplot as plt


def file_loader():

    """
    A function that allows the user to enter specific information on what the function should look for.
    It creates placements for the user's input, namely, the user can specify the dataset that should be used for the analysis.

    Returns:
        User's inputs. 
        args.data - will contain user's input for the dataset folder name.
    """

    parser = argparse.ArgumentParser(description = "Face detection")
    parser.add_argument("data", 
                    #required=True, 
                    help="Name of your dataset")

    args = parser.parse_args()
    return args

def get_directories(folder_name):

    path_to_grand_folder = os.getcwd()

    """
    This function creates a list of directories to the page of each paper.

    Input:
        - folder_name - a path to the folder with subfolders of each paper's pages.

    Returns:
        - A list of directories - path to every image (page) of the newspapers.
    """

    list_of_papers = sorted(os.listdir(path_to_grand_folder + "/in/{}".format(folder_name)))[:3]

    directories = []

    for paper_type in list_of_papers:
        paper_path = path_to_grand_folder + "/in/{}/".format(folder_name) + paper_type
        list_of_pages = sorted(os.listdir(paper_path))
        for page in list_of_pages:
            page_path = paper_path + "/" + page
            directories.append(page_path)

    return directories

def get_data_frame(dirs):

    """
    A function that creates vectors of values for the dataset and performs image detection.

    Output: a dataframe with face counts.
    """

    #Prepare data:

    titles = []
    years = []
    decades = []

    for paper in dirs:
        name = paper[-26:-23]
        year = paper[-22:-18]
        titles.append(name)
        years.append(year)

    for year in years:
        decade = int(year) // 10 * 10
        decades.append(decade)

    face_counts = []
    truncated_imgs = 0

    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()

    print("Face detection task has started:")

    for path in dirs:
        try: 
            img = Image.open(path)
            #boxes, _ = mtcnn.detect(img)
            boxes, _ = mtcnn.detect(img)
        except OSError:
            zero = 0
            face_counts.append(zero)
            truncated_imgs += 1
            print("Page {} from {} newspaper (year {}) is truncated. Skipping this one...".format(path[-9:-4], path[-26:-23], path[-22:-18]))
            continue
        #faces = boxes.shape[0]
        
        try:
            faces = boxes.shape[0]

        except AttributeError:
            zero = 0
            face_counts.append(zero)
            print("Found 0 faces in {} newspaper page {} from {}".format(path[-26:-23], path[-9:-4], path[-22:-18]))
            continue

            #face_counts.append(faces)
            #print("Found {} face(s) in {} newspaper page {} from {}".format(faces, path[-26:-23], path[-9:-4], path[-22:-18]))

        face_counts.append(faces)
        print("Found {} face(s) in {} newspaper page {} from {}".format(faces, path[-26:-23], path[-9:-4], path[-22:-18]))

    df = pd.DataFrame({
    'titles': titles,
    'years': years,
    'decades': decades,
    'face_counts': face_counts
    })
    print("Task done!")
    
    #print(truncated_imgs)
    return df

def face_per_page(data_frame):

    """
    A function that takes the dataframe and analyzes how many pages contain a face, no matter how many faces there are in the page.

    Input:
        - data_frame - data, which contains a column named "face_counts", containing a number of faces per page.

    Returns:
        - data_frame - a dataframe with additional column, which identifies whether ("1" for a yes, and "0" for a no) the page contains at least one face or not.
    """    

    face_counts_c = data_frame["face_counts"]

    face_per_page_c = []

    for count in face_counts_c:
        if count == 0:
            face_per_page_c.append(int(0))
        else:
            face_per_page_c.append(int(1))

    data_frame["face_per_page_c"] = face_per_page_c

    return data_frame

def get_summaries(data_frame): 

    titles = set(data_frame["titles"])

    for title in titles:

        paper_df = data_frame[data_frame["titles"] == title]

        paper_df = paper_df.groupby("decades")

        summaries = []

        # Iterate over each group
        for name, group in paper_df:
            total_faces = group['face_counts'].sum()  # Sum up all the picture counts per decade
            face_presence = round((group['face_per_page_c'] == 1).sum() / len(group) * 100, 1)  # Calculate the percentage of rows with a picture
            
            # Append the results as a dictionary to the result list
            summaries.append({
                'title': title,
                'decade': name, #unique values of the decade column
                'total_faces': total_faces,
                'face_presence': face_presence
            })
        local_path = os.getcwd()
        # Convert the result list to a DataFrame
        summary = pd.DataFrame(summaries)
        summary.to_csv("out/summary_{}.csv".format(title), index=False)

def get_plots(data_frame):    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting the first subplot
    axs[0].plot(data_frame["decade"], data_frame["face_presence"])
    axs[0].set_title("Variation in Percentage of Pages Featuring Human Faces, {} newspaper".format(data_frame["title"][0]))
    axs[0].set_xlabel("Decade")
    axs[0].set_ylabel("Percentage of Pages with Human Faces")
    axs[0].set_xticks(data_frame["decade"])
    axs[0].set_xticklabels(data_frame["decade"], rotation=90) 

    # Plotting the second subplot
    axs[1].plot(data_frame["decade"], data_frame["total_faces"])
    axs[1].set_title("Variation in Total Number of Human Faces, {} newspaper".format(data_frame["title"][0]))
    axs[1].set_xlabel("Decade")
    axs[1].set_ylabel("Total Number of Human Faces")
    axs[1].set_xticks(data_frame["decade"])
    axs[1].set_xticklabels(data_frame["decade"], rotation=90) 

    plt.tight_layout()
    plt.savefig("out/plots_{}".format(data_frame["title"][0]))
    plt.show()
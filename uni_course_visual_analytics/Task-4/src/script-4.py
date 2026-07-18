import os
from PIL import Image
import argparse

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

import pandas as pd
import matplotlib.pyplot as plt
import functions


def main():
    args = functions.file_loader()
    data_file = args.data

    paths = functions.get_directories(data_file)

    data = functions.get_data_frame(paths)

    updated_data = functions.face_per_page(data)

    functions.get_summaries(updated_data)

    GDL = pd.read_csv("out/summary_GDL.csv")
    IMP = pd.read_csv("out/summary_IMP.csv")
    JDG = pd.read_csv("out/summary_JDG.csv")

    functions.get_plots(GDL)
    functions.get_plots(IMP)
    functions.get_plots(JDG)

if __name__ == "__main__":
    main()
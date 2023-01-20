# AMLS_22-23_SN19002093

# External libraries: 

import os

import numpy as np

import torch

from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from torchvision import datasets

from torchvision import transforms

from PIL import Image

import matplotlib.pyplot as plt


# Tasks

Binary tasks: 

A1: Gender detection

A2: Emotion detection



Multi-class tasks: 

B1: Face shape classification

B2: Eye colous classification


# Organization

A1, A2, B1, B2 includes the code for one task. Four tasks use a same model. Comments and explanation are in A1. Other three tasks cancel most of comments to be clean. 

Datasets should include celeba (5000 jpg and label.csv), celeba_test (1000 jpg and label.csv), cartton_set (10000 png and label.csv), cartoon_set_test(2500 png and label.csv).

Four files of tasks (A1, A2, B1, B2) could be run together by "main.py". 

If running files by "main.py", please use the absolute path in all codes. 

If running files one by one, the relative path could be used. 

Sometimes "main.py" reports errors, please relaunch the Jupyter Notebook. 


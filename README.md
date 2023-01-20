# AMLS_22-23_SN19002093

# external libraries: 
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
Binary tasks: \n
A1: Gender detection\n
A2: Emotion detection\n

Multi-class tasks: \n
B1: Face shape classification\n
B2: Eye colous classification\n

# Organization
Four files of tasks could be run together by "main.py". \n
If running files by "main.py", please use the absolute path in all codes. \n
If running files one by one, the relative path could be used. \n
Sometimes "main.py" reports errors, please relaunch the Jupyter Notebook. \n

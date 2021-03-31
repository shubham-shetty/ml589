# Import Statements
import numpy as np
import sklearn as sk
from sklearn.tree import *
from sklearn.linear_model import * 
from sklearn.svm import *
from sklearn.metrics import *
from sklearn.neighbors import *
from sklearn.model_selection import KFold
from datetime import datetime
import math
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
import multiprocessing

# Load Data
data = np.load("data.npz")

X_trn = data["X_trn"]
y_trn = data["y_trn"]
X_tst = data["X_tst"]


# Common Functions

# Show image from data
def show(x):
    img = x.reshape((3,32,32)).transpose(1,2,0)
    plt.imshow(img)
    plt.axis('off')
    plt.draw()
    plt.pause(0.01)

# Write data to CSV file on local system
def write_csv(y_pred, filename):
    """Write a 1d numpy array to a Kaggle-compatible .csv file"""
    with open(filename, 'w+') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Category'])
        for idx, y in enumerate(y_pred):
            csv_writer.writerow([idx, y])

# Helper funtion to print table 
# Input variable table is a numpy array and headers is an array of column headers
def prettyPrintTable(table, headers) :
    for i, d in enumerate(table):
        if i == 0 :
            line = '|'.join(str(x).ljust(30) for x in headers)
            print(line)
            print('-' * len(line))

        line = '|'.join(str(x).ljust(30) for x in d)
        print(line)
    print()
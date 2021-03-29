import numpy as np
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier

from question_7 import KNN_classification
import csv

def write_csv(y_pred, filename):
    """Write a 1d numpy array to a Kaggle-compatible .csv file"""
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Category'])
        for idx, y in enumerate(y_pred):
            csv_writer.writerow([idx, y])


data = np.load("data.npz")
X_trn = data["X_trn"]
y_trn = data["y_trn"]
X_tst = data["X_tst"]
Y_pred = KNN_classification(11,X_trn,y_trn,X_tst)
print("Writing results to a Kaggle compatible csv file")
write_csv(Y_pred, 'sample_predictions.csv')








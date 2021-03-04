# Import Statements
import numpy as np
import sklearn as sk
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.tree import DecisionTreeRegressor


# Load Data
all_data = np.load("data.npz")
big_data = np.load("big_data.npz")


# Initialising for data.npz

## 3d training inputs
X_trn = all_data["X_trn"] 
## 1d training outputs
y_trn = all_data["y_trn"] 
## Test data inputs
X_val = all_data["X_val"] 
## Test data outputs
y_val = all_data["y_val"] 


# Initialising for big_data.npz

## 3d training inputs
X_big_trn = big_data["X_trn"] 
## 1d training outputs
y_big_trn = big_data["y_trn"] 
## Test data inputs
X_big_val = big_data["X_val"] 
## Test data outputs
y_big_val = big_data["y_val"] 


# Common Functions

## Euclidean Distance
def euc_dist(p1, p2, n):
    tot = 0
    for i in range(n):
        tot += (p1[i] - p2[i])**2
    return np.sqrt(tot)
    
## Loss Functions
### Mean Squared Error
def sq_error(a,b):
    n = len(a)
    err = 0
    for i in range(n):
        err += (a[i] - b[i])**2
    return err/n
### Mean Absolute Error
def abs_error(a,b):
    n = len(a)
    err = 0
    for i in range(n):
        err += abs(a[i] - b[i])
    return err/n

#Helper funtion to print table with 3 columns in a nice formatter manner
def prettyPrintTable(table, headerCol1, headerCol2, headerCol3) :
    for i, d in enumerate(table):
        if i == 0 :
            line = '|'.join(str(x).ljust(30) for x in (headerCol1, headerCol2, headerCol3))
            print(line)
            print('-' * len(line))

        line = '|'.join(str(x).ljust(30) for x in d)
        print(line)
            
#

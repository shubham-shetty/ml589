# Import Statements
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


# Load Data

## Training Data
X = np.loadtxt('x.csv')
Y = np.loadtxt('y.csv')

## Test Data
x = np.loadtxt('x_test.csv')
y = np.loadtxt('y_test.csv')


# Predictor Function
def f_m(x,m):
    if m>0:
        return x**m
    else:
        return 0
from question_16 import *
from preface import *
from datetime import datetime
import pandas as pd
import sys
import matplotlib.pyplot as plt
import math
import multiprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import logging
import os

def predictY(X,W,V,b,c):
    # Compute f(x) for all x.
    WX = np.matmul(W, np.transpose(X))
    B = np.array([b, ] * X.shape[0]).transpose()
    C = np.array([c, ] * X.shape[0]).transpose()
    B_plus_WX = np.add(B, WX)
    sigma = np.tanh(B_plus_WX)
    F = np.add(C, np.matmul(V, sigma))

    Y = np.zeros(X.shape[0], int)
    for i in range(Y.shape[0]) :
        Y[i] = int(np.argmax(F[:,i]))
    
    return Y

def gradient_descent_train(M,X,Y,X_test) :
    numLabels = 4
    momentum = 0.1
    l = 1
    stepSize = 0.0001
    numIter = 1000
    
    W = np.random.normal(0,1,(M,X.shape[1]))
    b = np.zeros(M)
    c = np.zeros(numLabels)
    V = np.random.normal(0,1,(numLabels, M))

    avg_dLdW = avg_dLdV = avg_dLdb = avg_dLdc = 0
    for i in range(numIter) :
        # ave_grad = (1 - momentum) * ave_grad + momentum * ∇h(w)
        dLdW, dLdV, dLdb, dLdc = prediction_grad_full(X,Y,W,V,b,c,l)
        avg_dLdW = (1 - momentum) * avg_dLdW + momentum * dLdW
        avg_dLdV = (1 - momentum) * avg_dLdV + momentum * dLdV
        avg_dLdb = (1 - momentum) * avg_dLdb + momentum * dLdb
        avg_dLdc = (1 - momentum) * avg_dLdc + momentum * dLdc
        
        # w = w - stepSize * ave_grad
        W = W - stepSize * avg_dLdW
        V = V - stepSize * avg_dLdV
        b = b - stepSize * avg_dLdb
        c = c - stepSize * avg_dLdc

        if i % 100 == 0 :
            print(f"{M} --> Iter {i} done")

    Y_pred = predictY(X_test, W, V, b, c)
    return Y_pred

if __name__ == '__main__':
    data = np.load("data.npz")
    X = data["X_trn"]
    Y = data["y_trn"]
    X_test = data["X_tst"]

    kf = KFold(n_splits=2,shuffle=True)
    kf.get_n_splits(X)

    kFoldForTraining = []
    kFoldForTesting = []

    # Split and store the splits so that the same splits can be used for all values of K.
    for train_index, test_index in kf.split(X):
        kFoldForTraining.append(train_index)
        kFoldForTesting.append(test_index)

    X_train_1, X_test_1 = X[kFoldForTraining[0]], X[kFoldForTesting[0]]
    Y_train_1, Y_test_1 = Y[kFoldForTraining[0]], Y[kFoldForTesting[0]]    

    X_train_2, X_test_2 = X[kFoldForTraining[1]], X[kFoldForTesting[1]]
    Y_train_2, Y_test_2 = Y[kFoldForTraining[1]], Y[kFoldForTesting[1]]    

    BestM = 0
    ClassErr = 1
    for m in (5, 40, 70) :
        Y_pred2 = gradient_descent_train(m,X_train_1, Y_train_1, X_train_2)
        Y_pred1 = gradient_descent_train(m,X_train_2, Y_train_2, X_train_1)

        classErrForThisM = ((1 - accuracy_score(Y_train_2, Y_pred2)) + (1 - accuracy_score(Y_train_1, Y_pred1)))/2

        if(ClassErr > classErrForThisM) :
            BestM = m
            ClassErr = classErrForThisM

    Y_pred = gradient_descent_train(BestM, X, Y, X_test)
    write_csv(Y_pred, "sample.csv")
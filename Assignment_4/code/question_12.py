import numpy as np
from matplotlib import pyplot as plt

from preface import X_trn, Y_trn, get_poly_expansion
from question_8 import get_poly_kernel
from question_10 import train_kernel_ridge
from question_11 import eval_kernel_ridge

def kernel_ridge_reg():
    for P in [1, 2, 3, 5, 10]:
        # Define polynomial kernel function
        k = get_poly_kernel(P)
        
        # Train Kernel Ridge Regression Model
        a = train_kernel_ridge(X_trn, Y_trn, 0.1, k)
        print(f"P: {P}")
        Y = []
        X_new = np.linspace(0,15,200)
        
        # Evaluate for new data
        for X in X_new:
            Y.append(eval_kernel_ridge(X_trn, X, a, k))
        Y = np.array(Y)
        
        # Plot results
        plt.figure(1, figsize=(6, 4))
        plt.scatter(X_trn, Y_trn, s=20, label="Training Output")
        plt.scatter(X_new, Y, s=10, label="Learned Function Output")
        plt.xlim([0, 15])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(np.arange(0, 16, 1))
        plt.legend()
        plt.show()

kernel_ridge_reg()
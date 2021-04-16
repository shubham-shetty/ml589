import numpy as np
from matplotlib import pyplot as plt

from preface import X_trn, Y_trn, get_poly_expansion
from question_6 import train_basis_expanded_ridge

def basis_expanded_ridge_reg():
    for P in [1, 2, 3, 5, 10]:
        # Define polynomial basis expansion function
        h = get_poly_expansion(P)
        
        # Train basis expanded ridge regression function
        W = train_basis_expanded_ridge(X_trn, Y_trn, 0.1, h)
        
        # Print optimal weights and generate plot
        print(f"P: {P}\nW: {W}\n")
        Y = []
        x_new = np.linspace(0,15,200)
        H = h(x_new)
        for x in x_new:
            Y.append(eval_basis_expanded_ridge(x,W,h))
        Y = np.array(Y)
        plt.figure(1, figsize=(6, 4))        
        plt.scatter(X_trn, Y_trn, s=20, label="Training Output")
        plt.scatter(x_new, Y, s=10, label="Learned Function Output")
        plt.xlim([0, 15])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(np.arange(0, 16, 1))
        plt.legend()
        plt.show()

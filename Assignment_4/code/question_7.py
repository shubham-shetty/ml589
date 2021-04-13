import numpy as np
from matplotlib import pyplot as plt

from preface import X_trn, Y_trn, get_poly_expansion
from question_6 import train_basis_expanded_ridge

for P in [1, 2, 3, 5, 10]:
    h = get_poly_expansion(P)
    W = train_basis_expanded_ridge(X_trn, Y_trn, 0.1, h)
    print(f"P: {P}; W: {W}\n")

    Y = []
    H = h(X_trn)
    for i in range(len(X_trn)):
        Y.append(np.dot(W, H[i]))
    Y = np.array(Y)

    plt.scatter(X_trn, Y_trn)
    plt.scatter(X_trn, Y)
    plt.xlim([0, 15])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(np.arange(0, 16, 1))
    plt.show()


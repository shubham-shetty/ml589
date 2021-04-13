import numpy as np
from matplotlib import pyplot as plt

from preface import X_trn, Y_trn, get_poly_expansion
from question_8 import get_poly_kernel
from question_10 import train_kernel_ridge
from question_11 import eval_kernel_ridge

# fixme 10 - singular matrix error
for P in [1, 2, 3, 5, 10]:
    k = get_poly_kernel(P)
    a = train_kernel_ridge(X_trn, Y_trn, 0.1, k)
    print(f"P: {P}\n")

    Y = []
    for i in range(len(X_trn)):
        Y.append(eval_kernel_ridge(X_trn, X_trn[i], a, k))
    Y = np.array(Y)

    plt.scatter(X_trn, Y_trn)
    plt.scatter(X_trn, Y)
    plt.xlim([0, 15])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(np.arange(0, 16, 1))
    plt.show()
    
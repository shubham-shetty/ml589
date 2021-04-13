from preface import *

def eval_kernel_ridge(X_trn, x, a, k) :
    N = X_trn.shape[0]
    y = 0

    # y = sum(a_n * k(X_trn[n],x)).
    for n in range(N) :
        y += a[n] * k(X_trn[n], x)

    return y

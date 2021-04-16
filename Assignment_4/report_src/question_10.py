from preface import *

def train_kernel_ridge(X,Y,l,k) :
    N = X.shape[0]
    
    # Compute K
    K = np.empty((N,N))
    for n in range(N) :
        for m in range(N) :
            K[n][m] = k(X[n], X[m])

    a = np.linalg.solve(K + l*np.identity(N), Y)
    return a

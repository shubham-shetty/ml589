from preface import *

def train_kernel_ridge(X,Y,λ,k) :
    N = X.shape[0]
    
    # Compute K
    K = np.empty((N,N))
    for n in range(N) :
        for m in range(N) :
            K[n][m] = k(X[n], X[m])

    α = np.linalg.solve(K + λ*np.identity(N), Y)
    return α
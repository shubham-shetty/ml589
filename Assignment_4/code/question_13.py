from preface import *
from question_8 import *
from question_10 import *

def is_kernel_valid(X_trn, Y_trn):
    print("P\tPositive Definite Term")
    for P in [1, 2, 3, 5, 10]:
        k = get_poly_kernel(P)
        a = train_kernel_ridge(X_trn, Y_trn, 0.1, k)
        N = X_trn.shape[0]
        K = np.empty((N,N))
        for n in range(N) :
            for m in range(N) :
                K[n][m] = k(X_trn[n], X_trn[m])
        mer = round(np.matmul(np.matmul(a.transpose(),K),a),10)
        print(f"{P}\t{mer}\n")

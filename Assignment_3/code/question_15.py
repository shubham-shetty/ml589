import autograd.numpy as np

def prediction_loss_full(X,Y,W,V,b,c,l):
    WX = np.matmul(W, np.transpose(X))
    B = np.array([b, ] * X.shape[0]).transpose()
    C = np.array([c, ] * X.shape[0]).transpose()
    B_plus_WX = np.add(B, WX)
    sigma = np.tanh(B_plus_WX)
    F = np.add(C, np.matmul(V, sigma))
    
    i = 0
    L = 0
    for i in range(Y.shape[0]) :
        softmax = np.log(np.sum(np.exp(F[:,i])))
        y = Y[i]
        L += -F[y][i] + softmax

    L += l*(np.sum(np.square(V)) + np.sum(np.square(W)))
    return L






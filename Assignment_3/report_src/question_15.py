from autograd import numpy as np

def prediction_loss_full(X,Y,W,V,b,c,l):
    WX = np.matmul(W, np.transpose(X))
    b = np.array([b, ] * X.shape[0]).transpose()
    c = np.array([c, ] * X.shape[0]).transpose()
    b_plus_WX = np.add(b, WX)
    sigma = np.tanh(b_plus_WX)
    f = np.add(c, np.matmul(V, sigma))

    L = 0
    for i in range(Y.shape[0]) :
        softmax = np.log(np.sum(np.exp(f[:,i])))
        y = Y[i]
        L += -f[y][i] + softmax
    
    L += l*(np.sum(np.square(V)) + np.sum(np.square(W)))
    return L

from autograd import grad
from autograd import numpy as np

def prediction_loss(x,y,W,V,b,c):
    Wx = np.matmul(W, x)
    b_plus_Wx = np.add(b, Wx)
    sigma = np.tanh(b_plus_Wx)
    f = np.add(c, np.matmul(V, sigma))
    softmax = 0
    for item in f:
        softmax += np.exp(item)

    softmax = np.log(softmax)
    return -f[y] + softmax

def prediction_grad_autograd(x,y,W,V,b,c):
    dLdW = grad(prediction_loss, 2)(x,y,W,V,b,c)
    dLdV = grad(prediction_loss, 3)(x,y,W,V,b,c)
    dLdb = grad(prediction_loss, 4)(x,y,W,V,b,c)
    dLdc = grad(prediction_loss, 5)(x,y,W,V,b,c)
    return dLdW, dLdV, dLdb, dLdc

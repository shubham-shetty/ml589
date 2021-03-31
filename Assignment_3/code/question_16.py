import autograd.numpy as np
from autograd import grad
from question_15 import prediction_loss_full


def prediction_grad_autograd_full(X,Y,W,V,b,c,l):
    dLdc_func = grad(prediction_loss_full,5)
    dldc = dLdc_func(X,Y,W,V,b,c,l)
    dLdW_func = grad(prediction_loss_full,2)
    dldW = dLdW_func(X,Y,W,V,b,c,l)
    dLdb_func = grad(prediction_loss_full,4)
    dLdb = dLdb_func(X,Y,W,V,b,c,l)
    dLdV_func = grad(prediction_loss_full,3)
    dldV = dLdV_func(X,Y,W,V,b,c,l)

    return dldW, dldV, dLdb, dldc
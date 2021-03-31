from autograd import grad

def prediction_grad_full(X,Y,W,V,b,c,l):
    dLdW = grad(prediction_loss_full, 2)(X, Y, W, V, b, c, l)
    dLdV = grad(prediction_loss_full, 3)(X, Y, W, V, b, c, l)
    dLdb = grad(prediction_loss_full, 4)(X, Y, W, V, b, c, l)
    dLdc = grad(prediction_loss_full, 5)(X, Y, W, V, b, c, l)

    return dLdW, dLdV, dLdb, dLdc

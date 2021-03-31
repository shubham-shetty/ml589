from autograd import grad

from code.question_11 import prediction_loss


def prediction_grad_autograd(x, y, W, V, b, c):
    dLdW = grad(prediction_loss, 2)(x, y, W, V, b, c)
    dLdV = grad(prediction_loss, 3)(x, y, W, V, b, c)
    dLdb = grad(prediction_loss, 4)(x, y, W, V, b, c)
    dLdc = grad(prediction_loss, 5)(x, y, W, V, b, c)
    return dLdW, dLdV, dLdb, dLdc

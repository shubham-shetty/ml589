from question_16 import *
from datetime import datetime
import pandas as pd
import sys

def gradient_descent_train() :
    data = np.load("data.npz")
    X = data["X_trn"]
    Y = data["y_trn"]
    numLabels = 4
    momentum = 0.1
    l = 1
    stepSize = 0.0001
    M = int(sys.argv[1])

    start = datetime.now()
    start_string = start.strftime("%d/%m/%Y %H:%M:%S")
    print(f"Time - {start_string}. Starting descent for M={M} now")
    W = np.random.normal(0,1,(M,X.shape[1]))
    b = np.zeros(M)
    c = np.zeros(numLabels)
    V = np.random.normal(0,1,(numLabels, M))
    l = 1
    avg_dLdW = avg_dLdV = avg_dLdb = avg_dLdc = 0
    errors = []
    for i in range(50) :
        # ave_grad = (1 - momentum) * ave_grad + momentum * ∇h(w)
        dLdW, dLdV, dLdb, dLdc = prediction_grad_autograd_full(X,Y,W,V,b,c,l)
        avg_dLdW = (1 - momentum) * avg_dLdW + momentum * dLdW
        avg_dLdV = (1 - momentum) * avg_dLdV + momentum * dLdV
        avg_dLdb = (1 - momentum) * avg_dLdb + momentum * dLdb
        avg_dLdc = (1 - momentum) * avg_dLdc + momentum * dLdc
        
        # w = w - stepSize * ave_grad
        W = W - stepSize * avg_dLdW
        V = V - stepSize * avg_dLdV
        b = b - stepSize * avg_dLdb
        c = c - stepSize * avg_dLdc

        errors.append([i, prediction_loss_full(X, Y, W, V, b, c, l)])
        
    end = datetime.now()
    print(f"Total time taken for M = {M} is {end - start}")
    df = pd.DataFrame(np.array(errors),columns=['Iterations', 'Regularized loss'])
    df.plot(x="Iterations", title="Regularized loss vs Iterations")

print("Hello")
gradient_descent_train()

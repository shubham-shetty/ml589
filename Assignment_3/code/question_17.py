from question_16 import *
from preface import *
from datetime import datetime
import pandas as pd
import sys
import matplotlib.pyplot as plt
import math
import multiprocessing

def gradient_descent_train(M) :
    data = np.load("data.npz")
    X = data["X_trn"]
    Y = data["y_trn"]
    numLabels = 4
    momentum = 0.1
    l = 1
    stepSize = 0.0001
    errors = []
    numIter = 1000
    start = datetime.now()
    start_string = start.strftime("%d/%m/%Y %H:%M:%S")
    print(f"Time - {start_string}. Starting descent for M={M} now")
    W = np.random.normal(0,1,(M,X.shape[1]))
    b = np.zeros(M)
    c = np.zeros(numLabels)
    V = np.random.normal(0,1,(numLabels, M))
    avg_dLdW = avg_dLdV = avg_dLdb = avg_dLdc = 0
    for i in range(numIter) :
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

        errors.append(prediction_loss_full(X, Y, W, V, b, c, l))

        if i % 100 == 0 :
            print(f"{M} --> Iteration {i} done")
        
    end = datetime.now()
    diff = end - start
    elapsed = int((diff.seconds * 1000) + (diff.microseconds / 1000))

    print(f"Total time taken for M = {M} is {diff}")
    return elapsed, errors

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=3)
    result_list = pool.map(gradient_descent_train, [5, 40, 70])
    
    runTime = []
    runTime.append([5, result_list[0][0]])
    runTime.append([40, result_list[1][0]])
    runTime.append([70, result_list[2][0]])

    prettyPrintTable(runTime, ["M","Run time in ms"])

    a = np.arange(1000)
    errors = np.array([a, result_list[0][1], result_list[1][1], result_list[2][1]]).T
    df = pd.DataFrame(np.array(errors),columns=['Iterations', 'Regularized loss for M=5', 'Regularized loss for M=50', 'Regularized loss for M=70'])
    df.plot(x="Iterations", title="Regularized loss vs Iterations")
    plt.savefig('q17.png')

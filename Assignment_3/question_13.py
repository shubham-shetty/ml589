import numpy as np
from question_12 import *

def printGradientValues() :
    x=np.array([1,2])
    y=1
    W=np.array([[0.5, -1],[-0.5, 1],[1, 0.5]])
    V=np.array([[-1, -1, 1],[1,1,1]])
    b=np.array([0,0,0])
    c=np.array([0,0])

    
    dLdW, dLdV, dLdb, dLdc = prediction_grad(x,y,W,V,b,c)

    print("Gradients are as below :")
    print("dLdW :")
    print(dLdW)
    print()
    print("dLdV :")
    print(dLdV)
    print()
    print("dLdb :")
    print(dLdb)
    print()
    print("dLdc :")
    print(dLdc)

from preface import *
from question_11 import predict_MAP

def MAP_MSE(X,Y,x_test,y_test):
    mse = np.sum(np.square((np.array(y_test) - np.array(list(map(lambda x:predict_MAP(x,X,Y), x_test))))))/100
    return mse
    
MAP_MSE(X,Y,x,y)

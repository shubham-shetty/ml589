def Bayes_MSE(X,Y,x_test,y_test):
    mse = np.sum(np.square((np.array(y_test) - np.array(list(map(lambda x:predict_Bayes(x,X,Y), x_test))))))/100
    return mse
    
print(f"Bayes_MSE: {Bayes_MSE(X,Y,x,y)}")
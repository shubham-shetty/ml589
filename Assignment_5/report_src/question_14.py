def f_m(x,m):
    # Predictor Function
    if m>0:
        return x**m
    else:
        return 0
        
def predict_Bayes(x,X,Y):
    m_i = [0,1,2,3,4,5,6,7,8,9]
    f = np.sum(np.array([posterior(X,Y,m)*f_m(x,m) for m in m_i]))
    return f
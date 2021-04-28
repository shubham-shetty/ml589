from preface import *
from question_7 import posterior

def predict_Bayes(x,X,Y):
    m_i = [0,1,2,3,4,5,6,7,8,9]
    f = np.sum(np.array([posterior(X,Y,m)*f_m(x,m) for m in m_i]))
    return f

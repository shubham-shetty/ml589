from preface import *
from question_9 import MAP

def predict_MAP(x,X,Y):
    m = MAP(X,Y)
    f = x ** m if m > 0 else 0
    return f

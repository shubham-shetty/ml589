from question_7 import posterior

def MAP(X,Y):
    m = 0
    pos_m = posterior(X, Y, m)
    for i in range(1,10) :
        pos_i = posterior(X, Y, i)
        if pos_i > pos_m :
            m = i
            pos_m = pos_i
    return m
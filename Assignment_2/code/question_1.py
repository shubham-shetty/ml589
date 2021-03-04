from preface import *

# KNN Regression
def KNN_reg_predict(X_trn, y_trn, x, K):
    
    # Algorithm - 
    # 1. Calculate distance of x from each element of training data
    # 2. Find K closest elements
    # 3. Take average of y for closest elements as final prediction
    
    # Calculating euclidean distance for each element
    dist_vector = []
    for X in X_trn:
        dist_vector.append(euc_dist(X, x, 3))
    
    # Sorting by distance to find closest neighbours
    dist_vector2 = list(enumerate(dist_vector))
    dist_sorted = sorted(dist_vector2, key=lambda x:x[1])
    
    # Taking average to return final prediction
    y_new=0
    for i in range(K):
        y_new += y_trn[dist_sorted[i][0]]/K
    return y_new
    
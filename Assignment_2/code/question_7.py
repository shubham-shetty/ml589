# Predicting Value via Regression Stump
def reg_stump_predict(x, dim, thresh, c_left, c_right):
    y = []
    for i in x:
        if i[dim] <= thresh:
            y.append(c_left)
        else:
            y.append(c_right)
    return y

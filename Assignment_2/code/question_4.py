# Training Ridge Regression Model
def linear_reg_train(X_trn, y_trn, l):
    xt = X_trn.transpose()
    xtx = np.matmul(xt, X_trn)
    xty = np.matmul(xt, y_trn)
    li = l*np.identity(X_trn.shape[1])
    w = np.linalg.solve(xtx+li, xty)
    return w

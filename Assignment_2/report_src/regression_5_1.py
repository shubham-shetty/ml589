# MSE for Linear Ridge Regression
def linear_reg_sq_error(X_trn, y_trn, X_val, y_val):
    l = [0, 0.001, 0.01, 0.1, 1, 10]
    eval_sq_error = []
    for i in l:
        linear_reg_trn = linear_reg_predict(X_trn, linear_reg_train(X_trn, y_trn, i))
        linear_reg_tst = linear_reg_predict(X_val, linear_reg_train(X_trn, y_trn, i))
        eval_sq_error.append([i, sq_error(linear_reg_trn, y_trn), sq_error(linear_reg_tst,y_val)])
    df = pd.DataFrame(np.array(eval_sq_error),columns=['Lambda (l)', 'Training Data Error', 'Test Data Error'])
    return(df)
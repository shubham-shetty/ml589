# MSE for KNN Regression
def KNN_reg_sq_error(X_trn, y_trn, X_val, y_val, K):
    eval_sq_error = []
    for i in range(1, K+1):
        prediction_trn = [KNN_reg_predict(X_trn, y_trn, j, i) for j in X_trn]
        prediction_tst = [KNN_reg_predict(X_trn, y_trn, j, i) for j in X_val]
        eval_sq_error.append([(i), sq_error(prediction_trn, y_trn), sq_error(prediction_tst,y_val)])
    df = pd.DataFrame(np.array(eval_sq_error),columns=['K', 'Training Data Error', 'Test Data Error'])
    return(df)
  
df = KNN_reg_sq_error(X_trn, y_trn, X_val, y_val, 10)
print(df)
df.plot(x="K", title="MSE for KNN Regression");

# MAE for KNN Regression
def KNN_reg_abs_error(X_trn, y_trn, X_val, y_val, K):
    eval_abs_error = []
    for i in range(1, K+1):
        prediction_trn = [KNN_reg_predict(X_trn, y_trn, j, i) for j in X_trn]
        prediction_tst = [KNN_reg_predict(X_trn, y_trn, j, i) for j in X_val]
        eval_abs_error.append([int(i), abs_error(prediction_trn, y_trn), abs_error(prediction_tst,y_val)])
    df = pd.DataFrame(np.array(eval_abs_error),columns=['K', 'Training Data Error', 'Test Data Error'])
    return(df)
  
df = KNN_reg_abs_error(X_trn, y_trn, X_val, y_val, 10)
print(df)
df.plot(x="K", title="MAE for KNN Regression");

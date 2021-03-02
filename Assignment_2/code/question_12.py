# Linear Ridge Regression for real data
def linear_ridge_reg_real(X_big_trn, y_big_trn, X_big_val, y_big_val):
    l = [0, 1, 10, 100, 1000, 10000]
    eval_sq_error = []
    for i in l:
        ridge_reg = linear_model.Ridge(alpha=i)
        ridge_reg.fit(X_big_trn, y_big_trn)
        linear_reg_trn = linear_reg_predict(X_big_trn, ridge_reg.coef_)
        linear_reg_tst = linear_reg_predict(X_big_val, ridge_reg.coef_)
        eval_sq_error.append([i, mean_squared_error(linear_reg_trn, y_big_trn), mean_squared_error(linear_reg_tst,y_big_val)])
    df = pd.DataFrame(np.array(eval_sq_error),columns=['Lambda (l)', 'Training Data Error', 'Test Data Error'])
    return(df)
  
df = linear_ridge_reg_real(X_big_trn, y_big_trn, X_big_val, y_big_val)
print(df)
df.plot(x="Lambda (l)", title="MSE for Linear Ridge Regression on Big Data");

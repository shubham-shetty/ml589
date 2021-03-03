df = linear_reg_sq_error(X_trn, y_trn, X_val, y_val)
print(df)
df.plot(x="Lambda (l)", title="MSE for Linear Ridge Regression");
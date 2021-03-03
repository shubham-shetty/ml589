df = linear_reg_abs_error(X_trn, y_trn, X_val, y_val)
print(df)
df.plot(x="Lambda (l)", title="MAE for Linear Ridge Regression");
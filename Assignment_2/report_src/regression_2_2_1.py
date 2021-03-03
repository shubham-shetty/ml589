df = KNN_reg_abs_error(X_trn, y_trn, X_val, y_val, 10)
print(df)
df.plot(x="K", title="MAE for KNN Regression");
# Regression Tree for real data
def reg_tree_real(X_big_trn, y_big_trn, X_big_val, y_big_val):
    max_depth = 5
    sq_errs = []
    
    # Fit model
    for i in range(1,max_depth+1):
        tree1 = DecisionTreeRegressor(max_depth=i)
        tree1.fit(X_big_trn, y_big_trn)
        # Predict
        y_pred_trn = tree1.predict(X_big_trn)
        y_pred_tst = tree1.predict(X_big_val)
        # Errors
        sq_errs.append([i, mean_squared_error(y_big_trn, y_pred_trn), mean_squared_error(y_big_val, y_pred_tst)])
    df = pd.DataFrame(np.array(sq_errs),columns=['Depth', 'Training Data Error', 'Test Data Error'])
    return df
  
df = reg_tree_real(X_big_trn, y_big_trn, X_big_val, y_big_val)
print(df)
df.plot(x="Depth", title="MSE for Regression Tree on Big Data");

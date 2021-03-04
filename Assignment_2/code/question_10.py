from preface import *

# KNN Regression for real data
def KNN_reg_real(X_big_trn, y_big_trn, X_big_val, y_big_val):
    K = [1, 2, 5, 10, 20, 50]
    sq_errs = []
    for i in K:
        neigh = KNeighborsRegressor(n_neighbors=i)
        neigh.fit(X_big_trn, y_big_trn)
        sq_errs.append([i, mean_squared_error(y_big_trn, neigh.predict(X_big_trn)), mean_squared_error(y_big_val, neigh.predict(X_big_val))])
    df = np.array(sq_errs)
    return df
  
df = KNN_reg_real(X_big_trn, y_big_trn, X_big_val, y_big_val)
print("MSE for KNN Regression on Big Data")
prettyPrintTable(df, "K", "Training Data Error", "Test Data Error")


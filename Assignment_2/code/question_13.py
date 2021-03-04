from preface import *
from question_3 import *

# Linear Lasso Regression for real data
def linear_lasso_reg_real(X_big_trn, y_big_trn, X_big_val, y_big_val):
    l = [0, 0.1, 1, 10, 100, 1000]
    N = X_big_trn.shape[0]
    eval_sq_error = []
    for i in l:
        lasso_reg = linear_model.Lasso(alpha=i/(2*N))
        lasso_reg.fit(X_big_trn, y_big_trn)
        linear_reg_trn = lasso_reg.predict(X_big_trn)
        linear_reg_tst = lasso_reg.predict(X_big_val)
        eval_sq_error.append([i, mean_squared_error(linear_reg_trn, y_big_trn), mean_squared_error(linear_reg_tst,y_big_val)])
    df = np.array(eval_sq_error)
    return(df)
  
df = linear_lasso_reg_real(X_big_trn, y_big_trn, X_big_val, y_big_val)
print("MSE for Linear Lasso Regression on Big Data")
prettyPrintTable(df, "Lambda (l)", "Training Data Error", "Test Data Error")

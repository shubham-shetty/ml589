from preface import *
from question_3 import *
from question_4 import *

# MSE for Linear Ridge Regression
def linear_reg_sq_error(X_trn, y_trn, X_val, y_val):
    l = [0, 0.001, 0.01, 0.1, 1, 10]
    eval_sq_error = []
    for i in l:
        linear_reg_trn = linear_reg_predict(X_trn, linear_reg_train(X_trn, y_trn, i))
        linear_reg_tst = linear_reg_predict(X_val, linear_reg_train(X_trn, y_trn, i))
        eval_sq_error.append([i, sq_error(linear_reg_trn, y_trn), sq_error(linear_reg_tst,y_val)])
    df = np.array(eval_sq_error)
    return(df)

df = linear_reg_sq_error(X_trn, y_trn, X_val, y_val)
print("MSE for Linear Ridge Regression on small data")
prettyPrintTable(df, "Lambda (l)", "Training Data Error", "Test Data Error")


# MAE for Linear Ridge Regression
def linear_reg_abs_error(X_trn, y_trn, X_val, y_val):
    l = [0, 0.001, 0.01, 0.1, 1, 10]
    eval_abs_error = []
    for i in l:
        linear_reg_trn = linear_reg_predict(X_trn, linear_reg_train(X_trn, y_trn, i))
        linear_reg_tst = linear_reg_predict(X_val, linear_reg_train(X_trn, y_trn, i))
        eval_abs_error.append([i, abs_error(linear_reg_trn, y_trn), abs_error(linear_reg_tst,y_val)])
    return np.array(eval_abs_error)
  
df = linear_reg_abs_error(X_trn, y_trn, X_val, y_val)
print("\n")
print("MAE for Linear Ridge Regression on small data")
prettyPrintTable(df, "Lambda (l)", "Training Data Error", "Test Data Error")

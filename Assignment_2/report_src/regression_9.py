# MSE & MAE for Regression Stump
def reg_stump_sq_error(X_trn, y_trn, X_val, y_val):
    eval_error = []
    dim, thresh, c_left, c_right = reg_stump_train(X_trn, y_trn)
    pred_trn = reg_stump_predict(X_trn, dim, thresh, c_left, c_right)
    pred_tst = reg_stump_predict(X_val, dim, thresh, c_left, c_right)
    eval_error.append([sq_error(pred_trn, y_trn), sq_error(pred_tst,y_val)])
    eval_error.append([abs_error(pred_trn, y_trn), abs_error(pred_tst,y_val)])
    df = pd.DataFrame(np.array(eval_error),columns=['Training Data Error', 'Test Data Error'],index=['MSE','MAE'])
    return df
    
# Generate Table
reg_stump_sq_error(X_trn, y_trn, X_val, y_val)
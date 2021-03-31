# Custom softmax function to obtain probabilities from confidence scores
def softmax(x):
    return np.exp(x)/sum(np.exp(x))


# Function to return required mtrics (0-1 misclassification error, Logistic loss, Hinge Loss)
def getMetrics(l, kf, clf):
   misc_error, ll_aggr, hl_aggr  = 0, 0, 0
   for trn_index, tst_index in kf.split(X_trn):
       # Split training data accordingly to training and validation data sets
       cv_x_trn, cv_x_tst = X_trn[trn_index], X_trn[tst_index]
       cv_y_trn, cv_y_tst = y_trn[trn_index], y_trn[tst_index]
       
       # Fit model to training set 
       model = clf.fit(cv_x_trn, cv_y_trn)
       
       # Decision function to generate confidence values
       decisions = model.decision_function(cv_x_tst)
       cv_pred_tst = np.argmax(decisions, axis=1)
       
       # Probability values for decisions
       probs = softmax(decisions)
       
       # Calculate miscalculation error for this particular fold
       error_1 = 0
       for x,y in zip(cv_y_tst, cv_pred_tst):
           if x!=y:
               error_1+=1
       misc_error += error_1/len(cv_pred_tst)
       
       
       # Calculate Log Loss for this particular fold
       ll = log_loss(cv_y_tst, probs)
       ll_aggr += ll
       
       # Calculate Hinge Loss for this particular fold
       hl = hinge_loss(cv_y_tst, decisions)
       hl_aggr += hl
       
   # Calculate average metrics across splits
   avg_misc_error = misc_error/kf.n_splits
   avg_ll = ll_aggr/kf.n_splits
   avg_hl = hl_aggr/kf.n_splits
   
   return avg_misc_error, avg_ll, avg_hl


# Validate Logistic Regression Model
def logistic_regressor():
   l_vals=[0.0001, 0.01, 1, 10, 100]
   
   # Define splits for K-Fold Cross-Validation
   n_splits=5
   # KFold Cross Validator
   kf = KFold(n_splits=n_splits, shuffle=True)
   
   misc_err = []
   log_loss = []
   hinge_loss = []
   
   for l in l_vals:
       log_reg = LogisticRegression(penalty='l2', C=1/(2*l), solver='sag', max_iter=100, multi_class='multinomial',
                                    n_jobs = -1)
       me, ll, hl = getMetrics(l, kf, log_reg)
       misc_err.append(me)
       log_loss.append(ll)
       hinge_loss.append(hl)
       
   print("Lambda\t\t\tError Value")
   for l in range(len(l_vals)):
       print(f"{l_vals[l]}\t\t\t{misc_err[l]}")
   print("\n")
   print("Lambda\t\t\tLog Loss")
   for l in range(len(l_vals)):
       print(f"{l_vals[l]}\t\t\t{log_loss[l]}")
   print("\n")
   print("Lambda\t\t\tHinge Loss")
   for l in range(len(l_vals)):
       print(f"{l_vals[l]}\t\t\t{hinge_loss[l]}")


# Validate Hinge Classification Model
def hinge_classifier():
    l_vals=[0.0001, 0.01, 1, 10, 100]
    
    # Define splits for K-Fold Cross-Validation
    n_splits=5
    # KFold Cross Validator
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    misc_err = []
    log_loss = []
    hinge_loss = []
    
    for l in l_vals:
        hinge_clf = LinearSVC(loss= 'hinge', C = 1/(2*l), max_iter=1000)
        me, ll, hl = getMetrics(l, kf, hinge_clf)
        misc_err.append(me)
        log_loss.append(ll)
        hinge_loss.append(hl)
        
    print("Lambda\t\t\tError Value")
    for l in range(len(l_vals)):
        print(f"{l_vals[l]}\t\t\t{misc_err[l]}")
    print("\n")
    print("Lambda\t\t\tLog Loss")
    for l in range(len(l_vals)):
        print(f"{l_vals[l]}\t\t\t{log_loss[l]}")
    print("\n")  
    print("Lambda\t\t\tHinge Loss")
    for l in range(len(l_vals)):
        print(f"{l_vals[l]}\t\t\t{hinge_loss[l]}")


if __name__ == "__main__":
	logistic_regressor()
	hinge_classifier()
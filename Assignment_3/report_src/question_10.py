def logistic_regression_final():
    l=100
    log_reg = LogisticRegression(penalty='l2', C=1/(2*l), solver='sag', max_iter=100, multi_class='multinomial',
                                     n_jobs = -1)
    model = log_reg.fit(X_trn, y_trn)
    y_pred = model.predict(X_tst)
    write_csv(y_pred, "Shubham_answer10.csv")

if __name__ == "__main__":
	logistic_regression_final()
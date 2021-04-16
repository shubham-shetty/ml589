import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from preface import X_trn_real, Y_trn_real
from sklearn.metrics import hinge_loss

n_splits = 5
kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
loss = [['gamma/lambda', 2, 20, 200], [1], [0.01], [0.001]]

for i, g in enumerate([1, 0.01, 0.001]):
    for l in [2,20,200] :
        clf = SVC(C=l, gamma=g)
        error = 0
        errorTable = []
        err_hinge_loss = 0
        for train_index, test_index in kf.split(X_trn_real):
            # Split train-test
            X_train, X_test = X_trn_real[train_index], X_trn_real[test_index]
            y_train, y_test = Y_trn_real[train_index], Y_trn_real[test_index]

            # Train the model
            model = clf.fit(X_train, y_train)
            predictions = model.decision_function(X_test)

            error += hinge_loss(y_test, predictions)
        
        loss[i + 1].append(error/5)

for row in loss:
        print('\t'.join([str(elem).center(24) for elem in row]))
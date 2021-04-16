from sklearn import svm
from sklearn.metrics import hinge_loss
from sklearn.model_selection import KFold

from preface import Y_trn_real, X_trn_real


def train_svm(X_trn, y_trn):
    l_vals = [2, 20, 200]
    splits = 5
    kf = KFold(n_splits=splits, shuffle=True)
    print("Lambda\t\tHinge Loss\n")
    for l in l_vals:
        clf = svm.SVC(kernel='linear', C=1 / (2 * l))
        sum_hinge_loss, avg_hinge_loss = 0, 0
        for train_index, test_index in kf.split(X_trn):
            # Split train-test
            X_train, X_test = X_trn[train_index], X_trn[test_index]
            y_train, y_test = y_trn[train_index], y_trn[test_index]

            # Train the model
            clf.fit(X_train, y_train)
            predictions = clf.decision_function(X_test)

            sum_hinge_loss += hinge_loss(y_test, predictions)
        avg_hinge_loss = sum_hinge_loss / splits
        print(f"{l}\t\t{avg_hinge_loss}")


print("\n\nMean validation-set hinge loss using linear kernel")
train_svm(X_trn_real, Y_trn_real)

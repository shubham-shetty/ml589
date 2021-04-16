from sklearn import svm
from sklearn.metrics import hinge_loss
from sklearn.model_selection import KFold

from preface import Y_trn_real, X_trn_real


def train_svm_poly(X_trn, y_trn, l, P, g):
    splits = 5
    kf = KFold(n_splits=splits, shuffle=True)
    clf = svm.SVC(kernel='poly', C=1 / (2 * l), degree=P, coef0=g, gamma=1)
    sum_hinge_loss = 0
    for train_index, test_index in kf.split(X_trn):
        # Split train-test
        X_train, X_test = X_trn[train_index], X_trn[test_index]
        y_train, y_test = y_trn[train_index], y_trn[test_index]

        # Train the model
        clf.fit(X_train, y_train)
        predictions = clf.decision_function(X_test)

        sum_hinge_loss += hinge_loss(y_test, predictions)
    return sum_hinge_loss / splits


def compute_hinge_loss(P):
    loss = [['gamma/lambda', 2, 20, 200], [0.001], [0.01], [1]]
    for i, g in enumerate([0.001, 0.01, 1]):
        for l in [2, 20, 200]:
            loss[i + 1].append(train_svm_poly(X_trn=X_trn_real, y_trn=Y_trn_real, l=l, P=P, g=g))

    for row in loss:
        print('\t'.join([str(elem).center(24) for elem in row]))

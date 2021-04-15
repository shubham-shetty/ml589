from sklearn import svm
from sklearn.model_selection import KFold

from preface import Y_trn_real, X_trn_real


def train_svm_poly(X_trn, y_trn, l, P, g):
    splits = 5
    kf = KFold(n_splits=splits, shuffle=True)
    clf = svm.SVC(kernel='poly', C=1 / l, degree=P, coef0=g)
    hinge_loss = 0
    for train_index, test_index in kf.split(X_trn):
        # Split train-test
        X_train, X_test = X_trn[train_index], X_trn[test_index]
        y_train, y_test = y_trn[train_index], y_trn[test_index]

        # Train the model
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        hinge_loss += sum([max(0, 1 - actual * predicted) for actual, predicted in zip(y_test, predictions)]) / len(
            X_test)
    return hinge_loss / splits


def compute_hinge_loss(P):
    loss = [['gamma/lambda', 2, 20, 200], [1], [0.01], [0.001]]
    for i, g in enumerate([1, 0.01, 0.001]):
        for l in [2, 20, 200]:
            loss[i + 1].append(train_svm_poly(X_trn=X_trn_real, y_trn=Y_trn_real, l=l, P=P, g=g))

    for row in loss:
        print('\t'.join([str(elem).center(24) for elem in row]))

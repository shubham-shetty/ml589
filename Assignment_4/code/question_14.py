from sklearn import svm
from sklearn.model_selection import KFold

from preface import Y_trn_real, X_trn_real


def train_svm(X_trn, y_trn, l):
    splits = 5
    kf = KFold(n_splits=splits, shuffle=True)
    clf = svm.SVC(kernel='linear', C=1 / l)
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


loss = []
for l in [2, 20, 200]:
    loss.append([l, train_svm(X_trn=X_trn_real, y_trn=Y_trn_real, l=l)])
print(loss)

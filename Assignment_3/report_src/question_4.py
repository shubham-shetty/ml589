import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

def train_classifier(X_trn, y_trn, depths):
    errors = np.zeros((6, 2))
    n_splits = 5
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
    for ind, depth in enumerate(depths):
        clf = DecisionTreeClassifier(random_state=None, max_depth=depth)
        error = 0
        for train_index, test_index in kf.split(X_trn):
            # Split train-test
            X_train, X_test = X_trn[train_index], X_trn[test_index]
            y_train, y_test = y_trn[train_index], y_trn[test_index]

            # Train the model
            model = clf.fit(X_train, y_train)
            predictions = model.predict(X_test)

            error += 1-sum([1 for actual, predicted in zip(y_test, predictions) if actual==predicted])/len(X_test)
        errors[ind] = depth, error/n_splits

    print(errors)


data = np.load("data.npz")
X_trn = data["X_trn"]
y_trn = data["y_trn"]
X_tst = data["X_tst"]
train_classifier(X_trn, y_trn, {1, 3, 6, 9, 12, 14})

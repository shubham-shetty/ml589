import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from preface import X_trn_real, Y_trn_real, X_val_real, write_csv
from sklearn.metrics import accuracy_score

# Calculate 0-1 generalization error for this model using 5 folds
splits = 5
kf = KFold(n_splits=splits, shuffle=True)
clf = SVC(kernel='poly', C=1 / (2 * 2), degree=3, coef0=1, gamma=1)
sum_generalization_error = 0
for train_index, test_index in kf.split(X_trn_real):
    # Split train-test
    X_train, X_test = X_trn_real[train_index], X_trn_real[test_index]
    y_train, y_test = Y_trn_real[train_index], Y_trn_real[test_index]

    # Train the model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    sum_generalization_error += (1 - accuracy_score(y_test, y_pred))
generalization_error = sum_generalization_error / splits
print(f"Expected 0-1 generalization error = {generalization_error}")

# Re - train the model
clf = SVC(kernel='poly', C=1/(2*2), degree=3, coef0=1, gamma=1)
clf.fit(X_trn_real, Y_trn_real)

# Predict on test data
predictions = clf.predict(X_val_real)

write_csv(predictions, "Kaggle.csv")

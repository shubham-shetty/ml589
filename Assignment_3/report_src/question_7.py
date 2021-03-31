from preface import *

# KNN Classification
def KNN_classification(numNeighbors, X_trn, y_trn, X_val):
    neigh = KNeighborsClassifier(n_neighbors=numNeighbors, n_jobs=-1)
    neigh.fit(X_trn, y_trn)
    y_pred = neigh.predict(X_val)
    return y_pred

# Evaluate what capacity of KNN would be optimum
def KNNModelSelection() :
    data = np.load("data.npz")
    X_trn = data["X_trn"]
    y_trn = data["y_trn"]
    X_tst = data["X_tst"]

    kf = KFold(n_splits=5,shuffle=True)
    kf.get_n_splits(X_trn)

    kFoldForTraining = []
    kFoldForTesting = []

    # Split and store the splits so that the same splits can be used for all values of K.
    for train_index, test_index in kf.split(X_trn):
        kFoldForTraining.append(train_index)
        kFoldForTesting.append(test_index)

    K = [1, 3, 5, 7, 9, 11]
    classificationError = []
    for i in K:
        # For each of the splits, run KNN classification
        classErrForThisValueOfK=0
        for splitNum in range(len(kFoldForTraining)) :
            X_train, X_test = X_trn[kFoldForTraining[splitNum]], X_trn[kFoldForTesting[splitNum]]
            y_train, y_test = y_trn[kFoldForTraining[splitNum]], y_trn[kFoldForTesting[splitNum]]
            y_pred = KNN_classification(i, X_train, y_train, X_test)
            classErrForThisValueOfK = classErrForThisValueOfK + (1 - accuracy_score(y_test, y_pred))
        
        # Store the mean classification error for this value of K in the numpy array
        classificationError.append([i, classErrForThisValueOfK/5.0])

    prettyPrintTable(classificationError, ["Num neighbors","Errors"])

if __name__ == '__main__':
    KNNModelSelection()
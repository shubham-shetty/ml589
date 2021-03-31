from question_4 import print_classifier_errors
from preface import *
from question_7 import KNNModelSelection
from question_13 import printGradientValues
from question_17 import gradient_descent_train

if __name__ == '__main__':
    # To compute errors for question 4
    print_classifier_errors()

    # To generate table for question 7
    KNNModelSelection()

    # To generate gradients for question 13
    print("Answer to question 13 -")
    printGradientValues()

    # To compute gradients for question 16
    # TODO Do we need to add an example?

    # To generate table and plot for question 17
    pool = multiprocessing.Pool(processes=3)
    result_list = pool.map(gradient_descent_train, [5, 40, 70])

    runTime = []
    runTime.append([5, result_list[0][0]])
    runTime.append([40, result_list[1][0]])
    runTime.append([70, result_list[2][0]])

    print("\n\nTotal time taken for 1000 iterations of gradient descent :")
    prettyPrintTable(runTime, ["M","Run time in ms"])
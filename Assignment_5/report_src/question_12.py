x = np.loadtxt('x_test.csv')
y = np.loadtxt('y_test.csv')

def MAP_plot(X,Y,x_test,y_test):
    y_pred = list(map(lambda x:predict_MAP(x,X,Y), x_test))
    plt.scatter(x_test, y_test, marker="x", label="Test True Output")
    plt.scatter(x_test, y_pred, marker="o", label="Test Prediction Output")
    plt.xlabel("Test Input x")
    plt.ylabel("Test Output y")
    plt.legend()
    plt.show()
    
    
MAP_plot(X,Y,x,y)
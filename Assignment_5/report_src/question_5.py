X = np.loadtxt('x.csv')
Y = np.loadtxt('y.csv')


def plot_likelihood(X,Y):
    x = [0,1,2,3,4,5,6,7,8,9]
    y = [likelihood(X,Y,m) for m in x]
    plt.bar(x, y)
    plt.xlabel("m")
    plt.ylabel("Likelihood")
    plt.xticks(np.arange(0, 10, 1))
    plt.show()
    
    
plot_likelihood(X,Y)
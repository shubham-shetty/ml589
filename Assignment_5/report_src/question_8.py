def plot_posterior(X,Y):
    x = [0,1,2,3,4,5,6,7,8,9]
    y = [posterior(X,Y,m) for m in x]
    #sum_all = np.sum(np.array([posterior(X,Y,m) for m in x]))
    #print(sum_all)
    plt.bar(x, y)
    plt.xlabel("m")
    plt.ylabel("Posterior")
    plt.xticks(np.arange(0, 10, 1))
    #plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()
    

plot_posterior(X,Y)
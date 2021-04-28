def likelihood(X,Y,m):
    p = np.product(np.array(list(map(lambda x,y: likelihood_single(x,y,m), X, Y))))
    return p
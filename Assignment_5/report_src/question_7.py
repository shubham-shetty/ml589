def posterior(X,Y,m):
    m_i = [0,1,2,3,4,5,6,7,8,9]
    normalizer = np.sum(np.array([prior(i)*likelihood(X,Y,i) for i in m_i]))
    p = (likelihood(X,Y,m)*prior(m))/normalizer
    return p
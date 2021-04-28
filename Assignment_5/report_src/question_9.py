def MAP(X,Y):
    m_i = [0,1,2,3,4,5,6,7,8,9]
    p = [posterior(X,Y,i) for i in m_i]
    m = m_i[p.index(max(p))]
    return m
def prediction_grad(x,y,W,V,b,c):
    # Compute f(x) for all x. f(x) = c + V.tanh(b + W.x)
    Wx = np.matmul(W,x)
    bPlusWx = np.add(b, Wx)
    fx = np.add(c, np.matmul(V, np.tanh(bPlusWx)))

    # Compute dLdf = - e_hat_y + g(f(x)). g(f(x)) = exp(f_y) / sumOfAll(exp(f))
    # e_hat_y is a unit vector with value 1 for y and 0 for rest. Compute negative unit vector
    numLables = c.shape[0]
    e = np.zeros((numLables, numLables))
    for i in range(numLables) :
        e[i][i] = -1

    # Compute g(f(x))
    sigma_exp_fv =  0
    for i in range(numLables) :
        sigma_exp_fv = sigma_exp_fv + np.exp(fx[i])
    gfx = np.zeros(numLables)
    for i in range(numLables) :
        gfx[i] = np.exp(fx[i]) / sigma_exp_fv
    
    dLdf = e[y] + gfx

    dLdc = dLdf

    h = np.tanh(bPlusWx)
    dLdV = np.outer(dLdf,h)

    derivativeOfTanh = 1 - np.tanh(bPlusWx)**2

    dLdb = derivativeOfTanh * np.matmul(np.transpose(V), dLdf)
    dLdW = np.outer(dLdb,x)

    return dLdW, dLdV, dLdb, dLdc
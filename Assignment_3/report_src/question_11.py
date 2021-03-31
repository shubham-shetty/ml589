def prediction_loss(x,y,W,V,b,c):
    # Compute f(x) for all x.
    Wx = np.matmul(W,x)
    bPlusWx = np.add(b, Wx)

    # fx = c + V.tanh(b + W.x)
    fx = np.add(c, np.matmul(V, np.tanh(bPlusWx)))

    expFx = 0
    for fi in fx :
        expFx = expFx + np.exp(fi)
    
    # Loss = -f_y(x) + log(sum(exp(f_i(x))))
    L = 0 - fx[y] + np.log(expFx)

    return L
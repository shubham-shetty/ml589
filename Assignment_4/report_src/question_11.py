def eval_kernel_ridge(x_trn, x, a, k):
    y = np.dot(a, np.array(list(map(lambda x1: k(x1, x), x_trn))))
    return y
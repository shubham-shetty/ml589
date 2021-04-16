def eval_basis_expanded_ridge(x, w, h):
    y = np.dot(w, h(x)[0])
    return y

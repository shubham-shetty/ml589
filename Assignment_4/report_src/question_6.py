def train_basis_expanded_ridge(X, Y, l, h):
    H = h(X)

    C = np.zeros((H[0].shape[0], H[0].shape[0]))
    for item in H:
        C = np.add(C, np.matmul(np.array([item]).T, np.array([item])))

    I = np.identity(len(C))

    w = np.linalg.solve(np.add(C, l * I),
                        np.matmul(np.transpose(H), Y))

    return w

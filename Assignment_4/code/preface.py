import numpy as np

from scipy.special import comb


def get_poly_expansion(P):
    def expand(X):
        tmp = [np.sqrt(comb(P, p)) * X ** p for p in range(P + 1)]
        return np.vstack(tmp).T

    return expand


data = np.load("data_synth.npz")
X_trn = data['X_trn']
Y_trn = data['Y_train']
X_val = data['X_val']
Y_val = data['Y_val']

data_real = np.load("data_real.npz")
X_trn_real = data_real['x_trn']
Y_trn_real = data_real['y_trn']
X_val_real = data_real['x_tst']

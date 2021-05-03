import matplotlib
import numpy as np
from matplotlib import cm
from scipy.linalg import eigh

from preface import face_data


def pca_dim_reduction(data):
    mean = data.mean(axis=1)
    d = 2500
    N = 100
    for k in {3, 5, 10, 30, 50, 100, 150, 300}:
        C = np.zeros((d, d))
        for i in range(len(data)):
            x_minus_x_bar = np.subtract(data[i], mean[i])
            C = np.add(C, np.matmul(x_minus_x_bar, np.array([x_minus_x_bar]).T))

        eigen_vectors, eigen_values = eigh(C, eigvals=(d - k, d - 1))

        Y = []
        for i in range(len(data)):
            y = []
            for eigen in eigen_vectors:
                y.append(np.dot(eigen, data[i]))
            Y.append(np.array(y))
        Y = np.array(Y)

        X_hat = []
        for item in Y:
            x_hat = np.zeros((d))
            for i in range(k):
                x_hat = np.add(x_hat, item[i] * eigen_vectors[i])
            X_hat.append(x_hat)
        X_hat = np.array(X_hat)

        # TODO change output location if required
        for i in range(N):
            matplotlib.image.imsave(f'face_{i}_restored_{k}.png', X_hat[i].reshape(50, 50), cmap=cm.Greys_r)


pca_dim_reduction(face_data)

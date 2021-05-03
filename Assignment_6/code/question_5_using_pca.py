import matplotlib
import numpy as np
from matplotlib import cm
from sklearn.decomposition import PCA

from preface import face_data


def pca_dim_reduction(data):
    for k in {3, 5, 10, 30, 50, 100}:
        pca = PCA()
        pca.fit(data)
        eigen_vectors = pca.components_[:k]
        eigen_values = pca.explained_variance_[:k]

        Y = []
        for i in range(len(data)):
            y = []
            for eigen in eigen_vectors:
                y.append(np.dot(eigen, data[i]))
            Y.append(np.array(y))
        Y = np.array(Y)

        # dim_reduced_face_data = pca.transform(data)
        # restored_face_data = dim_reduced_face_data.inverse_transform()

        X_hat = []
        for item in Y:
            x_hat = np.zeros((2500))
            for i in range(k):
                x_hat = np.add(x_hat, item[i] * eigen_vectors[i])
            X_hat.append(x_hat)
        X_hat = np.array(X_hat)
        for i in range(100):
            matplotlib.image.imsave(f'face_{i}_restored_{k}.png', X_hat[i].reshape(50, 50), cmap=cm.Greys_r)


pca_dim_reduction(face_data)

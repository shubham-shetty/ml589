import os

import matplotlib
from matplotlib import cm
from sklearn.decomposition import PCA

from preface import face_data


def pca_dim_reduction(data, k):
    pca = PCA(n_components=k)
    pca.fit(data)

    dim_reduced_face_data = pca.transform(data)
    restored_face_data = pca.inverse_transform(dim_reduced_face_data)

    for i in range(100):
        matplotlib.image.imsave(os.path.join('output', f'face_{i}_restored_{k}.png'),
                                restored_face_data[i].reshape(50, 50), cmap=cm.Greys_r)
    return dim_reduced_face_data, pca.components_


for k in [3, 5, 10, 30, 50, 100]:
    pca_dim_reduction(face_data, k)

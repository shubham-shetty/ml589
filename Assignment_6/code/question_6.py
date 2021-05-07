import numpy as np

from question_5 import pca_dim_reduction
from preface import face_data, prettyPrintTable


def compute_compression_rate(data):
    compression_rate = []
    for k in [3, 5, 10, 30, 50, 100]:
        dim_reduced_face_data, eigen_vectors = pca_dim_reduction(data, k)
        original_data_size = data.size * data.itemsize
        compressed_data_size = dim_reduced_face_data.size * dim_reduced_face_data.itemsize + eigen_vectors.size * eigen_vectors.itemsize
        compression_rate.append((k, compressed_data_size / original_data_size))
    return compression_rate


prettyPrintTable(np.array(compute_compression_rate(face_data)), ['k', 'Compression rate'])

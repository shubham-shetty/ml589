import os

import matplotlib.image as mpimg
import numpy as np

face_data = []
for i in range(100):
    img = mpimg.imread(os.path.join('faces', f'face_{i}.png'))
    img_vector = img.flatten()
    face_data.append(img_vector)

face_data = np.array(face_data)

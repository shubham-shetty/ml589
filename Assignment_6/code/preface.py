import os

import matplotlib.image as mpimg
import numpy as np

face_data = []
for file in os.listdir('faces'):
    img = mpimg.imread(os.path.join('faces', file))
    img_vector = img.flatten()
    face_data.append(img_vector)

face_data = np.array(face_data)

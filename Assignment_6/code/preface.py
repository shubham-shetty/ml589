import os

import matplotlib.image as mpimg
import numpy as np

face_data = []
for i in range(100):
    img = mpimg.imread(os.path.join('faces', f'face_{i}.png'))
    img_vector = img.flatten()
    face_data.append(img_vector)

face_data = np.array(face_data)


# Helper function to print table
# Input variable table is a numpy array and headers is an array of column headers
def prettyPrintTable(table, headers) :
    for i, d in enumerate(table):
        if i == 0 :
            line = '|'.join(str(x).ljust(30) for x in headers)
            print(line)
            print('-' * len(line))

        line = '|'.join(str(x).ljust(30) for x in d)
        print(line)
    print()

# Import Statements
import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import hinge_loss
from sklearn.svm import SVC


# Load Data
## 1D Data
data = np.load("data_synth.npz")
X_trn = data_synth['X_trn']           # 1D Input Training Dataset
Y_trn = data_synth['Y_train']         # 1D Output Training Dataset
X_val = data_synth['X_val']           # Test Input Dataset
Y_val = data_synth['Y_val']           # Test Output Dataset

## Big data for SVM testing
data_real = np.load("data_real.npz")
X_trn_real = data_real['x_trn']       # 686 training inputs of length 4
Y_trn_real = data_real['y_trn']       # 686 training outputs of length 4
X_val_real = data_real['x_tst']       # 686 test inputs


# Function to perform basis expansion
def get_poly_expansion(P):
    def expand(X):
        tmp = [np.sqrt(comb(P, p)) * X ** p for p in range(P + 1)]
        return np.vstack(tmp).T
    return expand
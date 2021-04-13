from preface import *

def get_poly_kernel(P):
    def k(x,xp):
        kernel_value = (1+np.inner(x, xp))**P
        return kernel_value
    return k
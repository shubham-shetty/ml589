from preface import *

def likelihood_single(x, y, m):
    mu = 0 if m == 0 else x ** m
    sigma = 0.1
    return norm.pdf(y, mu, sigma)

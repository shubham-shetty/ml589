from scipy.stats import norm


def likelihood_single(x, y, m):
    mu = 0 if m == 0 else x ** m
    sigma = 0.1
    # return (1 / (2 * math.pi * sigma)) * math.exp(-1 * ((y - mu) ** 2) / (2 * (sigma ** 2)))
    return norm.pdf(y, mu, sigma)

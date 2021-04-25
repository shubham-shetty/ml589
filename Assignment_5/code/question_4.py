from question_3 import likelihood_single


def likelihood(X, Y, m):
    p = 1
    for x, y in zip(X, Y):
        p *= likelihood_single(x, y, m)
    return p

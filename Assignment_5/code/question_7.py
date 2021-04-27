from question_1 import prior
from question_4 import likelihood


def posterior(X, Y, m):
    p = (likelihood(X, Y, m) * prior(m)) / sum([likelihood(X, Y, item) * prior(item) for item in range(10)])
    return p

from matplotlib import pyplot as plt

from preface import X, Y
from question_7 import posterior

P = []
for m in range(10):
    P.append(posterior(X, Y, m))

sums = sum(P)
P = [p / sums for p in P]
for m in range(10):
    print(f'm: {m}; posterior: {P[m]}')

plt.bar(list(range(10)), P, label="Posterior Bar Chart")
plt.xlabel("m")
plt.ylabel("posterior")
plt.legend()
plt.show()

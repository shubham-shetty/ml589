from matplotlib import pyplot as plt

from preface import X, Y
from question_3 import likelihood_single

P = []
for index, (x, y) in enumerate(zip(X, Y)):
    P.append(likelihood_single(x, y, index))
    print(f'm: {index}; likelihood: {P[-1]}')

plt.bar(list(range(10)), P, label="Likelihood Bar Chart")
plt.xlabel("m")
plt.ylabel("likelihood")
plt.legend()
plt.show()

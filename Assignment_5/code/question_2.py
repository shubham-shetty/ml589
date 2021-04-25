from matplotlib import pyplot as plt

from question_1 import prior

priors = []
for m in range(10):
    priors.append(prior(m))
    print("m", m, "prior(m)", priors[-1])

plt.bar(list(range(10)), priors, label="Priors Bar Chart")
plt.xlabel("m")
plt.ylabel("prior")
plt.legend()
plt.show()

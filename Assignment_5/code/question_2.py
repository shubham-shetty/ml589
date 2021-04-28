from preface import *
from question_1 import prior

def show_priors():
    for m in range(10):
        print("m", m, "prior(m)", prior(m))
        
    x = [0,1,2,3,4,5,6,7,8,9]
    y = [prior(m) for m in x]
    plt.bar(x, y, label="Priors Bar Chart")
    plt.xlabel("m")
    plt.ylabel("Prior")
    plt.xticks(np.arange(0, 10, 1))
    plt.show()
    
show_priors()

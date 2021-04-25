from preface import X, Y
from question_9 import MAP
from question_7 import posterior

m = MAP(X,Y)
print(m)
print(posterior(X,Y,m))
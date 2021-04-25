from preface import X, Y
from question_9 import MAP
from question_7 import posterior

m = MAP(X,Y)
print(f"MAP estimate m_MAP = {m}")
print(f"Posterior prob for m_MAP = {posterior(X,Y,m)}")
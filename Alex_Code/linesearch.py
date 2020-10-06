import numpy as np

def linesearch(x,p,g,fun,eta,gam,jmax):
    a = 1
    f0 = fun(x)
    aux = eta*np.dot(g,p)
    for j in range(jmax+1):
        xtry = x + a*p
        f1 = fun(xtry)
        if f1 < (f0 + a*aux):
            break
        else:
            a = a*gam
    return a, j
import numpy as np

def linesearch(x,p,g,I,func,eta,gam,jmax):
    a = 1
    f0 = fun(I,Y,w)
    aux = eta*np.dot(g,p)
    for j in range(jmax+1):
        xtry = x + a*p
        f1 = fun(I, Y, xtry)
        if f1 < (f0 + a*aux):
            break
        else:
            a = a*gam
    return a, j
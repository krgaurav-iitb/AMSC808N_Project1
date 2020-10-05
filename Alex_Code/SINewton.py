import numpy as np
from CG import CG

def SINewton(fun, gfun, Hvec, Y, w, batchsize = 64):
    rho = 0.1
    gam = 0.9
    jmax = round(np.ceil(np.log(1e-14)/np.log(gam)))
    eta = 0.5
    CGimax = 20
    n = Y.shape[0]
    bsz = np.minimum(n,batchsize)
    kmax = round(5e3)
    I = np.arange(n)
    f = np.zeros(kmax+1)
    f[0] = fun(I,Y,w)
    normgrad = np.zeros(kmax)
    nfail = 0
    nfailmax = 5*np.ceil(n/bsz)
    for k in range(kmax):
        Ig = np.random.permutation(I)[:bsz]
        IH = np.random.permutation(I)[:bsz]
        Mvec = lambda v: Hvec(IH,Y,w,v)
        b = gfun(Ig,Y,w)
        normgrad[k] = np.linalg.norm(b)
        s = CG(Mvec, -b, -b, CGimax, rho)
        a = 1
        f0 = fun(Ig,Y,w)
        aux = eta*np.dot(b,s)
        for j in range(jmax+1):
            wtry = w + a*s
            f1 = fun(Ig, Y, wtry)
            if f1 < (f0 + a*aux):
                """
                print('Linesearch: j=%d, f1 =%e, f0=%e' % (j,f1,f0))
                """
                break
            else:
                a = a*gam
        if j<jmax:
            w = wtry
        else:
            nfail += 1
        f[k+1] = fun(I,Y,w)
        """
        print('k = %d, a = %e, f = %e' % (k,a,f[k+1]))
        """
        if nfail > nfailmax:
            f = f[:k+2]
            normgrad = normgrad[:k+1]
            break
    print('k = %d, a = %e, f = %e' % (k,a,f[k+1]))
    return w,f,normgrad
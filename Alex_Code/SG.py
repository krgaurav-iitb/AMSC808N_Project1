import numpy as np
import time

def SG(fun, gfun, Y, x, afun, batchsize = 64, itermax = 10000):
    xiter = np.zeros((itermax+1,x.size))
    f = np.zeros(itermax+1)
    runtime = np.zeros(itermax)
    n = Y.shape[0]
    Iall = np.arange(n)
    tic = time.perf_counter()
    f[0] = fun(Iall,Y,x)
    xiter[0] = x
    grad_norm = np.zeros(itermax+1)
    bsz = np.minimum(n,batchsize)
    I = np.random.permutation(n)[:bsz]
    g = gfun(I,Y,x)
    grad_norm[0] = np.linalg.norm(g)
    toc = time.perf_counter()
    runtime[0] = toc - tic
    itr = 0
    while itr < itermax - 1:
        xiter[itr+1] = xiter[itr] - afun(itr)*g
        f[itr+1] = fun(Iall,Y,xiter[itr+1])
        itr += 1
        I = np.random.permutation(n)[:bsz]
        g = gfun(I,Y,xiter[itr])
        grad_norm[itr] = np.linalg.norm(g)
        toc = time.perf_counter()
        runtime[itr] = toc - tic
    xiter[itr+1] = xiter[itr] - afun(itr)*g
    f[itr+1] = fun(Iall,Y,xiter[itr+1])
    return xiter, f, grad_norm, runtime
    
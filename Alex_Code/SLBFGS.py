import numpy as np
from finddirection import finddirection
from linesearch import linesearch

def SLBFGS(fun, gfun, Y, x0, stepmethod = 'linesearch', m = 5, stepsperHupdate = 10, batchsizeg = 64, batchsizeH = 64, maxiter = 500, Hevalmethod = 'onestep'):
    n = Y.shape[0]
    gam = 0.9
    eta = 0.5
    tol = 1e-10
    jmax = round(np.ceil(np.log(1e-14)/(log(gam))))
    s = np.zeros(x0.size, m)
    y = np.zeros(x0.size, m)
    rho = np.zeros(m)
    f = np.zeros(maxiter+2)
    gradnorm  = np.zeros(maxiter+2)
    xiter = np.zeros((maxiter+2,x0.size))
    Iall = np.arange(n)
    f[0] = fun(Iall,Y,x0)
    xiter[0] = x0
    bszg = np.minimum(batchsizeg, n)
    bszH = np.minimum(batchsizeH, n)
    Ig = np.random.permutation(n)[:bszg]
    IH = np.random.permutation(n)[:bszH]
    g = gfun(Ig,Y,x)
    gH = gfun(IH, Y, x)
    grad_norm[0] = np.linalg.norm(g)
    a, j = linesearch(x, -g, g, Ig, fun, eta, gam, jmax)
    xnew = x - a*g
    gnew = gfun(Ig, Y, xnew)
    gnewH = gfun(IH, Y, xnew)
    s[:,0] = xnew - x
    y[:,0] = gnewH - gH
    rho(1) = 1/np.dot(s[:,0],y[:,0])
    x = xnew
    g = gnew
    nor = np.linalg.norm(g)
    xiter[1] = x
    grad_norm[1] = nor
    f[1] = fun(Iall, Y,x)
    itr = 1
    mitr = 1
    while nor > tol and itr < maxiter:
        if mitr < m:
            p = finddirection(g, s[:,:mitr], y[:,:itr],rho[:mitr])
        else:
            p = finddirection(g,s,y,rho)
        if stepmethod is 'linesearch':
            a,j = linesearch(x,p,g,Ig,func,eta,gam,jmax)
            if j == jmax:
                p = -g
                a,j = linesearch(x,p,g,Ig,func,eta,gam,jmax)
        else:
            a = stepmethod(itr)
        step = a*p
        xnew = x + step
        xiter[itr + 1] = xnew
        f[itr + 1] = fun(Iall,Y,xnew)
        if itr % stepsperHupdate == 0:
            IH = np.random.permutation(n)[:bszH]
            if Hevalmethod is 'onestep':
                snew = step
                ynew = gfun(IH, Y, xnew) - gfun(IH, Y, x)
            else:
                snew = xnew - xiter[itr + 1 - stepsperHupdate]
                ynew = gfun(IH, Y, xnew) - gfun(IH, Y, xiter[itr + 1 - stepsperHupdate])
            s = np.append(snew.reshape((-1,1)), s[:,:m-1], axis = 1)
            y = np.append(ynew.reshape((-1,1)), y[:,:m-1], axis = 1)
            rho = np.append(1/np.dot(snew,ynew), rho[:,:m-1])
            mitr += 1
        Ig = np.random.permutation(n)[:bszg]
        g = gfun(Ig, Y, xnew)
        nor = np.linalg.norm(g)
        gradnorm[itr + 1] = nor
        itr += 1
    if itr < maxiter:
        xiter = xiter[:itr+2]
        gradnorm = gradnorm[:itr+2]
        f = f[:itr+2]
       
    return xiter, f, gradnorm
        
        
function [point, slope] = SLBFGS(init, s, y, rho, m, Ng, Nh, M, func, grad, hess, grank, hrank)
%   init--initial guess
%   m--limited memory constant (default is 5)
%   Ng--batch size for gradient
%   Nh--batch size for hessian
%   S,Y--nonempty dim by m matrices whose columns are s_i and y_i
%   Rho--dim by 1 matrix whose entries are 1/s_i'y_i
%   grad--computing gradient
%   hess--computing hessian 
%   grank--rank of gradient
%   hrank--rank of hessian
gam = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor
xi_g = randi(grank, Ng, 1);
g = grad(xi_g,init);
tol = 1e-10;
nor = norm(g);
iter = 1;
x = init;
point = x; 
slope = g; 
while nor > tol
    % We compute the direction via the LBFGS routine 
    if iter < m 
        I = 1 : iter;
        p = finddirection(g,s(:,I),y(:,I),rho(I));
    else
        p = finddirection(g,s,y,rho);
    end
    % We compute the step size via ls (may be modified)
    [a,j] = ls(x,p,g,func,eta,gam,jmax);
    if j == jmax
        p = -g;
        [a,j] = ls(x,p,g,func,eta,gam,jmax);
    end
    % 
    step = a*p;
    xnew = x + step; %xnew is x_k+1; x is x_k 
    snew = step; %snew is sk_k+1 
    %Stochastic step for Hessian 
    if mod(iter,M) == 0
        xi_h = randi(hrank, Nh, 1);
        Htemp = hess(xi_h, x); %generate random hessian
        ynew = Htemp*snew; 
    else
        ynew = grad(xi_g,xnew) - g;
        snew = xnew - x;
    end
    s = [s snew];
    y = [y ynew];
    if size(s,2) > m
        s = s(:, 2:end);
        y = y(:, 2:end);
    end
    x = xnew; 
    xi_g = randi(grank, Ng, 1);
    g = grad(xi_g,x); % Random approximation to the gradient at current
                      % iterate 
    nor = norm(g);
    point = [point x];
    slope = [slope g];
    iter = iter + 1;
end

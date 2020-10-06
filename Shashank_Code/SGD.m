function [iters,gradz] = SGD(init,grad,step,rank,batchsize,m0,schedule)
    % init--initial guess
    % grad--function that evaluates gradient of f at current iterate
    % step--initial step size set to 0.3
    % rank--number of terms in the expansion of f
    % batchsize--batch size 
    % m0--length of first item in schedule
    % schedule--length of schedule
    % Default m0 = 10 and default schedule = 15
    % This is a decreasing stepsize routine
    n = rank;
    N = m0;
    NN = schedule;
    x = init; 
    iters = x;
    k = randperm(n,batchsize);
    g = grad(k,x);
    gradz = g;
    nor = norm(g);
    tol = 1e-10;
    iter = 1;
%     % Constant stepsize routine 
%     while ((nor > tol) && (iter < 10000))
%         k = randperm(n,batchsize);
%         g = grad(k,x);
%         x = x - step*g;
%         iters = [iters, x];
%         gradz = [gradz, g];
%         nor = norm(g);
%         iter = iter + 1;
%     end
    
    % Decreasing stepsize routine 
    for ii = 1 : NN
     s = step/(2^ii);
     nsteps = ceil(N*2^ii/ii);
        for i = 1 : nsteps
            k = randperm(n,batchsize);
            g = grad(k,x);
            x = x - s*g;
            iters = [iters, x];
            gradz = [gradz, g];
        end
    end
end
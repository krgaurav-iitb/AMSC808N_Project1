%% Soft Margin SVN vs Loss function-minimizing hyperplane
%
% Recall that the soft margin problem is as follows: 
%
% <latex>
% \begin{align} 
% \text{min}&\frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i \, \text{w.r.t} \\
% &y_i(w^{\top}x_i + b) \geq 1 - \xi_i 
% \end{align}
% </latex>
% 
% The corresponding dual problem is 
% 
% <latex>
% \begin{align}
% \text{min}&\frac{1}{2}D'\lambda - [1_{1\times n} 0_{1\times n}]\lambda\,
% \text{w.r.t}\\
% & C \geq \lambda_i \geq 0 \, 1 \leq i \leq 2n \\
% & \sum_{i=1}^{n}\lambda_iy_i = 0 \\
% & \lambda_i + \lambda _{i+n} = C \, 1 \leq i \leq n
% \end{align}
%</latex>
%
% Here $\lambda \in \mathbb{R}^{2n}$, $D' =
% \begin{bmatrix}(yy^{\top})\odot(XX^{\top}) & \\ & 0_{n\times
% n}\end{bmatrix}$. We can eliminate the equality constraint 
% $\lambda_i + \lambda _{i+n} = C$ by setting 
% $\lambda _{i+n} = C - \lambda_i$ and combining it with non-negativity 
% to get $C \geq \lambda_i$ for $1 \leq i \leq n$. 
% We can thus drop to solving for the first $n$ $\lambda$; the initial
% guess is $\lambda = 0_{n \times 1}$ and all the constraints are active.
% We also set the penalty constant $C$ to be $0.1$. 
%%
% Setting up arguments for ASM.m
c = 0.2; %Constant in the penalty function 
y = label; % the label vector 
n = length(y); % number of data points
%D = [(y*y').*((XX * XX')) zeros(n,n); zeros(n,n) zeros(n,n)]; %SPD matrix 
%                                                              in quadratic 
%                                                              program 
D = (y*y').*((XX * XX'));
%d = [ones(n,1); zeros(n,1)]; % linear term in Q.P
d = ones(n,1);
%Ap = [eye(n,n) eye(n,n)]; % the matrix A'
%App = [y' zeros(1,n)]; % the matrix A
App = y';
C = [eye(n,n); -eye(n,n); App];  %matrix of contraints
%b = [zeros(2*n + 1,1); c*ones(n,1)]; % vector of constraints
b = [zeros(n,1); -c*ones(n,1); 0];
grad = @(x) D*x - d; %gradient 
hess = @(x) D; %hessian 
%init = [zeros(n,1); c*ones(n,1)]; % initial guess
init = zeros(n,1);
%W = [1:n (2*n + 1):(3*n + 1)]; %set of working constraints at the initial point
W = 1:n;
W = W';
% The constraints matrix is a 2n+1 by n matrix 
% All constraints are active. 
% Time to run the solver! 
[lambs, lm] = ASM(init, grad, hess, C, b, W);
soln = lambs(:,end); % extracting the lambda vector  
wASM = (XX')*(y .* soln); % optimal w 
%Computing B via soft margin support vectors
avg = XX(find(soln == max(soln(find(y==1)))),:) ...
    + XX(find(soln == max(soln(find(y==-1)))),:);
B = -0.5*(avg * wASM);
wASM = [wASM; B];

%%% Plotting the classifier 

% Plotting the data
idem = find(y==-1);
igop = find(y==1);
figure;
hold on; grid;
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
X1 = (XX(:,1)-xmin)/(xmax-xmin);
X2 = (XX(:,2)-ymin)/(ymax-ymin);
X3 = (XX(:,3)-zmin)/(zmax-zmin);
XX = [X1,X2,X3];
plot3(XX(idem,1),XX(idem,2),XX(idem,3),'.','color','b','Markersize',20);
plot3(XX(igop,1),XX(igop,2),XX(igop,3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);

%Plotting the hyperplane

xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
nn = 50;
[xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
    linspace(zmin,zmax,nn));
plane = wASM(1)*xx+wASM(2)*yy+wASM(3)*zz+wASM(4);
plane2 = w(1)*xx+w(2)*yy+w(3)*zz+w(4);
p = patch(isosurface(xx,yy,zz,plane,0));
q = patch(isosurface(xx,yy,zz,plane2,0));
p.FaceColor = 'green';
p.EdgeColor = 'none';
q.FaceColor = 'red';
q.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);
%
%% Stochastic Gradient Descent: Convergence and Runtime statistics
%
% Recall that in a decreasing stepsize routine, we pick a stepsize sequence
% to be an element of $l^2(\mathbb{R})\setminus l^1(\mathbb{R})$;
% furthermore, if we intend for our stepsize to decay exponentially, then
% we must set a schedule that takes each stepsize $\alpha_k$ $m_k$ times,
% where $m_k = \mathcal{O}(k^{-1}2^{-k})$ so that the convergence property
% may be retained.
%
%% 
gradient = @(s,t)gfun0(s,Y,t,lam);
init = wASM;
step = 0.3;
rank = length(Y(:,1));
batchsize = 52;
m0 = 10;
schedule = 15;
[avgs, gradz] = SGD(init,gradient,step,rank,batchsize,m0,schedule);
% for i = 2:10
%     [trials,gz] = SGD(init,gradient,step,rank,batchsize,m0,schedule);
%     avgs = avgs + trials;
%     gradz = gradz + gz;
% end
% avgs = 0.1*avgs;
% gradz = 0.1*gradz;
% 
%% Stochastic L-BFGS: Implementation and Performance vs SIN and SG
%
% In L-BFGS, we maintain a store of $m$ pairs $(s_i, y_i)$ that enable us
% to calculate $H_{k}g_k$ where $H_k$ is some approximation to
% $\nabla^{2}f(x_k)$ and $g_k$ is some approximation to $\nabla f(x_k)$.
% Once we produce $x_{k+1} = x_k - \alpha_k H_{k}g_k$, then we can
% calculate the new $s_{k} = x_{k+1} - x_{k}$ and $y_k = \nabla
% f(x_{k+1})-\nabla f(x_{k}). In Stochastic L-BFGS, we periodically change
% the way for computing $y_k$ as follows: instead of letting $y_k = \nabla
% f(x_{k+1})-\nabla f(x_{k})$, we set $y_k = B_{k}s_k$ where $B_k$ is a
% (stochastic) approximation to the actual Hessian $\nabla^{2}f(x_k)$. This
% approach leads to the following implementation technicalities: 
% 
% # Frequency of the stochastic step: The less frequently we change the way
% to compute $(s_i, y_i)$, the more the method looks like L-BFGS. 
% # The approximation $B_k$: We assume that the Hessian has some rank $n$
% and we pick an approximation in a subspace spanned by a randomly chosen
% subset of the original basis indexed by $\xi^{H}_{k}$. The size of the $N_H$ 
% is the rank of the approximating subspace; smaller the $N_H$, the
% larger the variance in our approximate Hessian $B_k$. But we also do not
% want to make $N_H$ too close to $n$ as we would then lose the point of actually
% approximating the Hessian. 
% 
%% 
% Initializing with a linesearch routine like LBFGS

% n= 
func = @(x)fun(1:n,Y,x);
gam = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor
x = [-1; 1; -1; 1] %initial guess
g = gradient(1:n, x);
a = ls(x,-g,g,func,eta,gam,jmax);
xnew = x - a*g;
gnew = gfun(xnew);
s(:,1) = xnew - x;
y(:,1) = gnew - g;
x = xnew;


%% Toy examples, Sai Ravela (C) 2016-

% Marginals and MAP.
% The main condition is the spectral radius must be less than 1, or the 
% system must be diagonally dominant and therefore positive (semi)
% definite.

a = [3 0.1 0.01; 0.1 2 0.1; 0.01 0.1 1];
iz = inv(a);
mu = randn(3,1);
hin = iz*mu;
% Solve for posterior marginals using BP
[jii,hi,xmap]=gabp(iz,hin);
% Comparison of variances
[diag(a) 1./jii]'
% comparison with qp solution
xq=quadprog(iz,-hin);

% GGM solves a quadratic program
[hi./jii  mu xmap xq]'

% Posterior Marginals, with data assimilation
% We assume there is a measurement on one node, in the middle
H = [0 1 0]; 
y = H*mu + 0.5*randn;

Jin = iz + H'*(1/0.5)^2*H;
hin = hin + H'*(1/0.5)^2*y;
[jii,hi,xmaplus]=gabp(Jin,hin);
xqplus=quadprog(Jin,-hin);

% The least squares update rule is
xplus = mu+a*H'/(H*a*H'+0.5^2)*(y-H*mu);
[xmaplus xqplus xplus]'
% This the GGM implements the kalman filter and recursive least squares
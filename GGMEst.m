function [X] = GGMEst(S,lambda)
%% Sai Ravela, Stat Learning book, thanks to Peshkov
% Note changes below -- using QP, and setdiff is much simpler here
%         sedif = [1:i-1 i+1:p];Just leave i out.
optTol = 0.00001;
p = size(S,1);
maxIter = 10;
A = [eye(p-1,p-1);-eye(p-1,p-1)];
f = zeros(p-1,1);
% Initial W
W = S + lambda*eye(p,p); %chapter 17, page 636
options = optimset('LargeScale','off','Display','none');
qpArgs = {[],[],[],[],[],options};
for iter = 1:maxIter
    % Check Primal-Dual gap
    X = W^-1; % W should be PD
    gap = trace(S*X) + lambda*sum(sum(abs(X))) - p;
    fprintf('Iter = %d, OptCond = %.5f\n',iter,gap);
    if gap < optTol
        fprintf('Solution Found\n');
        break;
    end
    for i = 1:p
        % Compute Needed Partitions of W and S
        sedif = [1:i-1 i+1:p]; %Chapter 17, page 636
        s_12 = S(sedif,i);
        H = 2*W(sedif,sedif)^-1;
        b = lambda*ones(2*(p-1),1) + [s_12;-s_12];
        w = quadprog((H+H')/2,f,A,b,qpArgs{:});
        % Un-Permute
        W(sedif,i) = w;
        W(i,sedif) = w';
    end
end
%%Computational Statistics Example

load cereal;

[n,d] = size(cereal)
D=squareform(pdist(cereal,'cityblock'));

Q= -0.5*D.^2;
H = eye(n) - 1/n*ones(n,1)*ones(1,n);
B = H*Q*H;
[A,L] = eig(B);
[vals,inds] = sort(diag(L),'descend');
A = A(:,inds);
X = A(:,1:2)*diag(sqrt(vals(1:2)));
plot(X(:,1),X(:,2),'o');
text(X(:,1),X(:,2),labs);
title('MDS');

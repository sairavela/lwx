largesteig = zeros(100);
for j = 1:100
    N=500;
    M=rand(N);
    M=0.5*(M+M');
    L=100; %  magnitude
    M(M > .5) = 1;
    M(M <= .5) = -1;
    largesteig(j, j) = max(eig(M));
end
u = diag(largesteig);
histogram(u);

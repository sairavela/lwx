%% (C) F. Shao and S. Ravela, 2010
N = 1000;
% Generating points on the SwissRole manifold
tt = (3*pi/2)*(1+2*rand(1,N));  
height = 21*rand(1,N);
X = [tt.*cos(tt); height; tt.*sin(tt)]';
ColorVector = tt';

% Computing the Delaunay triangulation
INF = 1e6;
Tes = delaunay(X(1:N,1),X(1:N,2),X(1:N,3));

% Computing shortest distances between pairs of points
D = zeros(N,N);
for i=1:N
    for j=1:N
        D(i,j) = INF;
    end
end
for i=1:size(Tes,1)
    for j=1:4
        for k=1:4
            u=Tes(i,j);
            v=Tes(i,k);
            D(u,v)=sqrt((X(u,1)-X(v,1))^2+(X(u,2)-X(v,2))^2+(X(u,3)-X(v,3))^2);
        end
    end
end
for k=1:N
     D = min(D,repmat(D(:,k),[1 N])+repmat(D(k,:),[N 1])); 
end

% Compute the embedding
P=D.^2;
J=eye(N)-ones(N)/N;
B=-.5*J*P*J;
[vec,val]=eigs(B);
Y=[vec(1:N,1)*sqrt(val(1,1)) vec(1:N,2)*sqrt(val(2,2))];
scatter(Y(:,1), Y(:,2), 12, ColorVector, 'filled');
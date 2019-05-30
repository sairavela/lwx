%% KPCA Example with a kernel
% Sai Ravela (C) 2017
% No explicit feature map calculation

close all;
set(0,'DefaultFigureWindowStyle','docked')
N=100;
% Generate Data
th1 = rand(N,1)*2*pi;
th2 = rand(N,1)*2*pi;
amp1 = randn(N,1)*0.1+3;
amp2 = randn(N,1)*0.2+4;

x1=[amp1.*cos(th1) amp1.*sin(th1)];
x2=[amp2.*cos(th2) amp2.*sin(th2)];

data = [x1;x2];
[N,n]=size(data);

%Inner products
D=data*data';

%Kernel
K=D+diag(D)*diag(D)';

%Centering
K = K - 1/N*repmat(sum(K,1),[N 1])-1/N*repmat(sum(K,2),[1 N])+1/N^2*sum(K(:));
[u,s,v]=svd(K);

%Embedding
z=u'*K;

h1=figure(1);
scatter(data(:,1),data(:,2),12,[1*ones(100,1); 2*ones(100,1)])
figure(2); plot(ksdensity(z(1,:)),'r.')
figure(4);scatter(z(1,:),z(2,:),12,[1*ones(100,1); 2*ones(100,1)])
rg=0.1;
bestm = gmdistribution.fit(z(1:2,:)',2,'Start','randSample','Replicates',3,'Regularize',rg)
for i = 1:bestm.NComponents
    plot_gaussian_ellipsoid(bestm.mu(i,[1 2]),bestm.Sigma([1 2],[1 2],i),1,'k');
end
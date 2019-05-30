function [Jii,hi,xmap] = gabp(J,h)
% (C) Sai Ravela, 2016-
%% Given the information matrix J and potential vector h,
% produce the marginal posterior information and potential
eps = 1e-12;
maxiter = 5;
NH=zeros(size(J));
NH = abs(J)>0;

for i = 1:length(h)
    NH(i,i)=0;
end
nh = logical(NH);
M = zeros(size(J));
Mh = zeros(size(J));
for i = 1:length(h)
    M(i,:)= J(i,i);
    Mh(i,:)=h(i);
end
disp(size(h));
mo = zeros(size(M));
mh = zeros(size(Mh));
dJ=diag(J);
hs = zeros(size(h));
Jii = zeros(size(h)); hi = zeros(size(h));
% disp(size(M));
% disp(size(Mh));
% disp(size(nh));
% disp(size(J));
%disp(M);
%disp(Mh);
for iter = 1:maxiter
s=zeros(size(h));
for i = 1:length(h) %parallelizable
    s(i)=sum(J(i,nh(i,:))'./(M(nh(i,:),i)).*J(nh(i,:),i));
    hs(i) = sum(J(i,nh(i,:))'./(M(nh(i,:),i)).*(Mh(nh(i,:),i)));
    
    Jii(i)=dJ(i)-s(i); % marginal information
    mo(i,nh(i,:))=Jii(i)+(J(i,nh(i,:))'./M(nh(i,:),i).*J(nh(i,:),i))';
    hi(i) = h(i)-hs(i); % marginal potential
    mh(i,nh(i,:))=hi(i)+(J(i,nh(i,:))'./M(nh(i,:),i).*Mh(nh(i,:),i))';
end
xmap = hi./Jii;
if (sum(abs(M(:)-mo(:)))<eps)
    return
end
M=mo;Mh = mh; %iterate
end
disp(Jii);
disp(hi);
disp(xmap);

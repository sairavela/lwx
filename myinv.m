
%% Script for the inversion example.
% SR 2/2/17
%% machine Learning Questions
% How Many Samples
% What to Sample: Use all the data?
% How many parameters to consider?
% Which Features to use?
% Best Model, in the absence of a law.
% Regularization: Smoothness, Sparsity, Compactness, what?
% Invariance vs. Selectivity?
% Bias vs. Variance?
% What about noise?
% What is convergence rate?

x = -pi:pi/100:pi-pi/100; % OK
pl = -4*x.^-2;pl(101) = 0; % No DC

%Initialize
ker0 = zeros(200,1); 

% Calculate point spread function numerically
s = zeros(1,200); s(101) = 1;ds = del2(s);

l2=1; % Ridgify?
ns = 1000; % How many samples
nf=4; % Which features?
for i = 1:ns, % How many samples?
    sig = rand*99+1; % How to sample?
g = exp(-x.*x/2/(pi/sig*pi/sig));% What features to use?

%cheap version, what about noise?
phi=real(ifftshift(ifft((fft(fftshift(g)).*fftshift(pl)))));
d2p = del2(phi); d2p = d2p - min(d2p);

figure(1);
%subplot(221); plot(x,[g(:) d2p(:)]);
%subplot(222);plot(x,[phi(:)])
%subplot(223); 
%plot(x,[pl(:) 1./real(fftshift(fft(fftshift(ds(:)))))]);

tg = toeplitz(fftshift(g)); %Block circulant only in special cases!

% what does this do?
[u,s,v] = svd(tg);v = v(:,1:nf); s = s(1:nf,1:nf); u = u(:,1:nf);
lam = 0;
if (l2) 
    lam = mean(diag(s)).^2*eye(nf);
end
    ker=u*pinv(s.^2+lam)*s*u'*fftshift(phi(:)); % L2 on kernel?
% Why does L1 not make sense? What about compactness?
%ker = pinv(tg'*tg+eye(200))*tg'*fftshift(phi(:));

%subplot(224);plot(x,ker);
%kres = cconv(g(:),(ker(:)),200);
%subplot(222);plot(x,[phi(:) kres])

%figure(2);
%iker = pinv(toeplitz((ker')));
%subplot(121); imagesc(iker);
%subplot(122); imagesc(toeplitz(fftshift(ds)));
ker0 = ker0+ker;
end
iker = pinv(toeplitz((ker0))); % Don't worry about averaging here for now
subplot(121); plot(iker(100,:)./2/max(iker(:)));
subplot(122); plot(ds);

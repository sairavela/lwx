%% A toy example of using MCMC to sample from a posterior
% (C) Sai Ravela, MIT, 2012 -, all rights reserved.

% Truth data
ctr = [0.1 0; 0 0.2]; [u2,s2,v2] = svd(ctr);
parmc = u2'*s2*randn(2,1);
x = 2*rand(100,1)-1;
yc = parmc(1).*x+parmc(2)+randn(size(x))*trace(ctr)/2;
yc = yc - mean(yc);

c = [1 0; 0 1];  [u,s,v] = svd(c); %imprefect model for slope-intercept correlations


this=0; last = inf;plast = 0.0001;
fact = 1;
parmprev=0;

%% Just Monte Carlo

jj=1;
while(jj<50)
    parm  = parmc+u'*s*randn(2,1); %just MC
y  = parm(1).*x+parm(2);
plot(x,[y yc],'x');
axis([-1 1 -1 1]); 
pause(0.1);
jj=jj+1;
end

jj=1;
fact = 0.25;
        xx = 2*rand(100,1)-1; %% X component.
pause
% MCMC
while(jj<500)
    parm  = parmprev+randn(2,1)*fact;%proposal distribution
    y  = parm(1).*xx+parm(2);
    this = (sum((xx-x).*(xx-x)+(yc-y).*(yc-y)))./length(y);%likelihood
    a = rand;
    lratio=(exp(-this/0.04/0.04/2)/exp(-last/0.04/0.04/2))%likelihood ratio
    % we ignore both the "priors" and Q(x,x')= Q(x',x)
    if (a<lratio) % If alpha>=1  accept, otherwise with prob alpha
       last = this;
       parmprev=parm;
       xxl = xx;yyl = y;
       fact = fact*(0.95)+randn*0.001; %%Always some randomness, but we adapt proposal.
       plot(xx,y,'+g');hold on; plot(x,yc,'ob');hold off;
       axis([-1 1 -1 1]);
       pause(0.1);
       jj=jj+1;
else
    %plot(xx,y,'xr');hold on; plot(xxl,yyl,'+g');plot(x,yc,'ob');hold off;
    
    %axis([-1 1 -1 1]);
   % pause(0.1);
end
end
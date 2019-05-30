function emex;
%EM example for 2-Gaussian Mixture.
% Sai Ravela, 12.S990 F12.
% All rights unreserved.
% This code is a quick hack after a somewhat difficult class on EM in F12.


%%Create the true distributions
MU1 = [1 2]; SIGMA1 = [2 0; 0 .5];
MU2 = [-2 -3]; SIGMA2 = [1 0; 0 1];
%%Generate samples
X = [mvnrnd(MU1,SIGMA1,1000);mvnrnd(MU2,SIGMA2,1000)];

%%Initial guess for weights
alpha = [0.5 0.5];
%An initial starting point away from the true answer!
%You may want to randomize this and try to study behavior more generally.
mu = [MU1'-2 MU2'+2];
SIGMA1 = SIGMA1.*4;
SIGMA2 = SIGMA2.*.5;

%%Variables to keep track of convergence.
converged = 0;
prevz=0;incr = inf;

while(converged<100)
    %A somewhat ad-hoc convergence criteria, that allows some initial 
    %flucutuation to the likelihood change.
    if (converged>10 && (incr)<1e-6)
        break
    end
    %Plot
   scatter(X(:,1),X(:,2),10,'.'); hold on
    mix = floor(min(X(:,1)))-2; miy = floor(min(X(:,2)))-2;
    mx = ceil(max(X(:,1)))+2; my = ceil(max(X(:,2)))+2;
    
  ezcontour(@(x,y)pdfmix(x,y,mu,SIGMA1,SIGMA2,alpha));
  pause(0.1);
  hold off;
    
    %%E-Step -- calculate the conditional expectation of the missing data.
    for i = 1:size(X,1),
      w(i,1)=   alpha(1)*1/2/pi/det(SIGMA1)*exp(-0.5*(X(i,:)' - mu(:,1))'*inv(SIGMA1)*(X(i,:)' -mu(:,1)));
      w(i,2)=   alpha(2)*1/2/pi/det(SIGMA2)*exp(-0.5*(X(i,:)' - mu(:,2))'*inv(SIGMA2)*(X(i,:)' -mu(:,2)));
    end
    %Normalize weights over samples.
    w = w./repmat(sum(w,2),[1 2]);
    
    
    %%M-Step 
    alpha = sum(w)./size(X,1); % The new weight vector
    
    mu(1,:) = sum(w.*repmat(X(:,1),[1 2]))./sum(w);% The new means
    mu(2,:) = sum(w.*repmat(X(:,2),[1 2]))./sum(w);
    
    sig1 = zeros(2); sig2 = zeros(2);%Something temporary
    for i = 1:size(X,1)
        sig1 = sig1+w(i,1).*X(i,:)'*X(i,:);
        sig2 = sig2+w(i,2).*X(i,:)'*X(i,:);
    end
    sig1 = sig1./sum(w(:,1));
    sig2 = sig2./sum(w(:,2));
    SIGMA1 = sig1 - mu(:,1)*mu(:,1)';%The new covariances.
    SIGMA2 = sig2 - mu(:,2)*mu(:,2)';
    converged = converged+1;    %Increment iteration
    
    %Does likelihood improve? -- you can use log-likelihood also
    z = 0;
    for i = 1:size(X,1)
        z = z + pdfmix(X(i,1),X(i,2),mu,SIGMA1, SIGMA2, alpha);
    end
    incr = (z - prevz)./prevz;
    prevz = z;
end
end

%The conditional pdf.
function z = pdfmix(x,y,mu,SIGMA1,SIGMA2,alpha)
z=alpha(1)*1/2/pi/det(SIGMA1)*exp(-0.5*([x;y] - mu(:,1))'*inv(SIGMA1)*([x;y] -mu(:,1))) + ...
    alpha(2)*1/2/pi/det(SIGMA2)*exp(-0.5*([x;y] - mu(:,2))'*inv(SIGMA2)*([x;y] -mu(:,2)));
end


%% Sai Ravela (C) 2016-

%% We will use a  sampling scheme for the  graphical Model
% The structure of the graph is specified here (a chain)
% We produce a conditional distribution on the chain, see
% class notes, and sample from it. 
nodes = 9; % Five node network
for i = 1:nodes
    node(i).mean = 0; % 
end
for i = 4:6,
    node(i).mean = -5; %some nodes have a different mean.
end
edge = zeros(nodes);
edge(nodes,nodes) = 0.5;
for i = 1:nodes-1 
    edge(i,i+1) = 0.3; % Small covariance between nodes
    edge(i,i)=0.5; % relatively large variance (1) at nodes.
    % We could also have constructed this through the potential vector and
    % information matrix, see another example for that.
end
edge = edge+edge'; % Just make it symmetric.

% We will use a full KL expansion of the Graph, treating it as a GMRF.
mu = cell2mat({node(:).mean});
covc = edge; [u,s,v] = svd(covc);
samps = repmat(mu(:),[1 100])+sqrt(s)*u*randn(9,100);
plot(samps,'Color',[0.5 0.5 0.5]);hold on;  %% As a Gauss markov Field

% If the spectral radius is large i.e. off-diagonal terms 
% are big in the covariance, then the constructed GGM behaves differently
% than a KL expansion of GMRF. Check this with this trace
disp(trace((u*pinv(s)*u')*edge))
for i = 1:nodes
        node(i).samp = node(i).mean+sqrt(edge(i,i))*randn;
        % Generate initial marginal samples.
end 
plot(cell2mat({node(:).samp}),'b','LineWidth',1.5); % This is assuming independence
%this is a markov chain of memory 1, so that one flush
%forward and backward should have everyone see the neighbor.
% We are sampling, not inferring something
for jj = 1:100, % 
    for i = 2:nodes % Forward pass.
    % Calculate conditional mean and variance
    cmean = node(i).mean+edge(i-1,i)/edge(i-1,i-1)*(node(i-1).samp-node(i-1).mean);
    cvar =edge(i,i)-edge(i-1,i)/edge(i-1,i-1)*edge(i-1,i); % From the neighbor
    node(i).samp = cmean+sqrt(cvar)*randn;
    end
 for i = nodes-1:-1:1 % Backward pass
     % Generate the right conditional mean and variance
     cmean = node(i).mean+edge(i,i+1)/edge(i+1,i+1)*(node(i+1).samp-node(i+1).mean);
     cvar = edge(i,i)-edge(i,i+1)/edge(i+1,i+1)*edge(i,i+1);
         node(i).samp = cmean+sqrt(cvar)*randn;

end
end
plot(cell2mat({node(:).samp}),'g','LineWidth',3);

for i = 1:nodes
    sampdum(i) = mu(i)+0.5*randn;
end
plot(sampdum,'y');
hold off;
% softth_overlap - computes the soft-threshold operation
%                  corresponding to the overlapped trace norm
%
% Syntax
%  X = softth_overlap(Y, lambda, varargin)
%
% See also
%  pca, pcaspec, softth_overlap
% 
% Reference
% "Estimation of low-rank tensors via convex optimization"
% Ryota Tomioka, Kohei Hayashi, and Hisashi Kashima
% arXiv:1010.0789
% http://arxiv.org/abs/1010.0789
%
% "Statistical Performance of Convex Tensor Decomposition"
% Ryota Tomioka, Taiji Suzuki, Kohei Hayashi, Hisashi Kashima
% NIPS 2011
% http://books.nips.cc/papers/files/nips24/NIPS2011_0596.pdf
%
% Convex Tensor Decomposition via Structured Schatten Norm Regularization
% Ryota Tomioka, Taiji Suzuki
% NIPS 2013
% http://papers.nips.cc/paper/4985-convex-tensor-decomposition-via-structured-schatten-norm-regularization.pdf
%
% Copyright(c) 2010-2014 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt


% softth_overlap - Computes the soft-threshold operation with respect
%                  to the overlapped Schatten 1-norm
%
% Syntax
%  function X = softth_overlap(Y, lambda, varargin)
%
% Reference
% "On the extension of trace norm to tensors"
% Ryota Tomioka, Kohei Hayashi, and Hisashi Kashima
% arXiv:1010.0789
% http://arxiv.org/abs/1010.0789
% 
% Copyright(c) 2010 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt


function X = softth_overlap(Y, lambda, varargin)

opt=propertylist2struct(varargin{:});
opt=set_defaults(opt, 'eta', [], 'tol', 1e-3, 'verbose', 0,'yfact',10,'maxiter',2000);

sz=size(Y);
nd=ndims(Y);
m =prod(sz);

gamma=ones(1,nd)/nd;


if ~isempty(opt.eta)
  eta=opt.eta;
else
  eta=1/(opt.yfact*std(Y(:)));
end


Z=cell(1,nd);
A=cell(1,nd);
S=cell(1,nd);

for jj=1:nd
  szj = [sz(jj), prod(sz)/sz(jj)];
  A{jj} = zeros(szj);
  Z{jj} = zeros(szj);
end

X=Y;

kk=1;
nsv=10*ones(1,nd);
viol=inf*ones(1,nd);
while 1
  % Update X
  X1 = Y/lambda;
  for jj=1:nd
    X1 = X1 + flatten_adj(eta*Z{jj}-A{jj},sz,jj);
  end

  X=X1/(1/lambda + nd*eta);

  % Update Z
  for jj=1:nd
    [Z{jj},S{jj},nsv(jj)] = softth(A{jj}/eta+flatten(X,jj),gamma(jj)/eta,nsv(jj));
    
    % Check derivative
    % fprintf('max[%d]=%g\n',jj,max(svd(eta*(Z{jj}-flatten(X,jj)-A{jj}/eta))));
  end

  % Update A
  for jj=1:nd
    V=flatten(X,jj)-Z{jj};
    A{jj}=A{jj}+eta*V;
    viol(jj)=norm(V(:));
  end
  
  % Compute the objective
  fval(kk)=0.5*norm(X(:)-Y(:))^2/lambda;
  G=(X-Y)/lambda;
  for jj=1:nd
    fval(kk)=fval(kk)+gamma(jj)*sum(svd(flatten(X,jj)));
    G=G+flatten_adj(A{jj},sz,jj);
  end

  res(kk)=max([norm(G(:))/eta,viol]);

  if opt.verbose
    fprintf('k=%d fval=%g res=%g viol=%s eta=%g\n',...
            kk, fval(kk), res(kk), printvec(viol),eta);
  end
  
  if kk>=3 && res(kk)<opt.tol
    break;
  end

  
 
  if kk>opt.maxiter
    break;
  end
  
  
  kk=kk+1;
end

fprintf('k=%d fval=%g res=%g viol=%s eta=%g\n',...
        kk, fval(kk), res(kk), printvec(viol),eta);



function dval=evalDual(A, Y, lambda, gamma, sz)

nd=length(A);

fact=1;
for jj=1:nd
  ss=pcaspec(A{jj},1,10);
  fact=min(fact,gamma(jj)/ss);
end
%fprintf('fact=%g\n',fact);

As=zeros(sz);
for jj=1:nd
  A{jj}=A{jj}*fact;
  As=As+flatten_adj(A{jj},sz,jj);
  %fprintf('norm(A{[%d]})=%g\n',jj,norm(A{jj}));
end




dval = 0.5*lambda*norm(As(:))^2 - Y(:)'*As(:);


% matrixl1_adm - computes sparse+low-rank decomposition of
%                partially observed matrix
%
% Syntax
%  [X,Z,A,fval,res]=matrixl1_adm(X, I, yy, lambda, varargin)
%
% See also
%  tensor_as_matrix
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

function [X,Z,A,fval,res]=matrixl1_adm(X, I, yy, lambda, varargin)

opt=propertylist2struct(varargin{:});
opt=set_defaults(opt, 'eta',[], 'eta1', [], 'yfact', 10, 'gamma', [], 'tol', 1e-3, 'verbose', 0,'maxiter',2000);

if ~isempty(opt.gamma)
  gamma=opt.gamma;
else
  gamma=1;
end

if ~isempty(opt.eta)
  eta=opt.eta;
else
  eta=1/(opt.yfact*std(yy));
end

if ~isempty(opt.eta1)
  eta1=opt.eta1;
else
  eta1=1/(opt.yfact*std(yy));
end


sz=size(X);

m=length(yy);


Z=X;
A=zeros(size(X));

Y=zeros(sz);
ind=sub2ind(sz,I{:});
Y(ind)=yy;

delta = zeros(m,1);
beta  = zeros(m,1);

kk=1;
nsv=10;
dval=-inf;
while 1
  % X update
  X1=eta*Z-A;
  X1(ind)=X1(ind) + eta1*(yy-delta-beta/eta1);
  X = X1./(eta1*(Y~=0)+eta);

  % delta update
  [delta,ss] = l1_softth(yy-X(ind)-beta/eta1, 1/lambda/eta1);
  
  % Z update
  [Z,ss,nsv]=softth(X+A/eta,gamma/eta,nsv);

  % A update
  A=A+eta*(X-Z);
  
  % beta update
  beta = beta + eta1*(X(ind)+delta-yy);
  
  viol = [norm(X(:)-Z(:)), norm(X(ind)+delta-yy)];
  

  fval(kk)=gamma*sum(svd(X));
  if lambda>0
    fval(kk)=fval(kk)+sum(abs(X(ind)-yy))/lambda;
  end


  % gval(kk)=eta*norm(Z(:)-Z0(:)); %norm(G(:));
  dval = max(dval, -evalDual(A, beta, yy, lambda, gamma, ind));
  res(kk)=1-dval/fval(kk);
  % res(kk) = max(viol);
  
  if opt.verbose
    fprintf('[%d] fval=%g res=%g viol=%s\n', kk, fval(kk), res(kk), ...
            printvec(viol));
  end
  
  
  if res(kk)<opt.tol
    break;
  end
  
  if kk>opt.maxiter
    break;
  end

  kk=kk+1;
end

fprintf('[%d] fval=%g res=%g viol=%s eta=%g\n', kk, fval(kk), res(kk), ...
        printvec(viol),eta);


function dval=evalDual(A, beta, yy, lambda, gamma, ind)

sz=size(A);
ind_te=setdiff(1:prod(sz),ind);
A(ind)=-beta;
A(ind_te)=0;
ss=pcaspec(A,1,10);

fact=min([1,gamma/ss,1/lambda/max(abs(beta))]);
A=A*fact;
beta=beta*fact;

% fprintf('fact=%g\n',fact);

dval = yy'*beta;




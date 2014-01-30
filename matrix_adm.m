% matrix_adm - Computes reconstruction of a partly observed matrix
%
% Syntax
%  [X,Z,A,fval,res]=matrix_adm(X, I, yy, lambda, varargin)
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

function [X,Z,A,fval,res]=matrix_adm(X, I, yy, lambda, varargin)

opt=propertylist2struct(varargin{:});
opt=set_defaults(opt, 'eta',[], 'yfact', 10, 'gamma', [], 'tol', ...
                      1e-3, 'verbose', 0, 'relative',1,'maxiter',2000);

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


sz=size(X);

m=length(yy);


Z=zeros(size(X));
A=zeros(size(X));

B=zeros(sz);
ind=sub2ind(sz,I{:});
B(ind)=yy;

kk=1;
nsv=10;
while 1
  if lambda>0
    X = (B/lambda+eta*Z-A)./((B~=0)/lambda+eta);
  else
    X=Z-A/eta;
    X(ind)=yy;
  end
  
  Z0=Z;
  [Z,ss,nsv]=softth(X+A/eta,gamma/eta,nsv);

  A=A+eta*(X-Z);
  
  viol = norm(X(:)-Z(:));
  

  fval(kk)=gamma*sum(svd(X));
  if lambda>0
    fval(kk)=fval(kk)+0.5*sum((X(ind)-yy).^2)/lambda;
  end


  % gval(kk)=eta*norm(Z(:)-Z0(:)); %norm(G(:));
  gap=fval(kk)+evalDual(A, yy, lambda, gamma, ind);
  if opt.relative
    gap=gap/fval(kk);
  end
  res(kk)=gap;
  
  if opt.verbose
    fprintf('[%d] fval=%g res=%g viol=%g\n', kk, fval(kk), res(kk), ...
            viol);
  end
  
  
  if res(kk)<opt.tol
    break;
  end
  
  if kk>opt.maxiter
    break;
  end

  kk=kk+1;
end

fprintf('[%d] fval=%g res=%g viol=%g eta=%g\n', kk, fval(kk), res(kk), ...
        viol,eta);


function dval=evalDual(A, yy, lambda, gamma, ind)

sz=size(A);
ind_te=setdiff(1:prod(sz),ind);
A(ind_te)=0;
ss=pcaspec(A,1,10);
A=A/max(1,ss/gamma);

dval = 0.5*lambda*norm(A(ind))^2 - yy'*A(ind);


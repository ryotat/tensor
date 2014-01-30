% tensormix_adm - computes the reconstruction of a partially
%                 observed tensor via latent approach
%
% Syntax
%  [X,Z,A,fval,res]=tensormix_adm(X, I, yy, lambda, varargin)
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

function [X,Z,A,fval,res]=tensormix_adm(X, I, yy, lambda, varargin)

opt=propertylist2struct(varargin{:});
opt=set_defaults(opt, 'eta', [],'gamma',[], 'tol', 1e-3, 'verbose', 0,'yfact',10,'maxiter',2000);

sz=size(X);
nd=ndims(X);

if ~isempty(opt.gamma)
  gamma=opt.gamma;
else
  gamma=ones(1,nd);
end

if ~isempty(opt.eta)
  eta=opt.eta;
else
  eta=opt.yfact*std(yy);
end

ind=sub2ind(sz, I{:});
ind0=setdiff(1:prod(sz),ind);
m=size(yy,1);

Z=cell(1,nd);
V=cell(1,nd);
for jj=1:nd
  szj = [sz(jj), prod(sz)/sz(jj)];
  Z{jj} = zeros(szj);
  V{jj} = zeros(szj);
end

kk=1;
nsv=10*ones(1,nd);
alpha=yy;
A=zeros(sz); A(ind)=alpha;
while 1
  for jj=1:nd
    Ztmp = Z{jj}+eta*flatten(A,jj);
    [Z{jj},S{jj},nsv(jj)]=softth(Ztmp,gamma(jj)*eta,nsv(jj));
    V{jj}=(Ztmp-Z{jj})/eta;

    viol(jj)=norm(flatten(A,jj)-V{jj},'fro');
  end
 
  Zsum = zeros(sz);
  Vsum = zeros(sz);
  for jj=1:nd
    Zsum=Zsum+flatten_adj(Z{jj},sz,jj);
    Vsum=Vsum+flatten_adj(V{jj},sz,jj);
  end
  
  alpha = (yy-Zsum(ind)+eta*Vsum(ind))/(lambda+eta*nd);
  A(ind)=alpha;

   
  % Compute the objective
  if lambda>0
    fval(kk)=0.5*sum((Zsum(ind)-yy).^2)/lambda;
  else
    Zm=(Zsum(ind)-yy)/nd;
    fval(kk)=0;
  end
  
  gval(kk)=norm(lambda*alpha - yy + Zsum(ind));

  for jj=1:nd
    if lambda>0
      fval(kk)=fval(kk)+gamma(jj)*sum(S{jj});
    else
      Ztmp=flatten_adj(Z{jj},sz,jj);
      Ztmp(ind)=Ztmp(ind)-Zm;
      fval(kk)=fval(kk)+gamma(jj)*sum(svd(flatten(Ztmp,jj)));
    end
  end
  
  % Compute the dual objective
  fact=1;
  for jj=1:nd
    fact=min(fact, gamma(jj)/norm(flatten(A,jj)));
  end
  
  aa = alpha*fact;
  
  if kk>1
    dval=max(dval,-0.5*lambda*sum(aa.^2)+aa'*yy);
  else
    dval=-inf;
  end
  res(kk)=1-dval/fval(kk);
  
  if opt.verbose 
    fprintf('[%d] fval=%g res=%g viol=%s\n', kk, fval(kk), ...
          res(kk), printvec(viol));
%    fprintf('[%d] fval=%g dval=%g fact=%g\n', kk, fval(kk), ...
%          dval(kk), fact);
  end
%  if kk>1 && 1-dval(kk)/fval(kk)<tol
  if kk>1 && res(kk)<opt.tol % max(viol)<opt.tol && gval(kk)<opt.tol
   break;
  end
  
  if kk>opt.maxiter
    break;
  end

  kk=kk+1;
end

fprintf('[%d] fval=%g res=%g viol=%s eta=%g\n', kk, fval(kk), ...
        res(kk), printvec(viol),eta);

%fprintf('[%d] fval=%g dval=%g fact=%g\n', kk, fval(kk), ...
%        dval(kk), fact);

X=zeros(sz);
for jj=1:nd
  X = X + flatten_adj(Z{jj},sz,jj);
  Z{jj}=Z{jj}*nd;
end

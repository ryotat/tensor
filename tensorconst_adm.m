% tensorconst_adm - computes the reconstruction of a partially
%                   observed tensor using overlapped approach
%
% Syntax
%  [X,Z,A,fval,res] = tensorconst_adm(X, I, yy, lambda, varargin)
%
% See also
%  tensormix_adm, exp_completion
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

function [X,Z,A,fval,res] = tensorconst_adm(X, I, yy, lambda, varargin)

opt=propertylist2struct(varargin{:});
opt=set_defaults(opt, 'eta', [], 'gamma',[],'tol', 1e-3, 'verbose', 0,'yfact',10,'maxiter',2000);

sz=size(X);
nd=ndims(X);
m =length(I{1});

if ~isempty(opt.gamma)
  gamma=opt.gamma;
else
  gamma=ones(1,nd);
end

if ~isempty(opt.eta)
  eta=opt.eta;
else
  eta=1/(opt.yfact*std(yy));
end

if nd~=length(I)
  error('Number of dimensions mismatch.');
end

if m~=length(yy)
  error('Number of samples mismatch.');
end

Z=cell(1,nd);
A=cell(1,nd);
S=cell(1,nd);

for jj=1:nd
  szj = [sz(jj), prod(sz)/sz(jj)];
  A{jj} = zeros(szj);
  Z{jj} = zeros(szj);
end

B=zeros(sz);
ind=sub2ind(sz, I{:});
B(ind)=yy;

kk=1;
nsv=10*ones(1,nd);
while 1
  X1 = zeros(size(X));
  for jj=1:nd
    X1 = X1 - flatten_adj(A{jj}-eta*Z{jj},sz,jj);
  end
  
  if lambda>0
    X1(ind) = X1(ind) + yy/lambda;
    X=X1./((B~=0)/lambda + nd*eta);
  else
    X=X1/(eta*nd);
    X(ind)=yy;
  end
  

  % Check derivative
% $$$   D=zeros(size(X));
% $$$   D(ind)=X(ind)-yy;
% $$$   for jj=1:nd
% $$$     D=D+eta*flatten_adj(A{jj}/eta+flatten(X,jj)-Z{jj},sz,jj);
% $$$   end
% $$$   fprintf('gnorm=%g\n',norm(D(:)));
  
  for jj=1:nd
    [Z{jj},S{jj},nsv(jj)] = softth(A{jj}/eta+flatten(X,jj),gamma(jj)/eta,nsv(jj));
    
    % Check derivative
    % fprintf('max[%d]=%g\n',jj,max(svd(eta*(Z{jj}-flatten(X,jj)-A{jj}/eta))));
  end

  for jj=1:nd
    V=flatten(X,jj)-Z{jj};
    A{jj}=A{jj}+eta*V;
    viol(jj)=norm(V(:));
  end
  
  
  % Compute the objective
  G=zeros(size(X));
  fval(kk)=0;
  for jj=1:nd
    fval(kk)=fval(kk)+gamma(jj)*sum(svd(flatten(X,jj)));
    % fval(kk)=fval(kk)+gamma(jj)*sum(S{jj});
    G = G + flatten_adj(A{jj},sz,jj);
  end
  if lambda>0
    fval(kk)=fval(kk)+0.5*sum((X(ind)-yy).^2)/lambda;
    G(ind)=G(ind)+(X(ind)-yy)/lambda;
  else
    G(ind)=0;
  end

  res(kk)=1+evalDual(A, yy, lambda, gamma, sz, ind)/fval(kk);
  %  res(kk)=max([norm(G(:))/eta,viol]);% /norm(X(:));

  if opt.verbose
    fprintf('k=%d fval=%g res=%g viol=%s eta=%g\n',...
            kk, fval(kk), res(kk), printvec(viol),eta);
  end
  
  if kk>1 && res(kk)<opt.tol %  max(viol)<opt.tol && gval(kk)<opt.tol
    break;
  end
  
  if kk>opt.maxiter
    break;
  end
  
  
  kk=kk+1;
end

fprintf('k=%d fval=%g res=%g viol=%s eta=%g\n',...
        kk, fval(kk), res(kk), printvec(viol),eta);



function dval=evalDual(A, yy, lambda, gamma, sz, ind)

nd=length(A);

Am=zeros(sz);
for jj=1:nd
  Am=Am+flatten_adj(A{jj},sz,jj);
end

Am(ind)=0;
Am=Am/nd;

fact=1;
for jj=1:nd
  A{jj}=A{jj}-flatten(Am,jj);
  ss=pcaspec(A{jj},1,10);
  fact=min(fact,gamma(jj)/ss);
end
%fprintf('fact=%g\n',fact);

As=zeros(sz);
for jj=1:nd
  fact=min(1,gamma(jj)/ss);
  A{jj}=A{jj}*fact;
  As=As+flatten_adj(A{jj},sz,jj);
  % fprintf('fact[%d]=%g ',jj, max(1,ss/gamma(jj)));
end
%fprintf('norm(Am)=%g\n', norm(Am(:)));

% $$$ ind_test=setdiff(1:prod(sz), ind);
% $$$ V=zeros(sz);
% $$$ for jj=1:nd
% $$$   V=V+flatten_adj(A{jj},sz,jj);
% $$$ end
% $$$ fprintf('violation=%g\n',norm(V(ind_test)));


dval = 0.5*lambda*norm(As(ind))^2 - yy'*As(ind);


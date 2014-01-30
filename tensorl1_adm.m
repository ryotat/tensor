% tensorl1_adm - sparse + low-rank decomposition via overlapped approach
%
% Syntax
%  [X,Z,A,beta,fval,res] = tensorl1_adm(X, I, yy, lambda, varargin)
%
% See also
%  tensorconst_adm
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

function [X,Z,A,beta,fval,res] = tensorl1_adm(X, I, yy, lambda, varargin)

opt=propertylist2struct(varargin{:});
opt=set_defaults(opt, 'eta', [], 'eta1', [], 'gamma',[],'tol', 1e-3, 'verbose', 0,'yfact',10,'maxiter',2000);

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

if ~isempty(opt.eta1)
  eta1=opt.eta1;
else
  eta1=1/(opt.yfact*std(yy));
  % eta1=1/(opt.yfact*std(yy)*lambda^(1/4));
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

Y=zeros(sz);
ind=sub2ind(sz, I{:});
Y(ind)=yy;

delta = zeros(m,1);
beta  = zeros(m,1);

kk=1;
nsv=10*ones(1,nd);
dval=-inf;
while 1
  Xorig = X;

  % X update
  X1 = zeros(size(X));
  for jj=1:nd
    X1 = X1 + flatten_adj(eta*Z{jj}-A{jj},sz,jj);
  end
  
  X1(ind) = X1(ind) + eta1*(yy-delta-beta/eta1);
  X=X1./(eta1*(Y~=0)+nd*eta);

  % delta update
  [delta,ss] = l1_softth(yy-X(ind)-beta/eta1, 1/lambda/eta1);

  % Z update
  for jj=1:nd
    [Z{jj},S{jj},nsv(jj)] = softth(flatten(X,jj)+A{jj}/eta,gamma(jj)/eta,nsv(jj));
    
    % Check derivative
    % fprintf('max[%d]=%g\n',jj,max(svd(eta*(Z{jj}-flatten(X,jj)-A{jj}/eta))));
  end

  % Update A
  for jj=1:nd
    V=flatten(X,jj)-Z{jj};
    A{jj}=A{jj}+eta*V;
    viol(jj)=norm(V(:));
  end
  
  % Update beta
  beta = beta + eta1*(X(ind)+delta-yy);
  
  % Compute the objective
  G=zeros(size(X));
  fval(kk)=0;
  for jj=1:nd
    fval(kk)=fval(kk)+gamma(jj)*sum(svd(flatten(X,jj)));
    G = G + flatten_adj(A{jj},sz,jj);
  end
  if lambda>0
    fval(kk)=fval(kk)+sum(abs(yy-X(ind)))/lambda;
    G(ind)=G(ind)+(X(ind)-yy)/lambda;
  else
    G(ind)=0;
  end

  viol(nd+1) =norm(X(ind)+delta-yy);
  
  dval = max(dval, -evalDual(A, beta, yy, lambda, gamma, sz, ind));
  res(kk)=1-dval/fval(kk);
  % res(kk)=max([norm(G(:))/eta,viol]);% /norm(X(:));
  % res(kk)=max(viol);
  
  if opt.verbose
    fprintf('k=%d fval=%g res=%g viol=%s eta=%s\n',...
            kk, fval(kk), res(kk), printvec(viol),printvec([eta eta1]));
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



function dval=evalDual(A, beta, yy, lambda, gamma, sz, ind)

nd=length(A);

Am=zeros(sz);
for jj=1:nd
  Am=Am+flatten_adj(A{jj},sz,jj);
end

Am(ind)=Am(ind)+beta;
Am=Am/nd;

fact=1;
for jj=1:nd
  A{jj}=A{jj}-flatten(Am,jj);
  ss=pcaspec(A{jj},1,10);
  fact=min(fact,gamma(jj)/ss);
end
% fprintf('fact=%g\n',fact);

fact = min(fact, 1/lambda/max(abs(beta)));

As=zeros(sz);
for jj=1:nd
  A{jj}=A{jj}*fact;
  As=As+flatten_adj(A{jj},sz,jj);
  % fprintf('fact[%d]=%g ',jj, max(1,ss/gamma(jj)));
end
beta=beta*fact;
As(ind)=As(ind)+beta;

% fprintf('norm(Am)=%g fact=%g norm(As)=%g\n', norm(Am(:)), fact, norm(As(:)));

% $$$ ind_test=setdiff(1:prod(sz), ind);
% $$$ V=zeros(sz);
% $$$ for jj=1:nd
% $$$   V=V+flatten_adj(A{jj},sz,jj);
% $$$ end
% $$$ fprintf('violation=%g\n',norm(V(ind_test)));


dval = yy'*beta;


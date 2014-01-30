% softth - computes the proximity operator corresponding to the
%         trace norm
%
% Syntax
%  [vv,ss,nsv]=softth(vv, lambda, nsv, verbose);
%
% See also
%  pca, softth_overlap
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

function [vv,ss,nsv]=softth(vv, lambda, nsv, verbose);

if ~exist('verbose','var')
  verbose=0;
end


sz=size(vv);
nsv=min(min(sz),nsv+1);

if verbose
  t0=cputime;
  fprintf('sz=[%d %d]\n',sz(1), sz(2));
  fprintf('nsv=');
end

while 1
  if verbose
    fprintf('%d/',nsv);
  end
  [U,S,V]=pca(vv,min(min(sz),nsv),10);
 ss=diag(S);
  if min(ss)<lambda || nsv==min(sz)
    if verbose
      fprintf('min(ss)=%g time=%g\n',min(ss), cputime-t0);
    end
    break;
  else
    nsv=min(min(sz),round(nsv*2));
  end
end

ix=find(ss>=lambda);
ss=ss(ix)-lambda;
vv = U(:,ix)*diag(ss)*V(:,ix)';

nsv=length(ix);
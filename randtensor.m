% randtensor - randomly generates a low-rank tensor
%
% Syntax
%  X=randtensor(sz, dims)
%
% See also
%  randtensor3
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

function X=randtensor(sz, dims)

nd=length(sz);

X=randn(dims);

for jj=1:nd
  [U,R]=qr(randn(sz(jj),dims(jj)),0);

  sz1=size(X);
  sz1(jj)=sz(jj);
  X = flatten_adj(U*flatten(X,jj),sz1,jj);
end



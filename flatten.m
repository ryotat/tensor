% flatten - mode-k unfolding of a tensor
%
% Syntax
%  Z=flatten(X,ind)
%
% See also
%  flatten_adj
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

function Z=flatten(X,ind)

sz=size(X);

if isnumeric(ind)
  nd=max(ndims(X),ind);
  ind = {ind, [ind+1:nd, 1:ind-1]};
else
  nd=max(cell2mat(foreach(@max,ind)));
end


if length(ind{1})~=1 || ind{1}~=1
  X=permute(X,cell2mat(ind));
end

if length(ind{1})==1
  Z=X(:,:);
else
  Z=reshape(X,[prod(sz(ind{1})),prod(sz(ind{2}))]);
end

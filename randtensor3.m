% randtensor3 - randomly generates a 3-way low-rank tensor
%
% Syntax
%  X=randtensor3(sz, dims, const)
%
% Examples
%  X=randtensor3([50 50 20], [7 8 9]);  % tucker
%  X=randtensor3([50 50 20], 3);        % parafac
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

function X=randtensor3(sz, dims, const)

nd=length(sz);
U=cell(1,nd);
if exist('const','var') && const
  C=randn(dims);
  U{1}=qr(randn(sz(1),dims),0);
  U{2}=qr(randn(sz(2),dims),0);
  U{3}=rand(sz(3),dims);
  U{3}=bsxfun(@mrdivide, U{3}, sqrt(sum(U{3}.^2)));
else
  if length(dims)>1
    C=randn(dims);
    for jj=1:nd
      [U{jj},R]=qr(randn(sz(jj),dims(jj)),0);
    end
  else
    C=diag3(randn(dims));
    for jj=1:nd
      % [U{jj},R]=qr(randn(sz(jj),dims),0);
      U{jj}=randn(sz(jj),dims);
    end
 end
end
X=kolda3(C,U{:});

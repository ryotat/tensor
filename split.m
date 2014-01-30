% split - splits a string (or a cell array of string) into a cell array
%
% Syntax
%  B = split(A, sep)
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

function B = split(A, sep)

if ischar(A)
  A = {A};
end

B = cell(size(A));

for i=1:prod(size(A))
  s = findstr(A{i},sep);
  c = 1;
  B{i} = [];
  for j=1:length(s)
    B{i} = [B{i}, {A{i}(c:s(j)-1)}];
    c = s(j)+1;
  end
  if (c<=length(A{i}))
    B{i} = [B{i}, {A{i}(c:end)}];
  end
end

if prod(size(A))==1
  B = B{1};
end

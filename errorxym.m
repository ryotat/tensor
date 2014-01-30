% errorxym - plots multiple sequences of points with different
%            markers (without connecting lines)
%
% Syntax
%  errorxym(X, Ym, Ys, varargin)
%
% See also
%  errorxy
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

function errorxym(X, Ym, Ys, varargin)

if size(X,1)<size(X,2)
  X=X';
end

if size(Ym,1)~=length(X)
  error('size(Ym,1) must equal length(X)');
end

if ~isempty(Ys) && size(Ys,1)~=length(X)
  error('size(Ys,1) must equal length(X)');
end

markerorder = {'x','^','o','+','v','*'};
nm=length(markerorder);
for jj=1:size(Ym,2)
  hold on;
  if ~isempty(Ys)
    errorxy([X,Ym(:,jj),Ys(:,jj)],'Marker',markerorder{mod(jj-1,nm)+1},varargin{:});
  else
    plot(X, Ym(:,jj),markerorder{mod(jj-1,nm)+1},varargin{:});
  end
end

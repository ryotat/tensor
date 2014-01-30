% tensor_as_matrix - Computes the reconstruction of partly observed
%                    tensor via "As A Matrix" approach
%
% Syntax
%  [X,Z,fval,gval]=tensor_as_matrix(X, I, yy, lambda, varargin)
%
% See also
%  matrix_adm, matrixl1_adm
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

function [X,Z,fval,gval]=tensor_as_matrix(X, I, yy, lambda, varargin)
opt=propertylist2struct(varargin{:});
opt=set_defaults(opt, 'eta', [], 'tol', 1e-3, 'solver', @matrix_adm);


if ~exist('tol','var')
  tol=1e-3;
end

sz=size(X);
nd=ndims(X);

Z=cell(1,nd);
for ii=1:nd
  szp=[sz(ii:end) sz(1:ii-1)];
  Ip=[I(ii:end) I(1:ii-1)];
  J =sub2ind(szp(2:end), Ip{2:end});
  [Z1,Z{ii},Y,fval1,gval1]=opt.solver(zeros(szp(1),prod(szp(2:end))),{Ip{1}, J}, yy, lambda, opt);
  Z{ii}=Z{ii};
  fval(ii)=fval1(end);
  gval(ii)=gval1(end);
end

X=[];


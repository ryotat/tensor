% plot_nips2013 - plots figures for nips 2013 paper
%
% See also
%  plot_overlap_vs_latent_2013, plot_compare_denoising
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


plot_overlap_vs_latent_2013
ylim([0 70])
% printFigure(1,'overlap_vs_latent_2013.eps')

sz=[50 50 20];
sigmas=0.1;
lambda=[0.43 0.89 3.8; 0.89 3.8 11.3];
bHold=0;
plot_compare_denoising;

sz=[80 80 40];
sigmas=0.1;
lambda=1.6*[0.43 0.89 3.8; 0.89 3.8 11.3];
bHold=1;
plot_compare_denoising;

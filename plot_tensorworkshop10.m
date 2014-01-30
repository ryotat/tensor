% PLOT_TENSORWORKSHOP10 - Plots figure for TKML workshop 2010
%
% Example
%  load('result_compare5_new_50_50_20_7_8_9.mat')
%  plot_tensorworkshop10
%
% See also
%  exp_completion, tensorconst_adm, tensormix_adm
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

% load('result_compare5_50_50_20_7_8_9.mat')
nm = size(err,3);

if ~exist('tol','var')
  tol=1e-3;
end

figure, h=errorbar_logsafe(trfrac'*ones(1,nm), shiftdim(mean(err)), shiftdim(std(err)));

set(gca,'fontsize',14,'yscale','log');
ylim([1e-5 1e+2]);

set(h,'linewidth',2);
set(h(1:3),'color',[0 0 1]);
set(h(1),'linestyle','--');                                
set(h(2),'linestyle','-.');

col=get(gca,'colororder');
for ii=4:length(h), set(h(ii),'color', col(ii-2,:)); end

hold on;
plot(xlim, tol*[1 1], '--', 'color', [.5 .5 .5], 'linewidth',2);


grid on;
xlabel('Fraction of observed elements');
ylabel('Generalization error');
legend('As a Matrix (mode 1)',...
       'As a Matrix (mode 2)', ...
       'As a Matrix (mode 3)',...
       'Constraint',...
       'Mixture',...
       'Tucker (large)',...
       'Tucker (exact)',...
       'Optimization tolerance',...
       'Location','NorthEastOutside');

h=get(gca,'children');
set(gca,'children',h([2:end,1]));

set(gcf,'PaperSize',[20 20]);

% plot_completion_l1 - undocumented
%
% See also
%  exp_completion_l1
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




f = {[], @(x)23*x.^(1/4), @(x)10*x.^(1/2)};

for ii=1:3
  figure, imagesc(trfrac,log10(lambda(:,ii)), squeeze(mean(gen(:,:,ii,:))));
  set(gca,'fontsize',12,'clim',[0 1]);
  xlabel('Fraction of observed elements');
  ylabel('log(Regularization constant)');
  colorbar;
  hold on;
  if ~isempty(f{ii})
    plot(trfrac, log10(f{ii}(trfrac)), 'm--', 'linewidth', 2);
  end
  set(gcf,'paperpositionmode','auto','papersize',[20 20]);
end




% Compute ROC curve
ixm = 10; % 50% observed
ntr = round(prod(sz)*trfrac(ixm));
ncor = round(ntr*noisefr);
figure
for ii=1:3
  [mm,ilmd]=min(gen(1,:,ii,ixm))
  err0 = memo(1,ilmd,ii,ixm).err0;
  err  = memo(1,ilmd,ii,ixm).err;
  [ss,ix]=sort(-abs(err));
  
  tp=cumsum((abs(err0(ix))>0)/ncor);
  fp=cumsum((abs(err0(ix))==0)/(ntr-ncor));

  auc(ii)=diff(fp)'*tp(2:end);
  
  subplot(1,3,ii);
  plot(fp, tp, '-x', 'linewidth',2);
  set(gca,'fontsize',16);
  grid on;
  xlabel('False positive rate');
  ylabel('True positive rate');
  ylim([0.9 1]);
end


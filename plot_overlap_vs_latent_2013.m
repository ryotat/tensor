% plot_overlap_vs_latent_2013 - plots figure 1
%
% See also
%  exp_compare_full
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


rr=5:5:50;
err_best=zeros(length(rr), 3);
errbar_best=zeros(length(rr), 3);
lmd_best=zeros(length(rr), 3);
err_fix=zeros(length(rr),3);
ix_lmd_fix=[7, 11,1];

for jj=1:length(rr)
  r=rr(jj);
  S=load(sprintf('result_compare_full_50_50_20_%d_%d_3_nrep=10_sigma=0.1_demo.mat',r,r));
  
  errm=shiftdim(mean(S.err));
  errs=shiftdim(std(S.err));
  
  [mm,ix]=min(errm);
  
  err_best(jj,:)=mm;
  lmd_best(jj,:)=S.lambda(ix);
  
  errbar_best(jj,:)=diag(errs(ix,1:3))';
  
  err_fix(jj,:)=diag(errm(ix_lmd_fix,1:3))';

  errbar_fix(jj,:)=diag(errs(ix_lmd_fix,1:3))';
end

S=load('result_compare_full_50_50_20_40_40_3_nrep=10_sigma=0.1_demo.mat')

figure,
h=errorbar(rr'*ones(1,2), err_fix(:,1:2), errbar_fix(:,1:2), 'linewidth', 2);
hold on; plot(rr, err_best(:,1:2), '--','linewidth',2); 
grid on;
set(gca,'fontsize',16);
xlabel('Rank of the first two modes');
ylabel('Estimation error ||W-W*||_F');
title(sprintf('size=%s',printvec(S.sz)));
legend('Overlapped Schatten 1-norm','Latent Schatten 1-norm');
ylim([6 30]);
plot([40 40], ylim, '--', 'color', [.5 .5 .5], 'linewidth', 2);

ax=axes('position',[0.2 0.6 0.3 0.25]);
errm=shiftdim(mean(S.err(:,:,1:2)));
h=errorbar(S.lambda'*[1 1], errm, ...
           shiftdim(std(S.err(:,:,1:2))));
set(h,'linewidth',2);
set(gca,'xscale','log','fontsize',12);
grid on;

hold on;
plot([1;1]*S.lambda(ix_lmd_fix(1:2)), [0, 0; ...
                    diag(errm(ix_lmd_fix(1:2),1:2))'], 'm--',...
     'linewidth',2);

xlabel('Regularization constant \lambda');
ylabel('||W-W*||_F');
xlim(rangeof(S.lambda));
title(sprintf('rank=%s',printvec(S.dtrue)));
set(gcf,'papersize',[20 20],...
        'position', [32, 421, 832   470]);
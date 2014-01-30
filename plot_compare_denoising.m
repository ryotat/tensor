% plot_compare_denoising - plots the result of exp_denoising
%
% See also
%  exp_denoising, plot_nips2013
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


%sz=input('size=');
%sz=[60 60 30];
%sigmas=input('sigma='); % [0.01, 0.1];
%lambda=input('lambda=');
%bHold = input('bHold=');

nn=prod(sz);

K=length(sz);

for sigma=sigmas
  S=ls(sprintf('result_compare_full_%d_%d_%d_*sigma=%g.mat',sz(1),sz(2),sz(3),sigma));

  files=split(S,char(10));

  % Find best lambda
% $$$   ix=zeros(length(files),2);
% $$$   for ii=1:length(files)
% $$$     S=load(files{ii});
% $$$     [mm,ix(ii,:)]=min(shiftdim(mean(S.err)));
% $$$   end
% $$$   ix=floor(median(ix));
  S=load(files{1});
  ix=zeros(size(lambda));
  leg=cell(size(lambda));
  for ii=1:prod(size(lambda))
    [mm,ix(ii)]=min(abs(S.lambda-lambda(ii)));
    leg{ii}=sprintf('size=%s \\lambda=%.2f', printvec(sz), S.lambda(ix(ii)));
  end
  if ~bHold
    leg1=leg;
  end
  
  fprintf('sigma=%g lambda=%s (%s)\n', sigma, printvec(S.lambda(ix)), printvec(ix));

  clear X1 X2 Y1 Y2
  Y1=zeros(S.nrep, length(lambda), length(files));
  Y2=zeros(S.nrep, length(lambda), length(files));
  for ii=1:length(files)
    S=load(files{ii});
    X1(ii)=mean(sqrt(1./sz))^2*mean(sqrt(S.dtrue))^2;
    X2(:,ii)=min(sum(S.rank_mix,2),min(S.dtrue))/min(sz);
    Y1(:,:,ii)=S.err(:,ix(1,:),1).^2/nn;
    Y2(:,:,ii)=S.err(:,ix(2,:),2).^2/nn;
  end

  F=(1+sqrt(min(sz)*max(sz)/nn)+2/sigma*sqrt(min(sz)/nn))^2;

  
  if ~bHold
    figure;
  end
    
%  subplot(1,2,find(sigmas==sigma));
%  errorxy([mean(X)',mean(Y)',std(X)',std(Y)'],'ColXe',3,'ColYe',4,'WidthEB',2,'Marker','x','MarkSize',10)

  mY1=shiftdim(mean(Y1))';
  p=polyfit(X1', mY1(:,2), 1);
  if bHold
    axes(ax(1));
    hold on;
    errorxym(X1', mY1,[],'Color','r','MarkerSize',10,'LineWidth',2);
    plot([0 1], polyval(p,[0 1]), '--', 'color', [.5 .5 .5], 'linewidth',2);
    h=get(gca,'children');
    h=h(strcmp(get(get(gca,'children'),'linestyle'),'none'));
    legend(flipud(h),[leg1(1,:),leg(1,:)]);
  else
    subplot(1,3,1);
    errorxym(X1', mY1,[],'MarkerSize',10, 'LineWidth',2);
    hold on;
    plot([0 1], polyval(p,[0 1]), '--', 'color', [.5 .5 .5], 'linewidth',2);
    ylim([0 0.015]);
    grid on;
    set(gca,'fontsize',12);
    xlabel('Tucker rank complexity');
    ylabel('Mean squared error (overlap)')
    title('Overlapped approach');
    legend(leg{1,:});
    pos=get(gca,'position');
    set(gca,'position',[pos(1) 0.15, pos(3), 0.7])
    ax(1)=gca;
  end
  
  mY2=shiftdim(mean(Y2))';
  p=polyfit(mean(X2)', mY2(:,2), 1);
  if bHold
    axes(ax(2));
    hold on;
    errorxym(mean(X2)',mY2,[], 'Color','r','MarkerSize',10,'LineWidth',2);
    plot([0 1], polyval(p,[0 1]), '--', 'color', [.5 .5 .5], 'linewidth',2);
    h=get(gca,'children');
    h=h(strcmp(get(get(gca,'children'),'linestyle'),'none'));
    legend(flipud(h), [leg1(2,:),leg(2,:)]);
  else
    subplot(1,3,2);
    errorxym(mean(X2)', mY2,[],'MarkerSize', 10,'LineWidth',2);
    hold on;
    plot([0 1], polyval(p,[0 1]), '--', 'color', [.5 .5 .5], 'linewidth',2);
    ylim([0 0.015]);
    grid on;
    set(gca,'fontsize',12);
    xlabel('Latent rank complexity');
    ylabel('Mean squared error (latent)')
    title('Latent approach');
    legend(leg{2,:});
    pos=get(gca,'position');
    set(gca,'position',[pos(1) 0.15, pos(3), 0.7])
    ax(2)=gca;
  end
  
  if bHold
    axes(ax(3));
    hold on;
    errorxym(X1./mean(X2), shiftdim(mean(Y1)./mean(Y2))', [], 'Color','r' ,'linewidth',2, 'MarkerSize',10);
  else
    subplot(1,3,3);
    errorxym(X1./mean(X2), shiftdim(mean(Y1)./mean(Y2))',[] ,'linewidth',2, 'MarkerSize',10);
    xlim([0 max(xlim)]);
    ylim([0 max(ylim)]);
    set(gca,'fontsize',12);
    grid on;
    xlabel('TR complexity/LR complexity');
    ylabel('MSE (overlap) / MSE (latent)');
    title('Comparison');
    pos=get(gca,'position');
    set(gca,'position',[pos(1) 0.15, pos(3), 0.7])
    ax(3)=gca;
  end
% $$$   figure;
% $$$   hold on;
% $$$   for ii=1:length(files)
% $$$     S=load(files{ii});
% $$$     plot(mean(S.err(:,:,1)), mean(S.err(:,:,2)), '-x', 'linewidth',2);
% $$$   end
end

set(gcf,'position',[-368, 31, 1640, 628]);

% plot_nips2011 - plots figures for nips 2011 paper
%
% See also
%  plot_denoising, plot_threshold_vs_normalized_rank
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


if input('Plot noisy tensor decomposition?')==1
  tmps = {'result_full_sigma=0.01_50_50_20_*.mat',...
          'result_full_sigma=0.1_50_50_20_*.mat',...
          'result_full_sigma=0.01_100_100_50_*.mat',...
          'result_full_sigma=0.1_100_100_50_*.mat'};


  for ii=1:length(tmps)
    S=ls(tmps{ii},'-1');
    files=split(S,char(10))'
    
    lmd=input('lambda=');
    figure;
    [dim1,dim2,dim3,err,lambda]=plot_denoising(files,lmd(1),'bx');
    [dim1,dim2,dim3,err,lambda]=plot_denoising(files,lmd(2),'^g');
    [dim1,dim2,dim3,err,lambda]=plot_denoising(files,lmd(3),'ro');
    set(gca,'position',[0.13 0.15 0.775 0.8]);
    h=get(gca,'children');
    set(h(2),'color',[0 .5 0]);
    legend(h([3,2,1]),sprintf('\\lambda_M=%g/N',3*lmd(1)),...
           sprintf('\\lambda_M=%g/N',3*lmd(2)),...
           sprintf('\\lambda_M=%g/N',3*lmd(3)),...
           'Location','NorthWest');
    

    clear files, lmd;
  end
  keyboard;

end


if input('Plot tensor completion?')==1
  figure
  S=ls('result_50_50_20_*.mat','-1');
  files=split(S,char(10));
  [rs,frac,dim]=plot_threshold_vs_normalized_rank(files);
  h=get(gca,'children')
  h1=h(1);
  set(h1, 'color','r','marker','o','markersize',10);

  S=ls('result_100_100_50_*.mat','-1');
  files=split(S,char(10));
  [rs2,frac2,dim2]=plot_threshold_vs_normalized_rank(files);


  set(gca,'fontsize',16);
  grid on;


  h=get(gca,'children');
  % ix=find(cell2mat(foreach(@(x)isequal(x,'line'),get(h,'type'))));
  % h=h([1:16,18:28,30,31,17,29]);
  % set(gca,'children',h);
  % set(h(1),'marker','x','markersize',10);
  % set(h(2),'marker','o','color','r','markersize',10);
  set(h(1),'markersize',10);

  legend(h([2,1]),'size=[50 50 20]', 'size=[100 100 50]');
  set(gca,'position',[0.13 0.15 0.775 0.78]);
  
  xlabel('Normalized rank ||n^{-1}||_{1/2}||r||_{1/2}');
  ylabel('Fraction at err<=0.01')                           

end

%%%% matrix
figure

markers={'o','x','^','v'};
col=get(gca,'colororder');


pats = {'result_matrix_50_20_*.mat',...
        'result_matrix_100_40_*.mat',...
        'result_matrix_250_200*.mat'};

leg=repmat({[]},1,length(pats));
hh=zeros(1,length(pats));

for kk=1:length(pats)
S=ls(pats{kk},'-1');
files=split(S,char(10));
[rs,frac,dim]=plot_threshold_vs_normalized_rank(files);
h=get(gca,'children')
hh(kk)=h(1);

set(hh(kk), 'color',col(kk,:),'marker',markers{kk},'markersize',10);

load(files{1},'sz');
leg{kk}=sprintf('size=[%d %d]',sz(1),sz(2));

end

legend(hh, leg);
set(gca,'fontsize',16,...
        'position',[0.13 0.15 0.775 0.78]);
grid on;
ylim([0 1]);
xlabel('Normalized rank ||n^{-1}||_{1/2}||r||_{1/2}');
ylabel('Fraction at err<=0.01')                           

% plot_nips2011_final - undocumented
% 
% See also
%  plot_nips2011, plot_denoising
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


S1=load('result_compare5_50_50_20_7_8_9.mat');
figure, h=errorbar(S1.trfrac'*[1 1], shiftdim(mean(S1.err(:,:,[4,7]))), ...
                   shiftdim(std(S1.err(:,:,[4,7]))));
set(h,'linewidth',2);
set(h(2), 'linestyle','-.');
set(gca,'yscale','log', 'fontsize', 16, 'ytick', [1e-3 1]);

hold on;
plot(xlim, 1e-3*[1 1], '--', 'color', [.5 .5 .5],'linewidth',2);

ylim([1e-4 100]);
grid on;

legend('Convex', 'Tucker (exact)','Optimization tolerance','location','NorthEastOutside');

xlabel('Fraction of observed elements')
ylabel('Estimation error')

set(gcf,'PaperSize',[20 20]);

keyboard;

% $$$ patterns = {'result_full_sigma=0.01_50_50_20_*.mat',...
% $$$             'result_full_sigma=0.1_50_50_20_*.mat';...
% $$$             'result_full_sigma=0.01_100_100_50_*.mat',...
% $$$             'result_full_sigma=0.1_100_100_50_*.mat'};
% $$$ 


% Low noise
patterns{1} = {'result_full_sigma=0.01_50_50_20_*.mat',...
            'result_full_sigma=0.01_50_50_20_*.mat',...
            'result_full_sigma=0.01_50_50_20_*.mat';...
            'result_full_lmd=0.02_0.367_sigma=0.01_100_100_50_*.mat',...
            'result_full_lmd=0.02_0.367_sigma=0.01_100_100_50_*.mat',...
            'result_full_lmd=0.02_0.367_sigma=0.01_100_100_50_*.mat'};

            


lambda{1} = [0.01, 0.11 0.18; 0.02, 0.23  0.37]


% High noise
patterns{2} = {'result_full_sigma=0.1_50_50_20_*.mat',...
            'result_full_sigma=0.1_50_50_20_*.mat',...
            'result_full_sigma=0.1_50_50_20_*.mat';...
            'result_full_lmd=0.22_4_sigma=0.1_100_100_50_*.mat',...
            'result_full_lmd=0.22_4_sigma=0.1_100_100_50_*.mat',...
            'result_full_lmd=0.22_4_sigma=0.1_100_100_50_*.mat'}



lambda{2} = [0.11 0.78 2; 0.22 1.5 4]


for kk=1:length(lambda)
  lmd = lambda{kk};
  pats = patterns{kk};
figure;
color  = {'b' 'r'};
marker = {'x', '^', 'o','+', 'v', '*'};

for ii=1:size(lmd,1)
  for jj=1:size(lmd,2)
  [ret,S]=system(sprintf('ls -1 %s', pats{ii,jj}));
  files=split(S,char(10))';

  style = [color{ii}, marker{jj+(ii-1)*3}];
  [dim1,dim2,dim3,err,lambda_out(:,ii,jj)]=plot_denoising(files,lmd(ii,jj),style);
  end
  set(gca,'position',[0.13 0.15 0.775 0.8]);


end

leg=cell(3,2);
for jj=1:3
leg{jj,1}=sprintf('size=[50 50 20] \\lambda_M=%g/N',3*lmd(1,jj));
end
for jj=1:3
  leg{jj,2}=sprintf('size=[100 100 50] \\lambda_M=%g/N',3*lmd(2,jj));
end

h=get(gca,'children');
legend(h(6:-1:1),leg{:});
set(gca,'position',[0.1300    0.1000    0.7750    0.8000]);
set(gcf,'position',[156    55   694   610]);
shiftdim(median(lambda_out))
end



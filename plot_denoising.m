% plot_denoising - plot MSE against normalized rank
%
% Syntax
%  [dim1, dim2, dim3,err, lambda]=plot_denoising(files, lmd, style)
%
% See also
%  exp_denoising, plot_nips2011, plot_nips2011_final
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


%
%
function [dim1, dim2, dim3,err, lambda]=plot_denoising(files, lmd, style)

if ~exist('style','var')
  style='x';
end


ns=length(files);

frac=zeros(ns,1);

err=zeros(ns,1);
nn =zeros(ns,1);
for ii=1:ns
  S=load(files{ii});
  sz(ii,:)=S.sz;
  rs(ii,:)=S.dtrue;
  
  if exist('lmd','var')
    [mm,ix]=min((log(S.lambda)-log(lmd)).^2);
    err(ii)=mean(S.err(:,ix).^2)/prod(S.sz);
    lambda(ii)=S.lambda(ix);
  else
    [err(ii), ix]=min(mean(S.err.^2)/prod(S.sz));
    lambda(ii)=S.lambda(ix);
  end
end

fprintf('rangeof(lambda)=%s\n',printvec(rangeof(lambda)));

nn=prod(sz,2);

if size(rs,2)>1
rst = [min(rs(:,1), rs(:,2).*rs(:,3)),...
       min(rs(:,2), rs(:,1).*rs(:,3)),...
       min(rs(:,3), rs(:,1).*rs(:,2))];
else
  rst=[rs,rs];
end


% dim=rs*[50 50 20]'+rs(:,1).*rs(:,2).*rs(:,3) -sum(rs.^2,2);
%dim=max(rst,[],2);%sum(rst,2);
% dim=sum(rst,2);
% dim=sum(sqrt(rst./(ones(ns,1)*sz)),2).^2/(size(sz,2)^2*size(rst,2)^2);
dim1=(mean(sqrt(1./sz),2).*mean(sqrt(rst),2)).^2;
dim2=sum(sqrt(sz)+sqrt((nn*ones(1,3))./sz),2);
dim3=sum(sqrt(rst)/size(rst,2),2).^2;
%dim=nm';
% figure;
plot(dim1,err,style,'linewidth',2,'markersize',10);

if 0 % size(rs,2)>1
for ii=1:ns
  text(dim(ii),frac(ii),...
       sprintf('[%d %d %d]',rs(ii,1),rs(ii,2),rs(ii,3)));
end
end

ix=find(~isnan(err));
p=polyfit(dim1(ix),err(ix),1)
hold on;
plot(xlim,polyval(p,xlim),'--','color',[.5 .5 .5],'linewidth', 2)
h=get(gca,'children');
set(gca,'children',h([2:end,1]));

set(gca,'fontsize',16)
xlabel('Normalized rank')
ylabel('Mean squared error')

grid on;



